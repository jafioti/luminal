use std::sync::Arc;

use cudarc::{
    cublas::{sys::cublasOperation_t::*, CudaBlas},
    driver::{CudaDevice, CudaSlice, DevicePtr, DevicePtrMut},
};
use half::f16;
use petgraph::stable_graph::NodeIndex;

use crate::{
    op::{InputTensor, Operator},
    prelude::*,
};

/// Multiplies a MxK matrix with a KxN matrix, resulting in a MxN matrix
#[derive(Debug, Clone)]
pub struct CudaMatmul2D(CudaBlas, Arc<CudaDevice>);
impl PartialEq for CudaMatmul2D {
    fn eq(&self, _: &Self) -> bool {
        false
    }
}

impl Operator for CudaMatmul2D {
    fn process(&self, inp: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        let (a_shape, b_shape) = (inp[0].1.shape(), inp[1].1.shape());
        let (a_strides, b_strides) = (inp[0].1.strides(), inp[1].1.strides());
        let (m, k, n) = (
            a_shape[0].to_usize().unwrap() as i32,
            a_shape[1].to_usize().unwrap() as i32,
            b_shape[1].to_usize().unwrap() as i32,
        );
        let a = inp[0]
            .0
            .borrowed()
            .data
            .as_any()
            .downcast_ref::<CudaSlice<f16>>()
            .unwrap();
        let b = inp[1]
            .0
            .borrowed()
            .data
            .as_any()
            .downcast_ref::<CudaSlice<f16>>()
            .unwrap();
        let mut out = self.1.alloc_zeros::<f16>((m * n) as usize).unwrap();
        let (a_row_major, b_row_major) = (a_strides[0] > a_strides[1], b_strides[0] > b_strides[1]);
        let (transa, transb) = match (a_row_major, b_row_major) {
            (true, true) => (CUBLAS_OP_N, CUBLAS_OP_N),
            (false, false) => (CUBLAS_OP_T, CUBLAS_OP_T),
            (false, true) => (CUBLAS_OP_N, CUBLAS_OP_T),
            (true, false) => (CUBLAS_OP_T, CUBLAS_OP_N),
        };
        unsafe {
            cudarc::cublas::result::hgemm(
                *self.0.handle(),
                transa,
                transb,
                n,
                m,
                k,
                &f16::from_f32(1.0) as *const f16,
                *b.device_ptr() as *const f16,
                if b_row_major { n } else { k },
                *a.device_ptr() as *const f16,
                if a_row_major { k } else { m },
                &f16::from_f32(0.0) as *const f16,
                *out.device_ptr_mut() as *mut f16,
                n,
            )
            .unwrap();
        }

        vec![Tensor {
            data: Box::new(out),
        }]
    }
}

/// Multiplies a BxMxK matrix with a BxKxN matrix, resulting in a BxMxN matrix
#[derive(Debug, Clone)]
pub struct CudaBatchMatmul2D(CudaBlas, Arc<CudaDevice>);
impl PartialEq for CudaBatchMatmul2D {
    fn eq(&self, _: &Self) -> bool {
        false
    }
}

impl Operator for CudaBatchMatmul2D {
    fn process(&self, inp: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        let (a_shape, b_shape) = (inp[0].1.shape(), inp[1].1.shape());
        let (a_strides, b_strides) = (inp[0].1.strides(), inp[1].1.strides());
        let (batch_size, m, k, n) = (
            a_shape[0].to_usize().unwrap() as i32,
            a_shape[1].to_usize().unwrap() as i32,
            a_shape[2].to_usize().unwrap() as i32,
            b_shape[1].to_usize().unwrap() as i32,
        );
        let a = inp[0]
            .0
            .borrowed()
            .data
            .as_any()
            .downcast_ref::<CudaSlice<f16>>()
            .unwrap();
        let b = inp[1]
            .0
            .borrowed()
            .data
            .as_any()
            .downcast_ref::<CudaSlice<f16>>()
            .unwrap();
        let mut out = self
            .1
            .alloc_zeros::<f16>((m * n * batch_size) as usize)
            .unwrap();
        let (a_row_major, b_row_major) = (a_strides[1] > a_strides[2], b_strides[0] > b_strides[1]);
        let (transa, transb) = match (a_row_major, b_row_major) {
            (true, true) => (CUBLAS_OP_N, CUBLAS_OP_N),
            (false, false) => (CUBLAS_OP_T, CUBLAS_OP_T),
            (false, true) => (CUBLAS_OP_N, CUBLAS_OP_T),
            (true, false) => (CUBLAS_OP_T, CUBLAS_OP_N),
        };
        unsafe {
            cudarc::cublas::result::hgemm_strided_batched(
                *self.0.handle(),
                transa,
                transb,
                n,
                m,
                k,
                &f16::from_f32(1.0) as *const f16,
                *b.device_ptr() as *const f16,
                if b_row_major { n } else { k },
                0,
                *a.device_ptr() as *const f16,
                if a_row_major { k } else { m },
                a_strides[0] as i64,
                &f16::from_f32(0.0) as *const f16,
                *out.device_ptr_mut() as *mut f16,
                n,
                (m * n) as i64,
                batch_size,
            )
            .unwrap();
        }

        vec![Tensor {
            data: Box::new(out),
        }]
    }
}

#[derive(Default)]
pub struct CudaMatMulCompiler;

impl Compiler for CudaMatMulCompiler {
    fn compile(&self, graph: &mut Graph) {
        let dev = CudaDevice::new(0).unwrap();
        // Look for the matmul pattern
        let (mut sum_reduce, mut mul) = (NodeIndex::default(), NodeIndex::default());
        // Mul ([A, C(fake), B] | [A(fake), C, B]) -> SumReduce(2) -> [A, C]
        // Actually starts at [A,B] | [B, C]
        let s = SelectEdge::new(
            SelectOp::new()
                .ty::<CudaMul<f16>>()
                .shapes(vec![
                    vec![Dim::Unknown('A'), Dim::Unknown('C'), Dim::Unknown('B')],
                    vec![Dim::Unknown('A'), Dim::Unknown('C'), Dim::Unknown('B')],
                ])
                .fakes(vec![vec![false, true, false], vec![true, false, false]])
                .ptr(&mut mul),
            SelectOp::new()
                .ty::<CudaSumReduce<f16>>()
                .check(|o, _| {
                    if let Some(o) = o.as_any().downcast_ref::<CudaSumReduce<f16>>() {
                        o.2 == 2
                    } else {
                        false
                    }
                })
                .ptr(&mut sum_reduce),
        );
        for _ in s.search(graph) {
            if graph.no_delete.contains(&mul) {
                // The intermediate mul can't be deleted
                continue;
            }
            // Insert MatMul2D op
            let mut srcs = graph.get_sources(mul);
            // Undo expansions and permute
            srcs[0].1.remove_dim(1);
            srcs[1].1.remove_dim(0);
            srcs[1].1.permute(&[1, 0]);
            let new_op = graph
                .add_op(CudaMatmul2D(
                    CudaBlas::new(dev.clone()).unwrap(),
                    dev.clone(),
                ))
                .input(srcs[0].0, 0, srcs[0].1)
                .input(srcs[1].0, 0, srcs[1].1)
                .finish();

            // Create edges to dests
            move_outgoing_edge(sum_reduce, new_op, &mut graph.graph);
            move_references(
                &mut graph.id_remap,
                &mut graph.no_delete,
                &mut graph.to_retrieve,
                sum_reduce,
                new_op,
            );
            move_references(
                &mut graph.id_remap,
                &mut graph.no_delete,
                &mut graph.to_retrieve,
                mul,
                new_op,
            );

            // Remove the old ops
            graph.graph.remove_node(mul);
            graph.graph.remove_node(sum_reduce);
        }

        // Look for the batch matmul pattern
        let (mut sum_reduce, mut mul) = (NodeIndex::default(), NodeIndex::default());
        // Mul ([A, C(fake), B] | [A(fake), C, B]) -> SumReduce(2) -> [A, C]
        // Actually starts at [A,B] | [B, C]
        let s = SelectEdge::new(
            SelectOp::new()
                .ty::<CudaMul<f16>>()
                .shapes(vec![
                    vec![
                        Dim::Unknown('D'),
                        Dim::Unknown('A'),
                        Dim::Unknown('C'),
                        Dim::Unknown('B'),
                    ],
                    vec![
                        Dim::Unknown('D'),
                        Dim::Unknown('A'),
                        Dim::Unknown('C'),
                        Dim::Unknown('B'),
                    ],
                ])
                .fakes(vec![
                    vec![false, false, true, false],
                    vec![true, true, false, false],
                ])
                .ptr(&mut mul),
            SelectOp::new()
                .ty::<CudaSumReduce<f16>>()
                .check(|o, _| {
                    if let Some(o) = o.as_any().downcast_ref::<CudaSumReduce<f16>>() {
                        o.2 == 3
                    } else {
                        false
                    }
                })
                .ptr(&mut sum_reduce),
        );
        for _ in s.search(graph) {
            if graph.no_delete.contains(&mul) {
                // The intermediate mul can't be deleted
                continue;
            }
            // Insert BatchMatMul2D op
            let mut srcs = graph.get_sources(mul);
            // Undo expansions and permute
            srcs[0].1.remove_dim(2);
            srcs[1].1.remove_dim(1);
            srcs[1].1.remove_dim(0);
            srcs[1].1.permute(&[1, 0]);
            let new_op = graph
                .add_op(CudaBatchMatmul2D(
                    CudaBlas::new(dev.clone()).unwrap(),
                    dev.clone(),
                ))
                .input(srcs[0].0, 0, srcs[0].1)
                .input(srcs[1].0, 0, srcs[1].1)
                .finish();

            // Create edges to dests
            move_outgoing_edge(sum_reduce, new_op, &mut graph.graph);
            move_references(
                &mut graph.id_remap,
                &mut graph.no_delete,
                &mut graph.to_retrieve,
                sum_reduce,
                new_op,
            );
            move_references(
                &mut graph.id_remap,
                &mut graph.no_delete,
                &mut graph.to_retrieve,
                mul,
                new_op,
            );

            // Remove the old ops
            graph.graph.remove_node(mul);
            graph.graph.remove_node(sum_reduce);
        }
    }
}
