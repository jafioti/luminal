use std::sync::Arc;

use cudarc::{
    cublas::{sys::cublasOperation_t::*, CudaBlas, Gemm, GemmConfig, StridedBatchedConfig},
    driver::{CudaDevice, CudaSlice},
};
use petgraph::stable_graph::NodeIndex;

use crate::{op::Operator, prelude::*};

use super::prim::{CudaMul, CudaSumReduce};

/// Multiplies a MxK matrix with a KxN matrix, resulting in a MxN matrix
#[derive(Debug, Clone)]
pub struct CudaMatmul2D(CudaBlas, Arc<CudaDevice>);
impl PartialEq for CudaMatmul2D {
    fn eq(&self, _: &Self) -> bool {
        false
    }
}

impl Operator for CudaMatmul2D {
    fn process(
        &self,
        inp: Vec<(
            crate::op::InputTensor,
            crate::core::shape::simple_tracker::ShapeTracker,
        )>,
    ) -> Vec<Tensor> {
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
            .downcast_ref::<CudaSlice<f32>>()
            .unwrap();
        let b = inp[1]
            .0
            .borrowed()
            .data
            .as_any()
            .downcast_ref::<CudaSlice<f32>>()
            .unwrap();
        let mut out = self.1.alloc_zeros::<f32>((m * n) as usize).unwrap();
        let (a_row_major, b_row_major) = (a_strides[0] > a_strides[1], b_strides[0] > b_strides[1]);
        let (transa, transb) = match (a_row_major, b_row_major) {
            (true, true) => (CUBLAS_OP_N, CUBLAS_OP_N),
            (false, false) => (CUBLAS_OP_T, CUBLAS_OP_T),
            (false, true) => (CUBLAS_OP_N, CUBLAS_OP_T),
            (true, false) => (CUBLAS_OP_T, CUBLAS_OP_N),
        };
        unsafe {
            self.0
                .gemm(
                    GemmConfig {
                        transa,
                        transb,
                        m: n,
                        n: m,
                        k,
                        alpha: 1.0,
                        lda: if b_row_major { n } else { k },
                        ldb: if a_row_major { k } else { m },
                        beta: 0.0,
                        ldc: n,
                    },
                    b,
                    a,
                    &mut out,
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
    fn process(
        &self,
        inp: Vec<(
            crate::op::InputTensor,
            crate::core::shape::simple_tracker::ShapeTracker,
        )>,
    ) -> Vec<Tensor> {
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
            .downcast_ref::<CudaSlice<f32>>()
            .unwrap();
        let b = inp[1]
            .0
            .borrowed()
            .data
            .as_any()
            .downcast_ref::<CudaSlice<f32>>()
            .unwrap();
        let mut out = self
            .1
            .alloc_zeros::<f32>((m * n * batch_size) as usize)
            .unwrap();
        let (a_row_major, b_row_major) = (a_strides[1] > a_strides[2], b_strides[0] > b_strides[1]);
        let (transa, transb) = match (a_row_major, b_row_major) {
            (true, true) => (CUBLAS_OP_N, CUBLAS_OP_N),
            (false, false) => (CUBLAS_OP_T, CUBLAS_OP_T),
            (false, true) => (CUBLAS_OP_N, CUBLAS_OP_T),
            (true, false) => (CUBLAS_OP_T, CUBLAS_OP_N),
        };
        unsafe {
            self.0
                .gemm_strided_batched(
                    StridedBatchedConfig {
                        gemm: GemmConfig {
                            transa,
                            transb,
                            m: n,
                            n: m,
                            k,
                            alpha: 1.0,
                            lda: if b_row_major { n } else { k },
                            ldb: if a_row_major { k } else { m },
                            beta: 0.0,
                            ldc: n,
                        },
                        batch_size,
                        stride_a: 0,
                        stride_b: a_strides[0] as i64,
                        stride_c: (m * n) as i64,
                    },
                    b,
                    a,
                    &mut out,
                )
                .unwrap();
        }

        vec![Tensor {
            data: Box::new(out),
        }]
    }
}

#[derive(Default)]
pub struct CudaMatMulOptimizer;

impl GraphOptimizer for CudaMatMulOptimizer {
    fn optimize(&self, graph: &mut Graph) {
        let dev = CudaDevice::new(0).unwrap();
        // Look for the matmul pattern
        let s = GraphSelector::default();
        let (mut sum_reduce, mut mul) = (NodeIndex::default(), NodeIndex::default());
        // Mul ([A, C(fake), B] | [A(fake), C, B]) -> SumReduce(2) -> [A, C]
        // Actually starts at [A,B] | [B, C]
        s.edge(
            s.op()
                .ty::<CudaMul>()
                .shapes(vec![
                    vec![Dim::Unknown('A'), Dim::Unknown('C'), Dim::Unknown('B')],
                    vec![Dim::Unknown('A'), Dim::Unknown('C'), Dim::Unknown('B')],
                ])
                .fakes(vec![vec![false, true, false], vec![true, false, false]])
                .ptr(&mut mul),
            0,
            s.op()
                .ty::<CudaSumReduce>()
                .check(|o| {
                    if let Some(o) = o.as_any().downcast_ref::<CudaSumReduce>() {
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
        let s = GraphSelector::default();
        let (mut sum_reduce, mut mul) = (NodeIndex::default(), NodeIndex::default());
        // Mul ([A, C(fake), B] | [A(fake), C, B]) -> SumReduce(2) -> [A, C]
        // Actually starts at [A,B] | [B, C]
        s.edge(
            s.op()
                .ty::<CudaMul>()
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
            0,
            s.op()
                .ty::<CudaSumReduce>()
                .check(|o| {
                    if let Some(o) = o.as_any().downcast_ref::<CudaSumReduce>() {
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
