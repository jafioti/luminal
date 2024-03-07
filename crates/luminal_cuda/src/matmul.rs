use std::{marker::PhantomData, sync::Arc};

use luminal_cudarc::{
    cublas::{sys::cublasOperation_t::*, CudaBlas},
    driver::{CudaDevice, DevicePtr, DevicePtrMut},
};

use crate::{
    prim::{CudaMul, CudaSumReduce},
    CudaData, CudaFloat,
};
use luminal::{
    op::{InputTensor, Operator},
    prelude::*,
};

/// Multiplies a MxK matrix with a KxN matrix, resulting in a MxN matrix
#[derive(LuminalPrint, LuminalEqFalse, Clone)]
pub struct CudaMatmul2D<T>(Arc<CudaBlas>, Arc<CudaDevice>, PhantomData<T>);

impl<T: CudaFloat + 'static> Operator for CudaMatmul2D<T>
where
    CudaData<T>: Data,
{
    fn process(&mut self, inp: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        let (a_shape, b_shape) = (inp[0].1.shape(), inp[1].1.shape());
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
            .downcast_ref::<CudaData<T>>()
            .unwrap();
        let b = inp[1]
            .0
            .borrowed()
            .data
            .as_any()
            .downcast_ref::<CudaData<T>>()
            .unwrap();
        let mut out = self.1.alloc_zeros::<T>((m * n) as usize).unwrap();
        let (a_row_major, b_row_major) = (
            inp[0].1.indexes[1] > inp[0].1.indexes[0],
            inp[1].1.indexes[1] > inp[1].1.indexes[0],
        );
        let (transa, transb) = match (a_row_major, b_row_major) {
            (true, true) => (CUBLAS_OP_N, CUBLAS_OP_N),
            (false, false) => (CUBLAS_OP_T, CUBLAS_OP_T),
            (false, true) => (CUBLAS_OP_N, CUBLAS_OP_T),
            (true, false) => (CUBLAS_OP_T, CUBLAS_OP_N),
        };
        if T::is_f32() {
            unsafe {
                luminal_cudarc::cublas::result::sgemm(
                    *self.0.handle(),
                    transa,
                    transb,
                    n,
                    m,
                    k,
                    &1.0_f32 as *const f32,
                    *b.0.device_ptr() as *const f32,
                    if b_row_major { n } else { k },
                    *a.0.device_ptr() as *const f32,
                    if a_row_major { k } else { m },
                    &0.0_f32 as *const f32,
                    *out.device_ptr_mut() as *mut f32,
                    n,
                )
                .unwrap();
            }
        } else {
            unsafe {
                luminal_cudarc::cublas::result::hgemm(
                    *self.0.handle(),
                    transa,
                    transb,
                    n,
                    m,
                    k,
                    &f16::from_f32(1.0) as *const f16,
                    *b.0.device_ptr() as *const f16,
                    if b_row_major { n } else { k },
                    *a.0.device_ptr() as *const f16,
                    if a_row_major { k } else { m },
                    &f16::from_f32(0.0) as *const f16,
                    *out.device_ptr_mut() as *mut f16,
                    n,
                )
                .unwrap();
            }
        }

        vec![Tensor {
            data: Box::new(CudaData(out)),
        }]
    }
}

/// Multiplies a BxMxK matrix with a BxKxN matrix, resulting in a BxMxN matrix
#[derive(LuminalPrint, LuminalEqFalse, Clone)]
pub struct CudaBatchMatmul2D<T>(Arc<CudaBlas>, Arc<CudaDevice>, PhantomData<T>);

impl<T: CudaFloat + 'static> Operator for CudaBatchMatmul2D<T>
where
    CudaData<T>: Data,
{
    fn process(&mut self, inp: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        let (a_shape, b_shape) = (inp[0].1.shape(), inp[1].1.shape());
        let a_strides = inp[0].1.strides();
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
            .downcast_ref::<CudaData<T>>()
            .unwrap();
        let b = inp[1]
            .0
            .borrowed()
            .data
            .as_any()
            .downcast_ref::<CudaData<T>>()
            .unwrap();
        let mut out = self
            .1
            .alloc_zeros::<T>((m * n * batch_size) as usize)
            .unwrap();
        let (a_row_major, b_row_major) = (
            inp[0].1.indexes[2] > inp[0].1.indexes[1],
            inp[1].1.indexes[1] > inp[1].1.indexes[0],
        );
        let (transa, transb) = match (a_row_major, b_row_major) {
            (true, true) => (CUBLAS_OP_N, CUBLAS_OP_N),
            (false, false) => (CUBLAS_OP_T, CUBLAS_OP_T),
            (false, true) => (CUBLAS_OP_N, CUBLAS_OP_T),
            (true, false) => (CUBLAS_OP_T, CUBLAS_OP_N),
        };
        if T::is_f32() {
            unsafe {
                luminal_cudarc::cublas::result::sgemm_strided_batched(
                    *self.0.handle(),
                    transa,
                    transb,
                    n,
                    m,
                    k,
                    &1.0_f32 as *const f32,
                    *b.0.device_ptr() as *const f32,
                    if b_row_major { n } else { k },
                    0,
                    *a.0.device_ptr() as *const f32,
                    if a_row_major { k } else { m },
                    a_strides[0].to_usize().unwrap() as i64,
                    &0.0_f32 as *const f32,
                    *out.device_ptr_mut() as *mut f32,
                    n,
                    (m * n) as i64,
                    batch_size,
                )
                .unwrap();
            }
        } else {
            unsafe {
                luminal_cudarc::cublas::result::hgemm_strided_batched(
                    *self.0.handle(),
                    transa,
                    transb,
                    n,
                    m,
                    k,
                    &f16::from_f32(1.0) as *const f16,
                    *b.0.device_ptr() as *const f16,
                    if b_row_major { n } else { k },
                    0,
                    *a.0.device_ptr() as *const f16,
                    if a_row_major { k } else { m },
                    a_strides[0].to_usize().unwrap() as i64,
                    &f16::from_f32(0.0) as *const f16,
                    *out.device_ptr_mut() as *mut f16,
                    n,
                    (m * n) as i64,
                    batch_size,
                )
                .unwrap();
            }
        }

        vec![Tensor {
            data: Box::new(CudaData(out)),
        }]
    }
}

#[derive(Default)]
pub struct CudaMatMulCompiler<T>(PhantomData<T>);

impl<T: CudaFloat + 'static> Compiler for CudaMatMulCompiler<T>
where
    CudaData<T>: Data,
{
    fn compile<To: ToIdsMut>(&self, graph: &mut Graph, mut remap: To) {
        let dev = CudaDevice::new(0).unwrap();
        // Look for the matmul pattern
        // Mul ([A, C(fake), B] | [A(fake), C, B]) -> SumReduce(2) -> [A, C]
        // Actually starts at [A,B] | [B, C]
        let mut mul = op::<CudaMul<T>>();
        mul.shapes([['A', 'C', 'B'], ['A', 'C', 'B']]);
        mul.fakes([
            [Some(false), Some(true), Some(false)],
            [Some(true), Some(false), Some(false)],
        ]);
        let mut sum_reduce = unary::<CudaSumReduce<T>>(mul.clone());
        sum_reduce.check(|o, _| {
            if let Some(c) = o.as_any().downcast_ref::<CudaSumReduce<T>>() {
                c.dim == 0
            } else {
                false
            }
        });
        let mut s = sum_reduce.clone().search(graph);
        while s.next_match() {
            if s.check_no_delete(&[sum_reduce.id]) {
                // The intermediate mul can't be deleted
                continue;
            }
            let (mul, sum_reduce) = (s.get(&mul), s.get(&sum_reduce));
            // Insert MatMul2D op
            let mut srcs = graph.get_sources(mul);
            // Undo expansions and permute
            srcs[0].2.remove_dim(1);
            srcs[1].2.remove_dim(0);
            srcs[1].2.permute(&[1, 0]);
            let new_op = graph
                .add_op(CudaMatmul2D::<T>(
                    Arc::new(CudaBlas::new(dev.clone()).unwrap()),
                    dev.clone(),
                    Default::default(),
                ))
                .input(srcs[0].0, 0, srcs[0].2)
                .input(srcs[1].0, 0, srcs[1].2)
                .finish();

            // Create edges to dests
            move_outgoing_edge(sum_reduce, new_op, &mut graph.graph);
            move_references(
                &mut remap,
                &mut graph.no_delete,
                &mut graph.to_retrieve,
                sum_reduce,
                new_op,
            );
            move_references(
                &mut remap,
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
        // Mul ([A, C(fake), B] | [A(fake), C, B]) -> SumReduce(2) -> [A, C]
        // Actually starts at [A,B] | [B, C]
        let mut mul = op::<CudaMul<T>>();
        mul.shapes([['D', 'A', 'C', 'B'], ['D', 'A', 'C', 'B']]);
        mul.fakes([
            [Some(false), Some(false), Some(true), Some(false)],
            [Some(true), Some(true), Some(false), Some(false)],
        ]);
        let mut sum_reduce = unary::<CudaSumReduce<T>>(mul.clone());
        sum_reduce.check(|o, _| {
            if let Some(c) = o.as_any().downcast_ref::<CudaSumReduce<T>>() {
                c.dim == 3
            } else {
                false
            }
        });
        let mut s = sum_reduce.clone().search(graph);
        while s.next_match() {
            if s.check_no_delete(&[sum_reduce.id]) {
                // The intermediate mul can't be deleted
                continue;
            }
            let (mul, sum_reduce) = (s.get(&mul), s.get(&sum_reduce));
            // Insert BatchMatMul2D op
            let mut srcs = graph.get_sources(mul);
            // Undo expansions and permute
            srcs[0].2.remove_dim(2);
            srcs[1].2.remove_dim(1);
            srcs[1].2.remove_dim(0);
            srcs[1].2.permute(&[1, 0]);
            let new_op = graph
                .add_op(CudaBatchMatmul2D::<T>(
                    Arc::new(CudaBlas::new(dev.clone()).unwrap()),
                    dev.clone(),
                    Default::default(),
                ))
                .input(srcs[0].0, 0, srcs[0].2)
                .input(srcs[1].0, 0, srcs[1].2)
                .finish();

            // Create edges to dests
            move_outgoing_edge(sum_reduce, new_op, &mut graph.graph);
            move_references(
                &mut remap,
                &mut graph.no_delete,
                &mut graph.to_retrieve,
                sum_reduce,
                new_op,
            );
            move_references(
                &mut remap,
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
