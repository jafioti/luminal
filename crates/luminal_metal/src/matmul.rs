use std::{any::Any, marker::PhantomData, mem::size_of, sync::Arc};

use luminal::{
    op::{InputTensor, Operator},
    prelude::*,
    shape::symbolic::BigExpression,
};

use metal_rs::{objc::rc::autoreleasepool, *};

use crate::{
    compile_lib, get_buffer_from_tensor,
    prim::{MetalContiguous, MetalMul, MetalSumReduce},
    select_function_from_lib, MetalBuffer, MetalFloat, MetalKernel, MetalKernelWrapper, SetInt,
};

/// Multiplies a BxMxK matrix with a KxN matrix, resulting in a BxMxN matrix
#[derive(LuminalEqFalse, LuminalPrint, Clone)]
pub struct Matmul<T> {
    matmul_pipeline: ComputePipelineState,
    matvec_pipeline: ComputePipelineState,
    queue: CommandQueue,
    device: Device,
    _phantom: PhantomData<T>,
}

const BM: u64 = 8;
const BN: u64 = 32;
impl<T> MetalKernel for Matmul<T> {
    fn output_buffer_sizes(&self, input_shapes: &[ShapeTracker]) -> Vec<BigExpression> {
        let m = input_shapes[0].shape()[input_shapes[0].len() - 2].clone();
        let n = input_shapes[1].shape()[input_shapes[1].len() - 1].clone();
        let batch_size = input_shapes[0]
            .shape()
            .into_iter()
            .take(input_shapes[0].len() - 2)
            .product::<BigExpression>()
            .max(BigExpression::from(1));
        vec![batch_size * m * n * size_of::<T>()]
    }
    fn metal_forward(
        &self,
        inputs: &[(&Buffer, ShapeTracker)],
        command_buffer: &CommandBufferRef,
        _: &[&Buffer],
        output_buffers: &[&Buffer],
    ) {
        let (a_shape, b_shape) = (
            inputs[0]
                .1
                .shape()
                .into_iter()
                .map(|i| i.to_usize().unwrap())
                .collect::<Vec<_>>(),
            inputs[1]
                .1
                .shape()
                .into_iter()
                .map(|i| i.to_usize().unwrap())
                .collect::<Vec<_>>(),
        );
        let a_dims = a_shape.len();
        let m = a_shape[a_dims - 2];
        let batch_size = a_shape.iter().take(a_dims - 2).product::<usize>().max(1);
        let b_batch_size = b_shape
            .iter()
            .enumerate()
            .take(b_shape.len() - 2)
            .filter(|(i, _)| !inputs[1].1.fake[inputs[1].1.indexes[*i]])
            .map(|(_, i)| *i)
            .product::<usize>()
            .max(1);
        let b_dims = b_shape.len();
        let k = b_shape[b_dims - 2];
        let n = b_shape[b_dims - 1];

        let encoder =
            command_buffer.compute_command_encoder_with_descriptor(ComputePassDescriptor::new());
        if m == 1 && batch_size == 1 {
            // Matvec
            encoder.set_compute_pipeline_state(&self.matvec_pipeline);
            encoder.set_buffer(0, Some(inputs[1].0), 0);
            encoder.set_buffer(1, Some(inputs[0].0), 0);
            encoder.set_buffer(2, Some(output_buffers[0]), 0);
            encoder.set_i32(3, if m == 1 { k } else { m } as i32);
            encoder.set_i32(4, if m == 1 { n } else { m } as i32);
            encoder.set_i32(5, 0);
            encoder.set_i32(6, 0);
            encoder.set_threadgroup_memory_length(
                0,
                if inputs[1].1.indexes[inputs[1].1.len() - 1]
                    > inputs[1].1.indexes[inputs[1].1.len() - 2]
                {
                    BN * BM * 4
                } else {
                    BN * 8
                },
            );
            let b = if inputs[1].1.is_contiguous() { BN } else { BM };
            encoder.dispatch_thread_groups(
                MTLSize::new((n as u64 + b * 4 - 1).div_ceil(b * 4), 1, 1),
                MTLSize::new(BN, BM, 1),
            );
        } else {
            // Matmul
            encoder.set_compute_pipeline_state(&self.matmul_pipeline);

            // Set inputs
            encoder.set_buffer(0, Some(inputs[0].0), 0);
            encoder.set_buffer(1, Some(inputs[1].0), 0);
            encoder.set_buffer(2, Some(output_buffers[0]), 0);
            encoder.set_i32(3, m as i32);
            encoder.set_i32(4, n as i32);
            encoder.set_i32(5, k as i32);
            encoder.set_i32(6, (m * k) as i32); // A batch stride
            if inputs[1].1.len() > 2 // 3D or larger
                && inputs[1].1.fake[inputs[1].1.indexes[inputs[1].1.len() - 3]] // 3rd to last dimension is fake
                && inputs[1]
                    .1
                    .indexes
                    .iter()
                    .take(inputs[1].1.len().saturating_sub(4))
                    .any(|i| !inputs[1].1.fake[*i])
            // At least one non-fake dimension before 3rd to last
            {
                encoder.set_i32(7, (k * n) as i32); // B batch stride
                                                    // B batch size 2
                encoder.set_i32(8, b_shape[inputs[1].1.len() - 3] as i32);
            } else {
                encoder.set_i32(7, if b_batch_size == 1 { 0 } else { n * k } as i32); // B batch stride
                encoder.set_i32(8, 1); // B batch size
            }
            encoder.set_i32(9, (m * n) as i32); // C batch stride

            // Execute
            encoder.dispatch_thread_groups(
                MTLSize::new(
                    (n + 31).div_ceil(32) as u64,
                    (m + 31).div_ceil(32) as u64,
                    batch_size as u64,
                ),
                MTLSize::new(32, 2, 2),
            );
        }
        encoder.end_encoding();
    }
}

impl<T: 'static + Clone> Operator for Matmul<T> {
    fn process(&mut self, inp: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        autoreleasepool(|| {
            // Setup command queue / command buffer / encoder
            let command_buffer = self.queue.new_command_buffer();

            let (a_shape, b_shape) = (inp[0].1.shape(), inp[1].1.shape());
            let n = b_shape.last().unwrap().to_usize().unwrap();
            let batch_size = a_shape
                .iter()
                .map(|i| i.to_usize().unwrap())
                .take(a_shape.len() - 2)
                .product::<usize>();
            let m = a_shape[a_shape.len() - 2].to_usize().unwrap();

            let out = self.device.new_buffer(
                (batch_size * m * n * std::mem::size_of::<T>()) as u64,
                MTLResourceOptions::StorageModeShared,
            );

            self.metal_forward(
                &[
                    (get_buffer_from_tensor(&inp[0].0), inp[0].1),
                    (get_buffer_from_tensor(&inp[1].0), inp[1].1),
                ],
                command_buffer,
                &[],
                &[&out],
            );

            command_buffer.commit();
            command_buffer.wait_until_completed();

            vec![Tensor::new(MetalBuffer(out))]
        })
    }

    fn custom(&mut self, key: &str, _: Box<dyn Any>) -> Option<Box<dyn Any>> {
        if key == "metal" {
            return Some(Box::new(MetalKernelWrapper(Arc::new(Box::new(
                self.clone(),
            )))));
        }
        None
    }
}

#[derive(Default, Debug)]
pub struct MetalMatMulCompiler<T>(PhantomData<T>);

impl<T: MetalFloat> Compiler for MetalMatMulCompiler<T> {
    fn compile<To: ToIdsMut>(&self, graph: &mut Graph, mut remap: To) {
        let dev = Device::system_default().unwrap();
        let queue = dev.new_command_queue();

        // Look for the matmul pattern
        // Mul ([A, C(fake), B] | [A(fake), C, B]) -> SumReduce(2) -> [A, C]
        // Actually starts at [A,B] | [B, C]
        let mut mul2d = op::<MetalMul<T>>();
        mul2d.shapes([['M', 'N', 'K'], ['M', 'N', 'K']]);
        mul2d.fakes([
            [None, Some(true), Some(false)],
            [Some(true), Some(false), Some(false)],
        ]);
        let mut sr2d = op::<MetalSumReduce<T>>();
        sr2d.check(|o, _| {
            if let Some(o) = o.as_any().downcast_ref::<MetalSumReduce<T>>() {
                o.dim == 2
            } else {
                false
            }
        });
        let mut s2d = mul2d.clone().connect(sr2d.clone()).search(graph);
        let mut mul3d = op::<MetalMul<T>>();
        mul3d.shapes([['D', 'A', 'C', 'B'], ['D', 'A', 'C', 'B']]);
        mul3d.fakes([
            [Some(false), Some(false), Some(true), Some(false)],
            [None, Some(true), Some(false), Some(false)],
        ]);
        let mut sr3d = op::<MetalSumReduce<T>>();
        sr3d.check(|o, _| {
            if let Some(o) = o.as_any().downcast_ref::<MetalSumReduce<T>>() {
                o.dim == 2
            } else {
                false
            }
        });
        let mut s3d = mul3d.clone().connect(sr3d.clone()).search(graph);
        let mut mul4d = op::<MetalMul<T>>();
        mul4d.shapes([['E', 'D', 'A', 'C', 'B'], ['E', 'D', 'A', 'C', 'B']]);
        mul4d.fakes([
            [
                Some(false),
                Some(false),
                Some(false),
                Some(true),
                Some(false),
            ],
            [None, None, Some(true), Some(false), Some(false)],
        ]);
        let mut sr4d = op::<MetalSumReduce<T>>();
        sr4d.check(|o, _| {
            if let Some(o) = o.as_any().downcast_ref::<MetalSumReduce<T>>() {
                o.dim == 2
            } else {
                false
            }
        });
        let mut s4d = mul4d.clone().connect(sr4d.clone()).search(graph);
        let mut mul5d = op::<MetalMul<T>>();
        mul5d.shapes([
            ['F', 'E', 'D', 'A', 'C', 'B'],
            ['F', 'E', 'D', 'A', 'C', 'B'],
        ]);
        mul5d.fakes([
            [
                Some(false),
                Some(false),
                Some(false),
                Some(false),
                Some(true),
                Some(false),
            ],
            [None, None, None, Some(true), Some(false), Some(false)],
        ]);
        let mut sr5d = op::<MetalSumReduce<T>>();
        sr5d.check(|o, _| {
            if let Some(o) = o.as_any().downcast_ref::<MetalSumReduce<T>>() {
                o.dim == 2
            } else {
                false
            }
        });
        let mut s5d = mul5d.clone().connect(sr5d.clone()).search(graph);
        let matmul_library = compile_lib(&dev, include_str!("kernels/gemm.metal"));
        let matvec_library = compile_lib(&dev, include_str!("kernels/gemv.metal"));
        while s2d.next_match() || s3d.next_match() || s4d.next_match() || s5d.next_match() {
            let (mul, sum_reduce) = if s2d.matched {
                (s2d.get(&mul2d), s2d.get(&sr2d))
            } else if s3d.matched {
                (s3d.get(&mul3d), s3d.get(&sr3d))
            } else if s4d.matched {
                (s4d.get(&mul4d), s4d.get(&sr4d))
            } else {
                (s5d.get(&mul5d), s5d.get(&sr5d))
            };
            if graph.no_delete.contains(&mul) {
                // The intermediate mul can't be deleted
                continue;
            }
            // Insert Matmul op
            let srcs = graph.get_sources(mul);
            let (mut src1, mut src1_shape) = (srcs[0].0, srcs[0].2);
            let (mut src2, mut src2_shape) = (srcs[1].0, srcs[1].2);
            // Undo expansions and permute
            src1_shape.remove_dim(src1_shape.len() - 2);
            src2_shape.remove_dim(src2_shape.len() - 3);
            let mut dims = (0..src2_shape.len()).collect::<Vec<_>>();
            dims.swap(src2_shape.len() - 2, src2_shape.len() - 1);
            src2_shape.permute(&dims);
            // If src1 is padded or sliced, or batch dim isn't first, we need to make it contiguous
            if src1_shape
                .indexes
                .iter()
                .take(src1_shape.len() - 2)
                .enumerate()
                .any(|(a, b)| a != *b)
                || src1_shape.is_sliced()
                || src1_shape.is_padded()
            {
                src1 = graph
                    .add_op(MetalContiguous::<T>::new(
                        src1_shape,
                        dev.clone(),
                        queue.clone(),
                        &graph.dyn_map,
                    ))
                    .input(src1, 0, src1_shape)
                    .finish();
                src1_shape = src1_shape.contiguous();
            }
            // If src2 is padded or sliced, or batch dim isn't first, we need to make it contiguous
            if src2_shape
                .indexes
                .iter()
                .take(src2_shape.len() - 2)
                .filter(|i| !src2_shape.fake[**i])
                .enumerate()
                .any(|(a, b)| a != *b)
                || src2_shape.is_sliced()
                || src2_shape.is_padded()
            {
                src2 = graph
                    .add_op(MetalContiguous::<T>::new(
                        src2_shape,
                        dev.clone(),
                        queue.clone(),
                        &graph.dyn_map,
                    ))
                    .input(src2, 0, src2_shape)
                    .finish();
                src2_shape = src2_shape.contiguous();
            }
            let type_name = if T::is_f32() { "float32" } else { "float16" };
            let matmul_op = graph
                .add_op(Matmul::<T> {
                    matmul_pipeline: select_function_from_lib(
                        &matmul_library,
                        &format!( "gemm_{}{}_{type_name}_{type_name}_bm32_bn32_bk16_wm2_wn2_MN_naligned_K_taligned", if src1_shape.is_contiguous() {"n"} else {"t"}, if src2_shape.indexes[src2_shape.len() - 1] > src2_shape.indexes[src2_shape.len() - 2] {"n"} else {"t"}),
                        &dev
                    ),
                    matvec_pipeline: select_function_from_lib(
                        &matvec_library,
                        &format!(
                            "gemv_{}{type_name}_bm{BM}_bn{BN}_tm4_tn4",
                            if src2_shape.indexes[src2_shape.len() - 1] > src2_shape.indexes[src2_shape.len() - 2] { "t_" } else { "" }
                        ),
                        &dev
                    ),
                    queue: queue.clone(),
                    device: dev.clone(),
                    _phantom: Default::default()
                })
                .input(src1, 0, src1_shape)
                .input(src2, 0, src2_shape)
                .finish();

            // Create edges to dests
            move_outgoing_edge(sum_reduce, matmul_op, &mut graph.graph);
            move_references(
                &mut remap,
                &mut graph.no_delete,
                &mut graph.to_retrieve,
                sum_reduce,
                matmul_op,
            );

            // Remove the old ops
            graph.graph.remove_node(mul);
            graph.graph.remove_node(sum_reduce);
        }
    }
}

#[cfg(test)]
mod tests {
    use dfdx::{
        tensor::TensorFromVec,
        tensor_ops::{PermuteTo, TryMatMul},
    };
    use luminal::{
        prelude::*,
        tests::{assert_close_precision, random_vec},
    };

    use crate::MetalCompiler;
    #[test]
    fn test_matrix_vector() {
        const M: usize = 53;
        const N: usize = 256;
        let mut cx = Graph::new();
        let (a_vec, b_mat) = (random_vec(M), random_vec(M * N));
        let mut a = cx.named_tensor::<R2<1, M>>("Vec").set(a_vec.clone());
        let mut b = cx.named_tensor::<R2<N, M>>("Mat").set(b_mat.clone());
        let mut c = a.matmul(b.permute()).retrieve();

        cx.compile(
            <(GenericCompiler, MetalCompiler<f16>)>::default(),
            (&mut a, &mut b, &mut c),
        );
        cx.execute();

        let d_dev = dfdx::tensor::Cpu::default();
        let d_a = d_dev.tensor_from_vec(a_vec, (dfdx::shapes::Const::<M>,));
        let d_b =
            d_dev.tensor_from_vec(b_mat, (dfdx::shapes::Const::<N>, dfdx::shapes::Const::<M>));
        let d_c = d_a.matmul(d_b.permute());

        assert_close_precision(&c.data(), &d_c.as_vec(), 2);
    }

    #[test]
    fn test_batch_matrix_vector() {
        const M: usize = 256;
        const N: usize = 256;
        let mut cx = Graph::new();
        let (a_vec, b_mat) = (random_vec(M), random_vec(M * N));
        let mut a = cx.named_tensor::<R3<1, 1, M>>("Vec").set(a_vec.clone());
        let mut b = cx.named_tensor::<R2<M, N>>("Mat").set(b_mat.clone());
        let mut c = a.matmul(b).retrieve();

        cx.compile(
            <(GenericCompiler, MetalCompiler<f16>)>::default(),
            (&mut a, &mut b, &mut c),
        );
        cx.execute();

        let d_dev = dfdx::tensor::Cpu::default();
        let d_a = d_dev.tensor_from_vec(
            a_vec,
            (
                dfdx::shapes::Const::<1>,
                dfdx::shapes::Const::<1>,
                dfdx::shapes::Const::<M>,
            ),
        );
        let d_b =
            d_dev.tensor_from_vec(b_mat, (dfdx::shapes::Const::<M>, dfdx::shapes::Const::<N>));
        let d_c = d_a.matmul(d_b);

        assert_close_precision(&c.data(), &d_c.to_dtype::<f32>().as_vec(), 2);
    }
}
