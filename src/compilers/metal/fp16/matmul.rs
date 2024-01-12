use std::{mem::size_of, sync::Arc};

use half::f16;
use petgraph::stable_graph::NodeIndex;

use crate::{
    compilers::metal::{prim::*, *},
    op::{InputTensor, Operator},
    prelude::*,
};

use metal_rs::{objc::rc::autoreleasepool, *};

/// Multiplies a M vector with a MxN matrix, resulting in a N vector. Expects the matrix to be NxM row-major
#[derive(LuminalEq, LuminalPrint, Clone)]
pub struct MetalVecMat {
    kernel: ComputePipelineState,
    queue: CommandQueue,
    device: Device,
}

const BM: u64 = 8;
const BN: u64 = 32;
impl MetalVecMat {
    fn new(dev: &Device, queue: CommandQueue) -> Self {
        Self {
            kernel: compile_function(
                "kernel_vecmat",
                &format!(
                    "
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
#include <metal_simdgroup>
using namespace metal;

#define BM {BM}
#define BN {BN}
#define TM 4
#define TN 4

kernel void kernel_vecmat(
    const device half4* in_vec [[buffer(0)]],
    const device half4* mat [[buffer(1)]],
    device half4* out_vec [[buffer(2)]],
    const constant int& in_vec_size [[buffer(3)]],
    const constant int& out_vec_size [[buffer(4)]],
    threadgroup half4* tgp_memory [[threadgroup(0)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]]) {{

    int in_vec_size_divided_by_4 = in_vec_size / 4;
    int out_vec_size_divided_by_4 = out_vec_size / 4;
    uint out_col = tid.x * BN + lid.x;

    // Thread local accumulation results
    half4 result = 0;

    // Threadgroup accumulation results
    threadgroup half4* tgp_results = tgp_memory + lid.x * BM;

    // Per thread accumulation main loop
    for(int bm = lid.y; bm < in_vec_size_divided_by_4; bm += BM) {{
        #pragma unroll(TM)
        for(int tm = 0; tm < TM; tm++) {{
            result += mat[(bm * 4 + tm) * out_vec_size_divided_by_4 + out_col] * in_vec[bm][tm];
        }}
    }}

    // Threadgroup collection
    tgp_results[lid.y] = result;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Threadgroup accumulation and writing out results
    if(lid.y == 0 && out_col * TN < out_vec_size) {{
        #pragma unroll(BM)
        for(int i = 1; i < BM; i++) {{
            result += tgp_results[i];
        }}

        out_vec[out_col] = result;
    }}
}}"
                ),
                dev,
            ),
            queue,
            device: dev.clone(),
        }
    }
}

impl MetalKernel for MetalVecMat {
    fn output_buffer_sizes(&self, input_shapes: &[ShapeTracker]) -> Vec<BigExpression> {
        vec![input_shapes[1].shape()[1].clone() * size_of::<f16>()]
    }

    fn metal_forward(
        &self,
        inputs: &[(&Buffer, ShapeTracker)],
        command_buffer: &CommandBufferRef,
        _: &[&Buffer],
        output_buffers: &[&Buffer],
    ) {
        let (m, n) = (
            inputs[0].1.shape()[0].to_usize().unwrap(),
            inputs[1].1.shape()[1].to_usize().unwrap(),
        );

        let encoder =
            command_buffer.compute_command_encoder_with_descriptor(ComputePassDescriptor::new());

        // Set inputs
        encoder.set_buffer(0, Some(inputs[0].0), 0);
        encoder.set_buffer(1, Some(inputs[1].0), 0);
        encoder.set_buffer(2, Some(output_buffers[0]), 0);
        encoder.set_u32(3, m as u32);
        encoder.set_u32(4, n as u32);
        encoder.set_threadgroup_memory_length(0, BN * BM * 8);

        encoder.set_compute_pipeline_state(&self.kernel);
        encoder.dispatch_thread_groups(
            MTLSize {
                width: (n as u64).div_ceil(BN * 4),
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: BN,
                height: BM,
                depth: 1,
            },
        );
        encoder.end_encoding();
    }
}

impl Operator for MetalVecMat {
    fn process(&mut self, inp: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        autoreleasepool(|| {
            // Setup command queue / command buffer / encoder
            let command_buffer = self.queue.new_command_buffer();

            let n = inp[1].1.shape()[1].to_usize().unwrap();

            let out = self.device.new_buffer(
                (n * std::mem::size_of::<f16>()) as u64,
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

            vec![Tensor::new(out)]
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

/// Multiplies a BxMxK matrix with a KxN matrix, resulting in a BxMxN matrix
#[derive(LuminalEq, LuminalPrint, Clone)]
pub struct Matmul(ComputePipelineState, CommandQueue, Device);

impl Matmul {
    fn compile(dev: &Device) -> Library {
        dev.new_library_with_source(include_str!("gemm.metal"), &CompileOptions::new())
            .unwrap()

        // compile_function(
        //     "gemm_nn_float16_float16_bm32_bn32_bk16_wm2_wn2_MN_naligned_K_taligned",
        //     include_str!("gemm.metal"),
        //     dev,
        // )
    }
}

impl MetalKernel for Matmul {
    fn output_buffer_sizes(&self, input_shapes: &[ShapeTracker]) -> Vec<BigExpression> {
        let n = input_shapes[1].shape()[1].clone();
        let (batch_size, m) = if input_shapes[0].len() == 3 {
            (
                input_shapes[0].shape()[0].clone(),
                input_shapes[0].shape()[1].clone(),
            )
        } else {
            (1.into(), input_shapes[0].shape()[0].clone())
        };
        vec![BigExpression::from(m) * n * batch_size * size_of::<f16>()]
    }
    fn metal_forward(
        &self,
        inputs: &[(&Buffer, ShapeTracker)],
        command_buffer: &CommandBufferRef,
        _: &[&Buffer],
        output_buffers: &[&Buffer],
    ) {
        let (a_shape, b_shape) = (inputs[0].1.shape(), inputs[1].1.shape());
        let (k, n) = (
            b_shape[0].to_usize().unwrap(),
            b_shape[1].to_usize().unwrap(),
        );
        let (batch_size, m) = if a_shape.len() == 3 {
            (
                a_shape[0].to_usize().unwrap(),
                a_shape[1].to_usize().unwrap(),
            )
        } else {
            (1, a_shape[0].to_usize().unwrap())
        };

        let encoder =
            command_buffer.compute_command_encoder_with_descriptor(ComputePassDescriptor::new());
        encoder.set_compute_pipeline_state(&self.0);

        // Set inputs
        encoder.set_buffer(0, Some(inputs[0].0), 0);
        encoder.set_buffer(1, Some(inputs[1].0), 0);
        encoder.set_buffer(2, Some(output_buffers[0]), 0);
        encoder.set_i32(3, m as i32);
        encoder.set_i32(4, n as i32);
        encoder.set_i32(5, k as i32);
        encoder.set_i32(6, (m * k) as i32);
        encoder.set_i32(7, 0);
        encoder.set_i32(8, (m * n) as i32);

        // Execute
        encoder.dispatch_thread_groups(
            MTLSize {
                width: (n + 32 - 1).div_ceil(32) as u64,
                height: (m + 32 - 1).div_ceil(32) as u64,
                depth: batch_size as u64,
            },
            MTLSize {
                width: 32,
                height: 2,
                depth: 2,
            },
        );
        encoder.end_encoding();
    }
}

impl Operator for Matmul {
    fn process(&mut self, inp: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        autoreleasepool(|| {
            // Setup command queue / command buffer / encoder
            let command_buffer = self.1.new_command_buffer();

            let (a_shape, b_shape) = (inp[0].1.shape(), inp[1].1.shape());
            let n = b_shape[1].to_usize().unwrap();
            let (batch_size, m) = if a_shape.len() == 3 {
                (
                    a_shape[0].to_usize().unwrap(),
                    a_shape[1].to_usize().unwrap(),
                )
            } else {
                (0, a_shape[0].to_usize().unwrap())
            };

            let out = self.2.new_buffer(
                (batch_size * m * n * std::mem::size_of::<f16>()) as u64,
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

            vec![Tensor::new(out)]
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
pub struct MetalMatMulCompiler;

impl Compiler for MetalMatMulCompiler {
    fn compile<T: ToIdsMut>(&self, graph: &mut Graph, mut remap: T) {
        let dev = Device::system_default().unwrap();
        let queue = dev.new_command_queue();
        let (mut sum_reduce, mut mul) = (NodeIndex::default(), NodeIndex::default());

        // Look for vetmat pattern
        // Mul ([1(fake), N(fake), M] | [1(fake), N, M]) -> SumReduce(2) -> [N]
        let vecmat_pattern = SelectOp::new()
            .ty::<MetalMul<f16>>()
            .shapes(vec![
                vec![1.into(), 'N'.into(), 'M'.into()],
                vec![1.into(), 'N'.into(), 'M'.into()],
            ])
            .fakes(vec![
                vec![None, Some(true), Some(false)],
                vec![Some(true), Some(false), Some(false)],
            ])
            .ptr(&mut mul)
            .edge(
                SelectOp::new()
                    .check(|o, _| {
                        if let Some(o) = o.as_any().downcast_ref::<MetalSumReduce<f16>>() {
                            o.dim == 2
                        } else {
                            false
                        }
                    })
                    .ptr(&mut sum_reduce),
            );
        let batch_vecmat_pattern = SelectOp::new()
            .ty::<MetalMul<f16>>()
            .shapes(vec![
                vec![1.into(), 1.into(), 'N'.into(), 'M'.into()],
                vec![1.into(), 1.into(), 'N'.into(), 'M'.into()],
            ])
            .fakes(vec![
                vec![None, None, Some(true), Some(false)],
                vec![None, Some(true), Some(false), Some(false)],
            ])
            .ptr(&mut mul)
            .edge(
                SelectOp::new()
                    .check(|o, _| {
                        if let Some(o) = o.as_any().downcast_ref::<MetalSumReduce<f16>>() {
                            o.dim == 3
                        } else {
                            false
                        }
                    })
                    .ptr(&mut sum_reduce),
            );
        // Mul ([1, 1(fake?), N(fake), M] | [1, 1(fake), N, M]) -> SumReduce(2) -> [N]
        let mut s1 = vecmat_pattern.search(graph);
        let mut s2 = batch_vecmat_pattern.search(graph);
        while s1.next_match() || s2.next_match() {
            if graph.no_delete.contains(&mul) {
                // The intermediate mul can't be deleted
                continue;
            }
            // Insert VecMat op
            let srcs = graph.get_sources(mul);
            let (src1, mut src1_shape) = (srcs[0].0, srcs[0].2);
            let (mut src2, mut src2_shape) = (srcs[1].0, srcs[1].2);
            // Undo expansions and permute
            if src1_shape.dims.len() == 4 {
                src1_shape.remove_dim(2);
            }
            if src2_shape.dims.len() == 4 {
                src2_shape.remove_dim(1);
            }
            src1_shape.remove_dim(1);
            src1_shape.remove_dim(0);
            src2_shape.remove_dim(0);
            src2_shape.permute(&[1, 0]);
            // Src1: [M], Src2: [N, M]
            if !src2_shape.is_contiguous() || src2_shape.is_sliced() || src2_shape.is_padded() {
                src2 = graph
                    .add_op(MetalContiguous::<f16>::new(
                        src2_shape,
                        dev.clone(),
                        queue.clone(),
                        &mut HashMap::new(),
                        &graph.dyn_map,
                    ))
                    .input(src2, 0, src2_shape)
                    .finish();
                src2_shape = src2_shape.contiguous();
            }

            let matmul_op = graph
                .add_op(MetalVecMat::new(&dev, queue.clone()))
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
        // Look for the matmul pattern
        // Mul ([A, C(fake), B] | [A(fake), C, B]) -> SumReduce(2) -> [A, C]
        // Actually starts at [A,B] | [B, C]
        let mut single_searcher = SelectOp::new()
            .ty::<MetalMul<f16>>()
            .shapes(vec![
                vec!['M'.into(), 'N'.into(), 'K'.into()],
                vec!['M'.into(), 'N'.into(), 'K'.into()],
            ])
            .fakes(vec![
                vec![Some(false), Some(true), Some(false)],
                vec![Some(true), Some(false), Some(false)],
            ])
            .ptr(&mut mul)
            .edge(
                SelectOp::new()
                    .check(|o, _| {
                        if let Some(o) = o.as_any().downcast_ref::<MetalSumReduce<f16>>() {
                            o.dim == 2
                        } else {
                            false
                        }
                    })
                    .ptr(&mut sum_reduce),
            )
            .search(graph);
        let mut batch_searcher = SelectOp::new()
            .ty::<MetalMul<f16>>()
            .shapes(vec![
                vec!['D'.into(), 'A'.into(), 'C'.into(), 'B'.into()],
                vec!['D'.into(), 'A'.into(), 'C'.into(), 'B'.into()],
            ])
            .fakes(vec![
                vec![Some(false), Some(false), Some(true), Some(false)],
                vec![Some(true), Some(true), Some(false), Some(false)],
            ])
            .ptr(&mut mul)
            .edge(
                SelectOp::new()
                    .ty::<MetalSumReduce<f16>>()
                    .check(|o, _| {
                        if let Some(o) = o.as_any().downcast_ref::<MetalSumReduce<f16>>() {
                            o.dim == 3
                        } else {
                            false
                        }
                    })
                    .ptr(&mut sum_reduce),
            )
            .search(graph);
        let matmul_library = Matmul::compile(&dev);
        while single_searcher.next_match() || batch_searcher.next_match() {
            if graph.no_delete.contains(&mul) {
                // The intermediate mul can't be deleted
                continue;
            }
            // Insert Matmul op
            let srcs = graph.get_sources(mul);
            let (mut src1, mut src1_shape) = (srcs[0].0, srcs[0].2);
            let (mut src2, mut src2_shape) = (srcs[1].0, srcs[1].2);
            // Undo expansions and permute
            src1_shape.remove_dim(if src1_shape.len() == 4 { 2 } else { 1 });
            if src2_shape.len() == 4 {
                src2_shape.remove_dim(1);
            }
            src2_shape.remove_dim(0);
            src2_shape.permute(&[1, 0]);
            // Pad out N to multiple of 256 and K to 16
            let n_dim = Expression::from(src2_shape.shape()[1].clone());
            let k_dim = Expression::from(src2_shape.shape()[0].clone());
            let m_dim = if src1_shape.len() == 3 {
                Expression::from(src1_shape.shape()[1].clone())
            } else {
                Expression::from(src1_shape.shape()[0].clone())
            };
            let mut padded = false;
            let k_padding = if k_dim.to_usize().map(|i| i % 16 != 0).unwrap_or(true) {
                (k_dim + 15) / 16 * 16 - k_dim
            } else {
                0.into()
            };
            let m_padding = if m_dim.to_usize().map(|i| i % 32 != 0).unwrap_or(true) {
                padded = true;
                (m_dim + 31) / 32 * 32 - m_dim
            } else {
                0.into()
            };
            let n_padding = if n_dim.to_usize().map(|i| i % 256 != 0).unwrap_or(true) {
                padded = true;
                (n_dim + 255) / 256 * 256 - n_dim
            } else {
                0.into()
            };
            if src1_shape.len() == 2 {
                src1_shape.pad(&[(0.into(), m_padding), (0.into(), k_padding)]);
            } else {
                src1_shape.pad(&[
                    (0.into(), 0.into()),
                    (0.into(), m_padding),
                    (0.into(), k_padding),
                ]);
            }
            src2_shape.pad(&[(0.into(), k_padding), (0.into(), n_padding)]);
            if !src1_shape.is_contiguous() || src1_shape.is_sliced() || src1_shape.is_padded() {
                src1 = graph
                    .add_op(MetalContiguous::<f16>::new(
                        src1_shape,
                        dev.clone(),
                        queue.clone(),
                        &mut HashMap::new(),
                        &graph.dyn_map,
                    ))
                    .input(src1, 0, src1_shape)
                    .finish();
                src1_shape = src1_shape.contiguous();
            }
            if !src2_shape.is_contiguous() || src2_shape.is_sliced() || src2_shape.is_padded() {
                src2 = graph
                    .add_op(MetalContiguous::<f16>::new(
                        src2_shape,
                        dev.clone(),
                        queue.clone(),
                        &mut HashMap::new(),
                        &graph.dyn_map,
                    ))
                    .input(src2, 0, src2_shape)
                    .finish();
                src2_shape = src2_shape.contiguous();
            }
            let pipeline_state_descriptor = ComputePipelineDescriptor::new();
            pipeline_state_descriptor.set_compute_function(Some(
                &matmul_library
                    .get_function(
                        "gemm_nn_float16_float16_bm32_bn32_bk16_wm2_wn2_MN_naligned_K_taligned",
                        None,
                    )
                    .unwrap(),
            ));
            let pipeline = dev
                .new_compute_pipeline_state_with_function(
                    pipeline_state_descriptor.compute_function().unwrap(),
                )
                .unwrap();
            let mut matmul_op = graph
                .add_op(Matmul(pipeline, queue.clone(), dev.clone()))
                .input(src1, 0, src1_shape)
                .input(src2, 0, src2_shape)
                .finish();
            // Slice back to original size
            if padded {
                let new_shape = if src1_shape.len() == 3 {
                    let mut n = ShapeTracker::new(&[
                        Expression::from(src1_shape.shape()[0].clone()),
                        Expression::from(src1_shape.shape()[1].clone()),
                        Expression::from(src2_shape.shape()[1].clone()),
                    ]);
                    n.slice(&[
                        (0.into(), i32::MAX.into()),
                        (0.into(), m_dim),
                        (0.into(), n_dim),
                    ]);
                    n
                } else {
                    let mut n = ShapeTracker::new(&[
                        Expression::from(src1_shape.shape()[0].clone()),
                        Expression::from(src2_shape.shape()[1].clone()),
                    ]);
                    n.slice(&[(0.into(), i32::MAX.into()), (0.into(), n_dim)]);
                    n
                };
                matmul_op = graph
                    .add_op(MetalContiguous::<f16>::new(
                        new_shape,
                        dev.clone(),
                        queue.clone(),
                        &mut HashMap::new(),
                        &graph.dyn_map,
                    ))
                    .input(matmul_op, 0, new_shape)
                    .finish();
            }

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
    crate::test_imports!();
    #[test]
    fn test_matrix_vector() {
        const M: usize = 256;
        const N: usize = 256;
        let mut cx = Graph::new();
        let (a_vec, b_vec) = (random_vec(M), random_vec(M * N));
        let mut a = cx.named_tensor::<R2<1, M>>("Vec").set(a_vec.clone());
        let mut b = cx.named_tensor::<R2<M, N>>("Mat").set(b_vec.clone());
        let mut c = a.matmul(b).retrieve();

        cx.compile(
            GenericCompiler::<MetalFp16Compiler>::default(),
            (&mut a, &mut b, &mut c),
        );
        cx.execute();

        let d_dev = Cpu::default();
        let d_a = d_dev.tensor_from_vec(a_vec, (DConst::<M>,));
        let d_b = d_dev.tensor_from_vec(b_vec, (DConst::<M>, DConst::<N>));
        let d_c = d_a.matmul(d_b);

        assert_close_precision(&c.data(), &d_c.as_vec(), 2);
    }

    #[test]
    fn test_batch_matrix_vector() {
        const M: usize = 256;
        const N: usize = 256;
        let mut cx = Graph::new();
        let (a_vec, b_vec) = (random_vec(M), random_vec(M * N));
        let mut a = cx.named_tensor::<R3<1, 1, M>>("Vec").set(a_vec.clone());
        let mut b = cx.named_tensor::<R2<M, N>>("Mat").set(b_vec.clone());
        let mut c = a.matmul(b).retrieve();

        cx.compile(
            GenericCompiler::<MetalFp16Compiler>::default(),
            (&mut a, &mut b, &mut c),
        );
        // cx.display();
        cx.execute();

        let d_dev = Cpu::default();
        let d_a = d_dev.tensor_from_vec(a_vec, (DConst::<M>,));
        let d_b = d_dev.tensor_from_vec(b_vec, (DConst::<M>, DConst::<N>));
        let d_c = d_a.matmul(d_b);

        assert_close_precision(&c.data(), &d_c.to_dtype::<f32>().as_vec(), 2);
    }
}
