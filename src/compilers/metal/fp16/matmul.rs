use std::sync::Arc;

use half::f16;
use petgraph::stable_graph::NodeIndex;

use crate::{
    compilers::metal::*,
    op::{InputTensor, Operator},
    prelude::{symbolic::Expression, *},
};

use metal_rs::{objc::rc::autoreleasepool, *};

/// Multiplies a MxK matrix with a KxN matrix, resulting in a MxN matrix
#[derive(LuminalEq, LuminalPrint, Clone)]
pub struct MetalMatmul2D {
    simd_shader: ComputePipelineState,
    queue: CommandQueue,
    device: Device,
}

impl MetalMatmul2D {
    fn new(dev: &Device, queue: CommandQueue) -> Self {
        let simd_shader = compile_function(
            "kernel_matmul_2d",
            "
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
#include <metal_simdgroup>
using namespace metal;

kernel void kernel_matmul_2d(
    device const half *data1 [[buffer(0)]],
    device const half *data2 [[buffer(1)]],
    device half *a [[buffer(2)]],
    device uint& M [[buffer(3)]],
    device uint& N [[buffer(4)]],
    device uint& K [[buffer(5)]],
    uint3 block_pos [[threadgroup_position_in_grid]],
    uint3 global_pos [[thread_position_in_grid]]
) {
    a += block_pos.x * 32 * N + global_pos.y * 32;
    data1 += block_pos.x * 32 * K;
    data2 += global_pos.y * 32;

    simdgroup_float8x8 acc[4][4];
    for (uint i = 0; i < 4; ++i) {
        for (uint j = 0; j < 4; ++j) {
            acc[i][j] = simdgroup_float8x8(0);
        }
    }

    simdgroup_half8x8 A[4];
    simdgroup_half8x8 B[4];
    uint k8 = 8 * K;
    for (uint k = 0; k < K; k+=8) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        device const half *d1 = data1 + k;
        for (int i = 0; i < 4; ++i) {
            simdgroup_load(A[i], d1 + i * k8, K);
            simdgroup_load(B[i], data2 + k * N + i * 8, N);
        }

        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                simdgroup_multiply_accumulate(acc[i][j], A[j], B[i], acc[i][j]);
            }
        }
    }

    simdgroup_half8x8 temp = simdgroup_half8x8(0);
    simdgroup_half8x8 ident = simdgroup_half8x8(1);
    // Width
    for (int i = 0; i < 4; ++i) {
        uint n8i = i * 8 * N;
        // Height
        for (int j = 0; j < 4; ++j) {
            simdgroup_multiply(temp, acc[j][i], ident);
            simdgroup_store(temp, a+(8*j+n8i), N);
        }
    }
}",
            dev,
        );
        Self {
            simd_shader,
            queue,
            device: dev.clone(),
        }
    }
}

impl MetalKernelForward for MetalMatmul2D {
    fn metal_forward(
        &self,
        inputs: &[(&Buffer, ShapeTracker)],
        dev: &Device,
        command_buffer: &CommandBufferRef,
    ) -> Vec<Buffer> {
        let (a_shape, b_shape) = (inputs[0].1.shape(), inputs[1].1.shape());
        let (m, k, n) = (
            a_shape[0].to_usize().unwrap(),
            a_shape[1].to_usize().unwrap(),
            b_shape[1].to_usize().unwrap(),
        );

        let out = dev.new_buffer(
            (m * n * std::mem::size_of::<f16>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let encoder =
            command_buffer.compute_command_encoder_with_descriptor(ComputePassDescriptor::new());

        // Set inputs
        encoder.set_buffer(0, Some(inputs[0].0), 0);
        encoder.set_buffer(1, Some(inputs[1].0), 0);
        encoder.set_buffer(2, Some(&out), 0);
        encoder.set_int(3, m as u32);
        encoder.set_int(4, n as u32);
        encoder.set_int(5, k as u32);

        encoder.set_compute_pipeline_state(&self.simd_shader);
        encoder.dispatch_thread_groups(
            MTLSize {
                width: (m as u64).div_ceil(32),
                height: (n as u64).div_ceil(32 * 8),
                depth: 1,
            },
            MTLSize {
                width: 32,
                height: 8,
                depth: 1,
            },
        );
        encoder.end_encoding();

        vec![out]
    }
}

impl Operator for MetalMatmul2D {
    fn process(&self, inp: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        autoreleasepool(|| {
            let a = inp[0]
                .0
                .borrowed()
                .data
                .as_any()
                .downcast_ref::<Buffer>()
                .unwrap();
            let b = inp[1]
                .0
                .borrowed()
                .data
                .as_any()
                .downcast_ref::<Buffer>()
                .unwrap();

            // Setup command queue / command buffer / encoder
            let command_buffer = self.queue.new_command_buffer();

            let out = self
                .metal_forward(
                    &[(a, inp[0].1), (b, inp[1].1)],
                    &self.device,
                    command_buffer,
                )
                .pop()
                .unwrap();

            command_buffer.commit();
            command_buffer.wait_until_completed();

            vec![Tensor {
                data: Box::new(out),
            }]
        })
    }

    fn custom(&self, key: &str) -> Option<Box<dyn Any>> {
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
pub struct MetalBatchMatmul2D(ComputePipelineState, CommandQueue, Device);

impl MetalBatchMatmul2D {
    fn compile(dev: &Device, a_row_major: bool, b_row_major: bool) -> ComputePipelineState {
        compile_function(
                    "kernel_batch_matmul_2d",
                        &format!("
                        #include <metal_stdlib>
                        using namespace metal;
                        kernel void kernel_batch_matmul_2d(
                            device half *A [[buffer(0)]],
                            device half *B [[buffer(1)]],
                            device half *C [[buffer(2)]],
                            device uint& Batch [[buffer(3)]],
                            device uint& M [[buffer(4)]],
                            device uint& N [[buffer(5)]],
                            device uint& K [[buffer(6)]],
                            threadgroup half* shared_memory [[threadgroup(0)]],
                            uint3 global_pos [[thread_position_in_grid]],
                            uint3 local_pos [[thread_position_in_threadgroup]],
                            uint3 block_size [[threads_per_threadgroup]]
                        ) {{
                            float sum = 0.0f;
                        
                            threadgroup half* b_start = shared_memory + block_size.x * block_size.x;
                            uint local_x_block_x = local_pos.x * block_size.x;
                            uint shared_mem_addr = local_x_block_x + local_pos.y;
                            uint common_a_ind = global_pos.x * K + local_pos.y;
                            uint common_b_ind = global_pos.y * K + local_pos.x;
                            for (uint m = 0; m < K; m += block_size.x) {{
                                shared_memory[shared_mem_addr] = (local_pos.y + m < K) ? A[global_pos.z * M * K + {}] : 0.0h;
                                b_start[shared_mem_addr] = (local_pos.x + m < K) ? B[{}] : 0.0h;
                        
                                threadgroup_barrier(mem_flags::mem_threadgroup);
                        
                                #pragma unroll(8)
                                for (uint e = 0; e < block_size.x; ++e) {{
                                    sum = fast::fma((float)shared_memory[local_x_block_x + e], (float)b_start[e * block_size.x + local_pos.y], sum);
                                }}
                                threadgroup_barrier(mem_flags::mem_threadgroup);
                            }}
                            
                            if (global_pos.x < M && global_pos.y < N && global_pos.z < Batch) {{
                                C[global_pos.z * M * N + global_pos.x * N + global_pos.y] = (half)sum;
                            }}
                        }}", 
                        if a_row_major {"common_a_ind + m"} else {"(local_pos.y + m) * M + global_pos.x"}, 
                        if b_row_major {"(m + local_pos.x) * N + global_pos.y"} else {"common_b_ind + m"}
                    ),
                    dev,
                )
    }
}

impl MetalKernelForward for MetalBatchMatmul2D {
    fn metal_forward(
        &self,
        inputs: &[(&Buffer, ShapeTracker)],
        dev: &Device,
        command_buffer: &CommandBufferRef,
    ) -> Vec<Buffer> {
        let (a_shape, b_shape) = (inputs[0].1.shape(), inputs[1].1.shape());
        let (batch_size, m, k, n) = (
            a_shape[0].to_usize().unwrap(),
            a_shape[1].to_usize().unwrap(),
            a_shape[2].to_usize().unwrap(),
            b_shape[1].to_usize().unwrap(),
        );

        let out = dev.new_buffer(
            (batch_size * m * n * std::mem::size_of::<f16>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let encoder =
            command_buffer.compute_command_encoder_with_descriptor(ComputePassDescriptor::new());
        encoder.set_compute_pipeline_state(&self.0);

        // Set inputs
        encoder.set_buffer(0, Some(inputs[0].0), 0);
        encoder.set_buffer(1, Some(inputs[1].0), 0);
        encoder.set_buffer(2, Some(&out), 0);
        encoder.set_int(3, batch_size as u32);
        encoder.set_int(4, m as u32);
        encoder.set_int(5, n as u32);
        encoder.set_int(6, k as u32);
        encoder.set_threadgroup_memory_length(0, 8 * 8 * 2 * std::mem::size_of::<f16>() as u64);

        // Execute
        encoder.dispatch_thread_groups(
            MTLSize {
                width: (m as u64).div_ceil(8),
                height: (n as u64).div_ceil(8),
                depth: batch_size as u64,
            },
            MTLSize {
                width: 8,
                height: 8,
                depth: 1,
            },
        );
        encoder.end_encoding();

        vec![out]
    }
}

impl Operator for MetalBatchMatmul2D {
    fn process(&self, inp: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        autoreleasepool(|| {
            let a = inp[0]
                .0
                .borrowed()
                .data
                .as_any()
                .downcast_ref::<Buffer>()
                .unwrap();
            let b = inp[1]
                .0
                .borrowed()
                .data
                .as_any()
                .downcast_ref::<Buffer>()
                .unwrap();

            // Setup command queue / command buffer / encoder
            let command_buffer = self.1.new_command_buffer();

            let out = self
                .metal_forward(&[(a, inp[0].1), (b, inp[1].1)], &self.2, command_buffer)
                .pop()
                .unwrap();

            command_buffer.commit();
            command_buffer.wait_until_completed();

            vec![Tensor {
                data: Box::new(out),
            }]
        })
    }

    fn custom(&self, key: &str) -> Option<Box<dyn Any>> {
        if key == "metal" {
            return Some(Box::new(MetalKernelWrapper(Arc::new(Box::new(
                self.clone(),
            )))));
        }
        None
    }
}

#[derive(Default)]
pub struct MetalMatMulCompiler;

impl Compiler for MetalMatMulCompiler {
    fn compile(&self, graph: &mut Graph) {
        let dev = Device::system_default().unwrap();
        let queue = dev.new_command_queue();
        // Look for the matmul pattern
        // Mul ([A, C(fake), B] | [A(fake), C, B]) -> SumReduce(2) -> [A, C]
        // Actually starts at [A,B] | [B, C]
        let (mut sum_reduce, mut mul) = (NodeIndex::default(), NodeIndex::default());
        let s = SelectEdge::new(
            SelectOp::new()
                .ty::<MetalMul<f16>>()
                .shapes(vec![
                    vec!['A'.into(), 'C'.into(), 'B'.into()],
                    vec!['A'.into(), 'C'.into(), 'B'.into()],
                ])
                .fakes(vec![vec![false, true, false], vec![true, false, false]])
                .ptr(&mut mul),
            SelectOp::new()
                .ty::<MetalSumReduce<f16>>()
                .check(|o, _| {
                    if let Some(o) = o.as_any().downcast_ref::<MetalSumReduce<f16>>() {
                        o.3 == 2
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
            let srcs = graph.get_sources(mul);
            let (mut src1, mut src1_shape) = (srcs[0].0, srcs[0].2);
            let (mut src2, mut src2_shape) = (srcs[1].0, srcs[1].2);
            // Undo expansions and permute
            src1_shape.remove_dim(1);
            src2_shape.remove_dim(0);
            src2_shape.permute(&[1, 0]);

            // Pad out N to multiple of 256 and K to 16
            let n_dim = src2_shape.dims[src2_shape.indexes[1]];
            let k_dim = src1_shape.dims[src1_shape.indexes[1]];
            let k_padding = if k_dim.to_usize().map(|i| i < 16).unwrap_or(true) {
                Expression::from(16) - k_dim
            } else {
                0.into()
            };
            let mut padded = false;
            if n_dim.to_usize().map(|i| i % 256 != 0).unwrap_or(true) {
                padded = true;
                src2_shape.pad(&[
                    (0.into(), k_padding),
                    (0.into(), (n_dim + 255) / 256 * 256 - n_dim),
                ]);
            }
            if k_padding != 0.into() {
                src1_shape.pad(&[(0.into(), 0.into()), (0.into(), k_padding)]);
            }
            if !src1_shape.is_contiguous() || src1_shape.is_sliced() || src1_shape.is_padded() {
                src1 = graph
                    .add_op(MetalContiguous::<f16>::new(
                        src1_shape,
                        dev.clone(),
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
                        &mut HashMap::new(),
                        &graph.dyn_map,
                    ))
                    .input(src2, 0, src2_shape)
                    .finish();
                src2_shape = src2_shape.contiguous();
            }
            let mut matmul_op = graph
                .add_op(MetalMatmul2D::new(&dev, queue.clone()))
                .input(src1, 0, src1_shape)
                .input(src2, 0, src2_shape)
                .finish();

            // Slice back to original size
            if padded {
                let mut new_shape =
                    ShapeTracker::new(&[src1_shape.shape()[0], src2_shape.shape()[1]]);
                new_shape.slice(&[(0.into(), i32::MAX.into()), (0.into(), n_dim)]);
                matmul_op = graph
                    .add_op(MetalContiguous::<f16>::new(
                        new_shape,
                        dev.clone(),
                        &mut HashMap::new(),
                        &graph.dyn_map,
                    ))
                    .input(matmul_op, 0, new_shape)
                    .finish();
            }

            // Create edges to dests
            move_outgoing_edge(sum_reduce, matmul_op, &mut graph.graph);
            move_references(
                &mut graph.id_remap,
                &mut graph.no_delete,
                &mut graph.to_retrieve,
                sum_reduce,
                matmul_op,
            );

            // Remove the old ops
            graph.graph.remove_node(mul);
            graph.graph.remove_node(sum_reduce);
        }

        // Look for the batch matmul pattern
        // Mul ([A, C(fake), B] | [A(fake), C, B]) -> SumReduce(2) -> [A, C]
        // Actually starts at [A,B] | [B, C]
        let (mut sum_reduce, mut mul) = (NodeIndex::default(), NodeIndex::default());
        let s = SelectEdge::new(
            SelectOp::new()
                .ty::<MetalMul<f16>>()
                .shapes(vec![
                    vec!['D'.into(), 'A'.into(), 'C'.into(), 'B'.into()],
                    vec!['D'.into(), 'A'.into(), 'C'.into(), 'B'.into()],
                ])
                .fakes(vec![
                    vec![false, false, true, false],
                    vec![true, true, false, false],
                ])
                .ptr(&mut mul),
            SelectOp::new()
                .ty::<MetalSumReduce<f16>>()
                .check(|o, _| {
                    if let Some(o) = o.as_any().downcast_ref::<MetalSumReduce<f16>>() {
                        o.3 == 3
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
            let srcs = graph.get_sources(mul);
            let (mut src1, mut src1_shape) = (srcs[0].0, srcs[0].2);
            let (mut src2, mut src2_shape) = (srcs[1].0, srcs[1].2);
            // Undo expansions and permute
            src1_shape.remove_dim(2);
            src2_shape.remove_dim(1);
            src2_shape.remove_dim(0);
            src2_shape.permute(&[1, 0]);
            // println!("Src1: {:?}", src1_shape);
            // println!("Src2: {:?}", src2_shape);
            // // Pad out N to multiple of 256 and K to 16
            // let n_dim = src2_shape.dims[src2_shape.indexes[1]];
            // let k_dim = src1_shape.dims[src1_shape.indexes[2]];
            // let k_padding = if k_dim.to_usize().map(|i| i < 16).unwrap_or(true) {
            //     Expression::from(16) - k_dim
            // } else {
            //     0.into()
            // };
            // let mut padded = false;
            // if n_dim.to_usize().map(|i| i % 256 != 0).unwrap_or(true) {
            //     padded = true;
            //     src2_shape.pad(&[
            //         (0.into(), k_padding),
            //         (0.into(), (n_dim + 255) / 256 * 256 - n_dim),
            //     ]);
            // }
            // if k_padding != 0.into() {
            //     src1_shape.pad(&[
            //         (0.into(), 0.into()),
            //         (0.into(), 0.into()),
            //         (0.into(), k_padding),
            //     ]);
            // }
            if !src1_shape.is_contiguous() || src1_shape.is_sliced() || src1_shape.is_padded() {
                src1 = graph
                    .add_op(MetalContiguous::<f16>::new(
                        src1_shape,
                        dev.clone(),
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
                        &mut HashMap::new(),
                        &graph.dyn_map,
                    ))
                    .input(src2, 0, src2_shape)
                    .finish();
                src2_shape = src2_shape.contiguous();
            }
            let mut matmul_op = graph
                .add_op(MetalBatchMatmul2D(
                    MetalBatchMatmul2D::compile(
                        &dev,
                        src1_shape.indexes[1] < src1_shape.indexes[2],
                        src2_shape.indexes[0] < src2_shape.indexes[1],
                    ),
                    queue.clone(),
                    dev.clone(),
                ))
                .input(src1, 0, src1_shape)
                .input(src2, 0, src2_shape)
                .finish();
            // // Slice back to original size
            // if padded {
            //     let mut new_shape = ShapeTracker::new(&[
            //         src1_shape.shape()[0],
            //         src1_shape.shape()[1],
            //         src2_shape.shape()[1],
            //     ]);
            //     new_shape.slice(&[
            //         (0.into(), i32::MAX.into()),
            //         (0.into(), i32::MAX.into()),
            //         (0.into(), n_dim),
            //     ]);
            //     println!("Sliced: {:?}", new_shape);
            //     matmul_op = graph
            //         .add_op(MetalContiguous::<f16>::new(
            //             new_shape,
            //             dev.clone(),
            //             &mut HashMap::new(),
            //             &graph.dyn_map,
            //         ))
            //         .input(matmul_op, 0, new_shape)
            //         .finish();
            // }

            // Create edges to dests
            move_outgoing_edge(sum_reduce, matmul_op, &mut graph.graph);
            move_references(
                &mut graph.id_remap,
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
