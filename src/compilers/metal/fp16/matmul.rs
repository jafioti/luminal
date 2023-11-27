use std::sync::Arc;

use half::f16;
use petgraph::stable_graph::NodeIndex;

use crate::{
    compilers::metal::*,
    op::{InputTensor, Operator},
    prelude::*,
};

use metal_rs::{objc::rc::autoreleasepool, *};

/// Multiplies a MxK matrix with a KxN matrix, resulting in a MxN matrix
#[derive(LuminalEq, LuminalPrint, Clone)]
pub struct MetalMatmul2D(ComputePipelineState, CommandQueue, Device);

impl MetalMatmul2D {
    fn compile(dev: &Device, a_row_major: bool, b_row_major: bool) -> ComputePipelineState {
        compile_function(
            "kernel_matmul_2d",
            &format!(
                "
#include <metal_stdlib>
using namespace metal;

kernel void kernel_matmul_2d(
    device half *A [[buffer(0)]],
    device half *B [[buffer(1)]],
    device half *C [[buffer(2)]],
    device uint& M [[buffer(3)]],
    device uint& K [[buffer(4)]],
    device uint& N [[buffer(5)]],
    uint3 global_pos [[thread_position_in_grid]]
) {{
    uint row = global_pos.x;
    uint column = global_pos.y;

    if(row < M && column < N) {{
        float value = 0.0f;
        for(int i = 0; i < K; ++i) {{
            value = fast::fma((float)A[{}], (float)B[{}], value);
        }}
        C[row * N + column] = (half)value;
    }}
}}",
                if a_row_major {
                    "row * K + i"
                } else {
                    "i * M + row"
                },
                if b_row_major {
                    "i * N + column"
                } else {
                    "column * K + i"
                }
            ),
            dev,
        )
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
        encoder.set_compute_pipeline_state(&self.0);

        // Set inputs
        encoder.set_buffer(0, Some(inputs[0].0), 0);
        encoder.set_buffer(1, Some(inputs[1].0), 0);
        encoder.set_buffer(2, Some(&out), 0);
        encoder.set_int(3, m as u32);
        encoder.set_int(4, k as u32);
        encoder.set_int(5, n as u32);

        // Execute
        encoder.dispatch_thread_groups(
            MTLSize {
                width: (m as u64).div_ceil(16),
                height: (n as u64).div_ceil(16),
                depth: 1,
            },
            MTLSize {
                width: 16,
                height: 16,
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
                    vec![Dim::Unknown('A'), Dim::Unknown('C'), Dim::Unknown('B')],
                    vec![Dim::Unknown('A'), Dim::Unknown('C'), Dim::Unknown('B')],
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
            let mut srcs = graph.get_sources(mul);
            // Undo expansions and permute
            srcs[0].2.remove_dim(1);
            srcs[1].2.remove_dim(0);
            srcs[1].2.permute(&[1, 0]);
            let new_op = graph
                .add_op(MetalMatmul2D(
                    MetalMatmul2D::compile(
                        &dev,
                        srcs[0].2.indexes[0] < srcs[0].2.indexes[1],
                        srcs[1].2.indexes[0] < srcs[1].2.indexes[1],
                    ),
                    queue.clone(),
                    dev.clone(),
                ))
                .input(srcs[0].0, 0, srcs[0].2)
                .input(srcs[1].0, 0, srcs[1].2)
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
            let mut srcs = graph.get_sources(mul);
            // Undo expansions and permute
            srcs[0].2.remove_dim(2);
            srcs[1].2.remove_dim(1);
            srcs[1].2.remove_dim(0);
            srcs[1].2.permute(&[1, 0]);
            let new_op = graph
                .add_op(MetalBatchMatmul2D(
                    MetalBatchMatmul2D::compile(
                        &dev,
                        srcs[0].2.indexes[1] < srcs[0].2.indexes[2],
                        srcs[1].2.indexes[0] < srcs[1].2.indexes[1],
                    ),
                    queue.clone(),
                    dev.clone(),
                ))
                .input(srcs[0].0, 0, srcs[0].2)
                .input(srcs[1].0, 0, srcs[1].2)
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

            // Remove the old ops
            graph.graph.remove_node(mul);
            graph.graph.remove_node(sum_reduce);
        }
    }
}
