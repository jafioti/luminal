use std::sync::Arc;

use half::f16;
use petgraph::stable_graph::NodeIndex;

use crate::{
    compilers::metal::*,
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

impl MetalVecMat {
    fn new(dev: &Device, queue: CommandQueue) -> Self {
        Self {
            kernel: compile_function(
                "kernel_vecmat",
                "
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
#include <metal_simdgroup>
using namespace metal;

kernel void kernel_vecmat(
    device const half *data1 [[buffer(0)]],
    device const half *data2 [[buffer(1)]],
    device half *a [[buffer(2)]],
    device uint& M [[buffer(3)]],
    device uint& N [[buffer(4)]],
    uint3 global_pos [[thread_position_in_grid]]
) {
    if (global_pos.x < N) {
        float acc = 0.0;
        data2 += global_pos.x * M;
        for (uint i = 0; i < M; ++i) {
            acc = fast::fma(data1[i], data2[i], acc);
        }
        a[global_pos.x] = acc;
    }
}",
                dev,
            ),
            queue,
            device: dev.clone(),
        }
    }
}

impl MetalKernelForward for MetalVecMat {
    fn metal_forward(
        &self,
        inputs: &[(&Buffer, ShapeTracker)],
        dev: &Device,
        command_buffer: &CommandBufferRef,
    ) -> Vec<Buffer> {
        let (m, n) = (
            inputs[0].1.shape()[0].to_usize().unwrap(),
            inputs[1].1.shape()[0].to_usize().unwrap(),
        );

        let out = dev.new_buffer(
            (n * std::mem::size_of::<f16>()) as u64,
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

        encoder.set_compute_pipeline_state(&self.kernel);
        encoder.dispatch_thread_groups(
            MTLSize {
                width: (n as u64).div_ceil(256),
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 256,
                height: 1,
                depth: 1,
            },
        );
        encoder.end_encoding();

        vec![out]
    }
}

impl Operator for MetalVecMat {
    fn process(&self, inp: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        autoreleasepool(|| {
            // Setup command queue / command buffer / encoder
            let command_buffer = self.queue.new_command_buffer();

            let out = self
                .metal_forward(
                    &[
                        (get_buffer_from_tensor(&inp[0].0), inp[0].1),
                        (get_buffer_from_tensor(&inp[1].0), inp[1].1),
                    ],
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
    #pragma unroll(4)
    for (uint i = 0; i < 4; ++i) {
        #pragma unroll(4)
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
        #pragma unroll(4)
        for (int i = 0; i < 4; ++i) {
            simdgroup_load(A[i], d1 + i * k8, K);
            simdgroup_load(B[i], data2 + k * N + i * 8, N);
        }

        #pragma unroll(4)
        for (int i = 0; i < 4; ++i) {
            #pragma unroll(4)
            for (int j = 0; j < 4; ++j) {
                simdgroup_multiply_accumulate(acc[i][j], A[j], B[i], acc[i][j]);
            }
        }
    }

    simdgroup_half8x8 temp = simdgroup_half8x8(0);
    simdgroup_half8x8 ident = simdgroup_half8x8(1);
    // Width
    #pragma unroll(4)
    for (int i = 0; i < 4; ++i) {
        uint n8i = i * 8 * N;
        // Height
        #pragma unroll(4)
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
            // Setup command queue / command buffer / encoder
            let command_buffer = self.queue.new_command_buffer();

            let out = self
                .metal_forward(
                    &[
                        (get_buffer_from_tensor(&inp[0].0), inp[0].1),
                        (get_buffer_from_tensor(&inp[1].0), inp[1].1),
                    ],
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
    fn compile(dev: &Device) -> ComputePipelineState {
        compile_function(
            "kernel_batch_matmul_2d",
            "
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
#include <metal_simdgroup>
using namespace metal;

kernel void kernel_batch_matmul_2d(
    device const half *data1 [[buffer(0)]],
    device const half *data2 [[buffer(1)]],
    device half *a [[buffer(2)]],
    device uint& M [[buffer(3)]],
    device uint& N [[buffer(4)]],
    device uint& K [[buffer(5)]],
    uint3 block_pos [[threadgroup_position_in_grid]],
    uint3 global_pos [[thread_position_in_grid]]
) {
    a += M * N * block_pos.z + block_pos.x * 32 * N + global_pos.y * 32;
    data1 += M * K * block_pos.z + block_pos.x * 32 * K;
    data2 += global_pos.y * 32;

    simdgroup_float8x8 acc[4][4];
    #pragma unroll(4)
    for (uint i = 0; i < 4; ++i) {
        #pragma unroll(4)
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
        #pragma unroll(4)
        for (int i = 0; i < 4; ++i) {
            simdgroup_load(A[i], d1 + i * k8, K);
            simdgroup_load(B[i], data2 + k * N + i * 8, N);
        }

        #pragma unroll(4)
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                simdgroup_multiply_accumulate(acc[i][j], A[j], B[i], acc[i][j]);
            }
        }
    }

    simdgroup_half8x8 temp = simdgroup_half8x8(0);
    simdgroup_half8x8 ident = simdgroup_half8x8(1);
    // Width
    #pragma unroll(4)
    for (int i = 0; i < 4; ++i) {
        uint n8i = i * 8 * N;
        // Height
        #pragma unroll(4)
        for (int j = 0; j < 4; ++j) {
            simdgroup_multiply(temp, acc[j][i], ident);
            simdgroup_store(temp, a+(8*j+n8i), N);
        }
    }
}",
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
        encoder.set_int(3, m as u32);
        encoder.set_int(4, n as u32);
        encoder.set_int(5, k as u32);

        // Execute
        encoder.dispatch_thread_groups(
            MTLSize {
                width: (m as u64).div_ceil(32),
                height: (n as u64).div_ceil(32 * 8),
                depth: batch_size as u64,
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

impl Operator for MetalBatchMatmul2D {
    fn process(&self, inp: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        autoreleasepool(|| {
            // Setup command queue / command buffer / encoder
            let command_buffer = self.1.new_command_buffer();

            let out = self
                .metal_forward(
                    &[
                        (get_buffer_from_tensor(&inp[0].0), inp[0].1),
                        (get_buffer_from_tensor(&inp[1].0), inp[1].1),
                    ],
                    &self.2,
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

#[derive(Default)]
pub struct MetalMatMulCompiler;

impl Compiler for MetalMatMulCompiler {
    fn compile(&self, graph: &mut Graph) {
        let dev = Device::system_default().unwrap();
        let queue = dev.new_command_queue();
        let (mut sum_reduce, mut mul) = (NodeIndex::default(), NodeIndex::default());

        // Look for vetmat pattern
        // Mul ([1(fake), N(fake), M] | [1(fake), N, M]) -> SumReduce(2) -> [N]
        let vecmat_pattern = SelectEdge::new(
            SelectOp::new()
                .ty::<MetalMul<f16>>()
                .shapes(vec![
                    vec![1.into(), 'N'.into(), 'M'.into()],
                    vec![1.into(), 'N'.into(), 'M'.into()],
                ])
                .fakes(vec![
                    vec![None, Some(true), Some(false)],
                    vec![Some(true), Some(false), Some(false)],
                ])
                .ptr(&mut mul),
            SelectOp::new()
                .check(|o, _| {
                    if let Some(o) = o.as_any().downcast_ref::<MetalSumReduce<f16>>() {
                        o.3 == 2
                    } else {
                        false
                    }
                })
                .ptr(&mut sum_reduce),
        );
        let batch_vecmat_pattern = SelectEdge::new(
            SelectOp::new()
                .ty::<MetalMul<f16>>()
                .shapes(vec![
                    vec![1.into(), 1.into(), 'N'.into(), 'M'.into()],
                    vec![1.into(), 1.into(), 'N'.into(), 'M'.into()],
                ])
                .fakes(vec![
                    vec![None, None, Some(true), Some(false)],
                    vec![None, Some(true), Some(false), Some(false)],
                ])
                .ptr(&mut mul),
            SelectOp::new()
                .check(|o, _| {
                    if let Some(o) = o.as_any().downcast_ref::<MetalSumReduce<f16>>() {
                        o.3 == 3
                    } else {
                        false
                    }
                })
                .ptr(&mut sum_reduce),
        );
        // Mul ([1, 1(fake?), N(fake), M] | [1, 1(fake), N, M]) -> SumReduce(2) -> [N]
        for _ in vecmat_pattern
            .search(graph)
            .chain(batch_vecmat_pattern.search(graph))
        {
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
            // Src1: [M], Src2: [N, M]
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

            let matmul_op = graph
                .add_op(MetalVecMat::new(&dev, queue.clone()))
                .input(src1, 0, src1_shape)
                .input(src2, 0, src2_shape)
                .finish();

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

        // Look for the matmul pattern
        // Mul ([M, N(fake), K] | [M(fake), N, K]) -> SumReduce(2) -> [M, N]
        // or batch matmul where 1st or 2nd dim is 1
        // Mul ([1, M, N(fake), K] | [1, M(fake), N, K]) -> SumReduce(3) -> [1, M, N] // BMM batch size 1
        // Mul ([B, 1, N(fake), K] | [B, 1(fake), N, K]) -> SumReduce(3) -> [B, 1, N] // Batch vecmat
        let matmul_pattern = SelectEdge::new(
            SelectOp::new()
                .ty::<MetalMul<f16>>()
                .shapes(vec![
                    vec!['M'.into(), 'N'.into(), 'K'.into()],
                    vec!['M'.into(), 'N'.into(), 'K'.into()],
                ])
                .fakes(vec![
                    vec![Some(false), Some(true), Some(false)],
                    vec![Some(true), Some(false), Some(false)],
                ])
                .ptr(&mut mul),
            SelectOp::new()
                .check(|o, _| {
                    if let Some(o) = o.as_any().downcast_ref::<MetalSumReduce<f16>>() {
                        o.3 == 2
                    } else {
                        false
                    }
                })
                .ptr(&mut sum_reduce),
        );

        for _ in matmul_pattern.search(graph) {
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
            let m_dim = src1_shape.dims[src1_shape.indexes[0]];
            let k_padding = if k_dim.to_usize().map(|i| i % 16 != 0).unwrap_or(true) {
                (k_dim + 15) / 16 * 16 - k_dim
            } else {
                0.into()
            };
            let mut padded = false;
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
            src1_shape.pad(&[(0.into(), m_padding), (0.into(), k_padding)]);
            src2_shape.pad(&[(0.into(), k_padding), (0.into(), n_padding)]);
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
        let s = SelectEdge::new(
            SelectOp::new()
                .ty::<MetalMul<f16>>()
                .shapes(vec![
                    vec!['D'.into(), 'A'.into(), 'C'.into(), 'B'.into()],
                    vec!['D'.into(), 'A'.into(), 'C'.into(), 'B'.into()],
                ])
                .fakes(vec![
                    vec![Some(false), Some(false), Some(true), Some(false)],
                    vec![Some(true), Some(true), Some(false), Some(false)],
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
            // Pad out N to multiple of 256 and K to 16
            let n_dim = src2_shape.shape()[1];
            let k_dim = src1_shape.shape()[2];
            let m_dim = src1_shape.shape()[1];
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
            src1_shape.pad(&[
                (0.into(), 0.into()),
                (0.into(), m_padding),
                (0.into(), k_padding),
            ]);
            src2_shape.pad(&[(0.into(), k_padding), (0.into(), n_padding)]);
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
                    MetalBatchMatmul2D::compile(&dev),
                    queue.clone(),
                    dev.clone(),
                ))
                .input(src1, 0, src1_shape)
                .input(src2, 0, src2_shape)
                .finish();
            // Slice back to original size
            if padded {
                let mut new_shape = ShapeTracker::new(&[
                    src1_shape.shape()[0],
                    src1_shape.shape()[1],
                    src2_shape.shape()[1],
                ]);
                new_shape.slice(&[
                    (0.into(), i32::MAX.into()),
                    (0.into(), m_dim),
                    (0.into(), n_dim),
                ]);
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
    }
}

#[cfg(test)]
mod tests {
    crate::test_imports!();
    #[test]
    fn test_matrix_vector() {
        const M: usize = 13;
        const N: usize = 32;
        let mut cx = Graph::new();
        let (a_vec, b_vec) = (random_vec(M), random_vec(M * N));
        let a = cx.named_tensor::<R2<1, M>>("Vec").set(a_vec.clone());
        let b = cx.named_tensor::<R2<M, N>>("Mat").set(b_vec.clone());
        let c = a.matmul(b).retrieve();

        cx.compile(GenericCompiler::<MetalFp16Compiler>::default());
        cx.execute();

        let d_dev = Cpu::default();
        let d_a = d_dev.tensor_from_vec(a_vec, (DConst::<M>,));
        let d_b = d_dev.tensor_from_vec(b_vec, (DConst::<M>, DConst::<N>));
        let d_c = d_a.matmul(d_b);

        assert_close(&c.data(), &d_c.as_vec());
    }

    #[test]
    fn test_batch_matrix_vector() {
        const M: usize = 13;
        const N: usize = 32;
        let mut cx = Graph::new();
        let (a_vec, b_vec) = (random_vec(M), random_vec(M * N));
        let a = cx.named_tensor::<R3<1, 1, M>>("Vec").set(a_vec.clone());
        let b = cx.named_tensor::<R2<M, N>>("Mat").set(b_vec.clone());
        let c = a.matmul(b).retrieve();

        cx.compile(GenericCompiler::<MetalFp16Compiler>::default());
        // cx.display();
        cx.execute();

        let d_dev = Cpu::default();
        let d_a = d_dev.tensor_from_vec(a_vec, (DConst::<M>,));
        let d_b = d_dev.tensor_from_vec(b_vec, (DConst::<M>, DConst::<N>));
        let d_c = d_a.matmul(d_b);

        assert_close(&c.data(), &d_c.as_vec());
    }
}
