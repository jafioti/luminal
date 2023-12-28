use petgraph::stable_graph::NodeIndex;

use crate::{
    compilers::metal::{prim::*, *},
    op::{InputTensor, Operator},
    prelude::*,
};

use metal_rs::{objc::rc::autoreleasepool, *};

/// Multiplies a MxK matrix with a KxN matrix, resulting in a MxN matrix
#[derive(LuminalEq, LuminalPrint, Clone)]
pub struct MetalMatmul2D {
    naive_shader: ComputePipelineState,
    simd_shader: ComputePipelineState,
    device: Device,
}

impl MetalMatmul2D {
    fn new(dev: &Device) -> Self {
        let simd_shader = compile_function(
            "kernel_matmul_2d",
            "
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
#include <metal_simdgroup>
using namespace metal;

kernel void kernel_matmul_2d(
    device const float *data1 [[buffer(0)]],
    device const float *data2 [[buffer(1)]],
    device float *a [[buffer(2)]],
    device int& M [[buffer(3)]],
    device int& N [[buffer(4)]],
    device int& K [[buffer(5)]],
    uint3 block_pos [[threadgroup_position_in_grid]],
    uint3 global_pos [[thread_position_in_grid]]
) {
    a += block_pos.x * 32 * N + global_pos.y * 32;
    data1 += block_pos.x * 32 * K;
    data2 += global_pos.y * 32;

    simdgroup_float8x8 acc[4][4];
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
        acc[i][j] = simdgroup_float8x8(0);
        }
    }

    simdgroup_float8x8 A[4];
    simdgroup_float8x8 B[4];
    int k8 = 8 * K;
    for (int k = 0; k < K; k+=8) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        device const float *d1 = data1 + k;
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

    // Width
    for (int i = 0; i < 4; ++i) {
        int n8i = i * 8 * N;
        // Height
        for (int j = 0; j < 4; ++j) {
            simdgroup_store(acc[j][i], a+(8*j+n8i), N);
        }
    }
}",
            dev,
        );
        let naive_shader = compile_function(
            "kernel_matmul_2d_naive",
            "#include <metal_stdlib>
using namespace metal;

kernel void kernel_matmul_2d_naive(
    device float *A [[buffer(0)]],
    device float *B [[buffer(1)]],
    device float *C [[buffer(2)]],
    device int& M [[buffer(3)]],
    device int& N [[buffer(4)]],
    device int& K [[buffer(5)]],
    uint tid [[thread_position_in_grid]]
) {
    int row = tid / N;
    int column = tid % N;

    if(row < M && column < N) {
        float value = 0.0f;
        for(int i = 0; i < K; ++i) {
            int A_index = row * K + i;
            int B_index = i * N + column;
            value += A[A_index] * B[B_index];
        }
        C[row * N + column] = value;
    }
}",
            dev,
        );

        Self {
            naive_shader,
            simd_shader,
            device: dev.clone(),
        }
    }
}

impl Operator for MetalMatmul2D {
    fn process(&self, inp: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        autoreleasepool(|| {
            let (a_shape, b_shape) = (inp[0].1.shape(), inp[1].1.shape());
            let (m, k, n) = (
                a_shape[0].to_usize().unwrap(),
                a_shape[1].to_usize().unwrap(),
                b_shape[1].to_usize().unwrap(),
            );
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

            let out = self.device.new_buffer(
                (m * n * std::mem::size_of::<f32>()) as u64,
                MTLResourceOptions::StorageModeShared,
            );

            // Setup command queue / command buffer / encoder
            let command_queue = self.device.new_command_queue();
            let command_buffer = command_queue.new_command_buffer();
            let encoder = command_buffer
                .compute_command_encoder_with_descriptor(ComputePassDescriptor::new());

            // Set inputs
            encoder.set_buffer(0, Some(a), 0);
            encoder.set_buffer(1, Some(b), 0);
            encoder.set_buffer(2, Some(&out), 0);
            encoder.set_int(3, m as u32);
            encoder.set_int(4, n as u32);
            encoder.set_int(5, k as u32);

            if k >= 16 && n >= 256 && ((n != 0) && (n & (n - 1)) == 0) {
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
            } else {
                encoder.set_compute_pipeline_state(&self.naive_shader);
                encoder.dispatch_1d(n * m);
            }
            encoder.end_encoding();

            // Execute
            command_buffer.commit();
            command_buffer.wait_until_completed();

            vec![Tensor {
                data: Box::new(out),
            }]
        })
    }
}

/// Multiplies a BxMxK matrix with a BxKxN matrix, resulting in a BxMxN matrix
#[derive(LuminalEq, LuminalPrint, Clone)]
pub struct MetalBatchMatmul2D(ComputePipelineState, Device);

impl MetalBatchMatmul2D {
    fn compile(dev: &Device) -> ComputePipelineState {
        let mut code = "#include <metal_stdlib>
using namespace metal;

kernel void mkernel(
    device float *A [[buffer(0)]],
    device float *B [[buffer(1)]],
    device float *C [[buffer(2)]],
    device int& Batch [[buffer(3)]],
    device int& M [[buffer(4)]],
    device int& K [[buffer(5)]],
    device int& N [[buffer(6)]],
    device int& A_major [[buffer(7)]],
    device int& B_major [[buffer(8)]],
    device int& A_batch_stride [[buffer(9)]],
    device int& B_batch_stride [[buffer(10)]],
    device int& C_batch_stride [[buffer(11)]],
    uint tid [[thread_position_in_grid]]
) {
    int batch = tid / (M * N);
    int row = (tid % (M * N)) / N;
    int column = (tid % (M * N)) % N;

    if(batch < Batch && row < M && column < N) {
        float value = 0.0f;
        for(int i = 0; i < K; ++i) {
            int A_index = batch * A_batch_stride + (A_major ? (row * K + i) : (i * M + row)); // Row Major vs Column Major
            int B_index = batch * B_batch_stride + (B_major ? (i * N + column) : (column * K + i)); // Row Major vs Column Major
            value += A[A_index] * B[B_index];
        }
        C[batch * C_batch_stride + row * N + column] = value;
    }
}
"
        .to_string();
        let name = "kernel_batch_matmul_2d".to_string();
        code = code.replace("mkernel", "kernel_batch_matmul_2d");

        compile_function(&name, &code, dev)
    }
}

impl Operator for MetalBatchMatmul2D {
    fn process(&self, inp: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        autoreleasepool(|| {
            let (a_shape, b_shape) = (inp[0].1.shape(), inp[1].1.shape());
            let a_strides = inp[0].1.strides();
            let (a_row_major, b_row_major) = (
                inp[0].1.indexes[0] < inp[0].1.indexes[1],
                inp[1].1.indexes[0] < inp[1].1.indexes[1],
            );
            let (batch_size, m, k, n) = (
                a_shape[0].to_usize().unwrap(),
                a_shape[1].to_usize().unwrap(),
                a_shape[2].to_usize().unwrap(),
                b_shape[1].to_usize().unwrap(),
            );
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

            let out = self.1.new_buffer(
                (batch_size * m * n * std::mem::size_of::<f32>()) as u64,
                MTLResourceOptions::StorageModeShared,
            );

            // Setup command queue / command buffer / encoder
            let command_queue = self.1.new_command_queue();
            let command_buffer = command_queue.new_command_buffer();
            let encoder = command_buffer
                .compute_command_encoder_with_descriptor(ComputePassDescriptor::new());
            encoder.set_compute_pipeline_state(&self.0);

            // Set inputs
            encoder.set_buffer(0, Some(a), 0);
            encoder.set_buffer(1, Some(b), 0);
            encoder.set_buffer(2, Some(&out), 0);
            encoder.set_int(3, batch_size as u32);
            encoder.set_int(4, m as u32);
            encoder.set_int(5, k as u32);
            encoder.set_int(6, n as u32);
            encoder.set_int(7, a_row_major as u32);
            encoder.set_int(8, b_row_major as u32);
            encoder.set_int(9, a_strides[0].to_usize().unwrap() as u32);
            encoder.set_int(10, 0);
            encoder.set_int(11, (m * n) as u32);

            // Execute
            encoder.dispatch_1d(batch_size * n * m);
            encoder.end_encoding();
            command_buffer.commit();
            command_buffer.wait_until_completed();

            vec![Tensor {
                data: Box::new(out),
            }]
        })
    }
}

#[derive(Default)]
pub struct MetalMatMulCompiler;

impl Compiler for MetalMatMulCompiler {
    fn compile(&self, graph: &mut Graph) {
        let dev = Device::system_default().unwrap();
        // Look for the matmul pattern
        let (mut sum_reduce, mut mul) = (NodeIndex::default(), NodeIndex::default());
        // Mul ([A, C(fake), B] | [A(fake), C, B]) -> SumReduce(2) -> [A, C]
        // Actually starts at [A,B] | [B, C]
        let s = SelectEdge::new(
            SelectOp::new()
                .ty::<MetalMul<f32>>()
                .shapes(vec![
                    vec!['A'.into(), 'C'.into(), 'B'.into()],
                    vec!['A'.into(), 'C'.into(), 'B'.into()],
                ])
                .fakes(vec![
                    vec![Some(false), Some(true), Some(false)],
                    vec![Some(true), Some(false), Some(false)],
                ])
                .ptr(&mut mul),
            SelectOp::new()
                .ty::<MetalSumReduce<f32>>()
                .check(|o, _| {
                    if let Some(o) = o.as_any().downcast_ref::<MetalSumReduce<f32>>() {
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
            if !src1_shape.is_contiguous() {
                src1 = graph
                    .add_op(MetalContiguous::<f32>::new(
                        src1_shape,
                        dev.clone(),
                        &mut HashMap::default(),
                        &graph.dyn_map,
                    ))
                    .input(src1, 0, src1_shape)
                    .finish();
                src1_shape = src1_shape.contiguous();
            }
            if !src2_shape.is_contiguous() {
                src2 = graph
                    .add_op(MetalContiguous::<f32>::new(
                        src2_shape,
                        dev.clone(),
                        &mut HashMap::default(),
                        &graph.dyn_map,
                    ))
                    .input(src2, 0, src2_shape)
                    .finish();
                src2_shape = src2_shape.contiguous();
            }
            let new_op = graph
                .add_op(MetalMatmul2D::new(&dev))
                .input(src1, 0, src1_shape)
                .input(src2, 0, src2_shape)
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
                .ty::<MetalMul<f32>>()
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
                .ty::<MetalSumReduce<f32>>()
                .check(|o, _| {
                    if let Some(o) = o.as_any().downcast_ref::<MetalSumReduce<f32>>() {
                        o.3 == 3
                    } else {
                        false
                    }
                })
                .ptr(&mut sum_reduce),
        );
        let mut batched_matmul = None;
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
            if batched_matmul.is_none() {
                batched_matmul = Some(MetalBatchMatmul2D::compile(&dev));
            }
            let new_op = graph
                .add_op(MetalBatchMatmul2D(
                    batched_matmul.clone().unwrap(),
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
