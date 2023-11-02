use petgraph::stable_graph::NodeIndex;

use crate::{
    op::{InputTensor, Operator},
    optimizers::metal::*,
    prelude::*,
};

use super::prim::{MetalMul, MetalSumReduce};
use metal_rs::{objc::rc::autoreleasepool, *};

/// Multiplies a MxK matrix with a KxN matrix, resulting in a MxN matrix
#[derive(Debug, Clone)]
pub struct MetalMatmul2D(ComputePipelineState, Device);
impl PartialEq for MetalMatmul2D {
    fn eq(&self, _: &Self) -> bool {
        false
    }
}

impl MetalMatmul2D {
    fn compile(dev: &Device) -> ComputePipelineState {
        let mut code = "#include <metal_stdlib>
using namespace metal;

kernel void mkernel(
    device float *A [[buffer(0)]],
    device float *B [[buffer(1)]],
    device float *C [[buffer(2)]],
    device uint& M [[buffer(3)]],
    device uint& K [[buffer(4)]],
    device uint& N [[buffer(5)]],
    device uint& A_major [[buffer(6)]],
    device uint& B_major [[buffer(7)]],
    uint tid [[thread_position_in_grid]]
) {
    uint row = tid / N;
    uint column = tid % N;

    if(row < M && column < N) {
        float value = 0.0f;
        for(int i = 0; i < K; ++i) {
            uint A_index = A_major ? (row * K + i) : (i * M + row); // Row Major vs Column Major
            uint B_index = B_major ? (i * N + column) : (column * K + i); // Row Major vs Column Major
            value += A[A_index] * B[B_index];
        }
        C[row * N + column] = value;
    }
}
"
        .to_string();
        let name = "kernel_matmul_2d".to_string();
        code = code.replace("mkernel", "kernel_matmul_2d");

        compile_function(&name, &code, dev)
    }
}

impl Operator for MetalMatmul2D {
    fn process(&self, inp: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        autoreleasepool(|| {
            let (a_shape, b_shape) = (inp[0].1.shape(), inp[1].1.shape());
            let (a_strides, b_strides) = (inp[0].1.strides(), inp[1].1.strides());
            let (a_row_major, b_row_major) =
                (a_strides[0] > a_strides[1], b_strides[0] > b_strides[1]);
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

            let out = self.1.new_buffer(
                (m * n * std::mem::size_of::<f32>()) as u64,
                MTLResourceOptions::StorageModeManaged,
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
            encoder.set_int(3, m as u32);
            encoder.set_int(4, k as u32);
            encoder.set_int(5, n as u32);
            encoder.set_int(6, a_row_major as u32);
            encoder.set_int(7, b_row_major as u32);

            // Execute
            encoder.dispatch_n_elements(n * m);
            encoder.end_encoding();
            command_buffer.commit();
            command_buffer.wait_until_completed();

            vec![Tensor {
                data: Box::new(out),
            }]
        })
    }
}

/// Multiplies a BxMxK matrix with a BxKxN matrix, resulting in a BxMxN matrix
#[derive(Debug, Clone)]
pub struct MetalBatchMatmul2D(ComputePipelineState, Device);
impl PartialEq for MetalBatchMatmul2D {
    fn eq(&self, _: &Self) -> bool {
        false
    }
}

impl MetalBatchMatmul2D {
    fn compile(dev: &Device) -> ComputePipelineState {
        let mut code = "#include <metal_stdlib>
using namespace metal;

kernel void mkernel(
    device float *A [[buffer(0)]],
    device float *B [[buffer(1)]],
    device float *C [[buffer(2)]],
    device uint& Batch [[buffer(3)]],
    device uint& M [[buffer(4)]],
    device uint& K [[buffer(5)]],
    device uint& N [[buffer(6)]],
    device uint& A_major [[buffer(7)]],
    device uint& B_major [[buffer(8)]],
    device uint& A_batch_stride [[buffer(9)]],
    device uint& B_batch_stride [[buffer(10)]],
    device uint& C_batch_stride [[buffer(11)]],
    uint tid [[thread_position_in_grid]]
) {
    uint batch = tid / (M * N);
    uint row = (tid % (M * N)) / N;
    uint column = (tid % (M * N)) % N;

    if(batch < Batch && row < M && column < N) {
        float value = 0.0f;
        for(uint i = 0; i < K; ++i) {
            uint A_index = batch * A_batch_stride + (A_major ? (row * K + i) : (i * M + row)); // Row Major vs Column Major
            uint B_index = batch * B_batch_stride + (B_major ? (i * N + column) : (column * K + i)); // Row Major vs Column Major
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
            let (a_strides, b_strides) = (inp[0].1.strides(), inp[1].1.strides());
            let (a_row_major, b_row_major) =
                (a_strides[1] > a_strides[2], b_strides[0] > b_strides[1]);
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
                MTLResourceOptions::StorageModeManaged,
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
            encoder.set_int(9, a_strides[0] as u32);
            encoder.set_int(10, 0);
            encoder.set_int(11, (m * n) as u32);

            // Execute
            encoder.dispatch_n_elements(batch_size * n * m);
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
pub struct MetalMatMulOptimizer;

impl GraphOptimizer for MetalMatMulOptimizer {
    fn optimize(&self, graph: &mut Graph) {
        let dev = Device::system_default().unwrap();
        // Look for the matmul pattern
        let s = GraphSelector::default();
        let (mut sum_reduce, mut mul) = (NodeIndex::default(), NodeIndex::default());
        // Mul ([A, C(fake), B] | [A(fake), C, B]) -> SumReduce(2) -> [A, C]
        // Actually starts at [A,B] | [B, C]
        s.edge(
            s.op()
                .ty::<MetalMul>()
                .shapes(vec![
                    vec![Dim::Unknown('A'), Dim::Unknown('C'), Dim::Unknown('B')],
                    vec![Dim::Unknown('A'), Dim::Unknown('C'), Dim::Unknown('B')],
                ])
                .fakes(vec![vec![false, true, false], vec![true, false, false]])
                .ptr(&mut mul),
            s.op()
                .ty::<MetalSumReduce>()
                .check(|o, _| {
                    if let Some(o) = o.as_any().downcast_ref::<MetalSumReduce>() {
                        o.2 == 2
                    } else {
                        false
                    }
                })
                .ptr(&mut sum_reduce),
        );

        let mut matmul = None;
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
            if matmul.is_none() {
                matmul = Some(MetalMatmul2D::compile(&dev));
            }
            let new_op = graph
                .add_op(MetalMatmul2D(matmul.clone().unwrap(), dev.clone()))
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
                .ty::<MetalMul>()
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
            s.op()
                .ty::<MetalSumReduce>()
                .check(|o, _| {
                    if let Some(o) = o.as_any().downcast_ref::<MetalSumReduce>() {
                        o.2 == 3
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
            srcs[0].1.remove_dim(2);
            srcs[1].1.remove_dim(1);
            srcs[1].1.remove_dim(0);
            srcs[1].1.permute(&[1, 0]);
            if batched_matmul.is_none() {
                batched_matmul = Some(MetalBatchMatmul2D::compile(&dev));
            }
            let new_op = graph
                .add_op(MetalBatchMatmul2D(
                    batched_matmul.clone().unwrap(),
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
