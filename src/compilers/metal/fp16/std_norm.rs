use std::{mem::size_of, sync::Arc};

use half::f16;
use petgraph::stable_graph::NodeIndex;

use crate::{
    compilers::metal::{prim::*, *},
    op::{ConstantValue, InputTensor, Operator},
    prelude::*,
};

use super::mean_reduce::MetalMeanReduce;
use metal_rs::{objc::rc::autoreleasepool, *};

/// Special kernel for efficient std norming
#[derive(LuminalEq, LuminalPrint, Clone)]
pub struct MetalStdNorm {
    pipeline: ComputePipelineState,
    device: Device,
    queue: CommandQueue,
    epsilon: f32, // Epsilon
}

impl MetalStdNorm {
    fn new(epsilon: f32, device: Device, queue: CommandQueue) -> Self {
        let kernel_code = "#include <metal_stdlib>
#define SIMD_WIDTH 32

using namespace metal;
kernel void kernel_std_norm(
        device const  half * src0 [[buffer(0)]],
        device       half * dst [[buffer(1)]],
        constant   int64_t & row_size [[buffer(2)]],
        constant     float & eps [[buffer(3)]],
        threadgroup float  * buf [[threadgroup(0)]],
        uint threadgroup_pos[[threadgroup_position_in_grid]],
        uint simdgroup_pos[[thread_index_in_simdgroup]]) {
    device const half4 * x = (device const half4 *) (src0 + threadgroup_pos * row_size);

    float4 sumf = 0;

    // parallel sum
    for (int i = simdgroup_pos; i < row_size/4; i += SIMD_WIDTH) {
        sumf += (float4)x[i] * (float4)x[i];
    }
    float all_sum = sumf[0] + sumf[1] + sumf[2] + sumf[3];
    all_sum = simd_sum(all_sum);
    const float mean  = all_sum/row_size;
    const float scale = 1.0f/sqrt(mean + eps);

    device half4 * y = (device half4 *) (dst + threadgroup_pos * row_size);
    for (int i = simdgroup_pos; i < row_size/4; i += SIMD_WIDTH) {
        y[i] = (half4)(x[i] * scale);
    }
}";

        Self {
            pipeline: compile_function(&"kernel_std_norm", kernel_code, &device),
            device,
            queue,
            epsilon,
        }
    }
}

impl MetalKernel for MetalStdNorm {
    fn output_buffer_sizes(&self, input_shapes: &[ShapeTracker]) -> Vec<BigExpression> {
        vec![input_shapes[0].n_elements() * size_of::<f16>()]
    }

    fn metal_forward(
        &self,
        inputs: &[(&Buffer, ShapeTracker)],
        command_buffer: &CommandBufferRef,
        _: &[&Buffer],
        output_buffers: &[&Buffer],
    ) {
        let encoder =
            command_buffer.compute_command_encoder_with_descriptor(ComputePassDescriptor::new());
        encoder.set_compute_pipeline_state(&self.pipeline);
        let row_size = inputs[0].1.shape().last().unwrap().to_usize().unwrap();

        // Set inputs
        encoder.set_buffer(0, Some(inputs[0].0), 0);
        encoder.set_buffer(1, Some(output_buffers[0]), 0);
        encoder.set_i64(2, row_size as i64);
        encoder.set_f32(3, self.epsilon);
        let batch_size = inputs[0]
            .1
            .shape()
            .into_iter()
            .take(inputs[0].1.len() - 1)
            .map(|i| i.to_usize().unwrap())
            .product::<usize>();
        encoder.set_threadgroup_memory_length(0, 32 * size_of::<f32>() as u64);
        encoder.dispatch_thread_groups(
            MTLSize {
                width: batch_size as u64,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 32.min(row_size / 4) as u64,
                height: 1,
                depth: 1,
            },
        );
        encoder.end_encoding();
    }
}

impl Operator for MetalStdNorm {
    fn process(&mut self, tensors: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        autoreleasepool(|| {
            let command_buffer = self.queue.new_command_buffer();
            let a = tensors[0]
                .0
                .borrowed()
                .data
                .as_any()
                .downcast_ref::<Buffer>()
                .unwrap();
            let out = self.device.new_buffer(
                (tensors[0].1.n_elements().to_usize().unwrap() * size_of::<f16>()) as u64,
                MTLResourceOptions::StorageModeShared,
            );

            self.metal_forward(&[(a, tensors[0].1)], command_buffer, &[], &[&out]);

            command_buffer.commit();
            command_buffer.wait_until_completed();

            vec![Tensor::new(out)]
        })
    }

    fn custom(&mut self, key: &str, _: Box<dyn Any>) -> Option<Box<dyn Any>> {
        if key == "metal" {
            #[allow(clippy::arc_with_non_send_sync)]
            return Some(Box::new(MetalKernelWrapper(Arc::new(Box::new(
                self.clone(),
            )))));
        }
        None
    }
}

/// Replace the mean reduce pattern with a special kernel. This is meant to be ran **after** the FakeSumReduceCompiler.
#[derive(Default, Debug)]
pub struct StdNormCompiler;

impl Compiler for StdNormCompiler {
    fn compile<T: ToIdsMut>(&self, graph: &mut Graph, mut remap: T) {
        let dev = Device::system_default().unwrap();
        let queue = dev.new_command_queue();
        // Look for the RMSNorm pattern
        // mul(recip(sqrt(add(mean_reduce(mul(x, x)), 1e-6))), x)
        let (mut square, mut mean, mut add, mut sqrt, mut recip, mut mul, mut epsilon) = (
            NodeIndex::default(),
            NodeIndex::default(),
            NodeIndex::default(),
            NodeIndex::default(),
            NodeIndex::default(),
            NodeIndex::default(),
            NodeIndex::default(),
        );

        let s = SelectOp::new()
            .ty::<MetalMul<f16>>()
            .ptr(&mut square)
            .edge(SelectOp::new().ty::<MetalMeanReduce>().ptr(&mut mean))
            .edge(
                SelectOp::new()
                    .check(|op, _| {
                        if let Some(c) = op.as_any().downcast_ref::<MetalConstant<f16>>() {
                            if let ConstantValue::Float(v) = c.0 {
                                v <= 1e-4 && v > 0.0
                            } else {
                                false
                            }
                        } else {
                            false
                        }
                    })
                    .ptr(&mut epsilon)
                    .edge(SelectOp::new().ty::<MetalAdd<f16>>().ptr(&mut add)),
            )
            .edge(SelectOp::new().ty::<MetalSqrt<f16>>().ptr(&mut sqrt))
            .edge(SelectOp::new().ty::<MetalRecip<f16>>().ptr(&mut recip))
            .edge(SelectOp::new().ty::<MetalMul<f16>>().ptr(&mut mul));

        let mut searcher = s.search(graph);
        while searcher.next_match() {
            if check_no_delete(graph, &[add, sqrt, recip, mul, epsilon, square, mean]) {
                // An intermediate node can't be deleted
                continue;
            }
            let ConstantValue::Float(epsilon_num) = graph
                .graph
                .node_weight(epsilon)
                .unwrap()
                .as_any()
                .downcast_ref::<MetalConstant<f16>>()
                .unwrap()
                .0
            else {
                continue;
            };
            let (mut x, _, mut sh) = graph.get_sources(square)[0];
            if let Some(mean_reduce) = graph
                .graph
                .node_weight(mean)
                .unwrap()
                .as_any()
                .downcast_ref::<MetalMeanReduce>()
            {
                if mean_reduce.3 != sh.len() - 1 {
                    continue;
                }
            }
            if sh
                .shape()
                .last()
                .unwrap()
                .to_usize()
                .map(|i| i % 32 != 0 || i < 32)
                .unwrap_or(true)
            {
                continue;
            }
            if !graph.get_sources(square).iter().all(|(i, _, _)| *i == x) {
                continue;
            }
            if !graph.get_sources(mul).iter().any(|(i, _, _)| *i == x) {
                continue;
            }

            // Input must be contiguous
            if !sh.is_contiguous() || sh.is_sliced() || sh.is_padded() {
                x = graph
                    .add_op(MetalContiguous::<f16>::new(
                        sh,
                        dev.clone(),
                        queue.clone(),
                        &graph.dyn_map,
                    ))
                    .input(x, 0, sh)
                    .finish();
                sh = sh.contiguous();
            }

            // Insert RMSNorm op
            let rms_norm = graph
                .add_op(MetalStdNorm::new(epsilon_num, dev.clone(), queue.clone()))
                .input(x, 0, sh)
                .finish();

            // Create edges to dests
            move_outgoing_edge(mul, rms_norm, &mut graph.graph);
            move_references(
                &mut remap,
                &mut graph.no_delete,
                &mut graph.to_retrieve,
                mul,
                rms_norm,
            );

            // Remove the old ops
            graph.graph.remove_node(mul);
            graph.graph.remove_node(add);
            graph.graph.remove_node(recip);
            graph.graph.remove_node(epsilon);
            graph.graph.remove_node(sqrt);
            graph.graph.remove_node(square);
            graph.graph.remove_node(mean);
        }
    }
}
