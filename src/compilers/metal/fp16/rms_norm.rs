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

/// Special kernel for efficient rms norming
#[derive(LuminalEq, LuminalPrint, Clone)]
pub struct MetalRMSNorm {
    pipeline: ComputePipelineState,
    device: Device,
    queue: CommandQueue,
    epsilon: f32, // Epsilon
}

impl MetalRMSNorm {
    fn new(epsilon: f32, device: Device, queue: CommandQueue) -> Self {
        let kernel_code = "#include <metal_stdlib>
#define SIMD_WIDTH 32

using namespace metal;
kernel void kernel_rms_norm(
        device const  void * src0 [[buffer(0)]],
        device       half * dst [[buffer(1)]],
        constant   int64_t & ne00 [[buffer(2)]],
        constant  uint64_t & nb01 [[buffer(3)]],
        constant     float & eps [[buffer(4)]],
        threadgroup float  * buf [[threadgroup(0)]],
        uint tgpig[[threadgroup_position_in_grid]],
        uint tpitg[[thread_position_in_threadgroup]],
        uint sgitg[[simdgroup_index_in_threadgroup]],
        uint tiisg[[thread_index_in_simdgroup]],
        uint   ntg[[threads_per_threadgroup]]) {
    device const half4 * x = (device const half4 *) ((device const char *) src0 + tgpig*nb01);

    float4 sumf = 0;
    float all_sum = 0;

    // parallel sum
    for (int i00 = tpitg; i00 < ne00/4; i00 += ntg) {
        sumf += (float4)x[i00] * (float4)x[i00];
    }
    all_sum = sumf[0] + sumf[1] + sumf[2] + sumf[3];
    all_sum = simd_sum(all_sum);
    if (ntg > SIMD_WIDTH) {
        if (sgitg == 0) {
            buf[tiisg] = 0.0f;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tiisg == 0) {
            buf[sgitg] = all_sum;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        all_sum = buf[tiisg];
        all_sum = simd_sum(all_sum);
    }

    const float mean  = all_sum/ne00;
    const float scale = 1.0f/sqrt(mean + eps);

    device half4 * y = (device half4 *) (dst + tgpig*ne00);
    for (int i00 = tpitg; i00 < ne00/4; i00 += ntg) {
        y[i00] = (half4)(x[i00] * scale);
    }
}";

        Self {
            pipeline: compile_function(&"kernel_rms_norm", kernel_code, &device),
            device,
            queue,
            epsilon,
        }
    }
}

impl MetalKernel for MetalRMSNorm {
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
        let ne00 = inputs[0].1.shape()[2].to_usize().unwrap();
        let nb01 = inputs[0].1.strides()[1].to_usize().unwrap() * size_of::<f16>();

        // Set inputs
        encoder.set_buffer(0, Some(inputs[0].0), 0);
        encoder.set_buffer(1, Some(output_buffers[0]), 0);
        encoder.set_i64(2, ne00 as i64);
        encoder.set_u64(3, nb01 as u64);
        encoder.set_f32(4, self.epsilon);

        let mut nth = 32; // SIMD width
        while nth < ne00 / 4 && nth < 1024 {
            nth *= 2;
        }
        let n_rows = inputs[0]
            .1
            .shape()
            .into_iter()
            .take(2)
            .map(|i| i.to_usize().unwrap())
            .product::<usize>();
        encoder.set_threadgroup_memory_length(0, 32 * size_of::<f32>() as u64);
        encoder.dispatch_thread_groups(
            MTLSize {
                width: n_rows as u64,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: nth as u64,
                height: 1,
                depth: 1,
            },
        );
        encoder.end_encoding();
    }
}

impl Operator for MetalRMSNorm {
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
pub struct RMSNormCompiler;

impl Compiler for RMSNormCompiler {
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
                unreachable!()
            };
            let (mut x, _, mut sh) = graph.get_sources(square)[0];
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
                        &mut HashMap::new(),
                        &graph.dyn_map,
                    ))
                    .input(x, 0, sh)
                    .finish();
                sh = sh.contiguous();
            }

            // Insert RMSNorm op
            let rms_norm = graph
                .add_op(MetalRMSNorm::new(epsilon_num, dev.clone(), queue.clone()))
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
