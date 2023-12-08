use std::sync::Arc;

use half::f16;
use petgraph::stable_graph::NodeIndex;

use crate::{
    compilers::metal::*,
    op::{InputTensor, Operator},
    prelude::*,
};

use super::mean_reduce::MetalMeanReduce;
use metal_rs::{objc::rc::autoreleasepool, *};

/// Special kernel for efficient rms norming
#[derive(LuminalEq, LuminalPrint, Clone)]
pub struct MetalRMSNorm(
    ComputePipelineState, // Square-Mean kernel
    ComputePipelineState, // RMSNorm kernel
    Device,
    ShapeTracker, // Input shape
    *const HashMap<char, usize>,
);

impl MetalRMSNorm {
    fn new(dev: Device, inp_shape: ShapeTracker, dyn_map: *const HashMap<char, usize>) -> Self {
        let (idx_exp, valid_exp) = get_idx_valid_exps(inp_shape);
        let mut square_mean_code = format!(
            "
#include <metal_stdlib>
using namespace metal;
kernel void mkernel(device half *inp [[buffer(0)]], device float *out [[buffer(1)]], device uint& n_elements [[buffer(2)]], device uint& front_size [[buffer(3)]], device uint& back_size [[buffer(4)]], device uint& dim_size [[buffer(5)]], uint i_ [[thread_position_in_grid]]{}) {{
    if (i_ < n_elements) {{
        uint a_ = i_ / back_size;
        uint b_ = i_ % back_size;
        float reduce_value = 0.0;
        uint add_factor = a_ * dim_size * back_size + b_;
        for (uint c_ = 0; c_ < dim_size * back_size; c_ += back_size) {{
            uint idx = add_factor + c_;
            if (({valid_exp}) != 0) {{
                float val = (float)inp[{idx_exp}];
                reduce_value += (val * val);
            }}
        }}
        out[i_] = (reduce_value / (float)dim_size);
    }}
}}
", render_dyn_dim_inputs(&[inp_shape], 6),
        );
        let square_mean_code_name = format!("kernel_{}", hash(&square_mean_code));
        square_mean_code = square_mean_code.replace("mkernel", &square_mean_code_name);

        let mut meaned_shape = inp_shape;
        let meaned_size = meaned_shape.remove_dim(meaned_shape.len() - 1);
        meaned_shape.expand(meaned_shape.len(), meaned_size);
        let (meaned_idx_exp, _) = get_idx_valid_exps(meaned_shape);
        let mut rms_norm_code = format!("
#include <metal_stdlib>
using namespace metal;

kernel void mkernel(device float *inp [[buffer(0)]], device half *x [[buffer(1)]], device half *out [[buffer(2)]], device uint& n_elements [[buffer(3)]], uint idx [[thread_position_in_grid]]{}) {{
    if (idx < n_elements) {{
        float added = inp[{meaned_idx_exp}] + 1e-6f;
        float sq = sqrt(added);
        float recip = 1.0f / sq;
        out[idx] = (half)(recip * (float)x[{idx_exp}]);
    }}
}}", render_dyn_dim_inputs(&[inp_shape], 4));
        let rms_norm_code_name = format!("kernel_{}", hash(&rms_norm_code));
        rms_norm_code = rms_norm_code.replace("mkernel", &rms_norm_code_name);

        Self(
            compile_function(&square_mean_code_name, &square_mean_code, &dev),
            compile_function(&rms_norm_code_name, &rms_norm_code, &dev),
            dev,
            inp_shape,
            dyn_map,
        )
    }
}

impl MetalKernelForward for MetalRMSNorm {
    fn metal_forward(
        &self,
        inputs: &[(&Buffer, ShapeTracker)],
        dev: &Device,
        command_buffer: &CommandBufferRef,
    ) -> Vec<Buffer> {
        let mut meaned_shape = inputs[0].1;
        meaned_shape.remove_dim(meaned_shape.len() - 1);
        // Setup buffers
        let meaned = dev.new_buffer(
            (meaned_shape.n_elements() * std::mem::size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeManaged,
        );
        let front_size: usize = inputs[0]
            .1
            .shape()
            .iter()
            .take(meaned_shape.len())
            .map(|i| i.to_usize().unwrap())
            .product();
        let back_size = 1;
        let dim_size = inputs[0].1.shape()[meaned_shape.len()].to_usize().unwrap();

        let encoder =
            command_buffer.compute_command_encoder_with_descriptor(ComputePassDescriptor::new());
        encoder.set_compute_pipeline_state(&self.0);

        // Set inputs
        encoder.set_buffer(0, Some(inputs[0].0), 0);
        encoder.set_buffer(1, Some(&meaned), 0);
        encoder.set_int(2, meaned_shape.n_elements() as u32);
        encoder.set_int(3, front_size as u32);
        encoder.set_int(4, back_size as u32);
        encoder.set_int(5, dim_size as u32);
        input_dyn_dims(&[self.3], unsafe { self.4.as_ref().unwrap() }, encoder, 6);

        encoder.dispatch_1d(meaned_shape.n_elements());
        encoder.end_encoding();

        let out = dev.new_buffer(
            (inputs[0].1.n_elements() * std::mem::size_of::<f16>()) as u64,
            MTLResourceOptions::StorageModeManaged,
        );

        let encoder =
            command_buffer.compute_command_encoder_with_descriptor(ComputePassDescriptor::new());
        encoder.set_compute_pipeline_state(&self.1);

        // Set inputs
        encoder.set_buffer(0, Some(&meaned), 0);
        encoder.set_buffer(1, Some(inputs[0].0), 0);
        encoder.set_buffer(2, Some(&out), 0);
        encoder.set_int(3, inputs[0].1.n_elements() as u32);
        input_dyn_dims(&[self.3], unsafe { self.4.as_ref().unwrap() }, encoder, 4);

        // Execute
        encoder.dispatch_1d(inputs[0].1.n_elements());
        encoder.end_encoding();

        vec![out]
    }
}

impl Operator for MetalRMSNorm {
    fn process(&self, tensors: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        autoreleasepool(|| {
            let command_queue = self.2.new_command_queue();
            let command_buffer = command_queue.new_command_buffer();
            let a = tensors[0]
                .0
                .borrowed()
                .data
                .as_any()
                .downcast_ref::<Buffer>()
                .unwrap();

            let out = self
                .metal_forward(&[(a, tensors[0].1)], &self.2, command_buffer)
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

/// Replace the mean reduce pattern with a special kernel. This is meant to be ran **after** the FakeSumReduceCompiler.
#[derive(Default)]
pub struct RMSNormCompiler;

impl Compiler for RMSNormCompiler {
    fn compile(&self, graph: &mut Graph) {
        let dev = Device::system_default().unwrap();
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

        let s = SelectEdge::new(
            SelectEdge::new(
                SelectEdge::new(
                    SelectEdge::new(
                        SelectOp::new()
                            .check(|op, _| {
                                if let Some(c) = op.as_any().downcast_ref::<MetalConstant<f16>>() {
                                    c.0 == f16::from_f32(1e-6)
                                } else {
                                    false
                                }
                            })
                            .ptr(&mut epsilon),
                        SelectEdge::new(
                            SelectEdge::new(
                                SelectOp::new().ty::<MetalMul<f16>>().ptr(&mut square),
                                SelectOp::new().ty::<MetalMeanReduce>().ptr(&mut mean),
                            ),
                            SelectOp::new().ty::<MetalAdd<f16>>().ptr(&mut add),
                        ),
                    ),
                    SelectOp::new().ty::<MetalSqrt<f16>>().ptr(&mut sqrt),
                ),
                SelectOp::new().ty::<MetalRecip<f16>>().ptr(&mut recip),
            ),
            SelectOp::new().ty::<MetalMul<f16>>().ptr(&mut mul),
        );

        for _ in s.search(graph) {
            if graph.no_delete.contains(&add)
                || graph.no_delete.contains(&sqrt)
                || graph.no_delete.contains(&recip)
                || graph.no_delete.contains(&mul)
                || graph.no_delete.contains(&epsilon)
                || graph.no_delete.contains(&square)
                || graph.no_delete.contains(&mean)
            {
                // An intermediate node can't be deleted
                continue;
            }
            let x = graph.get_sources(square)[0];
            if !graph.get_sources(square).iter().all(|(i, _, _)| *i == x.0) {
                continue;
            }
            if !graph.get_sources(mul).iter().any(|(i, _, _)| *i == x.0) {
                continue;
            }

            // Insert RMSNorm op
            let rms_norm = graph
                .add_op(MetalRMSNorm::new(dev.clone(), x.2, &graph.dyn_map))
                .input(x.0, 0, x.2)
                .finish();

            // Create edges to dests
            move_outgoing_edge(mul, rms_norm, &mut graph.graph);
            move_references(
                &mut graph.id_remap,
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
