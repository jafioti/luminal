use half::f16;
use petgraph::stable_graph::NodeIndex;

use crate::{
    op::{InputTensor, Operator},
    optimizers::metal::*,
    prelude::*,
};

use super::{
    mean_reduce::MetalMeanReduce,
    prim::{MetalAdd, MetalCopyToDevice, MetalMul, MetalRecip, MetalSqrt},
};
use metal_rs::{objc::rc::autoreleasepool, *};

/// Special kernel for efficient mean reduction
#[derive(Debug, Clone)]
pub struct MetalRMSNormPostMean(ComputePipelineState, Device, ShapeTracker, ShapeTracker);
impl PartialEq for MetalRMSNormPostMean {
    fn eq(&self, _: &Self) -> bool {
        false
    }
}

impl MetalRMSNormPostMean {
    fn new(dev: Device, inp_shape: ShapeTracker, x_shape: ShapeTracker) -> Self {
        let (inp_idx, _) = get_idx_valid_exps(inp_shape);
        let (x_idx, _) = get_idx_valid_exps(x_shape);
        let mut code = format!("
#include <metal_stdlib>
using namespace metal;


kernel void mkernel(device half *inp [[buffer(0)]], device half *x [[buffer(1)]], device half *out [[buffer(2)]], device uint& n_elements [[buffer(3)]], uint idx [[thread_position_in_grid]]{}) {{
    if (idx < n_elements) {{
        out[idx] = (half)(1.0h / sqrt((float)inp[{inp_idx}] + 1e-6f) * (float)x[{x_idx}]);
    }}
}}", render_dyn_dim_inputs(&[inp_shape, x_shape], 4));
        code = code.replace("mkernel", "kernel_rmsnorm_post_mean");

        Self(
            compile_function("kernel_rmsnorm_post_mean", &code, &dev),
            dev,
            inp_shape,
            x_shape,
        )
    }
}

impl Operator for MetalRMSNormPostMean {
    fn process(&self, tensors: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        autoreleasepool(|| {
            let inp_size = tensors[1].1.n_physical_elements();
            // Setup buffers
            let a = tensors[0]
                .0
                .borrowed()
                .data
                .as_any()
                .downcast_ref::<Buffer>()
                .unwrap();
            let x = tensors[1]
                .0
                .borrowed()
                .data
                .as_any()
                .downcast_ref::<Buffer>()
                .unwrap();
            let out = self.1.new_buffer(
                (inp_size * std::mem::size_of::<f16>()) as u64,
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
            encoder.set_buffer(1, Some(x), 0);
            encoder.set_buffer(2, Some(&out), 0);
            encoder.set_int(3, inp_size as u32);
            input_dyn_dims(
                &[(self.2, tensors[0].1), (self.3, tensors[1].1)],
                encoder,
                4,
            );

            // Execute
            encoder.dispatch_n_elements(inp_size);
            encoder.end_encoding();
            command_buffer.commit();
            command_buffer.wait_until_completed();

            vec![Tensor {
                data: Box::new(out),
            }]
        })
    }
}

/// Replace the mean reduce pattern with a special kernel. This is meant to be ran **after** the FakeSumReduceOptimizer.
#[derive(Default)]
pub struct RMSNormOptimizer;

impl GraphOptimizer for RMSNormOptimizer {
    fn optimize(&self, graph: &mut Graph) {
        let dev = Device::system_default().unwrap();
        // Look for the RMSNorm pattern
        // mul(recip(sqrt(add(mean_reduce(mul(x, x)), 1e-6))), x)
        let s = GraphSelector::default();
        let (mut og_mul, mut add, mut sqrt, mut recip, mut mul, mut epsilon, mut copy_to) = (
            NodeIndex::default(),
            NodeIndex::default(),
            NodeIndex::default(),
            NodeIndex::default(),
            NodeIndex::default(),
            NodeIndex::default(),
            NodeIndex::default(),
        );

        let x = s.op();
        s.edge(
            x,
            0,
            s.edge(
                s.edge(
                    s.edge(
                        s.edge(
                            s.edge(
                                s.op().ty::<crate::op::Function>().ptr(&mut epsilon),
                                0,
                                s.op().ty::<MetalCopyToDevice>().ptr(&mut copy_to),
                            ),
                            0,
                            s.edge(
                                s.edge(
                                    s.edge(
                                        x,
                                        0,
                                        s.edge(x, 0, s.op().ty::<MetalMul>().ptr(&mut og_mul)),
                                    ),
                                    0,
                                    s.op().ty::<MetalMeanReduce>(),
                                ),
                                0,
                                s.op().ty::<MetalAdd>().ptr(&mut add),
                            ),
                        ),
                        0,
                        s.op().ty::<MetalSqrt>().ptr(&mut sqrt),
                    ),
                    0,
                    s.op().ty::<MetalRecip>().ptr(&mut recip),
                ),
                0,
                s.op().ty::<MetalMul>().ptr(&mut mul),
            ),
        );

        for _ in s.search(graph) {
            if graph.no_delete.contains(&add)
                || graph.no_delete.contains(&sqrt)
                || graph.no_delete.contains(&recip)
                || graph.no_delete.contains(&mul)
                || graph.no_delete.contains(&epsilon)
                || graph.no_delete.contains(&copy_to)
            {
                // An intermediate node can't be deleted
                continue;
            }

            // Insert RMSNorm op
            let x = graph.get_sources(og_mul)[0];
            graph.graph.remove_node(copy_to);
            let meaned = graph.get_sources(add)[0];
            let rms_norm = graph
                .add_op(MetalRMSNormPostMean::new(dev.clone(), meaned.1, x.1))
                .input(meaned.0, 0, meaned.1)
                .input(x.0, 0, x.1)
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
        }
    }
}
