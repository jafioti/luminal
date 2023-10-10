use half::f16;
use petgraph::stable_graph::NodeIndex;

use crate::{
    op::{InputTensor, Operator},
    optimizers::metal::*,
    prelude::*,
};

use super::prim::{MetalAdd, MetalMul, MetalSin, MetalConstant, MetalExp2};
use metal_rs::{objc::rc::autoreleasepool, *};

/// Special kernel for efficient mean reduction
#[derive(Debug, Clone)]
pub struct MetalCos(ComputePipelineState, Device);
impl PartialEq for MetalCos {
    fn eq(&self, _: &Self) -> bool {
        false
    }
}

impl MetalCos {
    fn new(dev: Device) -> Self {
        let mut code = 
            "
#include <metal_stdlib>
using namespace metal;
kernel void mkernel(device half *inp [[buffer(0)]], device half *out [[buffer(1)]], device uint& n_elements [[buffer(2)]], uint i_ [[thread_position_in_grid]]) {{
    if (i_ < n_elements) {{
        out[i_] = (half)cos((float)inp[i_]);
    }}
}}
".to_string();
        code = code.replace("mkernel", "kernel_metal_cos");

        Self(compile_function("kernel_metal_cos", &code, &dev), dev)
    }
}

impl Operator for MetalCos {
    fn process(&self, tensors: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        autoreleasepool(|| {
            let inp_size = tensors[0].1.n_physical_elements();

            // Setup buffers
            let a = tensors[0]
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
            encoder.set_buffer(1, Some(&out), 0);
            encoder.set_int(2, inp_size as u32);

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
pub struct MetalCosOptimizer;

impl GraphOptimizer for MetalCosOptimizer {
    fn optimize(&self, graph: &mut Graph) {
        let dev = Device::system_default().unwrap();
        // Look for the mean-reduce pattern
        // sin(add(mul(const_neg_one, x), const_pi_over_2))
        let s = GraphSelector::default();
        let (
            mut const_neg_one,
            mut const_pi,
            mut mul,
            mut add,
            mut sin,
            mut x,
        ) = (
            NodeIndex::default(),
            NodeIndex::default(),
            NodeIndex::default(),
            NodeIndex::default(),
            NodeIndex::default(),
            NodeIndex::default(),
        );

        s.edge(
            s.edge(
                s.op().check(|op, _| if let Some(c) = op.as_any().downcast_ref::<MetalConstant>() {
                    c.0 == (f16::PI / f16::from_f32(2.))
                } else {false}).ptr(&mut const_pi),
                0,
                s.edge(
                    s.edge(
                        s.op().ptr(&mut x),
                        0,
                        s.edge(
                                s.op().check(|op, _| if let Some(c) = op.as_any().downcast_ref::<MetalConstant>() {
                                    c.0 == f16::NEG_ONE
                                } else {false}).ptr(&mut const_neg_one),
                            0,
                            s.op().ty::<MetalMul>().ptr(&mut mul),
                        ),
                    ),
                    0,
                    s.op().ty::<MetalAdd>().ptr(&mut add),
                ),
            ),
            0,
            s.op().ty::<MetalSin>().ptr(&mut sin),
        );
        for _ in s.search(graph) {
            if graph.no_delete.contains(&const_neg_one)
                || graph.no_delete.contains(&const_pi)
                || graph.no_delete.contains(&mul)
                || graph.no_delete.contains(&add)
            {
                // An intermediate node can't be deleted
                continue;
            }

            // Insert cos op
            let shape = graph.graph.edges_directed(mul, petgraph::Direction::Incoming).next().unwrap().weight().2;
            let cos = graph
                .add_op(MetalCos::new(dev.clone()))
                .input(x, 0, shape)
                .finish();

            // Create edges to dests
            move_outgoing_edge(sin, cos, &mut graph.graph);
            move_references(
                &mut graph.id_remap,
                &mut graph.no_delete,
                &mut graph.to_retrieve,
                sin,
                cos,
            );

            // Remove the old ops
            graph.graph.remove_node(mul);
            graph.graph.remove_node(add);
            graph.graph.remove_node(const_neg_one);
            graph.graph.remove_node(const_pi);
            graph.graph.remove_node(sin);
        }
    }
}

/// Special kernel for efficient mean reduction
#[derive(Debug, Clone)]
pub struct MetalExp(ComputePipelineState, Device);
impl PartialEq for MetalExp {
    fn eq(&self, _: &Self) -> bool {
        false
    }
}

impl MetalExp {
    fn new(dev: Device) -> Self {
        let mut code = 
            "
#include <metal_stdlib>
using namespace metal;
kernel void mkernel(device half *inp [[buffer(0)]], device half *out [[buffer(1)]], device uint& n_elements [[buffer(2)]], uint i_ [[thread_position_in_grid]]) {{
    if (i_ < n_elements) {{
        out[i_] = exp(inp[i_]);
    }}
}}
".to_string();
        code = code.replace("mkernel", "kernel_metal_exp");

        Self(compile_function("kernel_metal_exp", &code, &dev), dev)
    }
}

impl Operator for MetalExp {
    fn process(&self, tensors: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        autoreleasepool(|| {
            let inp_size = tensors[0].1.n_physical_elements();

            // Setup buffers
            let a_inp = tensors[0]
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
            encoder.set_buffer(0, Some(a_inp), 0);
            encoder.set_buffer(1, Some(&out), 0);
            encoder.set_int(2, inp_size as u32);

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
pub struct MetalExpOptimizer;

impl GraphOptimizer for MetalExpOptimizer {
    fn optimize(&self, graph: &mut Graph) {
        let dev = Device::system_default().unwrap();
        // Look for the exp pattern
        // exp2(mul(x, const))
        let s = GraphSelector::default();
        let (
            mut constant,
            mut mul,
            mut exp2
        ) = (
            NodeIndex::default(),
            NodeIndex::default(),
            NodeIndex::default(),
        );

        s.edge(s.edge(s.op().check(|op, _| if let Some(c) = op.as_any().downcast_ref::<MetalConstant>() {
            c.0 == f16::from_f32(1.0 / f32::ln(2.))
        } else {false}).ptr(&mut constant), 0, s.op().ty::<MetalMul>().ptr(&mut mul)), 0, s.op().ty::<MetalExp2>().ptr(&mut exp2));

        for _ in s.search(graph) {
            if graph.no_delete.contains(&constant)
                || graph.no_delete.contains(&mul)
                || graph.no_delete.contains(&exp2)
            {
                // An intermediate node can't be deleted
                continue;
            }

            // Insert exp op
            let src = graph.get_sources(mul).into_iter().find(|(i, _)| *i != constant).unwrap();
            let exp = graph
                .add_op(MetalExp::new(dev.clone()))
                .input(src.0, 0, src.1)
                .finish();

            // Create edges to dests
            move_outgoing_edge(exp2, exp, &mut graph.graph);
            move_references(
                &mut graph.id_remap,
                &mut graph.no_delete,
                &mut graph.to_retrieve,
                exp2,
                exp,
            );

            // Remove the old ops
            graph.graph.remove_node(mul);
            graph.graph.remove_node(constant);
            graph.graph.remove_node(exp2);
        }
    }
}