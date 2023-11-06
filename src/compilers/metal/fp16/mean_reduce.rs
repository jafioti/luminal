use std::sync::Arc;

use half::f16;
use petgraph::stable_graph::NodeIndex;

use crate::{
    compilers::metal::*,
    op::{InputTensor, Operator},
    prelude::*,
};

use super::prim::{
    FakeSumReduce, MetalConstant, MetalKernelForward, MetalKernelWrapper, MetalMul, MetalRecip,
    MetalSumReduce,
};
use metal_rs::{objc::rc::autoreleasepool, *};

/// Special kernel for efficient mean reduction
#[derive(Debug, Clone)]
pub struct MetalMeanReduce(
    ComputePipelineState,
    CommandQueue,
    Device,
    usize,
    ShapeTracker,
);
impl PartialEq for MetalMeanReduce {
    fn eq(&self, _: &Self) -> bool {
        false
    }
}

impl MetalMeanReduce {
    fn new(dev: Device, queue: CommandQueue, dim: usize, shape: ShapeTracker) -> Self {
        let (idx_exp, valid_exp) = get_idx_valid_exps(shape);
        let mut code = format!(
            "
#include <metal_stdlib>
using namespace metal;
kernel void mkernel(device half *inp [[buffer(0)]], device half *out [[buffer(1)]], device uint& n_elements [[buffer(2)]], device uint& front_size [[buffer(3)]], device uint& back_size [[buffer(4)]], device uint& dim_size [[buffer(5)]], uint i_ [[thread_position_in_grid]]{}) {{
    if (i_ < n_elements) {{
        uint a_ = i_ / back_size;
        uint b_ = i_ % back_size;
        float reduce_value = 0.0;
        for (uint c_ = 0; c_ < dim_size; c_++) {{
            uint idx = a_ * dim_size * back_size + c_ * back_size + b_;
            if (({valid_exp}) != 0) {{
                reduce_value += (float)inp[{idx_exp}];
            }}
        }}
        out[i_] = (half)(reduce_value / (float)dim_size);
    }}
}}
", render_dyn_dim_inputs(&[shape], 6),
        );
        code = code.replace("mkernel", "kernel_mean_reduce");

        Self(
            compile_function("kernel_mean_reduce", &code, &dev),
            queue,
            dev,
            dim,
            shape,
        )
    }
}

impl MetalKernelForward for MetalMeanReduce {
    fn metal_forward(
        &self,
        inputs: &[(&Buffer, ShapeTracker)],
        dev: &Device,
        command_buffer: &CommandBufferRef,
    ) -> Vec<Buffer> {
        let mut sh = inputs[0].1;
        sh.remove_dim(self.3);
        let inp_size = sh.contiguous().n_elements();

        let out = dev.new_buffer(
            (inp_size * std::mem::size_of::<f16>()) as u64,
            MTLResourceOptions::StorageModeManaged,
        );
        let front_size: usize = inputs[0]
            .1
            .shape()
            .iter()
            .take(self.3)
            .map(|i| i.to_usize().unwrap())
            .product();
        let back_size: usize = inputs[0]
            .1
            .shape()
            .iter()
            .skip(self.3 + 1)
            .map(|i| i.to_usize().unwrap())
            .product();
        let dim_size = inputs[0].1.shape()[self.3].to_usize().unwrap();

        let encoder =
            command_buffer.compute_command_encoder_with_descriptor(ComputePassDescriptor::new());
        encoder.set_compute_pipeline_state(&self.0);

        // Set inputs
        encoder.set_buffer(0, Some(inputs[0].0), 0);
        encoder.set_buffer(1, Some(&out), 0);
        encoder.set_int(2, inp_size as u32);
        encoder.set_int(3, front_size as u32);
        encoder.set_int(4, back_size as u32);
        encoder.set_int(5, dim_size as u32);
        input_dyn_dims(&[(self.4, inputs[0].1)], encoder, 6);

        // Execute
        encoder.dispatch_n_elements(inp_size);
        encoder.end_encoding();

        vec![out]
    }
}

impl Operator for MetalMeanReduce {
    fn process(&self, tensors: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        autoreleasepool(|| {
            // Setup buffers
            let a = tensors[0]
                .0
                .borrowed()
                .data
                .as_any()
                .downcast_ref::<Buffer>()
                .unwrap();

            // Setup command queue / command buffer / encoder
            let command_buffer = self.1.new_command_buffer();

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
pub struct MeanReduceCompiler;

impl Compiler for MeanReduceCompiler {
    fn compile(&self, graph: &mut Graph) {
        let dev = Device::system_default().unwrap();
        let queue = dev.new_command_queue();
        // Look for the mean-reduce pattern
        // mul(recip(fake_sum_reduce(const_ones)), sum_reduce(x))
        let s = GraphSelector::default();
        let (mut one_const, mut fake_sum_reduce, mut recip, mut mul, mut sum_reduce) = (
            NodeIndex::default(),
            NodeIndex::default(),
            NodeIndex::default(),
            NodeIndex::default(),
            NodeIndex::default(),
        );

        s.edge(
            s.op().ty::<MetalSumReduce>().ptr(&mut sum_reduce),
            s.edge(
                s.edge(
                    s.edge(
                        s.op()
                            .check(|op, _| {
                                if let Some(c) = op.as_any().downcast_ref::<MetalConstant>() {
                                    c.0 == f16::ONE
                                } else {
                                    false
                                }
                            })
                            .ptr(&mut one_const),
                        s.op().ty::<FakeSumReduce>().ptr(&mut fake_sum_reduce),
                    ),
                    s.op().ty::<MetalRecip>().ptr(&mut recip),
                ),
                s.op().ty::<MetalMul>().ptr(&mut mul),
            ),
        );

        for _ in s.search(graph) {
            if graph.no_delete.contains(&sum_reduce)
                || graph.no_delete.contains(&one_const)
                || graph.no_delete.contains(&fake_sum_reduce)
                || graph.no_delete.contains(&recip)
            {
                // An intermediate node can't be deleted
                continue;
            }
            let dim = graph
                .graph
                .node_weight(sum_reduce)
                .unwrap()
                .as_any()
                .downcast_ref::<MetalSumReduce>()
                .unwrap()
                .3;
            // Insert MeanReduce op
            let src = graph.get_sources(sum_reduce)[0];
            let mean_reduce = graph
                .add_op(MetalMeanReduce::new(dev.clone(), queue.clone(), dim, src.1))
                .input(src.0, 0, src.1)
                .finish();

            // Create edges to dests
            move_outgoing_edge(mul, mean_reduce, &mut graph.graph);
            move_references(
                &mut graph.id_remap,
                &mut graph.no_delete,
                &mut graph.to_retrieve,
                mul,
                mean_reduce,
            );

            // Remove the old ops
            graph.graph.remove_node(mul);
            graph.graph.remove_node(sum_reduce);
            graph.graph.remove_node(recip);
            graph.graph.remove_node(one_const);
            graph.graph.remove_node(fake_sum_reduce);
        }
    }
}
