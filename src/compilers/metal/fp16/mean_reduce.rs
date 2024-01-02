use std::{mem::size_of, sync::Arc};

use half::f16;
use petgraph::stable_graph::NodeIndex;

use crate::{
    compilers::metal::{prim::*, *},
    op::{InputTensor, Operator},
    prelude::*,
};

use metal_rs::{objc::rc::autoreleasepool, *};

/// Special kernel for efficient mean reduction
#[derive(LuminalEq, LuminalPrint, Clone)]
pub struct MetalMeanReduce(
    ComputePipelineState,
    CommandQueue,
    Device,
    usize,
    ShapeTracker,
    *const HashMap<char, usize>,
);

impl MetalMeanReduce {
    fn new(
        dev: Device,
        queue: CommandQueue,
        dim: usize,
        shape: ShapeTracker,
        dyn_map: *const HashMap<char, usize>,
    ) -> Self {
        let (idx_exp, valid_exp) = get_idx_valid_exps(shape);
        let mut code = format!(
            "
#include <metal_stdlib>
using namespace metal;
kernel void mkernel(device half *inp [[buffer(0)]], device half *out [[buffer(1)]], device int& n_elements [[buffer(2)]], device int& front_size [[buffer(3)]], device int& back_size [[buffer(4)]], device int& dim_size [[buffer(5)]], uint i_ [[thread_position_in_grid]]{}) {{
    if (i_ < n_elements) {{
        int a_ = i_ / back_size;
        int b_ = i_ % back_size;
        float reduce_value = 0.0;
        for (int c_ = 0; c_ < dim_size; c_++) {{
            int idx = a_ * dim_size * back_size + c_ * back_size + b_;
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
            dyn_map,
        )
    }
}

impl MetalKernel for MetalMeanReduce {
    fn output_buffer_sizes(&self, input_shapes: &[ShapeTracker]) -> Vec<BigExpression> {
        let mut sh = input_shapes[0];
        sh.remove_dim(self.3);
        vec![sh.n_elements() * size_of::<f16>()]
    }
    fn metal_forward(
        &self,
        inputs: &[(&Buffer, ShapeTracker)],
        command_buffer: &CommandBufferRef,
        _: &[&Buffer],
        output_buffers: &[&Buffer],
    ) {
        let mut sh = inputs[0].1;
        sh.remove_dim(self.3);
        let inp_size = sh.n_elements().to_usize().unwrap();

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
        encoder.set_buffer(1, Some(output_buffers[0]), 0);
        encoder.set_int(2, inp_size as u32);
        encoder.set_int(3, front_size as u32);
        encoder.set_int(4, back_size as u32);
        encoder.set_int(5, dim_size as u32);
        input_dyn_dims(&[self.4], unsafe { self.5.as_ref().unwrap() }, encoder, 6);

        // Execute
        encoder.dispatch_1d(inp_size);
        encoder.end_encoding();
    }
}

impl Operator for MetalMeanReduce {
    fn process(&mut self, tensors: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        autoreleasepool(|| {
            // Setup buffers
            let mut sh = tensors[0].1;
            sh.remove_dim(self.3);
            let inp_size = sh.n_elements().to_usize().unwrap();
            let out = self.2.new_buffer(
                (inp_size * std::mem::size_of::<f16>()) as u64,
                MTLResourceOptions::StorageModeShared,
            );

            // Setup command queue / command buffer / encoder
            let command_buffer = self.1.new_command_buffer();

            self.metal_forward(
                &[(get_buffer_from_tensor(&tensors[0].0), tensors[0].1)],
                command_buffer,
                &[],
                &[&out],
            );

            command_buffer.commit();
            command_buffer.wait_until_completed();

            vec![Tensor {
                data: Box::new(out),
            }]
        })
    }

    fn custom(&self, key: &str) -> Option<Box<dyn Any>> {
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
pub struct MeanReduceCompiler;

impl Compiler for MeanReduceCompiler {
    fn compile<T: ToIds>(&self, graph: &mut Graph, remap: T) {
        let dev = Device::system_default().unwrap();
        let queue = dev.new_command_queue();
        // Look for the mean-reduce pattern
        // mul(recip(fake_sum_reduce(const_ones)), sum_reduce(x))
        let (mut fake_sum_reduce, mut recip, mut mul, mut sum_reduce) = (
            NodeIndex::default(),
            NodeIndex::default(),
            NodeIndex::default(),
            NodeIndex::default(),
        );

        let s = SelectEdge::new(
            SelectOp::new()
                .ty::<MetalSumReduce<f16>>()
                .ptr(&mut sum_reduce),
            SelectEdge::new(
                SelectEdge::new(
                    SelectOp::new()
                        .ty::<MetalConstant<f16>>()
                        .ptr(&mut fake_sum_reduce),
                    SelectOp::new().ty::<MetalRecip<f16>>().ptr(&mut recip),
                ),
                SelectOp::new().ty::<MetalMul<f16>>().ptr(&mut mul),
            ),
        );

        let mut searcher = s.search(graph);
        while searcher.next_match() {
            if graph.no_delete.contains(&sum_reduce)
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
                .downcast_ref::<MetalSumReduce<f16>>()
                .unwrap()
                .3;
            // Insert MeanReduce op
            let src = graph.get_sources(sum_reduce)[0];
            let mean_reduce = graph
                .add_op(MetalMeanReduce::new(
                    dev.clone(),
                    queue.clone(),
                    dim,
                    src.2,
                    &graph.dyn_map,
                ))
                .input(src.0, 0, src.2)
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
            graph.graph.remove_node(fake_sum_reduce);
        }
    }
}
