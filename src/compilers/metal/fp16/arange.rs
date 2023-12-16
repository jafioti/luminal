use std::sync::Arc;

use half::f16;
use petgraph::stable_graph::NodeIndex;

use crate::{
    compilers::metal::*,
    op::{Function as LFunction, InputTensor, Operator},
    prelude::*,
};

use metal_rs::{objc::rc::autoreleasepool, *};

/// Special kernel for efficient mean reduction
#[derive(LuminalEq, LuminalPrint, Clone)]
pub struct MetalARange(
    ComputePipelineState,
    CommandQueue,
    Device,
    BigExpression,
    *const HashMap<char, usize>,
);

impl MetalARange {
    fn new(
        dev: Device,
        queue: CommandQueue,
        dim: BigExpression,
        dyn_map: *const HashMap<char, usize>,
    ) -> Self {
        Self(
            compile_function("metal_arange", "
#include <metal_stdlib>
using namespace metal;
kernel void metal_arange(device half *out [[buffer(0)]], device uint& n_elements [[buffer(1)]], uint idx [[thread_position_in_grid]]) {
    if (idx < n_elements) {
        out[idx] = (half)idx;
    }
}", &dev),
            queue,
            dev,
            dim,
            dyn_map,
        )
    }
}

impl MetalKernelForward for MetalARange {
    fn metal_forward(
        &self,
        _: &[(&Buffer, ShapeTracker)],
        dev: &Device,
        command_buffer: &CommandBufferRef,
    ) -> Vec<Buffer> {
        // Calculate size
        let size = self.3.exec(unsafe { self.4.as_ref().unwrap() }).unwrap();
        let out = dev.new_buffer(
            (size * std::mem::size_of::<f16>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let encoder =
            command_buffer.compute_command_encoder_with_descriptor(ComputePassDescriptor::new());
        encoder.set_compute_pipeline_state(&self.0);

        // Set inputs
        encoder.set_buffer(0, Some(&out), 0);
        encoder.set_int(1, size as u32);

        // Execute
        encoder.dispatch_1d(size);
        encoder.end_encoding();

        vec![out]
    }
}

impl Operator for MetalARange {
    fn process(&self, _: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        autoreleasepool(|| {
            // Setup command queue / command buffer / encoder
            let command_buffer = self.1.new_command_buffer();

            let out = self
                .metal_forward(&[], &self.2, command_buffer)
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
            #[allow(clippy::arc_with_non_send_sync)]
            return Some(Box::new(MetalKernelWrapper(Arc::new(Box::new(
                self.clone(),
            )))));
        }
        None
    }
}

/// Replace the mean reduce pattern with a special kernel. This is meant to be ran **after** the FakeSumReduceCompiler.
#[derive(Default)]
pub struct ARangeCompiler;

impl Compiler for ARangeCompiler {
    fn compile(&self, graph: &mut Graph) {
        let dev = Device::system_default().unwrap();
        let queue = dev.new_command_queue();
        // Look for the mean-reduce pattern
        // mul(recip(fake_sum_reduce(const_ones)), sum_reduce(x))
        let mut arange_op = NodeIndex::default();

        let s = SelectEdge::from(
            SelectOp::new()
                .check(|o, _| {
                    if let Some(o) = o.as_any().downcast_ref::<LFunction>() {
                        o.0 == "ARange"
                    } else {
                        false
                    }
                })
                .ptr(&mut arange_op),
        );

        for _ in s.search(graph) {
            let src = graph.get_sources(arange_op)[0].2;
            let new_arange = graph
                .add_op(MetalARange::new(
                    dev.clone(),
                    queue.clone(),
                    src.shape()[2].into(),
                    &graph.dyn_map,
                ))
                .finish();

            // Create edges to dests
            move_outgoing_edge(arange_op, new_arange, &mut graph.graph);
            move_references(
                &mut graph.id_remap,
                &mut graph.no_delete,
                &mut graph.to_retrieve,
                arange_op,
                new_arange,
            );

            // Remove the old op
            graph.graph.remove_node(arange_op);
        }
    }
}
