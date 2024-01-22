use std::marker::PhantomData;

use objc::rc::autoreleasepool;
use petgraph::{stable_graph::NodeIndex, visit::EdgeRef, Direction};

use crate::{
    compilers::metal::{prim::*, *},
    constant_select_op,
    op::Operator,
    prelude::*,
};

use super::binary::MetalSub;

/// Sometimes CopyTo -> CopyFrom and CopyFrom -> CopyTo patterns remain, so let's clean them up
#[derive(LuminalPrint, Default)]
pub struct CopyCompiler<T>(PhantomData<T>);

impl<T: MetalFloat> Compiler for CopyCompiler<T> {
    fn compile<To: ToIdsMut>(&self, graph: &mut Graph, mut remap: To) {
        let (mut first, mut second) = (NodeIndex::default(), NodeIndex::default());
        let mut selector = SelectOp::new()
            .ty::<MetalCopyToDevice<T>>()
            .ptr(&mut first)
            .edge(
                SelectOp::new()
                    .ty::<MetalCopyToDevice<T>>()
                    .ptr(&mut second),
            )
            .search(graph);
        while selector.next_match() {
            // Ensure there are no dests from first that are not copies
            if graph
                .graph
                .edges_directed(first, petgraph::Direction::Outgoing)
                .filter(|e| {
                    let target = graph.graph.node_weight(e.target()).unwrap().as_any();
                    !target.is::<MetalCopyFromDevice<T>>() && !target.is::<MetalCopyToDevice<T>>()
                })
                .count()
                > 0
                || graph.no_delete.contains(&first)
            {
                continue;
            }
            let Some((source, _, _)) = graph.get_sources(first).pop() else {
                continue;
            };
            move_outgoing_edge(second, source, &mut graph.graph);
            move_references(
                &mut remap,
                &mut graph.no_delete,
                &mut graph.to_retrieve,
                second,
                source,
            );
            graph.graph.remove_node(second);
            for dest in graph
                .get_dests(first)
                .iter()
                .map(|(i, _)| *i)
                .collect::<Vec<_>>()
            {
                move_outgoing_edge(dest, source, &mut graph.graph);
                move_references(
                    &mut remap,
                    &mut graph.no_delete,
                    &mut graph.to_retrieve,
                    dest,
                    source,
                );
                graph.graph.remove_node(dest);
            }
            graph.graph.remove_node(first);
            selector.clear_cached_results();
        }
    }
}

/// Special kernel for producing aranges
#[derive(Clone, LuminalEqFalse)]
pub struct MetalARange<T: MetalFloat>(
    ComputePipelineState,
    CommandQueue,
    Device,
    BigExpression,
    *const HashMap<char, usize>,
    PhantomData<T>,
);

impl<T: MetalFloat> Debug for MetalARange<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "MetalARange({:?})", self.3)
    }
}

impl<T: MetalFloat> MetalARange<T> {
    fn new(
        dev: Device,
        queue: CommandQueue,
        dim: BigExpression,
        dyn_map: *const HashMap<char, usize>,
    ) -> Self {
        let type_name = T::type_name();
        Self(
            compile_function("metal_arange", &format!("
#include <metal_stdlib>
using namespace metal;
kernel void metal_arange(device {type_name} *out [[buffer(0)]], device int& n_elements [[buffer(1)]], uint idx [[thread_position_in_grid]]) {{
    if (idx < n_elements) {{
        out[idx] = ({type_name})idx;
    }}
}}"), &dev),
            queue,
            dev,
            dim,
            dyn_map,
            Default::default(),
        )
    }
}

impl<T: MetalFloat> MetalKernel for MetalARange<T> {
    fn output_buffer_sizes(&self, _: &[ShapeTracker]) -> Vec<BigExpression> {
        vec![self.3.clone() * std::mem::size_of::<f16>()]
    }
    fn metal_forward(
        &self,
        _: &[(&Buffer, ShapeTracker)],
        command_buffer: &CommandBufferRef,
        _: &[&Buffer],
        output_buffers: &[&Buffer],
    ) {
        // Calculate size
        let size = self.3.exec(unsafe { self.4.as_ref().unwrap() }).unwrap();

        let encoder =
            command_buffer.compute_command_encoder_with_descriptor(ComputePassDescriptor::new());
        encoder.set_compute_pipeline_state(&self.0);

        // Set inputs
        encoder.set_buffer(0, Some(output_buffers[0]), 0);
        encoder.set_u32(1, size as u32);

        // Execute
        encoder.dispatch_1d(size);
        encoder.end_encoding();
    }
}

impl<T: MetalFloat> Operator for MetalARange<T> {
    fn process(&mut self, _: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        autoreleasepool(|| {
            // Set up command buffer and output buffer
            let command_buffer = self.1.new_command_buffer();
            let size = self.3.exec(unsafe { self.4.as_ref().unwrap() }).unwrap();
            let out = self.2.new_buffer(
                (size * std::mem::size_of::<f16>()) as u64,
                MTLResourceOptions::StorageModeShared,
            );

            self.metal_forward(&[], command_buffer, &[], &[&out]);

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

/// Replace the arange pattern with a special kernel. This must be ran **after** the subtraction compiler
#[derive(Default, LuminalPrint)]
pub struct ARangeCompiler<T: MetalFloat>(PhantomData<T>);

impl<T: MetalFloat> Compiler for ARangeCompiler<T> {
    fn compile<To: ToIdsMut>(&self, graph: &mut Graph, _: To) {
        let dev = Device::system_default().unwrap();
        let queue = dev.new_command_queue();
        let (
            mut one_const,
            mut contig1,
            mut contig2,
            mut contig3,
            mut contig4,
            mut sum_reduce,
            mut subtraction_constant,
            mut subtraction,
        ) = (
            NodeIndex::default(),
            NodeIndex::default(),
            NodeIndex::default(),
            NodeIndex::default(),
            NodeIndex::default(),
            NodeIndex::default(),
            NodeIndex::default(),
            NodeIndex::default(),
        );

        // TODO: Make sure this actually checks the shape transformations to ensure pooling happens
        let contig = SelectOp::new().ty::<MetalContiguous<T>>();
        let pre_sub_pattern = constant_select_op!(1.0, T)
            .ptr(&mut one_const)
            .edge(contig.clone().ptr(&mut contig1))
            .edge(contig.clone().ptr(&mut contig2))
            .edge(contig.clone().ptr(&mut contig3))
            .edge(contig.clone().ptr(&mut contig4))
            .edge(
                SelectOp::new()
                    .ty::<MetalSumReduce<T>>()
                    .ptr(&mut sum_reduce),
            );
        let mut s1 = pre_sub_pattern
            .clone()
            .edge(
                constant_select_op!(1.0, T)
                    .ptr(&mut subtraction_constant)
                    .edge(SelectOp::new().ty::<MetalSub<T>>().ptr(&mut subtraction)),
            )
            .search(graph);
        let mut s2 = pre_sub_pattern
            .edge(
                constant_select_op!(-1.0, T)
                    .ptr(&mut subtraction_constant)
                    .edge(SelectOp::new().ty::<MetalAdd<T>>().ptr(&mut subtraction)),
            )
            .search(graph);

        while s1.next_match() || s2.next_match() {
            let arange_amount = {
                let sh = graph
                    .graph
                    .edge_weight(
                        graph
                            .graph
                            .edges_connecting(one_const, contig1)
                            .next()
                            .unwrap()
                            .id(),
                    )
                    .unwrap()
                    .as_data()
                    .unwrap()
                    .2;
                sh.dims[sh.indexes[sh.len() - 1]]
            };
            let arange_op = graph
                .add_op(MetalARange::<T>::new(
                    dev.clone(),
                    queue.clone(),
                    arange_amount.into(),
                    &graph.dyn_map,
                ))
                .finish();
            move_outgoing_edge(subtraction, arange_op, &mut graph.graph);

            graph.graph.remove_node(subtraction);
            graph.safe_remove_node(subtraction_constant, 0);
            graph.safe_remove_node(sum_reduce, 0);
            graph.safe_remove_node(contig4, 0);
            graph.safe_remove_node(contig3, 0);
            graph.safe_remove_node(contig2, 0);
            graph.safe_remove_node(contig1, 0);
            graph.safe_remove_node(one_const, 0);
            s1.clear_cached_results();
            s2.clear_cached_results();
        }
    }
}

#[derive(Debug, Default)]
pub struct ContiguousElimination<T>(PhantomData<T>);

impl<T: MetalFloat> Compiler for ContiguousElimination<T> {
    fn compile<To: ToIdsMut>(&self, graph: &mut Graph, mut remap: To) {
        // Look for contiguous calls going to ops that can accept non-contiguous inputs (marked non_contiguous)
        let (mut contig, mut op) = (NodeIndex::default(), NodeIndex::default());
        let pattern = SelectOp::new()
            .ty::<MetalContiguous<T>>()
            .ptr(&mut contig)
            .edge(
                SelectOp::new()
                    .check(|op, _| op.custom("non_contiguous", Box::new(())).is_some())
                    .ptr(&mut op),
            );
        let mut selector = pattern.search(graph);
        while selector.next_match() {
            if graph.no_delete.contains(&contig)
                || graph
                    .graph
                    .edges_directed(contig, Direction::Outgoing)
                    .count()
                    > 1
            {
                continue;
            }
            // Shape going from contig to op
            // let first_shape = graph
            //     .graph
            //     .edges_directed(contig, Direction::Incoming)
            //     .find_map(|e| e.weight().as_data())
            //     .unwrap()
            //     .2;
            let second_shape = graph
                .graph
                .edges_connecting(contig, op)
                .find_map(|e| e.weight().as_data())
                .unwrap()
                .2;
            // Here we should check if second shape and first shape are mergeable instead of just checking if second_shape is contiguous
            if second_shape.is_contiguous()
                && !second_shape.is_sliced()
                && !second_shape.is_padded()
            {
                let source = graph
                    .graph
                    .neighbors_directed(contig, petgraph::Direction::Incoming)
                    .next()
                    .unwrap();
                move_incoming_edge(contig, op, &mut graph.graph);
                move_references(
                    &mut remap,
                    &mut graph.no_delete,
                    &mut graph.to_retrieve,
                    contig,
                    source,
                );
                graph.graph.remove_node(contig);
                let new_shapes = graph
                    .get_sources(op)
                    .into_iter()
                    .map(|(_, _, s)| s)
                    .collect::<Vec<_>>();
                graph
                    .graph
                    .node_weight_mut(op)
                    .unwrap()
                    .custom("recompile_shapes", Box::new(new_shapes));
                selector.clear_cached_results();
            }
        }
    }
}
