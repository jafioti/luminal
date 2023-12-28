use std::marker::PhantomData;

use objc::rc::autoreleasepool;
use petgraph::{stable_graph::NodeIndex, visit::EdgeRef};

use crate::{
    compilers::metal::{prim::*, *},
    op::{ConstantValue, Operator},
    prelude::*,
};

use super::binary::MetalSub;

/// Sometimes CopyTo -> CopyFrom and CopyFrom -> CopyTo patterns remain, so let's clean them up
#[derive(LuminalPrint, Default)]
pub struct CopyCompiler<T>(PhantomData<T>);

impl<T: MetalFloat> Compiler for CopyCompiler<T> {
    fn compile(&self, graph: &mut Graph) {
        for (first, second) in graph
            .graph
            .edge_indices()
            .filter_map(|e| graph.graph.edge_endpoints(e))
            .filter(|(a, b)| {
                (graph
                    .graph
                    .node_weight(*a)
                    .unwrap()
                    .as_any()
                    .is::<MetalCopyToDevice<T>>()
                    && graph
                        .graph
                        .node_weight(*b)
                        .unwrap()
                        .as_any()
                        .is::<MetalCopyFromDevice<T>>())
                    || (graph
                        .graph
                        .node_weight(*a)
                        .unwrap()
                        .as_any()
                        .is::<MetalCopyFromDevice<T>>()
                        && graph
                            .graph
                            .node_weight(*b)
                            .unwrap()
                            .as_any()
                            .is::<MetalCopyToDevice<T>>())
            })
            .unique_by(|n| n.0)
            .unique_by(|n| n.1)
            .collect::<Vec<_>>()
        {
            if graph
                .graph
                .edges_directed(first, petgraph::Direction::Outgoing)
                .filter(|e| graph.graph.contains_node(e.target()))
                .filter(|e| {
                    !graph
                        .graph
                        .node_weight(e.target())
                        .unwrap()
                        .as_any()
                        .is::<MetalCopyFromDevice<T>>()
                        && !graph
                            .graph
                            .node_weight(e.target())
                            .unwrap()
                            .as_any()
                            .is::<MetalCopyToDevice<T>>()
                })
                .count()
                > 0
                || graph.no_delete.contains(&first)
            {
                continue;
            }
            let Some(source) = graph.get_sources(first).pop() else {
                continue;
            };
            move_outgoing_edge(second, source.0, &mut graph.graph);
            move_references(
                &mut graph.id_remap,
                &mut graph.no_delete,
                &mut graph.to_retrieve,
                second,
                source.0,
            );
            graph.graph.remove_node(second);
            graph.id_remap.retain(|_, v| *v != second);
            for dest in graph
                .get_dests(first)
                .iter()
                .map(|(i, _)| *i)
                .collect::<Vec<_>>()
            {
                move_outgoing_edge(dest, source.0, &mut graph.graph);
                move_references(
                    &mut graph.id_remap,
                    &mut graph.no_delete,
                    &mut graph.to_retrieve,
                    dest,
                    source.0,
                );
                graph.graph.remove_node(dest);
                graph.id_remap.retain(|_, v| *v != dest);
            }
            graph.graph.remove_node(first);
            graph.id_remap.retain(|_, v| *v != first);
        }
    }
}

/// Special kernel for efficient mean reduction
#[derive(LuminalEq, LuminalPrint, Clone)]
pub struct MetalARange<T: MetalFloat>(
    ComputePipelineState,
    CommandQueue,
    Device,
    BigExpression,
    *const HashMap<char, usize>,
    PhantomData<T>,
);

impl<T: MetalFloat> MetalARange<T> {
    fn new(
        dev: Device,
        queue: CommandQueue,
        dim: BigExpression,
        dyn_map: *const HashMap<char, usize>,
    ) -> Self {
        Self(
            compile_function("metal_arange", &format!("
#include <metal_stdlib>
using namespace metal;
kernel void metal_arange(device {} *out [[buffer(0)]], device int& n_elements [[buffer(1)]], uint idx [[thread_position_in_grid]]) {{
    if (idx < n_elements) {{
        out[idx] = ({})idx;
    }}
}}", T::type_name(), T::type_name()), &dev),
            queue,
            dev,
            dim,
            dyn_map,
            Default::default(),
        )
    }
}

impl<T: MetalFloat> MetalKernelForward for MetalARange<T> {
    fn output_buffer_sizes(&self, _: &[ShapeTracker]) -> Vec<BigExpression> {
        vec![self.3.clone() * std::mem::size_of::<f16>()]
    }
    fn metal_forward(
        &self,
        _: &[(&Buffer, ShapeTracker)],
        _: &Device,
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
        encoder.set_int(1, size as u32);

        // Execute
        encoder.dispatch_1d(size);
        encoder.end_encoding();
    }
}

impl<T: MetalFloat> Operator for MetalARange<T> {
    fn process(&self, _: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        autoreleasepool(|| {
            // Setup command queue / command buffer / encoder
            let command_buffer = self.1.new_command_buffer();
            let size = self.3.exec(unsafe { self.4.as_ref().unwrap() }).unwrap();
            let out = self.2.new_buffer(
                (size * std::mem::size_of::<f16>()) as u64,
                MTLResourceOptions::StorageModeShared,
            );

            self.metal_forward(&[], &self.2, command_buffer, &[], &[&out]);

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
#[derive(Default, LuminalPrint)]
pub struct ARangeCompiler<T: MetalFloat>(PhantomData<T>);

impl<T: MetalFloat> Compiler for ARangeCompiler<T> {
    fn compile(&self, graph: &mut Graph) {
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
        let s = SelectEdge::new(
            SelectEdge::new(
                SelectEdge::new(
                    SelectEdge::new(
                        SelectEdge::new(
                            SelectEdge::new(
                                SelectOp::new()
                                    .check(|o, _| {
                                        if let Some(c) =
                                            o.as_any().downcast_ref::<MetalConstant<f16>>()
                                        {
                                            if let ConstantValue::Float(f) = c.0 {
                                                f == 1.0
                                            } else {
                                                false
                                            }
                                        } else {
                                            false
                                        }
                                    })
                                    .ptr(&mut one_const),
                                SelectOp::new()
                                    .ty::<MetalContiguous<f16>>()
                                    .ptr(&mut contig1),
                            ),
                            SelectOp::new()
                                .ty::<MetalContiguous<f16>>()
                                .ptr(&mut contig2),
                        ),
                        SelectOp::new()
                            .ty::<MetalContiguous<f16>>()
                            .ptr(&mut contig3),
                    ),
                    SelectOp::new()
                        .ty::<MetalContiguous<f16>>()
                        .ptr(&mut contig4),
                ),
                SelectOp::new()
                    .ty::<MetalSumReduce<f16>>()
                    .ptr(&mut sum_reduce),
            ),
            SelectEdge::new(
                SelectOp::new()
                    .check(|o, _| {
                        if let Some(c) = o.as_any().downcast_ref::<MetalConstant<f16>>() {
                            if let ConstantValue::Float(f) = c.0 {
                                f == 1.0
                            } else {
                                false
                            }
                        } else {
                            false
                        }
                    })
                    .ptr(&mut subtraction_constant),
                SelectOp::new().ty::<MetalSub<f16>>().ptr(&mut subtraction),
            ),
        );

        for _ in s.search(graph) {
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

            graph.graph.remove_node(one_const);
            graph.graph.remove_node(contig1);
            graph.graph.remove_node(contig2);
            graph.graph.remove_node(contig3);
            graph.graph.remove_node(contig4);
            graph.graph.remove_node(sum_reduce);
            graph.graph.remove_node(subtraction);
            graph.graph.remove_node(subtraction_constant);
        }
    }
}
