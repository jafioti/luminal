use std::collections::HashMap;

use metal_rs::{Buffer, Device};
use petgraph::{stable_graph::NodeIndex, Direction::Incoming};

use crate::{op::Operator, prelude::*};

use super::prim::MetalKernelWrapper;

#[derive(Default)]
pub struct CommonBufferOptimizer;

impl GraphOptimizer for CommonBufferOptimizer {
    fn optimize(&self, graph: &mut crate::prelude::Graph) {
        // Look for successive unary metal kernels implementing MetalKernelForward
        let (mut op1, mut op2) = (NodeIndex::default(), NodeIndex::default());
        let s = GraphSelector::default();
        s.edge(
            s.op()
                .check(|op, inps| op.custom("metal").is_some() && inps.len() == 1)
                .ptr(&mut op1),
            0,
            s.op()
                .check(|op, inps| op.custom("metal").is_some() && inps.len() == 1)
                .ptr(&mut op2),
        );
        for _ in s.search(graph) {
            // Replace ops with CommandBufferOp
            let mut node1: Box<dyn Operator> = Box::new(CommandBufferOp(
                petgraph::stable_graph::StableGraph::default(),
                &graph.dyn_map,
            )); // Dummy op to get op1 out
            std::mem::swap(graph.graph.node_weight_mut(op1).unwrap(), &mut node1);

            let node2_shape = graph.get_sources(op2)[0].1;

            move_references(
                &mut graph.id_remap,
                &mut graph.no_delete,
                &mut graph.to_retrieve,
                op2,
                op1,
            );
            move_outgoing_edge(op2, op1, &mut graph.graph);

            let node2 = graph.graph.remove_node(op2).unwrap();

            let mut internal_graph = petgraph::stable_graph::StableGraph::default();
            let node1 =
                internal_graph.add_node(*node1.custom("metal").unwrap().downcast().unwrap());
            let node2 =
                internal_graph.add_node(*node2.custom("metal").unwrap().downcast().unwrap());
            internal_graph.add_edge(node1, node2, node2_shape);
            let mut command_buffer_op: Box<dyn Operator> =
                Box::new(CommandBufferOp(internal_graph, &graph.dyn_map));

            std::mem::swap(
                graph.graph.node_weight_mut(op1).unwrap(),
                &mut command_buffer_op,
            );
        }
        // Look for metal kernel -> CommonBuffer
        let s = GraphSelector::default();
        s.edge(
            s.op()
                .check(|op, inps| op.custom("metal").is_some() && inps.len() == 1)
                .ptr(&mut op1),
            0,
            s.op().ty::<CommandBufferOp>().ptr(&mut op2),
        );
        for _ in s.search(graph) {
            if graph.get_dests(op1).len() != 1 {
                // There can only be one dest
                continue;
            }
            let node1_shape = graph.get_sources(op1)[0].1;
            move_references(
                &mut graph.id_remap,
                &mut graph.no_delete,
                &mut graph.to_retrieve,
                op1,
                op2,
            );
            move_incoming_edge(op1, op2, &mut graph.graph);
            let op = graph.graph.remove_node(op1).unwrap();
            let internal_graph = &mut graph
                .graph
                .node_weight_mut(op2)
                .unwrap()
                .as_any_mut()
                .downcast_mut::<CommandBufferOp>()
                .unwrap()
                .0;
            let first_node = petgraph::algo::toposort(&*internal_graph, None).unwrap()[0];
            let new_node =
                internal_graph.add_node(*op.custom("metal").unwrap().downcast().unwrap());
            internal_graph.add_edge(new_node, first_node, node1_shape);
        }
        // Look for CommonBuffer -> metal kernel
        let s = GraphSelector::default();
        s.edge(
            s.op().ty::<CommandBufferOp>().ptr(&mut op1),
            0,
            s.op()
                .check(|op, inps| op.custom("metal").is_some() && inps.len() == 1)
                .ptr(&mut op2),
        );
        for _ in s.search(graph) {
            if graph.get_dests(op1).len() != 1 {
                // There can only be one dest
                continue;
            }
            let node2_shape = graph.get_sources(op2)[0].1;
            move_references(
                &mut graph.id_remap,
                &mut graph.no_delete,
                &mut graph.to_retrieve,
                op2,
                op1,
            );
            move_outgoing_edge(op2, op1, &mut graph.graph);
            let op = graph.graph.remove_node(op2).unwrap();
            let internal_graph = &mut graph
                .graph
                .node_weight_mut(op1)
                .unwrap()
                .as_any_mut()
                .downcast_mut::<CommandBufferOp>()
                .unwrap()
                .0;
            let last_node = *petgraph::algo::toposort(&*internal_graph, None)
                .unwrap()
                .last()
                .unwrap();
            let new_node =
                internal_graph.add_node(*op.custom("metal").unwrap().downcast().unwrap());
            internal_graph.add_edge(last_node, new_node, node2_shape);
        }
    }
}

#[derive(Debug)]
struct CommandBufferOp(
    petgraph::stable_graph::StableGraph<MetalKernelWrapper, ShapeTracker>,
    *const HashMap<char, usize>,
);
impl PartialEq for CommandBufferOp {
    fn eq(&self, _: &Self) -> bool {
        false
    }
}

impl Operator for CommandBufferOp {
    fn process(&self, inp: Vec<(crate::op::InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        // Setup input buffers
        let first_shapes = inp.iter().map(|(_, s)| *s).collect::<Vec<_>>();
        let mut buffers = inp
            .into_iter()
            .map(|(t, _)| {
                t.borrowed()
                    .data
                    .as_any()
                    .downcast_ref::<Buffer>()
                    .unwrap()
                    .clone()
            })
            .collect::<Vec<_>>();
        // Create command buffer
        let dev = Device::system_default().unwrap();
        let command_queue = dev.new_command_queue();
        let command_buffer = command_queue.new_command_buffer();

        // For each operation
        for op_id in petgraph::algo::toposort(&self.0, None).unwrap() {
            // Fill in dyn shapes
            let mut shapes = self
                .0
                .edges_directed(op_id, Incoming)
                .map(|e| *e.weight())
                .collect::<Vec<_>>();
            if shapes.is_empty() {
                shapes = first_shapes.clone();
            }
            for shape in &mut shapes {
                *shape = shape.resolve_global_dyn_dims(unsafe { self.1.as_ref().unwrap() });
            }
            // Run through metal kernel forward
            buffers = self.0.node_weight(op_id).unwrap().0.metal_forward(
                &buffers.iter().zip(shapes.into_iter()).collect::<Vec<_>>(),
                &dev,
                command_buffer,
            );
        }
        // Execute command buffer
        command_buffer.commit();
        command_buffer.wait_until_completed();

        // Return buffers as tensors
        buffers
            .into_iter()
            .map(|buff| Tensor {
                data: Box::new(buff),
            })
            .collect()
    }
}
