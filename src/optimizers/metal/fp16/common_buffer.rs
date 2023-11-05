use std::collections::HashMap;

use itertools::Itertools;
use metal_rs::{objc::rc::autoreleasepool, Buffer, Device};
use petgraph::{
    stable_graph::NodeIndex,
    visit::EdgeRef,
    Direction::{self},
};

use crate::{
    op::{InputTensor, Operator},
    prelude::*,
};

use super::prim::MetalKernelWrapper;

#[derive(Default)]
pub struct CommonBufferOptimizer;

impl GraphOptimizer for CommonBufferOptimizer {
    fn optimize(&self, graph: &mut crate::prelude::Graph) {
        // Replace all metal kernels
        let s = GraphSelector::default();
        let mut op1 = NodeIndex::default();
        s.op()
            .check(|op, _| op.custom("metal").is_some())
            .ptr(&mut op1);
        for _ in s.search(graph) {
            // Create internal graph
            let inputs = graph
                .graph
                .edges_directed(op1, Direction::Incoming)
                .map(|e| (e.source(), *e.weight()))
                .collect::<Vec<_>>();
            let outputs = graph
                .graph
                .edges_directed(op1, Direction::Outgoing)
                .map(|e| (e.target(), *e.weight()))
                .collect::<Vec<_>>();
            let op_node = graph.graph.remove_node(op1).unwrap();
            let operator: MetalKernelWrapper =
                *op_node.custom("metal").unwrap().downcast().unwrap();
            let mut internal_graph = petgraph::stable_graph::StableGraph::default();
            let node = internal_graph.add_node(operator);
            let new_op = graph
                .add_op(CommandBufferOp {
                    internal_graph,
                    dyn_map: &graph.dyn_map,
                    input_map: HashMap::from([(
                        node,
                        inputs
                            .iter()
                            .filter_map(|i| i.1.as_data())
                            .map(|(a, _, b)| (a, a, b))
                            .collect(),
                    )]),
                    outputs: (0..=outputs
                        .iter()
                        .filter_map(|i| i.1.as_data())
                        .map(|(_, i, _)| i)
                        .max()
                        .unwrap_or_default())
                        .map(|i| (node, i))
                        .collect(),
                })
                .finish();
            for (src, weight) in inputs {
                graph.graph.add_edge(src, new_op, weight);
            }
            for (trg, weight) in outputs {
                graph.graph.add_edge(new_op, trg, weight);
            }
            move_references(
                &mut graph.id_remap,
                &mut graph.no_delete,
                &mut graph.to_retrieve,
                op1,
                new_op,
            );
        }
        // Look for CommonBuffer -> CommonBuffer
        let (mut op1, mut op2) = (NodeIndex::default(), NodeIndex::default());
        let s = GraphSelector::default();
        s.edge(
            s.op().ty::<CommandBufferOp>().ptr(&mut op1),
            s.op().ty::<CommandBufferOp>().ptr(&mut op2),
        );
        for _ in s.clone().search(graph) {
            // Make sure all non-direct outputs of op1 don't reach op2
            if graph
                .graph
                .edges_directed(op1, Direction::Outgoing)
                .filter(|e| e.target() != op2)
                .all(|e| !check_if_reaches_node(graph, e.target(), op2))
            {
                merge_command_buffer_ops(op1, op2, graph);
            }
        }
    }
}

fn check_if_reaches_node(graph: &Graph, start: NodeIndex, target: NodeIndex) -> bool {
    graph
        .graph
        .edges_directed(start, Direction::Outgoing)
        .any(|e| e.target() == target || check_if_reaches_node(graph, e.target(), target))
}

fn merge_nodes(
    graph_1: &CommandBufferOp,
    graph_2: &mut CommandBufferOp,
) -> HashMap<NodeIndex, NodeIndex> {
    let mut old_to_new = HashMap::default();
    for node in graph_1.internal_graph.node_indices() {
        let new_node = graph_2
            .internal_graph
            .add_node(graph_1.internal_graph.node_weight(node).unwrap().clone());
        old_to_new.insert(node, new_node);
    }
    old_to_new
}

fn merge_internal_connections(
    graph_1: &CommandBufferOp,
    graph_2: &mut CommandBufferOp,
    inputs_from_graph_1: &HashMap<u8, u8>,
    old_to_new: &HashMap<NodeIndex, NodeIndex>,
) {
    for (node, mut sources) in graph_2.input_map.clone() {
        for x in (0..sources.len()).rev() {
            let (source_inp_ind, graph_source_id, shape) = sources[x];
            if let Some(i) = inputs_from_graph_1.get(&graph_source_id) {
                // This source is coming from the other graph, output index i
                let graph_1_internal_id = graph_1.outputs[*i as usize];
                graph_2.internal_graph.add_edge(
                    old_to_new[&graph_1_internal_id.0],
                    node,
                    (source_inp_ind, graph_1_internal_id.1, shape),
                );
                sources.remove(x);
            }
        }
        // Update input map
        if sources.is_empty() {
            graph_2.input_map.remove(&node);
        } else {
            *graph_2.input_map.get_mut(&node).unwrap() = sources;
        }
    }
}

fn remove_unused_graph_1_outputs(
    graph: &mut Graph,
    graph_1: &mut CommandBufferOp,
    graph_1_index: NodeIndex,
    graph_2_index: NodeIndex,
) {
    let edges_out = graph
        .graph
        .edges_directed(graph_1_index, Direction::Outgoing)
        .filter(|e| e.target() != graph_2_index)
        .map(|e| e.weight().1)
        .collect::<Vec<_>>();
    for i in (0..graph_1.outputs.len() as u8).rev() {
        if edges_out.iter().all(|o| *o != i) {
            graph_1.outputs.remove(i as usize);
            // Decrement all edges coming from outputs higher than i
            for edge in graph
                .graph
                .edges_directed(graph_1_index, Direction::Outgoing)
                .filter(|e| e.target() != graph_2_index && e.weight().1 >= i)
                .map(|e| e.id())
                .collect::<Vec<_>>()
            {
                graph.graph.edge_weight_mut(edge).unwrap().1 -= 1;
            }
        }
    }
}

fn merge_outputs(
    graph_1: &CommandBufferOp,
    graph_2: &mut CommandBufferOp,
    old_to_new: &HashMap<NodeIndex, NodeIndex>,
) {
    for output in &graph_1.outputs {
        graph_2.outputs.push((old_to_new[&output.0], output.1));
    }
}

fn transfer_output_edges(
    graph: &mut Graph,
    graph_1_index: NodeIndex,
    graph_2_index: NodeIndex,
    graph_1: &CommandBufferOp,
    graph_2_outputs: &[(NodeIndex, u8)],
    old_to_new: &HashMap<NodeIndex, NodeIndex>,
) {
    for (target, weight) in graph
        .graph
        .edges_directed(graph_1_index, Direction::Outgoing)
        .filter(|e| e.target() != graph_2_index)
        .map(|e| (e.target(), *e.weight()))
        .collect::<Vec<_>>()
    {
        let graph_1_output = graph_1.outputs[weight.1 as usize];
        let graph_2_output_ind = graph_2_outputs
            .iter()
            .enumerate()
            .find(|(_, i)| **i == (old_to_new[&graph_1_output.0], graph_1_output.1))
            .unwrap()
            .0;
        graph.graph.add_edge(
            graph_2_index,
            target,
            (weight.0, graph_2_output_ind as u8, weight.2),
        );
    }
}

fn transfer_input_edges(graph: &mut Graph, graph_1_index: NodeIndex, graph_2_index: NodeIndex) {
    let mut n_graph_2_sources = graph
        .graph
        .edges_directed(graph_2_index, Direction::Incoming)
        .filter(|e| e.source() != graph_1_index)
        .count() as u8;
    for (source, weight) in graph
        .graph
        .edges_directed(graph_1_index, Direction::Incoming)
        .map(|e| (e.source(), *e.weight()))
        .collect::<Vec<_>>()
    {
        if graph
            .graph
            .edges_directed(graph_2_index, Direction::Incoming)
            .all(|e| e.source() != source || e.weight().1 != weight.1)
        {
            // This edge isn't already going to graph 2
            graph.graph.add_edge(
                source,
                graph_2_index,
                (n_graph_2_sources, weight.1, weight.2),
            );
            n_graph_2_sources += 1;
        }
    }
}

fn merge_inputs_internal(
    graph_1: &CommandBufferOp,
    graph_2: &mut CommandBufferOp,
    old_to_new: &HashMap<NodeIndex, NodeIndex>,
    graph_1_inputs: &[(NodeIndex, u8)],
    graph_2_inputs: &[(NodeIndex, u8)],
) {
    // Loop through new nodes
    for (old_node, new_node) in old_to_new {
        // Loop through edge inputs in old graph (don't need to be added to input map)
        for (source, weight) in graph_1
            .internal_graph
            .edges_directed(*old_node, Direction::Incoming)
            .map(|e| (e.source(), e.weight()))
        {
            graph_2
                .internal_graph
                .add_edge(old_to_new[&source], *new_node, *weight);
        }
        // Loop through input map inputs in old graph (need to be added to input map)
        for (node, inputs) in &graph_1.input_map {
            if !inputs.is_empty() {
                graph_2.input_map.insert(old_to_new[node], vec![]);
            }
            for input in inputs {
                let outer_node = graph_1_inputs[input.1 as usize];
                if let Some(ind) = graph_2_inputs
                    .iter()
                    .enumerate()
                    .find(|(_, i)| **i == outer_node)
                    .map(|(i, _)| i)
                {
                    graph_2
                        .input_map
                        .get_mut(&old_to_new[node])
                        .unwrap()
                        .push((input.0, ind as u8, input.2));
                }
            }
        }
    }
}

/// Merge op1 into op2
fn merge_command_buffer_ops(op1: NodeIndex, op2: NodeIndex, graph: &mut Graph) {
    // Add nodes to graph 2 from graph 1
    let mut graph_1 = (*graph
        .graph
        .node_weight(op1)
        .unwrap()
        .as_any()
        .downcast_ref::<CommandBufferOp>()
        .unwrap())
    .clone();
    // if graph_1.internal_graph.node_indices().count() > 2
    //     || graph
    //         .graph
    //         .node_weight_mut(op2)
    //         .unwrap()
    //         .as_any_mut()
    //         .downcast_mut::<CommandBufferOp>()
    //         .unwrap()
    //         .internal_graph
    //         .node_indices()
    //         .count()
    //         > 2
    // {
    //     return;
    // }
    let old_to_new = merge_nodes(
        &graph_1,
        graph
            .graph
            .node_weight_mut(op2)
            .unwrap()
            .as_any_mut()
            .downcast_mut::<CommandBufferOp>()
            .unwrap(),
    );
    let inputs_from_graph_1 = graph
        .graph
        .edges_directed(op2, Direction::Incoming)
        .filter(|e| e.source() == op1)
        .map(|e| (e.weight().0, e.weight().1))
        .collect();
    merge_internal_connections(
        &graph_1,
        graph
            .graph
            .node_weight_mut(op2)
            .unwrap()
            .as_any_mut()
            .downcast_mut::<CommandBufferOp>()
            .unwrap(),
        &inputs_from_graph_1,
        &old_to_new,
    );
    remove_unused_graph_1_outputs(graph, &mut graph_1, op1, op2);
    merge_outputs(
        &graph_1,
        graph
            .graph
            .node_weight_mut(op2)
            .unwrap()
            .as_any_mut()
            .downcast_mut::<CommandBufferOp>()
            .unwrap(),
        &old_to_new,
    );
    let graph_2_outputs = graph
        .graph
        .node_weight_mut(op2)
        .unwrap()
        .as_any_mut()
        .downcast_mut::<CommandBufferOp>()
        .unwrap()
        .outputs
        .clone();
    transfer_output_edges(graph, op1, op2, &graph_1, &graph_2_outputs, &old_to_new);
    transfer_input_edges(graph, op1, op2);
    let graph_1_inputs = graph
        .graph
        .edges_directed(op1, Direction::Incoming)
        .sorted_by_key(|e| e.weight().0)
        .map(|e| (e.source(), e.weight().1))
        .collect::<Vec<_>>();
    let graph_2_inputs = graph
        .graph
        .edges_directed(op2, Direction::Incoming)
        .filter(|e| e.source() != op1)
        .sorted_by_key(|e| e.weight().0)
        .map(|e| (e.source(), e.weight().1))
        .collect::<Vec<_>>();
    merge_inputs_internal(
        &graph_1,
        graph
            .graph
            .node_weight_mut(op2)
            .unwrap()
            .as_any_mut()
            .downcast_mut::<CommandBufferOp>()
            .unwrap(),
        &old_to_new,
        &graph_1_inputs,
        &graph_2_inputs,
    );
    move_references(
        &mut graph.id_remap,
        &mut graph.no_delete,
        &mut graph.to_retrieve,
        op1,
        op2,
    );
    graph.graph.remove_node(op1);
}

#[derive(Debug, Clone)]
struct CommandBufferOp {
    /// Edge weight: (input index, output index, Shape)
    internal_graph: petgraph::stable_graph::StableGraph<MetalKernelWrapper, (u8, u8, ShapeTracker)>,
    dyn_map: *const HashMap<char, usize>,
    /// Map for input nodes to CommandBufferOp inputs
    input_map: HashMap<NodeIndex, Vec<(u8, u8, ShapeTracker)>>,
    /// Orderings of the outputs
    outputs: Vec<(NodeIndex, u8)>,
}
impl PartialEq for CommandBufferOp {
    fn eq(&self, _: &Self) -> bool {
        false
    }
}

impl Operator for CommandBufferOp {
    fn process(&self, inp: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        autoreleasepool(|| {
            // Create command buffer
            let dev = Device::system_default().unwrap();
            let command_queue = dev.new_command_queue();
            let command_buffer = command_queue.new_command_buffer();

            // Dynamic dependency tracking (will be scrapped once we switch to shared buffers)
            let mut remaining_consumers: HashMap<(NodeIndex, u8), usize> = self
                .internal_graph
                .node_indices()
                .flat_map(|i| {
                    self.internal_graph
                        .edges_directed(i, Direction::Outgoing)
                        .group_by(|k| k.weight().1)
                        .into_iter()
                        .map(|(ind, edges)| ((i, ind), edges.count()))
                        .collect::<Vec<_>>()
                })
                .collect();
            let mut intermediate_buffers: HashMap<(NodeIndex, u8), Buffer> = HashMap::new();

            // For each operation
            for op_id in petgraph::algo::toposort(&self.internal_graph, None).unwrap() {
                // Run through metal kernel forward
                let mut input_buffers = vec![];
                let (mut i, mut j) = (0, 0);
                let edges = self
                    .internal_graph
                    .edges_directed(op_id, Direction::Incoming)
                    .sorted_by_key(|e| e.weight().0)
                    .map(|e| (e.source(), *e.weight()))
                    .collect::<Vec<_>>();
                let inputs = self
                    .input_map
                    .get(&op_id)
                    .cloned()
                    .unwrap_or_default()
                    .into_iter()
                    .sorted_by_key(|(i, _, _)| *i)
                    .collect::<Vec<_>>();
                let mut to_remove = vec![];
                while i + j < edges.len() + inputs.len() {
                    if i < edges.len() && edges[i].1 .0 as usize == input_buffers.len() {
                        input_buffers.push((
                            intermediate_buffers
                                .get(&(edges[i].0, edges[i].1 .1))
                                .unwrap(),
                            edges[i].1 .2,
                        ));
                        let rem = remaining_consumers
                            .get_mut(&(edges[i].0, edges[i].1 .1))
                            .unwrap();
                        *rem -= 1;
                        if *rem == 0 {
                            to_remove.push((edges[i].0, edges[i].1 .1));
                        }
                        i += 1;
                    } else {
                        input_buffers.push((
                            inp[inputs[j].1 as usize]
                                .0
                                .borrowed()
                                .data
                                .as_any()
                                .downcast_ref::<Buffer>()
                                .unwrap(),
                            inputs[j].2,
                        ));
                        j += 1;
                    }
                }

                let new_buffers = self
                    .internal_graph
                    .node_weight(op_id)
                    .unwrap()
                    .0
                    .metal_forward(
                        &input_buffers
                            .iter()
                            .map(|(buff, sh)| {
                                (*buff, sh.resolve_global_dyn_dims(unsafe { &*self.dyn_map }))
                            })
                            .collect::<Vec<_>>(),
                        &dev,
                        command_buffer,
                    );
                intermediate_buffers.extend(
                    new_buffers
                        .into_iter()
                        .enumerate()
                        .map(|(i, buffer)| ((op_id, i as u8), buffer)),
                );

                for k in to_remove {
                    if !self.outputs.contains(&k) {
                        intermediate_buffers.remove(&k);
                    }
                }
            }
            // Execute command buffer
            command_buffer.commit();
            command_buffer.wait_until_completed();

            // Return buffers as tensors
            self.outputs
                .iter()
                .map(|id| Tensor {
                    data: Box::new(intermediate_buffers.remove(id).unwrap()),
                })
                .collect()
        })
    }
}
