use rustc_hash::{FxHashMap, FxHashSet};

use itertools::Itertools;
use petgraph::{stable_graph::NodeIndex, visit::EdgeRef, Direction};

use crate::prelude::{Graph, SerializeModule, Serializer};

use super::compiler_utils::ToIds;

/// A module that can initialize it's variables on the graph
pub trait InitModule {
    fn initialize(cx: &mut Graph) -> Self;
}

/// A module with a forward pass
pub trait Module<I> {
    type Output;
    fn forward(&self, input: I) -> Self::Output;
}

/// Mapping from weight name to node id
pub fn state_dict<M: SerializeModule>(model: &M) -> FxHashMap<String, NodeIndex> {
    let mut s = Serializer::default();
    model.serialize(&mut s);
    s.state
}

/// Set of weight node ids
pub fn state_set<M: SerializeModule>(model: &M) -> Vec<NodeIndex> {
    state_dict(model)
        .into_iter()
        .sorted_by_key(|(k, _)| k.clone())
        .map(|(_, v)| v)
        .collect()
}

/// Transfer data from one set of nodes in one graph to another set in another graph
pub fn transfer_data<A: ToIds, B: ToIds>(
    srcs: A,
    src_graph: &mut Graph,
    dests: B,
    dest_graph: &mut Graph,
) {
    for (src, dest) in srcs.to_ids().into_iter().zip(dests.to_ids().into_iter()) {
        let mut output_num = 0;
        while let Some(tensor) = src_graph.tensors.remove(&(src, output_num)) {
            dest_graph.tensors.insert((dest, output_num), tensor);
            output_num += 1;
        }
    }
}

/// Transfer data from one set of nodes to another set in the same graph
pub fn transfer_data_same_graph<A: ToIds, B: ToIds>(srcs: A, dests: B, graph: &mut Graph) {
    for (src, dest) in srcs.to_ids().into_iter().zip(dests.to_ids().into_iter()) {
        let mut output_num = 0;
        while let Some(tensor) = graph.tensors.remove(&(src, output_num)) {
            graph.tensors.insert((dest, output_num), tensor);
            output_num += 1;
        }
    }
}

/// Delete all incoming nodes to this set of nodes
pub fn delete_inputs<T: ToIds>(nodes: T, graph: &mut Graph) {
    for node in nodes.to_ids() {
        delete_upstream(graph, node);
    }
    graph.toposort();
}

fn delete_upstream(graph: &mut Graph, node: NodeIndex) {
    for e in graph
        .graph
        .edges_directed(node, petgraph::Direction::Incoming)
        .filter(|e| !e.weight().is_schedule())
        .map(|e| e.source())
        .collect::<Vec<_>>()
    {
        delete_upstream(graph, e);
        graph.graph.remove_node(e);
    }
}

/// Get the downstream set from an original set, in a deterministic order
pub fn downstream<T: ToIds>(nodes: T, graph: &Graph) -> Vec<NodeIndex> {
    let orig_set = nodes.to_ids().into_iter().collect::<FxHashSet<_>>();
    let mut fin = vec![];
    let mut added = FxHashSet::default();
    // Loop through nodes
    for mut node in nodes.to_ids() {
        // Go downstream as far as possible along a single stream of ops
        while graph
            .graph
            .edges_directed(node, Direction::Outgoing)
            .filter(|e| !e.weight().is_schedule())
            .count()
            == 1
        {
            let new_node = graph
                .graph
                .edges_directed(node, Direction::Outgoing)
                .next()
                .unwrap()
                .target();
            if !is_from_set(new_node, graph, &orig_set) {
                break;
            }
            node = new_node;
        }
        if !added.contains(&node) {
            added.insert(node);
            fin.push(node);
        }
    }
    fin
}

fn is_from_set(node: NodeIndex, graph: &Graph, set: &FxHashSet<NodeIndex>) -> bool {
    // Reverse dfs upward
    let mut stack = vec![node];
    while let Some(node) = stack.pop() {
        if !set.contains(&node) {
            let mut new_nodes = graph
                .graph
                .edges_directed(node, Direction::Incoming)
                .filter(|e| !e.weight().is_schedule())
                .map(|e| e.source())
                .collect::<Vec<_>>();
            if new_nodes.is_empty() {
                // Node isn't from set and doesn't have upstream nodes
                return false;
            }
            stack.append(&mut new_nodes);
        }
    }
    true
}
