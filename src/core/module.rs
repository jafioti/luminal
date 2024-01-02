use std::collections::HashMap;

use petgraph::{stable_graph::NodeIndex, visit::EdgeRef};

use crate::prelude::{Graph, SerializeModule, Serializer};

/// A module that can initialize it's variables on the graph
pub trait InitModule {
    fn initialize(cx: &mut Graph) -> Self;
}

/// A module with a forward pass
pub trait Module<I> {
    type Output;
    fn forward(&self, input: I) -> Self::Output;
}

pub fn state_dict<M: SerializeModule>(model: &M) -> HashMap<String, NodeIndex> {
    let mut s = Serializer::default();
    model.serialize(&mut s);
    s.state
}

pub fn transfer_weights<M: SerializeModule>(
    src_model: &M,
    src_graph: &mut Graph,
    dest_model: &M,
    dest_graph: &mut Graph,
) {
    let src_state_dict = state_dict(src_model);
    let dest_state_dict = state_dict(dest_model);
    for (key, src_ind) in src_state_dict {
        let dest_ind = *dest_state_dict
            .get(&key)
            .unwrap_or_else(|| panic!("{key} was in the source model but not in the dest model!"));
        let mut output_num = 0;
        loop {
            if let Some(tensor) = src_graph.tensors.remove(&(src_ind, output_num)) {
                dest_graph.tensors.insert((dest_ind, output_num), tensor);
            } else {
                break;
            }
            output_num += 1;
        }
        if output_num == 0 {
            panic!("{key} tensor wasn't found in the source graph!");
        }
    }
}

/// Delete all incoming nodes to this set of nodes
pub fn delete_inputs(nodes: &[NodeIndex], graph: &mut Graph) {
    for node in nodes {
        delete_upstream(graph, *node);
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

pub fn keep_weights<M: SerializeModule>(model: &M, graph: &mut Graph) {
    for node in state_dict(model).values() {
        graph.no_delete.insert(*node);
    }
}
