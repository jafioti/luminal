use std::collections::HashMap;

use petgraph::{stable_graph::NodeIndex, visit::EdgeRef};

use crate::prelude::{remap_id, Graph, SerializeModule, Serializer};

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
    for (key, mut src_ind) in src_state_dict {
        src_ind = remap_id(src_ind, &src_graph.id_remap);
        let dest_ind = remap_id(
            *dest_state_dict.get(&key).unwrap_or_else(|| {
                panic!("{key} was in the source model but not in the dest model!")
            }),
            &dest_graph.id_remap,
        );
        let mut output_num = 0;
        loop {
            if let Some(tensor) = src_graph.tensors.remove(&(src_ind, output_num)) {
                dest_graph.tensors.insert((dest_ind, output_num), tensor);
            } else {
                break;
            }
            output_num += 1;
        }
    }
}

/// Delete all incoming nodes to the states of the model
pub fn delete_inputs<M: SerializeModule>(model: &M, graph: &mut Graph) {
    for node in state_dict(model).values() {
        delete_upstream(graph, remap_id(*node, &graph.id_remap));
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
        let id = remap_id(*node, &graph.id_remap);
        graph.no_delete.insert(id);
    }
}
