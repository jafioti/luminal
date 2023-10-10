use petgraph::visit::EdgeRef;

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

pub fn transfer_weights<M: SerializeModule>(
    src_model: &M,
    src_graph: &mut Graph,
    dest_model: &M,
    dest_graph: &mut Graph,
) {
    let src_state_dict = {
        let mut s = Serializer::default();
        src_model.serialize(&mut s);
        s.state
    };
    let dest_state_dict = {
        let mut s = Serializer::default();
        dest_model.serialize(&mut s);
        s.state
    };
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
pub fn delete_loads<M: SerializeModule>(model: &M, graph: &mut Graph) {
    let mut s = Serializer::default();
    model.serialize(&mut s);
    for node in s.state.values() {
        for e in graph
            .graph
            .edges_directed(
                remap_id(*node, &graph.id_remap),
                petgraph::Direction::Incoming,
            )
            .map(|e| e.source())
            .collect::<Vec<_>>()
        {
            graph.graph.remove_node(e);
        }
    }
    graph.toposort();
}

pub fn mark_weights<M: SerializeModule>(model: &M, graph: &mut Graph) {
    let mut s = Serializer::default();
    model.serialize(&mut s);
    for node in s.state.values() {
        graph.no_delete.insert(*node);
    }
}
