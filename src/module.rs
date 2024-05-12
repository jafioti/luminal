use rustc_hash::{FxHashMap, FxHashSet};

use itertools::Itertools;
use petgraph::{stable_graph::NodeIndex, visit::EdgeRef, Direction};

use crate::prelude::*;

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
pub fn param_dict(model: impl SerializeModule) -> FxHashMap<String, NodeIndex> {
    let mut s = Serializer::default();
    model.serialize(&mut s);
    s.state
}

/// Set of weight node ids
pub fn params(model: impl SerializeModule) -> Vec<NodeIndex> {
    param_dict(model)
        .into_iter()
        .sorted_by_key(|(k, _)| k.clone())
        .map(|(_, v)| v)
        .collect()
}

/// Transfer data from one set of nodes in one graph to another set in another graph
pub fn transfer_data(
    srcs: impl ToIds,
    src_graph: &mut Graph,
    dests: impl ToIds,
    dest_graph: &mut Graph,
) {
    for (src, dest) in srcs.to_ids().into_iter().zip(dests.to_ids().into_iter()) {
        let mut output_num = 0;
        while let Some(tensor) = src_graph.tensors.remove(&(src, output_num)) {
            dest_graph.tensors.insert((dest, output_num), tensor);
            output_num += 1;
        }
        if output_num == 0 {
            panic!("No source tensor found for node {}", src.index());
        }
    }
}

/// Transfer data from one set of nodes to another set in the same graph
pub fn transfer_data_same_graph(srcs: impl ToIds, dests: impl ToIds, graph: &mut Graph) {
    for (src, dest) in srcs.to_ids().into_iter().zip(dests.to_ids().into_iter()) {
        let mut output_num = 0;
        while let Some(tensor) = graph.tensors.remove(&(src, output_num)) {
            graph.tensors.insert((dest, output_num), tensor);
            output_num += 1;
        }
        if output_num == 0 {
            panic!("No source tensor found for node {}", src.index());
        }
    }
}

/// Delete all incoming nodes to this set of nodes
pub fn delete_inputs(nodes: impl ToIds, graph: &mut Graph) {
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
pub fn downstream(nodes: impl ToIds, graph: &Graph) -> Vec<NodeIndex> {
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

// Tuple impls

impl<X> Module<X> for () {
    type Output = X;
    fn forward(&self, input: X) -> Self::Output {
        input
    }
}

impl<X, M: Module<X, Output = X>> Module<X> for Vec<M> {
    type Output = X;
    fn forward(&self, mut x: X) -> Self::Output {
        for layer in self {
            x = layer.forward(x);
        }
        x
    }
}

impl<X, M: Module<X, Output = X>> Module<X> for &[M] {
    type Output = X;
    fn forward(&self, mut x: X) -> Self::Output {
        for layer in self.iter() {
            x = layer.forward(x);
        }
        x
    }
}

impl<const N: usize, X, M: Module<X, Output = X>> Module<X> for [M; N] {
    type Output = X;
    fn forward(&self, mut x: X) -> Self::Output {
        for layer in self.iter() {
            x = layer.forward(x);
        }
        x
    }
}

macro_rules! tuple_impls {
    ([$($name:ident),+] [$($idx:tt),+], $last:ident, [$($rev_tail:ident),*]) => {
        impl<
            Input,
            $last:
            $(Module::<$rev_tail ::Output>, $rev_tail: )*
            Module<Input>
        > Module<Input> for ($($name,)+) {
            type Output = $last ::Output;

            /// Calls forward sequentially on each module in the tuple.
            fn forward(&self, x: Input) -> Self::Output {
                $(let x = self.$idx.forward(x);)+
                x
            }
        }

        impl<$($name: InitModule,)+> InitModule for ($($name,)+) {
            fn initialize(cx: &mut Graph) -> Self {
                (
                $($name::initialize(cx),)+
                )
            }
        }

        impl<$($name: SerializeModule,)+> SerializeModule for ($($name,)+) {
            fn serialize(&self, s: &mut Serializer) {
                $(s.module(&format!("layer{}", $idx), &self.$idx);)+
            }
        }
    };
}

tuple_impls!([M1][0], M1, []);
tuple_impls!([M1, M2] [0, 1], M2, [M1]);
tuple_impls!([M1, M2, M3] [0, 1, 2], M3, [M2, M1]);
tuple_impls!([M1, M2, M3, M4] [0, 1, 2, 3], M4, [M3, M2, M1]);
tuple_impls!([M1, M2, M3, M4, M5] [0, 1, 2, 3, 4], M5, [M4, M3, M2, M1]);
tuple_impls!([M1, M2, M3, M4, M5, M6] [0, 1, 2, 3, 4, 5], M6, [M5, M4, M3, M2, M1]);
tuple_impls!([M1, M2, M3, M4, M5, M6, M7] [0, 1, 2, 3, 4, 5, 6], M7, [M6, M5, M4, M3, M2, M1]);
tuple_impls!([M1, M2, M3, M4, M5, M6, M7, M8] [0, 1, 2, 3, 4, 5, 6, 7], M8, [M7, M6, M5, M4, M3, M2, M1]);
tuple_impls!([M1, M2, M3, M4, M5, M6, M7, M8, M9] [0, 1, 2, 3, 4, 5, 6, 7, 8], M9, [M8, M7, M6, M5, M4, M3, M2, M1]);
tuple_impls!([M1, M2, M3, M4, M5, M6, M7, M8, M9, M10] [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], M10, [M9, M8, M7, M6, M5, M4, M3, M2, M1]);

/// Tell luminal how to represent the module as a dict of (String, NodeIndex)'s
pub trait SerializeModule {
    fn serialize(&self, s: &mut Serializer);
}

impl<T: SerializeModule> SerializeModule for &T {
    fn serialize(&self, s: &mut Serializer) {
        (*self).serialize(s)
    }
}

/// Serializer keeps track of the tensors and modules that make up a model
#[derive(Debug, Default)]
pub struct Serializer {
    current_path: Vec<String>,
    pub state: FxHashMap<String, NodeIndex>,
}

impl Serializer {
    pub fn tensor<S: Shape>(&mut self, name: &str, tensor: GraphTensor<S>) {
        if !name.is_empty() {
            // Add new path component
            self.current_path.push(name.to_string());
        }
        // Insert tensor id
        self.state.insert(self.current_path.join("/"), tensor.id);
        if !name.is_empty() {
            // Remove new path component
            self.current_path.pop();
        }
    }
    pub fn module(&mut self, name: &str, module: impl SerializeModule) {
        if !name.is_empty() {
            // Add new path component
            self.current_path.push(name.to_string());
        }
        // Serialize
        module.serialize(self);
        if !name.is_empty() {
            // Remove new path component
            self.current_path.pop();
        }
    }
}
