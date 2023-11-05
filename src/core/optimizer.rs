use std::{
    any::{Any, TypeId},
    collections::{
        hash_map::{DefaultHasher, Entry},
        HashMap, HashSet, VecDeque,
    },
    hash::Hasher,
    sync::{Arc, Mutex},
};

use itertools::Itertools;
use petgraph::{
    stable_graph::{NodeIndex, StableGraph},
    visit::EdgeRef,
    Direction,
};

use crate::{
    graph::Graph,
    op::Operator,
    prelude::{Dependency, Dim, ShapeTracker},
};

pub trait GraphOptimizer {
    /// Run an optimization pass
    fn optimize(&self, graph: &mut Graph);
}

impl GraphOptimizer for () {
    fn optimize(&self, _: &mut Graph) {}
}

macro_rules! tuple_impls {
    ([$($name:ident),+] , [$($idx:tt),+]) => {
        impl<
        $($name:
            GraphOptimizer, )+
        > GraphOptimizer for ($($name,)+) {
            fn optimize(&self, graph: &mut Graph) {
                $(self.$idx.optimize(graph);)+
            }
        }
    };
}

tuple_impls!([M1], [0]);
tuple_impls!([M1, M2], [0, 1]);
tuple_impls!([M1, M2, M3], [0, 1, 2]);
tuple_impls!([M1, M2, M3, M4], [0, 1, 2, 3]);
tuple_impls!([M1, M2, M3, M4, M5], [0, 1, 2, 3, 4]);
tuple_impls!([M1, M2, M3, M4, M5, M6], [0, 1, 2, 3, 4, 5]);
tuple_impls!([M1, M2, M3, M4, M5, M6, M7], [0, 1, 2, 3, 4, 5, 6]);
tuple_impls!([M1, M2, M3, M4, M5, M6, M7, M8], [0, 1, 2, 3, 4, 5, 6, 7]);
tuple_impls!(
    [M1, M2, M3, M4, M5, M6, M7, M8, M9],
    [0, 1, 2, 3, 4, 5, 6, 7, 8]
);
tuple_impls!(
    [M1, M2, M3, M4, M5, M6, M7, M8, M9, M10],
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
);

// Helpers

/// Transfer all external references from one node to another (this may happen because one node is about to be removed / merged into another)
pub fn move_references(
    id_remap: &mut HashMap<NodeIndex, NodeIndex>,
    no_delete: &mut HashSet<NodeIndex<u32>>,
    to_retrieve: &mut HashSet<NodeIndex<u32>>,
    src: NodeIndex,
    trg: NodeIndex,
) {
    // Only remap if it isn't already remapped, otherwise it would invalidate the past remappig
    if let Entry::Vacant(e) = id_remap.entry(src) {
        // Create remap
        e.insert(trg);
        // Transfer no_delete
        if no_delete.remove(&src) {
            no_delete.insert(trg);
        }
        // Transfer to_retrieve
        if to_retrieve.remove(&src) {
            to_retrieve.insert(trg);
        }
    }
}

pub fn move_outgoing_edge<N, E: Clone>(
    from: NodeIndex,
    to: NodeIndex,
    graph: &mut StableGraph<N, E>,
) {
    // Carry over outgoing edges from node to other_node
    for (weight, target) in graph
        .edges_directed(from, petgraph::Direction::Outgoing)
        .map(|e| (e.weight().clone(), e.target()))
        .collect::<Vec<_>>()
    {
        graph.add_edge(to, target, weight);
    }
}

pub fn move_incoming_edge<N, E: Clone>(
    from: NodeIndex,
    to: NodeIndex,
    graph: &mut StableGraph<N, E>,
) {
    // Carry over incoming edges from node to other_node
    for (weight, source) in graph
        .edges_directed(from, petgraph::Direction::Incoming)
        .map(|e| (e.weight().clone(), e.source()))
        .collect::<Vec<_>>()
    {
        graph.add_edge(source, to, weight);
    }
}

pub trait TraitObjEq {
    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;
    fn is_equal(&self, _: &dyn TraitObjEq) -> bool;
}

// Implement TraitObjEq for all 'static types implementing PartialEq.
impl<S: 'static + PartialEq> TraitObjEq for S {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn is_equal(&self, other: &dyn TraitObjEq) -> bool {
        // Do a type-safe casting. If the types are different,
        // return false, otherwise test the values for equality.
        other
            .as_any()
            .downcast_ref::<S>()
            .map_or(false, |a| self == a)
    }
}

type SelectorWeight = (
    Option<TypeId>,                                     // Type constraint
    Option<fn(&dyn Operator, &[ShapeTracker]) -> bool>, // Check constraint
    Option<Vec<Vec<Dim>>>,                              // Shape constraint
    Option<Vec<Vec<bool>>>,                             // Fake constraint
    Vec<*mut NodeIndex>,                                // Pointers
);

type SelectionGraph = petgraph::Graph<SelectorWeight, Option<u8>>;

// Graph Selector
#[derive(Default, Clone)]
pub struct GraphSelector {
    #[allow(clippy::type_complexity)]
    graph: Arc<Mutex<SelectionGraph>>,
}

pub struct GraphSearch {
    selector: GraphSelector,
    graph: *const Graph,
    already_returned: HashSet<Vec<(NodeIndex, NodeIndex)>>,
    found: VecDeque<HashMap<NodeIndex, NodeIndex>>, // Queue of found patterns
    graph_hash: u64,
}

fn calculate_hash<T: std::hash::Hash>(t: &T) -> u64 {
    let mut s = DefaultHasher::new();
    t.hash(&mut s);
    s.finish()
}

fn hash_graph(graph: &Graph) -> u64 {
    calculate_hash(&(
        graph.graph.node_indices().collect::<Vec<_>>(),
        graph
            .graph
            .edge_indices()
            .map(|e| graph.graph.edge_endpoints(e))
            .collect::<Vec<_>>(),
    ))
}

impl Iterator for GraphSearch {
    type Item = ();

    fn next(&mut self) -> Option<Self::Item> {
        // Look through graph for pattern from selector
        let graph = unsafe { self.graph.as_ref().unwrap() };
        let graph_hash = hash_graph(graph);
        if graph_hash != self.graph_hash {
            self.graph_hash = graph_hash;
            self.found.clear();
        }
        if self.found.is_empty() {
            let Some(select_start) = petgraph::algo::toposort(&*self.selector.graph.lock().unwrap(), None).unwrap().pop() else {
                return None;
            };
            for node in graph.graph.node_indices() {
                for m in search_new(
                    select_start,
                    &self.selector.graph.lock().unwrap(),
                    node,
                    &graph.graph,
                    &mut HashSet::new(),
                    &mut HashMap::default(),
                ) {
                    let key = m
                        .iter()
                        .sorted_by_key(|(i, _)| **i)
                        .map(|(a, b)| (*a, *b))
                        .collect::<Vec<_>>();
                    if !self.already_returned.contains(&key) {
                        self.found.push_back(m);
                    }
                }
            }
        }

        match self.found.pop_front() {
            Some(pattern) => {
                self.already_returned.insert(
                    pattern
                        .iter()
                        .sorted_by_key(|(i, _)| **i)
                        .map(|(a, b)| (*a, *b))
                        .collect(),
                );
                // Apply pattern to ptrs
                let selector_graph = self.selector.graph.lock().unwrap();
                for (selector_node, ptr) in selector_graph.node_indices().flat_map(|n| {
                    selector_graph
                        .node_weight(n)
                        .unwrap()
                        .4
                        .iter()
                        .map(move |i| (n, *i))
                }) {
                    unsafe {
                        *ptr = pattern[&selector_node];
                    }
                }
                Some(())
            }
            None => None,
        }
    }
}

/// Find all matching patterns
fn search_new(
    selector_node: NodeIndex,
    selector_graph: &SelectionGraph,
    graph_node: NodeIndex,
    graph: &StableGraph<Box<dyn Operator>, Dependency>,
    used: &mut HashSet<NodeIndex>,
    selector_used: &mut HashMap<NodeIndex, NodeIndex>,
) -> Vec<HashMap<NodeIndex, NodeIndex>> {
    if !test_node(
        selector_graph.node_weight(selector_node).unwrap(),
        graph,
        graph_node,
    ) {
        return vec![];
    }

    selector_used.insert(selector_node, graph_node);
    used.insert(graph_node);

    if selector_used.len() == selector_graph.node_count() {
        let m = selector_used.clone();
        used.remove(&graph_node);
        selector_used.remove(&selector_node);
        return vec![m];
    }

    let mut new_matches = vec![];
    // Loop through outgoing edges
    for graph_target in graph
        .edges_directed(graph_node, Direction::Outgoing)
        .map(|e| e.target())
        .filter(|e| !used.contains(e))
        .collect::<Vec<_>>()
    {
        for selector_target in selector_graph
            .edges_directed(selector_node, Direction::Outgoing)
            .map(|e| e.target())
            .filter(|e| !selector_used.contains_key(e))
            .collect::<Vec<_>>()
        {
            let matches = search_new(
                selector_target,
                selector_graph,
                graph_target,
                graph,
                used,
                selector_used,
            );
            new_matches.extend(matches);
        }
    }

    // Loop through incoming edges
    for graph_source in graph
        .edges_directed(graph_node, Direction::Incoming)
        .map(|e| e.source())
        .filter(|e| !used.contains(e))
        .collect::<Vec<_>>()
    {
        for selector_source in selector_graph
            .edges_directed(selector_node, Direction::Incoming)
            .map(|e| e.source())
            .filter(|e| !selector_used.contains_key(e))
            .collect::<Vec<_>>()
        {
            let matches = search_new(
                selector_source,
                selector_graph,
                graph_source,
                graph,
                used,
                selector_used,
            );
            new_matches.extend(matches);
        }
    }

    // Reset used maps
    used.remove(&graph_node);
    selector_used.remove(&selector_node);

    new_matches
}

fn test_node(
    (type_id, check, shape, fakes, _): &SelectorWeight,
    graph: &StableGraph<Box<dyn Operator>, Dependency>,
    graph_node: NodeIndex,
) -> bool {
    let current_weight = graph.node_weight(graph_node).unwrap();
    // Test type
    if let Some(ty) = type_id {
        if current_weight.as_any().type_id() != *ty {
            return false;
        }
    }
    let input_shapes = graph
        .edges_directed(graph_node, petgraph::Direction::Incoming)
        .filter_map(|e| e.weight().as_data())
        .sorted_by_key(|e| e.0)
        .map(|e| e.2)
        .collect::<Vec<_>>();

    // Test shape
    if let Some(shape) = shape {
        let mut shape_map = HashMap::new();
        if shape.len() != input_shapes.len() {
            return false;
        }
        for (a_sh, b_sh) in shape.iter().zip(input_shapes.iter()) {
            if a_sh.len() != b_sh.dims.len() {
                return false;
            }
            for (a, b) in a_sh.iter().zip(b_sh.shape().iter()) {
                match a {
                    Dim::Known(n) => {
                        if *b != Dim::Known(*n) {
                            return false;
                        }
                    }
                    Dim::Unknown(c) => {
                        if let Some(expected) = shape_map.get(c) {
                            if b != expected {
                                return false;
                            }
                        } else {
                            shape_map.insert(*c, *b);
                        }
                    }
                }
            }
        }
    }
    // Test fakes
    if let Some(fakes) = fakes {
        for (a_sh, b_sh) in fakes.iter().zip(input_shapes.iter()) {
            for (a, b) in a_sh.iter().zip(b_sh.indexes.iter().map(|i| b_sh.fake[*i])) {
                if *a != b {
                    return false;
                }
            }
        }
    }

    // Run check
    if let Some(check) = check {
        if !check(current_weight.as_ref(), &input_shapes) {
            return false;
        }
    }
    true
}

// #[allow(clippy::type_complexity)]
// fn search(
//     selector_node: NodeIndex,
//     selector_graph: &SelectionGraph,
//     graph_node: NodeIndex,
//     graph: &StableGraph<Box<dyn Operator>, (u8, u8, ShapeTracker)>,
//     used: &mut HashSet<NodeIndex>,
//     selector_used: &mut HashSet<NodeIndex>,
//     assignment_order: &mut Vec<NodeIndex>,
// ) -> bool {
//     let selector_weight = selector_graph.node_weight(selector_node).unwrap();
//     if !test_node(
//         selector_graph.node_weight(selector_node).unwrap(),
//         graph,
//         graph_node,
//     ) {
//         return false;
//     }

//     // Used is to make sure we don't use the same node from the source graph twice, which prevents cycles
//     used.insert(graph_node);
//     selector_used.insert(selector_node);
//     // Match outgoing
//     for (select_outgoing, select_output) in selector_graph
//         .edges_directed(selector_node, Direction::Outgoing)
//         .map(|e| (e.target(), e.weight()))
//         .filter(|i| !selector_used.contains(&i.0))
//         .collect::<Vec<_>>()
//         .into_iter()
//     {
//         if let Some((target, _)) = graph
//             .edges_directed(graph_node, Direction::Outgoing)
//             .map(|e| (e.target(), e.weight().1))
//             .filter(|i| !used.contains(&i.0) && select_output.map(|o| i.1 == o).unwrap_or(true))
//             .collect::<Vec<_>>()
//             .into_iter()
//             .find(|(i, _)| {
//                 search(
//                     select_outgoing,
//                     selector_graph,
//                     *i,
//                     graph,
//                     used,
//                     selector_used,
//                     assignment_order,
//                 )
//             })
//         {
//             used.insert(target);
//         } else {
//             used.remove(&graph_node);
//             selector_used.remove(&selector_node);
//             return false;
//         }
//     }
//     // Match incoming
//     for (select_incoming, select_output) in selector_graph
//         .edges_directed(selector_node, Direction::Incoming)
//         .map(|e| (e.source(), e.weight()))
//         .filter(|i| !selector_used.contains(&i.0))
//         .collect::<Vec<_>>()
//         .into_iter()
//     {
//         if let Some((target, _)) = graph
//             .edges_directed(graph_node, Direction::Incoming)
//             .map(|e| (e.source(), e.weight().1))
//             .filter(|i| !used.contains(&i.0) && select_output.map(|o| i.1 == o).unwrap_or(true))
//             .collect::<Vec<_>>()
//             .into_iter()
//             .find(|i| {
//                 search(
//                     select_incoming,
//                     selector_graph,
//                     i.0,
//                     graph,
//                     used,
//                     selector_used,
//                     assignment_order,
//                 )
//             })
//         {
//             used.insert(target);
//         } else {
//             used.remove(&graph_node);
//             selector_used.remove(&selector_node);
//             return false;
//         }
//     }
//     // All checks out
//     for ptr in &selector_weight.4 {
//         unsafe {
//             **ptr = graph_node;
//         }
//     }
//     assignment_order.push(graph_node);
//     true
// }

impl GraphSelector {
    /// Create a new op to select
    pub fn op(&self) -> OpSelector {
        let mut m_self = self.graph.lock().unwrap();
        let id = m_self.add_node((None, None, None, None, vec![]));
        OpSelector { graph: self, id }
    }

    /// Specify an edge between ops
    pub fn edge(&self, o1: OpSelector, o2: OpSelector) -> OpSelector {
        let mut m_self = self.graph.lock().unwrap();
        m_self.add_edge(o1.id, o2.id, None);
        o2
    }

    /// Specify an edge between ops with a specified output
    pub fn edge_output(&self, o1: OpSelector, output: u8, o2: OpSelector) -> OpSelector {
        let mut m_self = self.graph.lock().unwrap();
        m_self.add_edge(o1.id, o2.id, Some(output));
        o2
    }

    /// Start searching a graph
    pub fn search(self, graph: &Graph) -> GraphSearch {
        GraphSearch {
            selector: self,
            already_returned: HashSet::default(),
            found: VecDeque::default(),
            graph_hash: hash_graph(graph),
            graph,
        }
    }
}

#[derive(Clone, Copy)]
pub struct OpSelector {
    graph: *const GraphSelector,
    id: NodeIndex,
}

impl OpSelector {
    /// Constrain the op to a type
    pub fn ty<O: Operator + 'static>(self) -> OpSelector {
        let graph = unsafe { self.graph.as_ref().unwrap() };
        let mut m_graph = graph.graph.lock().unwrap();
        m_graph.node_weight_mut(self.id).unwrap().0 = Some(TypeId::of::<O>());
        self
    }
    /// Constrain the op to a type
    pub fn type_id(self, type_id: TypeId) -> OpSelector {
        let graph = unsafe { self.graph.as_ref().unwrap() };
        let mut m_graph = graph.graph.lock().unwrap();
        m_graph.node_weight_mut(self.id).unwrap().0 = Some(type_id);
        self
    }
    /// Constrain the op to a checking function
    pub fn check(self, check: fn(&dyn Operator, &[ShapeTracker]) -> bool) -> Self {
        let graph = unsafe { self.graph.as_ref().unwrap() };
        let mut m_graph = graph.graph.lock().unwrap();
        m_graph.node_weight_mut(self.id).unwrap().1 = Some(check);
        self
    }
    /// Constrain the op to input shapes
    pub fn shapes<S: Into<Vec<Vec<Dim>>>>(self, shapes: S) -> Self {
        let graph = unsafe { self.graph.as_ref().unwrap() };
        let mut m_graph = graph.graph.lock().unwrap();
        m_graph.node_weight_mut(self.id).unwrap().2 = Some(shapes.into());
        self
    }
    /// Constrain the op to input shape fakes
    pub fn fakes<S: Into<Vec<Vec<bool>>>>(self, fakes: S) -> Self {
        let graph = unsafe { self.graph.as_ref().unwrap() };
        let mut m_graph = graph.graph.lock().unwrap();
        m_graph.node_weight_mut(self.id).unwrap().3 = Some(fakes.into());
        self
    }
    /// Register a pointer to set if the op is matched
    pub fn ptr(self, ptr: *mut NodeIndex) -> Self {
        let graph = unsafe { self.graph.as_ref().unwrap() };
        let mut m_graph = graph.graph.lock().unwrap();
        m_graph.node_weight_mut(self.id).unwrap().4.push(ptr);
        self
    }
}

#[cfg(test)]
mod tests {
    use petgraph::adj::NodeIndex;

    use crate::{
        op::{Exp2, Log2, Mul, SumReduce},
        prelude::Graph,
    };

    use super::GraphSelector;

    #[test]
    fn test_graph_selector() {
        let cx = Graph::default();
        // Exp -> Log or Log -> Exp
        let (mut exp, mut log) = (NodeIndex::default(), NodeIndex::default());
        let selector1 = GraphSelector::default();
        selector1.edge(
            selector1.op().ty::<Log2>().ptr(&mut log),
            selector1.op().ty::<Exp2>().ptr(&mut exp),
        );
        let selector2 = GraphSelector::default();
        selector2.edge(
            selector2.op().ty::<Exp2>().ptr(&mut exp),
            selector2.op().ty::<Log2>().ptr(&mut log),
        );

        assert_eq!(
            selector1.search(&cx).chain(selector2.search(&cx)).next(),
            None
        );

        // Matmul
        let s = GraphSelector::default();
        s.edge(s.op().ty::<Mul>(), s.op().ty::<SumReduce>());

        assert_eq!(s.search(&cx).next(), None);
    }
}
