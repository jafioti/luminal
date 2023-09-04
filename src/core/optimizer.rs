use std::{
    any::{Any, TypeId},
    collections::{HashMap, HashSet},
    sync::{Arc, Mutex},
};

use petgraph::{
    stable_graph::{NodeIndex, StableGraph},
    visit::EdgeRef,
};

use crate::{graph::Graph, op::Operator, prelude::Dim};

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
    // Create remap
    id_remap.insert(src, trg);
    // Transfer no_delete
    if no_delete.remove(&src) {
        no_delete.insert(trg);
    }
    // Transfer to_retrieve
    if to_retrieve.remove(&src) {
        to_retrieve.insert(trg);
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

// Graph Selector
#[derive(Default)]
pub struct GraphSelector {
    #[allow(clippy::type_complexity)]
    graph: Arc<
        Mutex<
            petgraph::Graph<
                (
                    Option<TypeId>,
                    Option<Box<dyn TraitObjEq>>,
                    Option<Vec<Dim>>,
                    Vec<*mut NodeIndex>,
                    Option<usize>,
                ),
                (),
            >,
        >,
    >,
}

pub struct GraphSearch {
    selector: GraphSelector,
    already_returned: HashSet<NodeIndex>,
    graph: *const Graph,
}

impl Iterator for GraphSearch {
    type Item = ();

    fn next(&mut self) -> Option<Self::Item> {
        // Look through graph for pattern from selector
        let Some(select_start) = self.selector.graph.lock().unwrap().node_indices().next() else {
            return None;
        };
        let graph = unsafe { self.graph.as_ref().unwrap() };
        let mut used = HashSet::new();
        let mut selector_used = HashSet::new();
        for node in graph.graph.node_indices() {
            if self.already_returned.contains(&node) {
                continue;
            }
            if search(
                select_start,
                &self.selector.graph.lock().unwrap(),
                node,
                &graph.graph,
                &mut used,
                &mut selector_used,
            ) {
                self.already_returned.insert(node);
                return Some(());
            }
        }
        None
    }
}

#[allow(clippy::type_complexity)]
fn search(
    selector_node: NodeIndex,
    selector_graph: &petgraph::Graph<
        (
            Option<TypeId>,
            Option<Box<dyn TraitObjEq>>,
            Option<Vec<Dim>>,
            Vec<*mut NodeIndex>,
            Option<usize>,
        ),
        (),
    >,
    graph_node: petgraph::stable_graph::NodeIndex,
    graph: &StableGraph<Box<dyn Operator>, (u8, crate::core::shape::simple_tracker::ShapeTracker)>,
    used: &mut HashSet<NodeIndex>,
    selector_used: &mut HashSet<NodeIndex>,
) -> bool {
    let selector_weight = selector_graph.node_weight(selector_node).unwrap();
    let current_weight = graph.node_weight(graph_node).unwrap();
    // Test type
    if let Some(ty) = &selector_weight.0 {
        if current_weight.as_any().type_id() != *ty {
            return false;
        }
    }
    // Test value
    if let Some(value) = &selector_weight.1 {
        if !current_weight.is_equal(value.as_ref()) {
            return false;
        }
    }
    // Test shape
    if let Some(shape) = &selector_weight.2 {
        // if shape.len() != current_weight.1.len() {
        //     return false;
        // }
        // for (a, b) in shape.iter().zip(current_weight.1.iter()) {
        //     if a != b {
        //         return false;
        //     }
        // }
    }
    // Used is to make sure we don't use the same node from the source graph twice, which prevents cycles
    used.insert(graph_node);
    selector_used.insert(selector_node);
    // Match outgoing
    for select_outgoing in selector_graph
        .edges_directed(selector_node, petgraph::Direction::Outgoing)
        .map(|e| e.target())
        .filter(|i| !selector_used.contains(i))
        .collect::<Vec<_>>()
        .into_iter()
    {
        if let Some(target) = graph
            .edges_directed(graph_node, petgraph::Direction::Outgoing)
            .map(|e| e.target())
            .filter(|i| !used.contains(i))
            .collect::<Vec<_>>()
            .into_iter()
            .find(|i| {
                search(
                    select_outgoing,
                    selector_graph,
                    *i,
                    graph,
                    used,
                    selector_used,
                )
            })
        {
            used.insert(target);
        } else {
            used.remove(&graph_node);
            selector_used.remove(&selector_node);
            return false;
        }
    }
    // Match incoming
    for select_incoming in selector_graph
        .edges_directed(selector_node, petgraph::Direction::Incoming)
        .map(|e| e.source())
        .filter(|i| !selector_used.contains(i))
        .collect::<Vec<_>>()
        .into_iter()
    {
        if let Some(target) = graph
            .edges_directed(graph_node, petgraph::Direction::Outgoing)
            .map(|e| e.target())
            .filter(|i| !used.contains(i))
            .collect::<Vec<_>>()
            .into_iter()
            .find(|i| {
                search(
                    select_incoming,
                    selector_graph,
                    *i,
                    graph,
                    used,
                    selector_used,
                )
            })
        {
            used.insert(target);
        } else {
            used.remove(&graph_node);
            selector_used.remove(&selector_node);
            return false;
        }
    }
    // All checks out
    for ptr in &selector_weight.3 {
        unsafe {
            **ptr = graph_node;
        }
    }
    true
}

impl GraphSelector {
    /// Create a new op to select
    pub fn op(&self) -> OpSelector {
        let mut m_self = self.graph.lock().unwrap();
        let id = m_self.add_node((None, None, None, vec![], None));
        OpSelector { graph: self, id }
    }

    /// Specify an edge between ops
    pub fn edge(&self, o1: OpSelector, o2: OpSelector) -> OpSelector {
        let mut m_self = self.graph.lock().unwrap();
        m_self.add_edge(o1.id, o2.id, ());
        o2
    }

    /// Start searching a graph
    pub fn search(self, graph: &Graph) -> GraphSearch {
        GraphSearch {
            selector: self,
            already_returned: HashSet::new(),
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
    /// Constrain the op to a value
    pub fn value<O: Operator + 'static>(self, value: O) -> Self {
        let graph = unsafe { self.graph.as_ref().unwrap() };
        let mut m_graph = graph.graph.lock().unwrap();
        m_graph.node_weight_mut(self.id).unwrap().0 = Some(TypeId::of::<O>());
        m_graph.node_weight_mut(self.id).unwrap().1 = Some(Box::new(value));
        self
    }
    /// Constrain the op to an output shape
    pub fn shape(self, shape: &[Dim]) -> Self {
        let graph = unsafe { self.graph.as_ref().unwrap() };
        let mut m_graph = graph.graph.lock().unwrap();
        m_graph.node_weight_mut(self.id).unwrap().2 = Some(shape.to_vec());
        self
    }
    /// Constrain the op to an output shape of a dimension
    pub fn dim(self, dim: usize) -> Self {
        let graph = unsafe { self.graph.as_ref().unwrap() };
        let mut m_graph = graph.graph.lock().unwrap();
        m_graph.node_weight_mut(self.id).unwrap().4 = Some(dim);
        self
    }
    /// Register a pointer to set if the op is matched
    pub fn ptr(self, ptr: *mut NodeIndex) -> Self {
        let graph = unsafe { self.graph.as_ref().unwrap() };
        let mut m_graph = graph.graph.lock().unwrap();
        m_graph.node_weight_mut(self.id).unwrap().3.push(ptr);
        self
    }
}

// #[cfg(test)]
// mod tests {
//     use petgraph::adj::NodeIndex;

//     use crate::{
//         op::{Exp2, Log2, Mul, SumReduce},
//         prelude::Graph,
//     };

//     use super::GraphSelector;

//     #[test]
//     fn test_graph_selector() {
//         let cx = Graph::default();
//         // Exp -> Log or Log -> Exp
//         let (mut exp, mut log) = (NodeIndex::default(), NodeIndex::default());
//         let selector1 = GraphSelector::default();
//         selector1.edge(
//             selector1.op().ty::<Log2>().ptr(&mut log),
//             selector1.op().ty::<Exp2>().ptr(&mut exp),
//         );
//         let selector2 = GraphSelector::default();
//         selector2.edge(
//             selector2.op().ty::<Exp2>().ptr(&mut exp),
//             selector2.op().ty::<Log2>().ptr(&mut log),
//         );

//         assert_eq!(
//             selector1.search(&cx).chain(selector2.search(&cx)).next(),
//             None
//         );

//         // Matmul
//         let s = GraphSelector::default();
//         s.edge(
//             s.edge(
//                 s.op().ty::<Expand>(),
//                 s.edge(
//                     s.edge(s.op().ty::<Permute>(), s.op().ty::<Expand>()),
//                     s.op().ty::<Mul>(),
//                 ),
//             ),
//             s.op().ty::<SumReduce>(),
//         );

//         assert_eq!(s.search(&cx).next(), None);
//     }
// }
