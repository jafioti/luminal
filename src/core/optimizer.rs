use std::{
    any::{Any, TypeId},
    collections::{hash_map::Entry, HashMap, HashSet},
    sync::{Arc, Mutex},
};

use itertools::Itertools;
use petgraph::{
    stable_graph::{NodeIndex, StableGraph},
    visit::EdgeRef,
};

use crate::{
    graph::Graph,
    op::Operator,
    prelude::{Dim, ShapeTracker},
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
                    Option<TypeId>,                                     // Type constraint
                    Option<fn(&dyn Operator, &[ShapeTracker]) -> bool>, // Check constraint
                    Option<Vec<Vec<Dim>>>,                              // Shape constraint
                    Option<Vec<Vec<bool>>>,                             // Fake constraint
                    Vec<*mut NodeIndex>,                                // Pointers
                ),
                u8,
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
            Option<TypeId>,                                     // Type constraint
            Option<fn(&dyn Operator, &[ShapeTracker]) -> bool>, // Check constraint
            Option<Vec<Vec<Dim>>>,                              // Shape constraint
            Option<Vec<Vec<bool>>>,                             // Fake constraint
            Vec<*mut NodeIndex>,                                // Pointers
        ),
        u8,
    >,
    graph_node: petgraph::stable_graph::NodeIndex,
    graph: &StableGraph<Box<dyn Operator>, (u8, u8, crate::core::shape::tracker::ShapeTracker)>,
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
    let input_shapes = graph
        .edges_directed(graph_node, petgraph::Direction::Incoming)
        .sorted_by_key(|e| e.weight().0)
        .map(|e| e.weight().2)
        .collect::<Vec<_>>();

    // Test shape
    if let Some(shape) = &selector_weight.2 {
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
    if let Some(fakes) = &selector_weight.3 {
        for (a_sh, b_sh) in fakes.iter().zip(input_shapes.iter()) {
            for (a, b) in a_sh.iter().zip(b_sh.indexes.iter().map(|i| b_sh.fake[*i])) {
                if *a != b {
                    return false;
                }
            }
        }
    }

    // Run check
    if let Some(check) = &selector_weight.1 {
        if !check(current_weight.as_ref(), &input_shapes) {
            return false;
        }
    }

    // Used is to make sure we don't use the same node from the source graph twice, which prevents cycles
    used.insert(graph_node);
    selector_used.insert(selector_node);
    // Match outgoing
    for (select_outgoing, select_output) in selector_graph
        .edges_directed(selector_node, petgraph::Direction::Outgoing)
        .map(|e| (e.target(), e.weight()))
        .filter(|i| !selector_used.contains(&i.0))
        .collect::<Vec<_>>()
        .into_iter()
    {
        if let Some((target, _)) = graph
            .edges_directed(graph_node, petgraph::Direction::Outgoing)
            .map(|e| (e.target(), e.weight().1))
            .filter(|i| !used.contains(&i.0) && i.1 == *select_output)
            .collect::<Vec<_>>()
            .into_iter()
            .find(|i| {
                search(
                    select_outgoing,
                    selector_graph,
                    i.0,
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
    for (select_incoming, select_output) in selector_graph
        .edges_directed(selector_node, petgraph::Direction::Incoming)
        .map(|e| (e.source(), e.weight()))
        .filter(|i| !selector_used.contains(&i.0))
        .collect::<Vec<_>>()
        .into_iter()
    {
        if let Some((target, _)) = graph
            .edges_directed(graph_node, petgraph::Direction::Outgoing)
            .map(|e| (e.target(), e.weight().1))
            .filter(|i| !used.contains(&i.0) && i.1 == *select_output)
            .collect::<Vec<_>>()
            .into_iter()
            .find(|i| {
                search(
                    select_incoming,
                    selector_graph,
                    i.0,
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
    for ptr in &selector_weight.4 {
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
        let id = m_self.add_node((None, None, None, None, vec![]));
        OpSelector { graph: self, id }
    }

    /// Specify an edge between ops
    pub fn edge(&self, o1: OpSelector, output: u8, o2: OpSelector) -> OpSelector {
        let mut m_self = self.graph.lock().unwrap();
        m_self.add_edge(o1.id, o2.id, output);
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
            0,
            selector1.op().ty::<Exp2>().ptr(&mut exp),
        );
        let selector2 = GraphSelector::default();
        selector2.edge(
            selector2.op().ty::<Exp2>().ptr(&mut exp),
            0,
            selector2.op().ty::<Log2>().ptr(&mut log),
        );

        assert_eq!(
            selector1.search(&cx).chain(selector2.search(&cx)).next(),
            None
        );

        // Matmul
        let s = GraphSelector::default();
        s.edge(s.op().ty::<Mul>(), 0, s.op().ty::<SumReduce>());

        assert_eq!(s.search(&cx).next(), None);
    }
}
