use std::{
    collections::{HashMap, HashSet},
    sync::{Arc, Mutex},
};

use petgraph::{
    stable_graph::{NodeIndex, StableGraph},
    visit::EdgeRef,
};

use crate::{graph::Graph, op::Operator, prelude::RealDim};

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

// Graph Selector
#[derive(Default)]
pub struct GraphSelector {
    #[allow(clippy::type_complexity)]
    graph: Arc<
        Mutex<petgraph::Graph<(Box<dyn Operator>, Option<Vec<RealDim>>, Vec<*mut NodeIndex>), ()>>,
    >,
}

pub struct GraphSearch {
    selector: GraphSelector,
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
        for node in graph.graph.node_indices() {
            if search(
                select_start,
                &self.selector.graph.lock().unwrap(),
                node,
                &graph.graph,
                NodeIndex::default(),
            ) {
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
        (Box<dyn Operator>, Option<Vec<RealDim>>, Vec<*mut NodeIndex>),
        (),
    >,
    graph_node: petgraph::stable_graph::NodeIndex,
    graph: &StableGraph<(Box<dyn Operator>, Vec<RealDim>), u8>,
    coming_from: NodeIndex,
) -> bool {
    let selector_weight = selector_graph.node_weight(selector_node).unwrap();
    let current_weight = graph.node_weight(graph_node).unwrap();
    if current_weight.0.as_any().type_id() != selector_weight.0.as_any().type_id() {
        return false;
    }
    if let Some(shape) = &selector_weight.1 {
        if shape.len() != current_weight.1.len() {
            return false;
        }
        for (a, b) in shape.iter().zip(current_weight.1.iter()) {
            if a != b {
                return false;
            }
        }
    }
    // Match outgoing
    let mut used = HashSet::new();
    for select_outgoing in
        selector_graph.edges_directed(selector_node, petgraph::Direction::Outgoing)
    {
        if select_outgoing.target() == coming_from {
            continue;
        }
        if let Some(target) = graph
            .edges_directed(graph_node, petgraph::Direction::Outgoing)
            .map(|e| e.target())
            .filter(|i| !used.contains(i))
            .find(|i| {
                search(
                    select_outgoing.target(),
                    selector_graph,
                    *i,
                    graph,
                    selector_node,
                )
            })
        {
            used.insert(target);
        } else {
            return false;
        }
    }
    // Match incoming
    let mut used = HashSet::new();
    for select_incoming in
        selector_graph.edges_directed(selector_node, petgraph::Direction::Incoming)
    {
        if select_incoming.source() == coming_from {
            continue;
        }
        if let Some(target) = graph
            .edges_directed(graph_node, petgraph::Direction::Outgoing)
            .map(|e| e.target())
            .filter(|i| !used.contains(i))
            .find(|i| {
                search(
                    select_incoming.source(),
                    selector_graph,
                    *i,
                    graph,
                    selector_node,
                )
            })
        {
            used.insert(target);
        } else {
            return false;
        }
    }
    // All checks out
    for ptr in &selector_weight.2 {
        unsafe {
            **ptr = graph_node;
        }
    }
    true
}

impl GraphSelector {
    pub fn op<O: Operator + 'static>(&self, op: O) -> OpSelector {
        let mut m_self = self.graph.lock().unwrap();
        let id = m_self.add_node((Box::new(op), None, vec![]));
        OpSelector { graph: self, id }
    }

    pub fn edge(&self, o1: OpSelector, o2: OpSelector) -> OpSelector {
        let mut m_self = self.graph.lock().unwrap();
        m_self.add_edge(o1.id, o2.id, ());
        o2
    }

    pub fn search(self, graph: &Graph) -> GraphSearch {
        GraphSearch {
            selector: self,
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
    pub fn shape(self, shape: &[RealDim]) -> Self {
        let graph = unsafe { self.graph.as_ref().unwrap() };
        let mut m_graph = graph.graph.lock().unwrap();
        m_graph.node_weight_mut(self.id).unwrap().1 = Some(shape.to_vec());
        self
    }

    pub fn ptr(self, ptr: *mut NodeIndex) -> Self {
        let graph = unsafe { self.graph.as_ref().unwrap() };
        let mut m_graph = graph.graph.lock().unwrap();
        m_graph.node_weight_mut(self.id).unwrap().2.push(ptr);
        self
    }
}

#[cfg(test)]
mod tests {
    use petgraph::adj::NodeIndex;

    use crate::{
        op::{Exp2, Expand, Log2, Mul, Permute, SumReduce},
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
            selector1.op(Log2).ptr(&mut log),
            selector1.op(Exp2).ptr(&mut exp),
        );
        let selector2 = GraphSelector::default();
        selector2.edge(
            selector2.op(Exp2).ptr(&mut exp),
            selector2.op(Log2).ptr(&mut log),
        );

        assert_eq!(
            selector1.search(&cx).chain(selector2.search(&cx)).next(),
            None
        );

        // Matmul
        let s = GraphSelector::default();
        s.edge(
            s.edge(
                s.op(Expand::default()),
                s.edge(
                    s.edge(s.op(Permute::default()), s.op(Expand::default())),
                    s.op(Mul),
                ),
            ),
            s.op(SumReduce::default()),
        );

        assert_eq!(s.search(&cx).next(), None);
    }
}
