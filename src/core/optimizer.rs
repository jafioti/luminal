use std::sync::{Arc, Mutex};

use petgraph::stable_graph::NodeIndex;

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

// Graph Selector
#[derive(Default)]
pub struct GraphSelector {
    #[allow(clippy::type_complexity)]
    graph: Arc<
        Mutex<
            petgraph::Graph<
                (
                    Box<dyn Operator>,
                    Option<Vec<RealDim>>,
                    Option<*mut NodeIndex>,
                ),
                (),
            >,
        >,
    >,
}

impl GraphSelector {
    pub fn op<O: Operator + 'static>(&self, op: O) -> OpSelector {
        let mut m_self = self.graph.lock().unwrap();
        let id = m_self.add_node((Box::new(op), None, None));
        OpSelector { graph: self, id }
    }

    pub fn edge(&self, o1: OpSelector, o2: OpSelector) -> OpSelector {
        let mut m_self = self.graph.lock().unwrap();
        m_self.add_edge(o1.id, o2.id, ());
        o2
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
        m_graph.node_weight_mut(self.id).unwrap().2 = Some(ptr);
        self
    }
}

#[cfg(test)]
mod tests {
    use petgraph::adj::NodeIndex;

    use crate::op::{Exp2, Expand, Log2, Mul, Permute, SumReduce};

    use super::GraphSelector;

    #[test]
    fn test_graph_selector() {
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
    }
}
