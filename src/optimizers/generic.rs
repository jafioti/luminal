use std::{any::TypeId, collections::HashMap};

use itertools::Itertools;
use petgraph::{stable_graph::NodeIndex, visit::EdgeRef, Direction};

use crate::{
    op::{Exp2, Function, Log2, Recip},
    prelude::*,
};

/// Generic platform-agnostic optimizations. It's a good idea to use these all the time.
pub type GenericOptimizer = (UnarySequentialOpt, RemoveUnusedNodes, CSE);

/// Eliminate complementary unary sequential operations like `x.log().exp()`
#[derive(Default)]
pub struct UnarySequentialOpt;

impl GraphOptimizer for UnarySequentialOpt {
    fn optimize(&self, graph: &mut Graph) {
        // Here are all the complementary ops that should be removed
        let sequences = [
            (TypeId::of::<Log2>(), TypeId::of::<Exp2>()),
            (TypeId::of::<Recip>(), TypeId::of::<Recip>()),
        ];
        let (mut first, mut last) = (NodeIndex::default(), NodeIndex::default());
        for selector_graph in sequences
            .into_iter()
            .flat_map(|(f, l)| {
                // Construct two searches: in order and reversed
                let a = GraphSelector::default();
                a.edge(
                    a.op().type_id(f).ptr(&mut first),
                    a.op().type_id(l).ptr(&mut last),
                );
                let b = GraphSelector::default();
                b.edge(
                    b.op().type_id(l).ptr(&mut first),
                    b.op().type_id(f).ptr(&mut last),
                );
                [a, b]
            })
            .collect::<Vec<_>>()
        {
            for _ in selector_graph.search(graph) {
                if graph.no_delete.contains(&first)
                    || graph
                        .graph
                        .edges_directed(first, Direction::Outgoing)
                        .count()
                        != 1
                {
                    // Either first is marked as no_delete or there are other nodes depending on first
                    continue;
                }
                // Remove current node and next node
                let pre_node = graph
                    .graph
                    .edges_directed(first, petgraph::Direction::Incoming)
                    .next()
                    .unwrap()
                    .source();

                move_outgoing_edge(last, pre_node, &mut graph.graph);
                move_references(
                    &mut graph.id_remap,
                    &mut graph.no_delete,
                    &mut graph.to_retrieve,
                    last,
                    pre_node,
                );
                graph.graph.remove_node(first);
                graph.graph.remove_node(last);
            }
        }
    }
}

/// [Common subexpression elimination](https://en.wikipedia.org/wiki/Common_subexpression_elimination)
#[derive(Default)]
pub struct CSE;

impl GraphOptimizer for CSE {
    fn optimize(&self, graph: &mut Graph) {
        // Look for nodes that have the exact same srcs
        // Loop cause I'm lazy
        loop {
            let mut eliminated = false;
            let mut srcs_set = HashMap::new();
            for node in graph.graph.node_indices().collect_vec() {
                let mut srcs = graph
                    .graph
                    .edges_directed(node, petgraph::Direction::Incoming)
                    .map(|e| e.source())
                    .collect_vec();

                if srcs.is_empty()
                    || graph
                        .graph
                        .node_weight(node)
                        .unwrap()
                        .as_any()
                        .is::<Function>()
                {
                    continue;
                }

                // If order doesn't matter, make sure different ordered srcs match by sorting
                srcs.sort();

                if let Some(other_node) = srcs_set.get(&srcs) {
                    let a = graph.graph.node_weight(node).unwrap();
                    let b = graph.graph.node_weight(*other_node).unwrap();
                    let a_src_shapes = graph
                        .get_sources(node)
                        .into_iter()
                        .map(|(_, a)| a)
                        .collect_vec();
                    let b_src_shapes = graph
                        .get_sources(*other_node)
                        .into_iter()
                        .map(|(_, a)| a)
                        .collect_vec();
                    if a.as_any().type_id() == b.as_any().type_id() && a_src_shapes == b_src_shapes
                    // If the op, input shapes, and output shape is the same, we can combine them (UNCLEAR IF THIS IS TRUE, NEED PROPER PartialEq)
                    {
                        // Carry over outgoing edges from node to other_node
                        move_outgoing_edge(node, *other_node, &mut graph.graph);
                        // Transfer all references to node over to other node
                        move_references(
                            &mut graph.id_remap,
                            &mut graph.no_delete,
                            &mut graph.to_retrieve,
                            node,
                            *other_node,
                        );
                        // Remove node
                        graph.graph.remove_node(node);
                        eliminated = true;
                        break;
                    }
                }
                srcs_set.insert(srcs, node);
            }

            if !eliminated {
                break;
            }
        }
    }
}

/// Remove unused nodes
#[derive(Default)]
pub struct RemoveUnusedNodes;

impl GraphOptimizer for RemoveUnusedNodes {
    fn optimize(&self, graph: &mut Graph) {
        // Reverse topo sort
        for node in graph.graph.node_indices().collect::<Vec<_>>() {
            if graph
                .graph
                .edges_directed(node, petgraph::Direction::Outgoing)
                .count()
                == 0
                && !graph.no_delete.contains(&node)
            {
                // No dependencies and not marked for retrieval, so remove
                graph.graph.remove_node(node);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::prelude::*;
    #[test]
    fn test_log_exp() {
        let mut cx = Graph::new();
        let a = cx.new_tensor::<R0>("I");
        let b = a.log_2().exp_2();
        b.mark();

        cx.optimize(GenericOptimizer::default());
        assert_eq!(cx.graph.node_count(), 1);
    }
}
