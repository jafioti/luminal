use std::collections::HashMap;

use itertools::Itertools;
use petgraph::{stable_graph::NodeIndex, visit::EdgeRef, Direction};

use crate::{
    op::{Exp2, Function, Log2, Permute},
    prelude::*,
};

/// Generic platform-agnostic optimizations. It's a good idea to use these all the time.
pub type GenericOptimizer = (UnarySequentialOpt, RemoveUnusedNodes, PermuteOptimizer, CSE);

/// Optimizations specific to permutes
pub type PermuteOptimizer = (CombinePermutes, NoOpPermutes);

/// Eliminate complementary unary sequential operations like `x.log().exp()`
#[derive(Default)]
pub struct UnarySequentialOpt;

impl GraphOptimizer for UnarySequentialOpt {
    fn optimize(&self, graph: &mut Graph) {
        // Scan through unary sequential eliminations
        let (mut first, mut last) = (NodeIndex::default(), NodeIndex::default());
        let a = GraphSelector::default();
        a.edge(
            a.op().ty::<Log2>().ptr(&mut first),
            a.op().ty::<Exp2>().ptr(&mut last),
        );
        let b = GraphSelector::default();
        b.edge(
            b.op().ty::<Exp2>().ptr(&mut first),
            b.op().ty::<Log2>().ptr(&mut last),
        );
        for _ in a.search(graph).chain(b.search(graph)) {
            if graph.no_delete.contains(&first) || graph.no_delete.contains(&last) {
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

/// Combine sequential permutes
#[derive(Default)]
pub struct CombinePermutes;

impl GraphOptimizer for CombinePermutes {
    fn optimize(&self, graph: &mut Graph) {
        // Scan through nodes
        let (mut first, mut last) = (NodeIndex::default(), NodeIndex::default());
        let s = GraphSelector::default();
        s.edge(
            s.op().ty::<Permute>().ptr(&mut first),
            s.op().ty::<Permute>().ptr(&mut last),
        );
        for _ in s.search(graph) {
            if graph.no_delete.contains(&first) || graph.no_delete.contains(&last) {
                continue;
            }
            if graph
                .graph
                .edges_directed(first, petgraph::Direction::Outgoing)
                .count()
                != 1
            {
                continue;
            }
            let permute_node = graph
                .get_op(first)
                .unwrap()
                .as_any()
                .downcast_ref::<Permute>()
                .unwrap();
            let other_permute = graph
                .get_op(last)
                .unwrap()
                .as_any()
                .downcast_ref::<Permute>()
                .unwrap();
            // Compute new permute indicies
            graph
                .graph
                .node_weight_mut(first)
                .unwrap()
                .0
                .as_any_mut()
                .downcast_mut::<Permute>()
                .unwrap()
                .0 = other_permute.0.iter().map(|i| permute_node.0[*i]).collect();
            // Remove other node
            move_outgoing_edge(last, first, &mut graph.graph);
            move_references(
                &mut graph.id_remap,
                &mut graph.no_delete,
                &mut graph.to_retrieve,
                last,
                first,
            );
            graph.graph.remove_node(last);
        }
    }
}

/// Get rid of permutes that do nothing, ex: (0, 1, 2, 3)
#[derive(Default)]
pub struct NoOpPermutes;

impl GraphOptimizer for NoOpPermutes {
    fn optimize(&self, graph: &mut Graph) {
        // Scan through nodes
        for id in graph.graph.node_indices().collect_vec() {
            if graph.no_delete.contains(&id) {
                continue;
            }
            if let Some(permute_node) = graph.get_op(id).unwrap().as_any().downcast_ref::<Permute>()
            {
                if permute_node.0.iter().enumerate().all(|(a, b)| a == *b) {
                    // Remove permute
                    if let Some(src) = graph
                        .graph
                        .edges_directed(id, Direction::Incoming)
                        .map(|e| e.source())
                        .next()
                    {
                        move_outgoing_edge(id, src, &mut graph.graph);
                        move_references(
                            &mut graph.id_remap,
                            &mut graph.no_delete,
                            &mut graph.to_retrieve,
                            id,
                            src,
                        );
                        graph.graph.remove_node(id);
                    }
                }
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
                        .0
                        .as_any()
                        .is::<Function>()
                {
                    continue;
                }

                // If order doesn't matter, make sure  different ordered srcs match by sorting
                srcs.sort();

                if let Some(other_node) = srcs_set.get(&srcs) {
                    let a = graph.graph.node_weight(node).unwrap();
                    let b = graph.graph.node_weight(*other_node).unwrap();
                    let a_src_shapes = graph
                        .get_sources(node)
                        .into_iter()
                        .map(|(_, (_, a))| a)
                        .collect_vec();
                    let b_src_shapes = graph
                        .get_sources(*other_node)
                        .into_iter()
                        .map(|(_, (_, a))| a)
                        .collect_vec();
                    if a.0.as_any().type_id() == b.0.as_any().type_id()
                        && a.1 == b.1
                        && a_src_shapes == b_src_shapes
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
        for node in petgraph::algo::toposort(&graph.graph, None)
            .unwrap()
            .into_iter()
            .rev()
        {
            if graph
                .graph
                .edges_directed(node, petgraph::Direction::Outgoing)
                .count()
                == 0
                && !graph.to_retrieve.contains(&node)
            {
                // No dependencies and not marked for retrieval, so remove
                graph.graph.remove_node(node);
            }
        }
    }
}
