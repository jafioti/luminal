use std::collections::HashMap;

use itertools::Itertools;
use petgraph::{visit::EdgeRef, Direction};

use crate::{
    op::{Exp2, Log2},
    prelude::*,
};

/// Generic platform-agnostic optimizations. It's a good idea to use these all the time.
pub type GenericOptimizer = (UnarySequentialOpt, CSE, RemoveUnusedNodes);

/// Eliminate complementary unary sequential operations like `x.log().exp()`
#[derive(Default)]
pub struct UnarySequentialOpt;

impl GraphOptimizer for UnarySequentialOpt {
    fn optimize(&self, graph: &mut Graph) {
        // Scan through unary sequential eliminations
        for id in graph.graph.node_indices().collect_vec() {
            if graph.no_delete.contains(&id) {
                continue;
            }
            for outgoing_target in graph
                .graph
                .edges_directed(id, petgraph::Direction::Outgoing)
                .map(|i| i.target())
                .collect_vec()
            {
                let op = graph.get_op(id).unwrap();
                let other = graph.get_op(outgoing_target).unwrap();
                if (op.as_any().is::<Exp2>() && other.as_any().is::<Log2>())
                    || (op.as_any().is::<Log2>() && other.as_any().is::<Exp2>())
                {
                    // Remove current node and next node
                    let pre_node = graph
                        .graph
                        .edges_directed(id, petgraph::Direction::Incoming)
                        .next()
                        .unwrap()
                        .source();

                    for (edge_weight, outgoing_edge_target) in graph
                        .graph
                        .edges_directed(outgoing_target, Direction::Outgoing)
                        .map(|e| (*e.weight(), e.target()))
                        .collect_vec()
                    {
                        graph
                            .graph
                            .add_edge(pre_node, outgoing_edge_target, edge_weight);
                    }

                    Graph::move_references(
                        &mut graph.id_remap,
                        &mut graph.no_delete,
                        &mut graph.to_retrieve,
                        outgoing_target,
                        pre_node,
                    );
                    graph.graph.remove_node(id);
                    graph.graph.remove_node(outgoing_target);
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

                if srcs.is_empty() || graph.graph.node_weight(node).unwrap().0.name() == "Input" {
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
                        for (weight, target) in graph
                            .graph
                            .edges_directed(node, petgraph::Direction::Outgoing)
                            .map(|e| (*e.weight(), e.target()))
                            .collect_vec()
                        {
                            graph.graph.add_edge(*other_node, target, weight);
                        }
                        // Transfer all references to node over to other node
                        Graph::move_references(
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
