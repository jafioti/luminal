use std::collections::HashMap;

use itertools::Itertools;
use petgraph::{visit::EdgeRef, Direction};

use crate::prelude::*;

// Platform agnostic optimizations

pub type GeneralOpt = (UnarySequentialOpt, CSE);

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
                let (op, _) = graph.graph.node_weight(id).unwrap();
                if (op.name() == "Exp2"
                    && graph.graph.node_weight(outgoing_target).unwrap().0.name() == "Log2")
                    || (op.name() == "Log2"
                        && graph.graph.node_weight(outgoing_target).unwrap().0.name() == "Exp2")
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

/// Common subexpression elimination (https://en.wikipedia.org/wiki/Common_subexpression_elimination)
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
                    if graph.graph.node_weight(node).unwrap().0.name()
                        == graph.graph.node_weight(*other_node).unwrap().0.name()
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
