use std::collections::HashMap;

use itertools::Itertools;
use petgraph::{visit::EdgeRef, Direction};

use crate::{
    op::{Exp2, Log2, Operator},
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
                let (op, _, _) = graph.graph.node_weight(id).unwrap();
                if (check_op::<Exp2>(op)
                    && check_op::<Log2>(&graph.graph.node_weight(outgoing_target).unwrap().0))
                    || (check_op::<Log2>(op)
                        && check_op::<Exp2>(&graph.graph.node_weight(outgoing_target).unwrap().0))
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
                    if format!("{:?}", graph.graph.node_weight(node).unwrap())
                        == format!("{:?}", graph.graph.node_weight(*other_node).unwrap())
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

#[allow(clippy::borrowed_box)]
fn check_op<T: 'static>(op: &Box<dyn Operator>) -> bool {
    op.as_any().downcast_ref::<T>().is_some()
}
