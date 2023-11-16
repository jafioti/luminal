use std::{
    any::TypeId,
    collections::{HashMap, HashSet},
};

use itertools::Itertools;
use petgraph::{
    stable_graph::{NodeIndex, StableGraph},
    visit::EdgeRef,
    Direction,
};

use crate::{
    op::{Exp2, Function, Log2, Operator, Recip},
    prelude::*,
};

/// Generic platform-agnostic optimizations. It's a good idea to use these all the time.
pub type GenericCompiler = (
    UnarySequentialElimination,
    // RemoveUnusedNodes, // Broken right now, unclear why
    CSE,
);

/// Eliminate complementary unary sequential operations like `x.log().exp()`
#[derive(Default)]
pub struct UnarySequentialElimination;

impl Compiler for UnarySequentialElimination {
    fn compile(&self, graph: &mut Graph) {
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

impl Compiler for CSE {
    fn compile(&self, graph: &mut Graph) {
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

impl Compiler for RemoveUnusedNodes {
    fn compile(&self, graph: &mut Graph) {
        // Reverse topo sort
        for node in graph.graph.node_indices().collect::<Vec<_>>() {
            if graph
                .graph
                .edges_directed(node, petgraph::Direction::Outgoing)
                .count()
                == 0
                && !graph.no_delete.contains(&node)
            {
                // No dependencies and not marked for no_delete, so remove
                graph.graph.remove_node(node);
            }
        }
    }
}

#[derive(Default)]
pub struct DepthFirst;

impl Compiler for DepthFirst {
    fn compile(&self, graph: &mut Graph) {
        fn toposort(
            id: NodeIndex,
            graph: &StableGraph<Box<dyn Operator>, Dependency>,
            visited: &mut HashSet<NodeIndex>,
        ) -> (Vec<NodeIndex>, usize, bool) {
            if visited.contains(&id) {
                return (vec![], 0, false);
            }
            // Loop through node sources
            let stacks = graph
                .edges_directed(id, Direction::Incoming)
                .sorted_by_key(|e| e.source())
                .map(|e| toposort(e.source(), graph, visited))
                .collect::<Vec<_>>();
            let num_stacks = stacks.len();

            let mut final_stack = vec![];
            let mut complete = true;
            for (mut stack, _, c) in stacks.into_iter().sorted_by_key(|(_, _, b)| !*b) {
                final_stack.append(&mut stack);
                complete &= c;
            }
            final_stack.push(id);
            visited.insert(id);

            (final_stack, num_stacks, complete)
        }

        // Depth-first toposort
        let mut visited = HashSet::default();
        let mut pre_sorted = petgraph::algo::toposort(&graph.graph, None).unwrap();
        pre_sorted.reverse();
        let mut stacks = vec![];
        for node in pre_sorted {
            if !visited.contains(&node) {
                stacks.push(toposort(node, &graph.graph, &mut visited));
            }
        }
        let mut nodes = vec![];
        for (mut stack, _, _) in stacks.into_iter().sorted_by_key(|(_, _, b)| !*b) {
            nodes.append(&mut stack);
        }

        // Insert schedule deps
        for i in 0..nodes.len() - 1 {
            graph.add_schedule_dependency(nodes[i], nodes[i + 1]);
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
        b.retrieve();

        cx.compile(GenericCompiler::default());
        assert_eq!(cx.graph.node_count(), 1);
    }
}
