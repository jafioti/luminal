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
    op::{Exp2, Function, Log2, MaxReduce, Operator, Recip, SumReduce},
    prelude::*,
};

pub type GenericCompiler<Inner = ((),)> = (PreGenericCompiler, Inner, PostGenericCompiler);

/// Generic platform-agnostic optimizations. It's a good idea to use these all the time.
pub type PostGenericCompiler = (
    UnarySequentialElimination,
    // RemoveUnusedNodes, // Broken right now, unclear why
    CSE,
);

pub type PreGenericCompiler = (RemoveSingleReductions,);

/// Eliminate complementary unary sequential operations like `x.log().exp()`
#[derive(Debug, Default)]
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
                let a = SelectEdge::new(
                    SelectOp::new().type_id(f).ptr(&mut first),
                    SelectOp::new().type_id(l).ptr(&mut last),
                );
                let b = SelectEdge::new(
                    SelectOp::new().type_id(l).ptr(&mut first),
                    SelectOp::new().type_id(f).ptr(&mut last),
                );
                [a, b]
            })
            .collect::<Vec<_>>()
        {
            let mut searcher = selector_graph.search(graph);
            while searcher.next_match() {
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
        let mut eliminated = true;
        while eliminated {
            eliminated = false;
            let mut srcs_set = HashMap::new();
            for node in graph.graph.node_indices().collect_vec() {
                let srcs = graph
                    .graph
                    .edges_directed(node, petgraph::Direction::Incoming)
                    .filter(|e| !e.weight().is_schedule())
                    .sorted_by_key(|e| e.weight().as_data().unwrap().0)
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

                if let Some(other_node) = srcs_set.get(&srcs) {
                    let a = graph.graph.node_weight(node).unwrap();
                    let b = graph.graph.node_weight(*other_node).unwrap();
                    let a_src_shapes = graph
                        .get_sources(node)
                        .into_iter()
                        .map(|(_, _, a)| a)
                        .collect_vec();
                    let b_src_shapes = graph
                        .get_sources(*other_node)
                        .into_iter()
                        .map(|(_, _, a)| a)
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
        }
    }
}

/// Remove maxreduces and sumreduces that don't do anything
#[derive(Default)]
pub struct RemoveSingleReductions;

impl Compiler for RemoveSingleReductions {
    fn compile(&self, graph: &mut Graph) {
        for node in graph.graph.node_indices().collect::<Vec<_>>() {
            let dim = if let Some(red) = graph
                .graph
                .node_weight(node)
                .unwrap()
                .as_any()
                .downcast_ref::<SumReduce>()
            {
                Some(red.0)
            } else {
                graph
                    .graph
                    .node_weight(node)
                    .unwrap()
                    .as_any()
                    .downcast_ref::<MaxReduce>()
                    .map(|red| red.0)
            };
            if let Some(dim) = dim {
                if graph
                    .graph
                    .edges_directed(node, Direction::Incoming)
                    .next()
                    .map(|e| {
                        e.weight()
                            .as_data()
                            .map(|w| {
                                w.2.dims[w.2.indexes[dim]]
                                    .to_usize()
                                    .map(|i| i == 1)
                                    .unwrap_or_default()
                            })
                            .unwrap_or_default()
                    })
                    .unwrap_or_default()
                {
                    let upstream = graph
                        .graph
                        .neighbors_directed(node, Direction::Incoming)
                        .next()
                        .unwrap();
                    move_references(
                        &mut graph.id_remap,
                        &mut graph.no_delete,
                        &mut graph.to_retrieve,
                        node,
                        upstream,
                    );
                    move_outgoing_edge(node, upstream, &mut graph.graph);
                    graph.graph.remove_node(node);
                }
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

/// Mark no_delete as far downstream as possible
#[derive(Debug, Default)]
pub struct RemapDownstream(pub Vec<NodeIndex>);

impl Compiler for RemapDownstream {
    fn compile(&self, graph: &mut Graph) {
        // Loop through state dict tensors marked as no_delete
        for mut node in self
            .0
            .iter()
            .map(|i| remap_id(*i, &graph.id_remap))
            .collect::<Vec<_>>()
        {
            // Go downstream as far as possible along a single stream of ops
            while graph
                .graph
                .edges_directed(node, Direction::Outgoing)
                .count()
                == 1
            {
                let new_node = graph
                    .graph
                    .edges_directed(node, Direction::Outgoing)
                    .next()
                    .unwrap()
                    .target();
                if graph
                    .graph
                    .edges_directed(new_node, Direction::Incoming)
                    .count()
                    > 1
                {
                    break;
                }
                // Remap node to new node
                graph.id_remap.insert(node, new_node);
                node = new_node;
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
        let a = cx.tensor::<R0>();
        let b = a.log_2().exp_2();
        b.retrieve();

        cx.compile(GenericCompiler::<()>::default());
        assert_eq!(cx.graph.node_count(), 1);
    }
}
