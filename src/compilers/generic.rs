use std::collections::{HashMap, HashSet};

use itertools::Itertools;
use petgraph::{
    algo::toposort,
    stable_graph::{NodeIndex, StableGraph},
    visit::EdgeRef,
    Direction,
};

use crate::{
    op::{Add, Function, MaxReduce, Mul, Operator, Recip, SumReduce},
    prelude::*,
};

/// Generic platform-agnostic optimizations. It's a good idea to use these all the time.
pub type GenericCompiler = (
    //RemoveSingleReductions,
    RemoveUnusedNodes,
    ArithmeticElimination,
    CSE,
);

/// [Common subexpression elimination](https://en.wikipedia.org/wiki/Common_subexpression_elimination)
#[derive(Default)]
pub struct CSE;

impl Compiler for CSE {
    type Output = ();
    fn compile<T: ToIdsMut>(&self, graph: &mut Graph, mut ids: T) {
        // Look for nodes that have the exact same srcs
        // Loop cause I'm lazy
        let mut eliminated = true;
        while eliminated {
            eliminated = false;
            let mut srcs_set: HashMap<Vec<NodeIndex>, Vec<NodeIndex>> = HashMap::new();
            for node in graph.graph.node_indices().collect_vec() {
                if graph
                    .graph
                    .node_weight(node)
                    .unwrap()
                    .as_any()
                    .is::<Function>()
                {
                    continue;
                }
                let srcs = graph
                    .graph
                    .edges_directed(node, petgraph::Direction::Incoming)
                    .filter(|e| !e.weight().is_schedule())
                    .sorted_by_key(|e| e.weight().as_data().unwrap().0)
                    .map(|e| e.source())
                    .collect_vec();

                if let Some(other_nodes) = srcs_set.get(&srcs) {
                    for other_node in other_nodes {
                        let a = graph.graph.node_weight(node).unwrap();
                        let Some(b) = graph.graph.node_weight(*other_node) else {
                            continue;
                        };
                        if !a.is_equal(b.as_ref()) {
                            continue;
                        }
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
                        if a_src_shapes != b_src_shapes {
                            continue;
                        }
                        // If the op, input shapes, and output shape is the same, we can combine them (UNCLEAR IF THIS IS TRUE, NEED PROPER PartialEq)
                        // Carry over outgoing edges from node to other_node
                        move_outgoing_edge(node, *other_node, &mut graph.graph);
                        // Transfer all references to node over to other node
                        remap(node, *other_node, &mut ids, graph);
                        // Remove node
                        graph.graph.remove_node(node);
                        eliminated = true;
                        break;
                    }
                    if eliminated {
                        break;
                    }
                }
                if let Some(nodes) = srcs_set.get_mut(&srcs) {
                    nodes.push(node);
                } else {
                    srcs_set.insert(srcs, vec![node]);
                }
            }
            srcs_set.clear();
        }
    }
}

/// Remove maxreduces and sumreduces that don't do anything
#[derive(Default)]
pub struct RemoveSingleReductions;

impl Compiler for RemoveSingleReductions {
    type Output = ();
    fn compile<T: ToIdsMut>(&self, graph: &mut Graph, mut ids: T) {
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
                    remap(node, upstream, &mut ids, graph);
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
    type Output = ();
    fn compile<T: ToIdsMut>(&self, graph: &mut Graph, _: T) {
        // Reverse topo sort
        for node in toposort(&graph.graph, None).unwrap().into_iter().rev() {
            if graph.edges_directed(node, Direction::Outgoing).count() == 0
                && !graph.no_delete.contains(&node)
            {
                // No dependencies and not marked for no_delete, so remove
                graph.remove_node(node);
            }
        }
    }
}

#[derive(Default)]
pub struct DepthFirst;

impl Compiler for DepthFirst {
    type Output = ();
    fn compile<T: ToIdsMut>(&self, graph: &mut Graph, _: T) {
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
    type Output = ();
    fn compile<T: ToIdsMut>(&self, graph: &mut Graph, mut remap: T) {
        let set = self.0.iter().copied().collect::<HashSet<_>>();
        // Loop through state dict tensors marked as no_delete
        for mut node in self.0.iter().copied() {
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
                if !is_from_set(new_node, graph, &set) {
                    break;
                }
                // Remap node to new node
                for id in remap.to_ids_mut() {
                    if *id == node {
                        *id = new_node;
                    }
                }
                node = new_node;
            }
        }
    }
}

fn is_from_set(node: NodeIndex, graph: &Graph, set: &HashSet<NodeIndex>) -> bool {
    // Reverse dfs upward
    let mut stack = vec![node];
    while let Some(node) = stack.pop() {
        if !set.contains(&node) {
            let mut new_nodes = graph
                .graph
                .edges_directed(node, Direction::Incoming)
                .filter(|e| !e.weight().is_schedule())
                .map(|e| e.source())
                .collect_vec();
            if new_nodes.is_empty() {
                // Node isn't from set and doesn't have upstream nodes
                return false;
            }
            stack.append(&mut new_nodes);
        }
    }
    true
}

/// **Reduces arithmetic expressions**
///
/// - Current: x + 0 => x, x * 1 => x
/// - TODO: x / x => 1, x - x => 0, x * 0 => 0, x - 0 => x, x * 0 => 0, 0 / x => 0
#[derive(Debug, Default)]
pub struct ArithmeticElimination;

impl Compiler for ArithmeticElimination {
    type Output = ();
    fn compile<T: ToIdsMut>(&self, graph: &mut Graph, mut ids: T) {
        // x + 0, 0 + x
        let zero = constant(0.);
        let inp = node();
        let add1 = binary::<Add>(zero.clone(), inp.clone());
        let add2 = binary::<Add>(inp.clone(), zero.clone());
        let mut s1 = add1.clone().search(graph);
        let mut s2 = add2.clone().search(graph);
        while s1.next_match() || s2.next_match() {
            let (inp, zero, add) = if s1.matched {
                (s1.get(&inp), s1.get(&zero), s1.get(&add1))
            } else {
                (s2.get(&inp), s2.get(&zero), s2.get(&add2))
            };
            if graph.no_delete.contains(&zero) {
                continue;
            }
            // Carry over outgoing edges
            let input_shape = graph
                .graph
                .edges_connecting(inp, add)
                .find_map(|e| e.weight().as_data())
                .unwrap()
                .2;
            if input_shape.is_reshaped() {
                // If any output shape is non-contiguous, we need to keep the op for it's contiguous functionality TODO: replace with explicit contiguous op here
                if graph
                    .graph
                    .edges_connecting(inp, add)
                    .filter_map(|e| e.weight().as_data())
                    .any(|(_, _, sh)| sh.is_reshaped())
                {
                    continue;
                }
                for (weight, target) in graph
                    .graph
                    .edges_directed(add, petgraph::Direction::Outgoing)
                    .map(|e| (*e.weight(), e.target()))
                    .collect::<Vec<_>>()
                {
                    if let Some(weight) = weight.as_data() {
                        graph.graph.add_edge(
                            inp,
                            target,
                            Dependency::Data {
                                input_order: weight.0,
                                output_order: weight.1,
                                shape: input_shape,
                            },
                        );
                    }
                }
            } else {
                move_outgoing_edge(add, inp, &mut graph.graph);
            }
            remap(add, inp, &mut ids, graph);
            if graph
                .graph
                .edges_directed(zero, Direction::Outgoing)
                .count()
                == 1
            {
                graph.graph.remove_node(zero);
            }
            graph.graph.remove_node(add);
        }
        // x * 1, 1 * x
        let one = constant(1.);
        let inp = node();
        let mul1 = binary::<Mul>(one.clone(), inp.clone());
        let mul2 = binary::<Mul>(inp.clone(), one.clone());
        let mut s1 = mul1.clone().search(graph);
        let mut s2 = mul2.clone().search(graph);
        while s1.next_match() || s2.next_match() {
            let (inp, one, mul) = if s1.matched {
                (s1.get(&inp), s1.get(&one), s1.get(&mul1))
            } else {
                (s2.get(&inp), s2.get(&one), s2.get(&mul2))
            };
            if graph.no_delete.contains(&one) {
                continue;
            }
            // Carry over outgoing edges
            let input_shape = graph
                .graph
                .edges_connecting(inp, mul)
                .find_map(|e| e.weight().as_data())
                .unwrap()
                .2;
            if input_shape.is_reshaped() {
                // If any output shape is non-contiguous, we need to keep the op for it's contiguous functionality TODO: replace with explicit contiguous op here
                if graph
                    .graph
                    .edges_connecting(inp, mul)
                    .filter_map(|e| e.weight().as_data())
                    .any(|(_, _, sh)| sh.is_reshaped())
                {
                    continue;
                }
                for (weight, target) in graph
                    .graph
                    .edges_directed(mul, petgraph::Direction::Outgoing)
                    .map(|e| (*e.weight(), e.target()))
                    .collect::<Vec<_>>()
                {
                    if let Some(weight) = weight.as_data() {
                        graph.graph.add_edge(
                            inp,
                            target,
                            Dependency::Data {
                                input_order: weight.0,
                                output_order: weight.1,
                                shape: input_shape,
                            },
                        );
                    }
                }
            } else {
                move_outgoing_edge(mul, inp, &mut graph.graph);
            }
            remap(mul, inp, &mut ids, graph);
            graph.safe_remove_node(one, 1);
            graph.graph.remove_node(mul);
        }
        // recip(recip(x))
        let inp = node();
        let intermediate = unary::<Recip>(inp.clone());
        let out = unary::<Recip>(intermediate.clone());
        let mut s = out.clone().search(graph);
        while s.next_match() {
            let (inp, intermediate, out) = (s.get(&inp), s.get(&intermediate), s.get(&out));
            if graph.no_delete.contains(&intermediate) {
                continue;
            }
            // Carry over outgoing edges
            let input_shape = graph
                .graph
                .edges_connecting(inp, intermediate)
                .find_map(|e| e.weight().as_data())
                .unwrap()
                .2;
            if input_shape.is_reshaped() {
                // If any output shape is non-contiguous, we need to keep the op for it's contiguous functionality TODO: replace with explicit contiguous op here
                if graph
                    .graph
                    .edges_connecting(inp, intermediate)
                    .filter_map(|e| e.weight().as_data())
                    .any(|(_, _, sh)| sh.is_reshaped())
                    || graph
                        .graph
                        .edges_connecting(intermediate, out)
                        .filter_map(|e| e.weight().as_data())
                        .any(|(_, _, sh)| sh.is_reshaped())
                {
                    continue;
                }
                for (weight, target) in graph
                    .graph
                    .edges_directed(intermediate, petgraph::Direction::Outgoing)
                    .map(|e| (*e.weight(), e.target()))
                    .collect::<Vec<_>>()
                {
                    if let Some(weight) = weight.as_data() {
                        graph.graph.add_edge(
                            inp,
                            target,
                            Dependency::Data {
                                input_order: weight.0,
                                output_order: weight.1,
                                shape: input_shape,
                            },
                        );
                    }
                }
            } else {
                move_outgoing_edge(out, inp, &mut graph.graph);
            }
            remap(intermediate, inp, &mut ids, graph);
            remap(out, inp, &mut ids, graph);
            graph.remove_node(out);
            graph.safe_remove_node(intermediate, 0);
        }
    }
}
