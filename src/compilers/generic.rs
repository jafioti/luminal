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
    op::{
        Add, Constant, ConstantValue, Exp2, Function, Log2, MaxReduce, Mul, Operator, Recip,
        SumReduce,
    },
    prelude::*,
};

/// Generic platform-agnostic optimizations. It's a good idea to use these all the time.
pub type GenericCompiler<Inner = ()> = (PreGenericCompiler, Inner, PostGenericCompiler);

pub type PreGenericCompiler = (
    RemoveSingleReductions,
    ArithmeticElimination,
    UnarySequentialElimination,
);

pub type PostGenericCompiler = (
    // RemoveUnusedNodes, // Broken right now, unclear why
    // CSE,
    (),
);

/// Eliminate complementary unary sequential operations like `x.log().exp()`
#[derive(Debug, Default)]
pub struct UnarySequentialElimination;

impl Compiler for UnarySequentialElimination {
    fn compile<T: ToIdsMut>(&self, graph: &mut Graph, mut remap: T) {
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
                let (f_sel, l_sel) = (
                    SelectOp::new().type_id(f).ptr(&mut first),
                    SelectOp::new().type_id(l).ptr(&mut last),
                );
                [f_sel.clone().edge(l_sel.clone()), l_sel.edge(f_sel)]
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
                    &mut remap,
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
    fn compile<T: ToIdsMut>(&self, graph: &mut Graph, mut remap: T) {
        // Look for nodes that have the exact same srcs
        // Loop cause I'm lazy
        let mut eliminated = true;
        while eliminated {
            eliminated = false;
            let mut srcs_set: HashMap<Vec<NodeIndex>, Vec<NodeIndex>> = HashMap::new();
            for node in graph.graph.node_indices().collect_vec() {
                let srcs = graph
                    .graph
                    .edges_directed(node, petgraph::Direction::Incoming)
                    .filter(|e| !e.weight().is_schedule())
                    .sorted_by_key(|e| e.weight().as_data().unwrap().0)
                    .map(|e| e.source())
                    .collect_vec();

                if graph
                    .graph
                    .node_weight(node)
                    .unwrap()
                    .as_any()
                    .is::<Function>()
                {
                    continue;
                }

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
                        move_references(
                            &mut remap,
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
        }
    }
}

/// Remove maxreduces and sumreduces that don't do anything
#[derive(Default)]
pub struct RemoveSingleReductions;

impl Compiler for RemoveSingleReductions {
    fn compile<T: ToIdsMut>(&self, graph: &mut Graph, mut remap: T) {
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
                        &mut remap,
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
    fn compile<T: ToIdsMut>(&self, graph: &mut Graph, _: T) {
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

#[cfg(test)]
mod tests {
    use crate::prelude::*;
    #[test]
    fn test_log_exp() {
        let mut cx = Graph::new();
        let a = cx.tensor::<R0>();
        let _ = a.log2().exp2().retrieve();

        cx.compile(GenericCompiler::<()>::default(), ());
        assert_eq!(cx.graph.node_count(), 1);
    }
}

/// **Reduces arithmetic expressions**
///
/// - Current: x + 0 => x, x * 1 => x
/// - TODO: x / x => 1, x - x => 0, x * 0 => 0, x - 0 => x, x * 0 => 0, 0 / x => 0
#[derive(Debug, Default)]
pub struct ArithmeticElimination;

impl Compiler for ArithmeticElimination {
    fn compile<T: ToIdsMut>(&self, graph: &mut Graph, mut remap: T) {
        // x + 0, 0 + x
        let (mut x, mut add, mut zero) = (
            NodeIndex::default(),
            NodeIndex::default(),
            NodeIndex::default(),
        );
        let zero_pat = SelectOp::new()
            .check(|o, _| {
                if let Some(o) = o.as_any().downcast_ref::<Constant>() {
                    if let ConstantValue::Float(c) = o.0 {
                        c == 0.0
                    } else {
                        false
                    }
                } else {
                    false
                }
            })
            .ptr(&mut zero);
        let mut selector1 = SelectOp::new()
            .ptr(&mut x)
            .edge(
                zero_pat
                    .clone()
                    .edge(SelectOp::new().ty::<Add>().ptr(&mut add)),
            )
            .search(graph);
        let mut selector2 = zero_pat
            .edge(
                SelectOp::new()
                    .ptr(&mut x)
                    .edge(SelectOp::new().ty::<Add>().ptr(&mut add)),
            )
            .search(graph);
        while selector1.next_match() || selector2.next_match() {
            if graph.no_delete.contains(&zero) {
                continue;
            }
            // Carry over outgoing edges
            let input_shape = graph
                .graph
                .edges_connecting(x, add)
                .find_map(|e| e.weight().as_data())
                .unwrap()
                .2;
            if !input_shape.is_contiguous() || input_shape.is_padded() || input_shape.is_sliced() {
                // If any output shape is non-contiguous, we need to keep the op for it's contiguous functionality TODO: replace with explicit contiguous op here
                if graph
                    .graph
                    .edges_directed(add, Direction::Outgoing)
                    .filter_map(|e| e.weight().as_data())
                    .any(|(_, _, sh)| !sh.is_contiguous() || sh.is_padded() || sh.is_sliced())
                {
                    continue;
                }
                for (weight, target) in graph
                    .graph
                    .edges_directed(add, petgraph::Direction::Outgoing)
                    .map(|e| (e.weight().clone(), e.target()))
                    .collect::<Vec<_>>()
                {
                    if let Some(weight) = weight.as_data() {
                        graph.graph.add_edge(
                            x,
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
                move_outgoing_edge(add, x, &mut graph.graph);
            }
            move_references(
                &mut remap,
                &mut graph.no_delete,
                &mut graph.to_retrieve,
                add,
                x,
            );
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
        let (mut a, mut mul, mut one) = (
            NodeIndex::default(),
            NodeIndex::default(),
            NodeIndex::default(),
        );
        let one_pat = SelectOp::new()
            .check(|o, _| {
                if let Some(o) = o.as_any().downcast_ref::<Constant>() {
                    if let ConstantValue::Float(c) = o.0 {
                        c == 1.0
                    } else {
                        false
                    }
                } else {
                    false
                }
            })
            .ptr(&mut one);
        let mut selector1 = SelectOp::new()
            .ptr(&mut a)
            .edge(
                one_pat
                    .clone()
                    .edge(SelectOp::new().ty::<Mul>().ptr(&mut mul)),
            )
            .search(graph);
        let mut selector2 = one_pat
            .edge(
                SelectOp::new()
                    .ptr(&mut a)
                    .edge(SelectOp::new().ty::<Mul>().ptr(&mut mul)),
            )
            .search(graph);
        while selector1.next_match() || selector2.next_match() {
            if graph.no_delete.contains(&one) {
                continue;
            }
            // Carry over outgoing edges
            let input_shape = graph
                .graph
                .edges_connecting(a, mul)
                .find_map(|e| e.weight().as_data())
                .unwrap()
                .2;
            if !input_shape.is_contiguous() || input_shape.is_padded() || input_shape.is_sliced() {
                // If any output shape is non-contiguous, we need to keep the op for it's contiguous functionality TODO: replace with explicit contiguous op here
                if graph
                    .graph
                    .edges_directed(mul, Direction::Outgoing)
                    .filter_map(|e| e.weight().as_data())
                    .any(|(_, _, sh)| !sh.is_contiguous() || sh.is_padded() || sh.is_sliced())
                {
                    continue;
                }
                for (weight, target) in graph
                    .graph
                    .edges_directed(mul, petgraph::Direction::Outgoing)
                    .map(|e| (e.weight().clone(), e.target()))
                    .collect::<Vec<_>>()
                {
                    if let Some(weight) = weight.as_data() {
                        graph.graph.add_edge(
                            a,
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
                move_outgoing_edge(mul, a, &mut graph.graph);
            }
            move_references(
                &mut remap,
                &mut graph.no_delete,
                &mut graph.to_retrieve,
                mul,
                a,
            );
            if graph.graph.edges_directed(one, Direction::Outgoing).count() == 1 {
                graph.graph.remove_node(one);
            }
            graph.graph.remove_node(mul);
        }
        // graph.display();
    }
}
