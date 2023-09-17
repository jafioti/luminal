#![allow(clippy::needless_range_loop)]

use crate::{
    graph_tensor::GraphTensor,
    op::{self, InputTensor, Operator, Print},
    optimizer::GraphOptimizer,
    shape::*,
    tensor::Tensor,
};
use std::collections::{HashMap, HashSet, VecDeque};

use colored::Colorize;
use itertools::Itertools;
use petgraph::{graph::NodeIndex, stable_graph::StableGraph, visit::EdgeRef, Direction};

#[derive(Debug, Default)]
pub struct Graph {
    pub tensors: HashMap<NodeIndex, Tensor>,
    pub(crate) id_remap: HashMap<NodeIndex, NodeIndex>,
    pub dyn_map: HashMap<char, usize>,
    pub graph:
        StableGraph<Box<dyn Operator>, (u8, crate::core::shape::simple_tracker::ShapeTracker)>,
    pub no_delete: HashSet<NodeIndex>,
    /// Mark tensors that need to be retrieved later (mostly for optimizers to insert copy back calls, the graph itself doesn't treat these differently)
    pub(crate) to_retrieve: HashSet<NodeIndex>,
    /// A list of current node to run, source nodes, and view nodes to delete after execution.
    #[allow(clippy::type_complexity)]
    pub(crate) linearized_graph: Option<
        Vec<(
            NodeIndex,
            Vec<(NodeIndex, crate::core::shape::simple_tracker::ShapeTracker)>,
        )>,
    >,
}

impl Graph {
    pub fn new() -> Graph {
        Graph::default()
    }

    pub fn add_op<O: Operator + 'static>(&mut self, op: O) -> NewOp {
        self.graph.free_node = NodeIndex::end(); // Prevent reuse of deleted indexes (screws up remapping)
        NewOp {
            new_op_id: self.graph.add_node(Box::new(op)),
            graph_ref: self,
            num_srcs: 0,
        }
    }

    pub fn get_tensor(&mut self, id: NodeIndex) -> Option<Tensor> {
        // Walk through remap
        self.tensors.remove(&remap_id(id, &self.id_remap))
    }

    pub fn get_tensor_ref(&self, id: NodeIndex) -> Option<&Tensor> {
        // Walk through remap
        self.tensors.get(&remap_id(id, &self.id_remap))
    }

    pub fn set_dyn_dim(&mut self, dim: char, val: usize) {
        self.dyn_map.insert(dim, val);
    }

    pub fn new_tensor<S: Shape>(&mut self, name: &str) -> GraphTensor<S> {
        self.graph.free_node = NodeIndex::end(); // Prevent reuse of deleted indexes (screws up remapping)
        GraphTensor {
            id: self.graph.add_node(Box::new(op::Function(
                name.to_string(),
                Box::new(|_| panic!("You must set a value for this tensor!")),
            ))),
            graph_ref: self,
            shape: S::to_tracker(),
            _phantom: Default::default(),
        }
    }

    /// Run the full suite of optimizations
    pub fn optimize<O: GraphOptimizer>(&mut self, optimizer: O) {
        optimizer.optimize(self);
        self.toposort();
    }

    /// Refresh the internally sorted graph
    fn toposort(&mut self) {
        // Depth-first toposort
        let mut visited = HashSet::default();
        let mut pre_sorted = petgraph::algo::toposort(&self.graph, None).unwrap();
        pre_sorted.reverse();
        let mut stacks = vec![];
        for node in pre_sorted {
            if !visited.contains(&node) {
                stacks.push(toposort(node, &self.graph, &mut visited));
            }
        }
        let mut nodes = vec![];
        for (mut stack, _, _) in stacks.into_iter().sorted_by_key(|(_, _, b)| !*b) {
            nodes.append(&mut stack);
        }
        let mut v = Vec::with_capacity(nodes.len());
        for node in nodes {
            let src_ids = self
                .graph
                .edges_directed(node, Direction::Incoming)
                .sorted_by_key(|e| e.weight().0)
                .map(|i| (i.source(), i.weight().1))
                .collect_vec();
            v.push((node, src_ids));
        }
        self.linearized_graph = Some(v);
    }

    /// Clear any remaining tensors that may be around from old executions
    pub fn reset(&mut self) {
        // (This is where we should do the tensor caching!)
        for t in self.tensors.keys().copied().collect_vec() {
            if !self.no_delete.contains(&t) {
                self.tensors.remove(&t);
            }
        }
    }

    /// Clear all nodes not required to produce output nodes, stop looking past inputs.
    ///
    /// Pruning doesn't respect nodes marked as no_delete.
    pub fn prune<O: IntoIterator<Item = NodeIndex>, I: IntoIterator<Item = NodeIndex>>(
        &mut self,
        outputs: O,
        inputs: I,
    ) {
        let mut keep_nodes = HashSet::default();
        let input_nodes: HashSet<NodeIndex> = inputs.into_iter().collect();
        for n in outputs {
            reverse_dfs_mark(n, self, &mut keep_nodes, &input_nodes);
        }
        for node in self.graph.node_indices().collect_vec() {
            if !keep_nodes.contains(&node)
                && !self.graph.node_weight(node).unwrap().as_any().is::<Print>()
            {
                self.graph.remove_node(node);
                self.no_delete.remove(&node);
                self.to_retrieve.remove(&node);
            }
        }
        self.linearized_graph = None;
    }

    /// Swap the tensors with these ids
    pub fn swap_tensors<A: Shape, B: Shape>(&mut self, a: GraphTensor<A>, b: GraphTensor<B>) {
        // Swap tensors
        let a_t = self.tensors.remove(&a.id);
        let b_t = self.tensors.remove(&b.id);
        if let Some(a_t) = a_t {
            self.tensors.insert(b.id, a_t);
        }
        if let Some(b_t) = b_t {
            self.tensors.insert(a.id, b_t);
        }
    }

    /// Execute the graph.
    pub fn execute(&mut self) {
        self.reset();
        // Track the number of views pointing to each tensor so we know when to clear
        if self.linearized_graph.is_none() {
            self.toposort();
        }
        let mut remaining_consumers: HashMap<NodeIndex, usize> = self
            .graph
            .node_indices()
            .map(|i| (i, self.graph.edges_directed(i, Direction::Outgoing).count()))
            .collect();

        for (node, src_ids) in self.linearized_graph.as_ref().unwrap().iter() {
            if self.tensors.contains_key(node) {
                continue;
            }

            // This needs to be done in a weird way with sperate queues to satisfy the borrow checker (all mutable calls to self.tensors happen before references are taken)
            let mut owned = VecDeque::default();
            let mut refs = VecDeque::default();
            for (i, (id, _)) in src_ids.iter().enumerate() {
                if remaining_consumers[id] == 1 && !self.no_delete.contains(id) {
                    owned.push_back((InputTensor::Owned(self.tensors.remove(id).unwrap()), i));
                }
            }
            for (i, (id, _)) in src_ids.iter().enumerate() {
                if remaining_consumers[id] != 1 || self.no_delete.contains(id) {
                    refs.push_back((InputTensor::Borrowed(self.tensors.get(id).unwrap()), i));
                }
            }
            let mut srcs = vec![];
            for (i, (_, st)) in src_ids.iter().enumerate() {
                srcs.push((
                    if owned.front().map(|(_, ind)| *ind == i).unwrap_or_default() {
                        owned.pop_front().unwrap().0
                    } else {
                        refs.pop_front().unwrap().0
                    },
                    *st,
                ));
            }

            // Substitute in the dyn dims
            for (_, st) in srcs.iter_mut() {
                *st = st.resolve_global_dyn_dims(&self.dyn_map);
            }

            // All sources are ready
            // Execute
            let tensor = self.graph.node_weight(*node).unwrap().process(srcs);
            self.tensors.insert(*node, tensor);

            // Check if we can delete the source tensors now
            for (source, _) in src_ids {
                *remaining_consumers.get_mut(source).unwrap() -= 1;
            }
        }
    }

    /// Execute the graph with debug prints
    pub fn execute_debug(&mut self) {
        self.reset();
        // Track the number of views pointing to each tensor so we know when to clear
        if self.linearized_graph.is_none() {
            self.toposort();
        }
        let mut remaining_consumers: HashMap<NodeIndex, usize> = self
            .graph
            .node_indices()
            .map(|i| (i, self.graph.edges_directed(i, Direction::Outgoing).count()))
            .collect();

        for (node, src_ids) in self.linearized_graph.as_ref().unwrap().iter() {
            if self.tensors.contains_key(node) {
                continue;
            }

            // This needs to be done in a weird way with sperate queues to satisfy the borrow checker (all mutable calls to self.tensors happen before references are taken)
            let mut owned = VecDeque::default();
            let mut refs = VecDeque::default();
            for (i, (id, _)) in src_ids.iter().enumerate() {
                if remaining_consumers[id] == 1 && !self.no_delete.contains(id) {
                    owned.push_back((InputTensor::Owned(self.tensors.remove(id).unwrap()), i));
                }
            }
            for (i, (id, _)) in src_ids.iter().enumerate() {
                if remaining_consumers[id] != 1 || self.no_delete.contains(id) {
                    refs.push_back((InputTensor::Borrowed(self.tensors.get(id).unwrap()), i));
                }
            }
            let mut srcs = vec![];
            for (i, (_, st)) in src_ids.iter().enumerate() {
                srcs.push((
                    if owned.front().map(|(_, ind)| *ind == i).unwrap_or_default() {
                        owned.pop_front().unwrap().0
                    } else {
                        refs.pop_front().unwrap().0
                    },
                    *st,
                ));
            }

            // Substitute in the dyn dims
            for (_, st) in srcs.iter_mut() {
                *st = st.resolve_global_dyn_dims(&self.dyn_map);
            }

            // All sources are ready
            let now = std::time::Instant::now();
            let op_name = format!("{:?}", self.graph.node_weight(*node).unwrap());
            let mut shapes_string = srcs
                .iter()
                .map(|(_, s)| {
                    format!(
                        "{:?}",
                        s.shape()
                            .into_iter()
                            .map(|i| i.to_usize().unwrap())
                            .collect::<Vec<_>>()
                    )
                })
                .join(", ");
            if !shapes_string.is_empty() {
                shapes_string = format!(" ({shapes_string})");
            }
            print!("{}", op_name.bold().bright_green());
            print!("{shapes_string}");
            // Execute
            let tensor = self.graph.node_weight(*node).unwrap().process(srcs);
            println!(
                "{:.>1$}",
                if now.elapsed().as_secs() > 1 {
                    format!("{:.2}s", now.elapsed().as_secs_f32()).bold()
                } else if now.elapsed().as_millis() > 1 {
                    format!("{}ms", now.elapsed().as_millis()).bold()
                } else {
                    format!("{}µs", now.elapsed().as_micros()).bold()
                },
                term_size::dimensions().unwrap().0 - op_name.len() - shapes_string.len(),
            );
            self.tensors.insert(*node, tensor);

            // Check if we can delete the source tensors now
            for (source, _) in src_ids {
                *remaining_consumers.get_mut(source).unwrap() -= 1;
            }
        }
    }

    /// Execute the graph without deleting intermediate tensors
    pub fn execute_no_delete(&mut self) {
        self.reset();
        // Track the number of views pointing to each tensor so we know when to clear;
        if self.linearized_graph.is_none() {
            self.toposort();
        }
        for (node, src_ids) in self.linearized_graph.as_ref().unwrap().iter() {
            if self.tensors.contains_key(node) {
                continue;
            }
            let mut srcs = src_ids
                .iter()
                .map(|(id, st)| (InputTensor::Borrowed(self.tensors.get(id).unwrap()), *st))
                .collect_vec();

            // Substitute in the dyn dims
            for (_, st) in srcs.iter_mut() {
                *st = st.resolve_global_dyn_dims(&self.dyn_map);
            }

            // All sources are ready, execute
            let tensor = self.graph.node_weight(*node).unwrap().process(srcs);
            self.tensors.insert(*node, tensor);
        }
    }

    /// Convert to debug-viewable graph
    pub fn debug_graph(
        &self,
        show_shapes: bool,
    ) -> petgraph::stable_graph::StableGraph<String, u8, petgraph::Directed, u32> {
        let mut new_graph = petgraph::stable_graph::StableGraph::default();
        let mut id_map = HashMap::new();
        for (id, node) in self.graph.node_indices().zip(self.graph.node_weights()) {
            id_map.insert(id, new_graph.add_node(format!("{node:?}")));
        }

        for node in self.graph.node_indices() {
            for edge in self
                .graph
                .edges_directed(node, petgraph::Direction::Outgoing)
                .sorted_by_key(|e| e.weight().0)
            {
                new_graph.add_edge(
                    id_map[&edge.source()],
                    id_map[&edge.target()],
                    edge.weight().0,
                );
                if show_shapes && new_graph.contains_node(id_map[&edge.target()]) {
                    new_graph
                        .node_weight_mut(id_map[&edge.target()])
                        .unwrap()
                        .push_str(&format!(" | {:?}", edge.weight().1.shape()));
                }
            }
        }

        new_graph
    }

    pub fn display(&self) {
        display_graph(&self.debug_graph(false));
    }

    pub fn display_shapes(&self) {
        display_graph(&self.debug_graph(true));
    }

    /// Get the sources of a node given it's id
    #[allow(clippy::type_complexity, clippy::borrowed_box)]
    pub fn get_sources(&self, node_id: NodeIndex) -> Vec<(NodeIndex, ShapeTracker)> {
        self.graph
            .edges_directed(node_id, Direction::Incoming)
            .sorted_by_key(|e| e.weight().0)
            .map(|e| (e.source(), e.weight().1))
            .collect()
    }

    /// Get the dests of a node given it's id
    #[allow(clippy::type_complexity, clippy::borrowed_box)]
    pub fn get_dests(&self, node_id: NodeIndex) -> Vec<(NodeIndex, &Box<dyn Operator>)> {
        self.graph
            .edges_directed(node_id, Direction::Outgoing)
            .sorted_by_key(|e| e.weight().0)
            .map(|e| e.target())
            .map(|n| (n, self.graph.node_weight(n).unwrap()))
            .collect()
    }
}

/// View a debug graph in the browser
pub fn display_graph(
    graph: &petgraph::stable_graph::StableGraph<String, u8, petgraph::Directed, u32>,
) {
    let url = format!(
        "https://dreampuf.github.io/GraphvizOnline/#{}",
        urlencoding::encode(
            &petgraph::dot::Dot::with_config(&graph, &[petgraph::dot::Config::EdgeNoLabel,])
                .to_string()
        )
    );
    if let Err(e) = webbrowser::open(&url) {
        panic!("Error displaying graph: {:?}", e);
    }
}

pub trait JoinGraph {
    fn join(
        self,
        rhs: &petgraph::stable_graph::StableGraph<String, u8, petgraph::Directed, u32>,
    ) -> Self;
}

impl JoinGraph for petgraph::stable_graph::StableGraph<String, u8, petgraph::Directed, u32> {
    /// Join two debug graphs together
    fn join(
        mut self,
        rhs: &petgraph::stable_graph::StableGraph<String, u8, petgraph::Directed, u32>,
    ) -> Self {
        let mut id_map = HashMap::new(); // We track the node id remapping here so they don't overlap
        for (index, node) in rhs.node_indices().zip(rhs.node_weights()) {
            id_map.insert(index, self.add_node(node.clone()));
        }

        for node in rhs.node_indices() {
            for edge in rhs.edges_directed(node, petgraph::Direction::Outgoing) {
                self.add_edge(
                    id_map[&edge.source()],
                    id_map[&edge.target()],
                    *edge.weight(),
                );
            }
        }

        self
    }
}

pub struct NewOp<'a> {
    new_op_id: NodeIndex,
    graph_ref: &'a mut Graph,
    num_srcs: u8,
}

impl<'a> NewOp<'a> {
    pub fn finish(self) -> NodeIndex {
        self.new_op_id
    }

    pub fn input(
        mut self,
        id: NodeIndex,
        shape: crate::core::shape::simple_tracker::ShapeTracker,
    ) -> Self {
        self.graph_ref.graph.add_edge(
            remap_id(id, &self.graph_ref.id_remap),
            self.new_op_id,
            (self.num_srcs, shape),
        );
        self.num_srcs += 1;
        self
    }
}

/// Walk through remap to get new id
pub(crate) fn remap_id(
    mut id: NodeIndex,
    map: &HashMap<NodeIndex<u32>, NodeIndex<u32>>,
) -> NodeIndex {
    while let Some(new_id) = map.get(&id) {
        id = *new_id;
    }
    id
}

fn toposort(
    id: NodeIndex,
    graph: &StableGraph<Box<dyn Operator>, (u8, crate::core::shape::simple_tracker::ShapeTracker)>,
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

fn reverse_dfs_mark(
    curr_node: NodeIndex,
    cx: &Graph,
    marked: &mut HashSet<NodeIndex>,
    input_nodes: &HashSet<NodeIndex>,
) {
    marked.insert(curr_node);
    if !input_nodes.contains(&curr_node) {
        for i in cx
            .graph
            .edges_directed(curr_node, Direction::Incoming)
            .map(|e| e.source())
            .filter(|i| !marked.contains(i))
            .collect_vec()
        {
            reverse_dfs_mark(i, cx, marked, input_nodes);
        }
    }
}
