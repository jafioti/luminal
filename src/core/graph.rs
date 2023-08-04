#![allow(clippy::needless_range_loop)]

use crate::{
    graph_tensor::GraphTensor,
    op::{self, Operator},
    optimizer::GraphOptimizer,
    prelude::TensorView,
    shape::*,
    tensor::Tensor,
};
use std::collections::{HashMap, HashSet};

use itertools::Itertools;
use petgraph::{graph::NodeIndex, stable_graph::StableGraph, visit::EdgeRef, Direction};

#[derive(Debug, Default)]
pub struct Graph {
    pub(crate) tensors: HashMap<NodeIndex, Tensor>,
    pub(crate) views: HashMap<NodeIndex, TensorView>,
    pub(crate) id_remap: HashMap<NodeIndex, NodeIndex>,
    #[allow(clippy::type_complexity)]
    pub(crate) graph: StableGraph<(Box<dyn Operator>, Vec<RealDim>), u8>,
    pub(crate) no_delete: HashSet<NodeIndex>,
    pub(crate) to_retrieve: HashSet<NodeIndex>,
    /// A list of current node to run, source nodes.
    pub(crate) linearized_graph: Option<Vec<(NodeIndex, Vec<NodeIndex>)>>,
}

impl Graph {
    pub fn new() -> Graph {
        Graph::default()
    }

    pub fn add_op<O: Operator + 'static>(&mut self, op: O, output_shape: Vec<RealDim>) -> NewOp {
        self.graph.free_node = NodeIndex::end(); // Prevent reuse of deleted indexes (screws up remapping)
        NewOp {
            new_op_id: self.graph.add_node((Box::new(op), output_shape)),
            graph_ref: self,
            num_srcs: 0,
        }
    }

    pub(crate) fn get_op(&self, id: NodeIndex) -> Option<&dyn Operator> {
        self.graph.node_weight(id).map(|n| n.0.as_ref())
    }

    pub fn get_tensor(&mut self, id: NodeIndex) -> Option<Tensor> {
        let id = self.get_view(id)?.tensor_id;
        self.tensors.remove(&id)
    }

    pub fn get_tensor_ref(&self, id: NodeIndex) -> Option<&Tensor> {
        let view = self.get_view(id)?;
        self.tensors.get(&view.tensor_id)
    }

    pub fn get_view(&self, mut id: NodeIndex) -> Option<&TensorView> {
        // Walk through remaps
        while let Some(new_id) = self.id_remap.get(&id) {
            id = *new_id;
        }

        self.views.get(&id)
    }

    pub fn new_tensor<S: Shape>(&mut self, name: &str) -> GraphTensor<S> {
        self.graph.free_node = NodeIndex::end(); // Prevent reuse of deleted indexes (screws up remapping)
        let tensor = GraphTensor {
            id: self.graph.add_node((
                Box::new(op::Function(
                    name.to_string(),
                    Box::new(|_, _| panic!("You must set a value for this tensor!")),
                )),
                S::realized_shape(),
            )),
            graph_ref: self,
            _phantom: Default::default(),
        };
        self.no_delete.insert(tensor.id); // This gets set because we want to keep inputs around to run the graph multiple times
        tensor
    }

    /// Run the full suite of optimizations
    pub fn optimize<O: GraphOptimizer>(&mut self, optimizer: O) {
        optimizer.optimize(self);
        self.toposort();
    }

    /// Refresh the internally sorted graph
    fn toposort(&mut self) {
        // Depth-first toposort
        let nodes = petgraph::algo::toposort(&self.graph, None).unwrap();
        let mut v = Vec::with_capacity(nodes.len());
        for node in nodes {
            let src_ids = self
                .graph
                .edges_directed(node, Direction::Incoming)
                .sorted_by_key(|e| e.weight())
                .map(|i| i.source())
                .collect_vec();
            v.push((node, src_ids));
        }
        self.linearized_graph = Some(v);
    }

    /// Clear any remaining tensors that may be around from old executions
    pub fn reset(&mut self) {
        // (This is where we should do the tensor caching!)
        self.tensors.clear();
        self.views.clear();
    }

    /// Execute the graph.
    pub fn execute(&mut self) {
        // Track the number of dependencies each node has so we know when to clear
        let mut dependencies: HashMap<NodeIndex, usize> = self
            .graph
            .node_indices()
            .map(|n| (n, self.graph.edges_directed(n, Direction::Outgoing).count()))
            .collect();
        let mut views_pointing: HashMap<NodeIndex, usize> = self
            .views
            .iter()
            .group_by(|(_, v)| v.tensor_id)
            .into_iter()
            .map(|(v, i)| (v, i.count()))
            .collect();
        if self.linearized_graph.is_none() {
            self.toposort();
        }
        for (node, src_ids) in self.linearized_graph.as_ref().unwrap().iter() {
            if self.views.contains_key(node) {
                continue;
            }
            let srcs = src_ids
                .iter()
                .map(|i| {
                    let view = self.views.get(i).unwrap().clone();
                    (self.tensors.get(&view.tensor_id).unwrap(), view)
                })
                .collect_vec();

            // All sources are ready, execute
            let (t, v) = self
                .graph
                .node_weight(*node)
                .unwrap()
                .0
                .process(srcs, *node);
            if let Some(tensor) = t {
                self.tensors.insert(*node, tensor);
            }
            if let Some(vp) = views_pointing.get_mut(&v.tensor_id) {
                *vp += 1;
            } else {
                views_pointing.insert(v.tensor_id, 1);
            }
            self.views.insert(*node, v);

            // Check if we can delete the source tensors now
            for source in src_ids.iter().filter(|n| !self.no_delete.contains(n)) {
                let deps = dependencies.get_mut(source).unwrap();
                *deps -= 1;
                if *deps == 0 {
                    // No more dependencies for this view, let's remove it
                    if let Some(view) = self.views.remove(source) {
                        let vp = views_pointing.get_mut(&view.tensor_id).unwrap();
                        *vp -= 1;
                        if *vp == 0 {
                            // No views pointing at this tensor, remove it
                            self.tensors.remove(&view.tensor_id);
                        }
                    }
                }
            }
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
            id_map.insert(
                id,
                new_graph.add_node(if show_shapes {
                    format!("{node:?}")
                } else {
                    format!("{:?}", node.0)
                }),
            );
        }

        for node in self.graph.node_indices() {
            for edge in self
                .graph
                .edges_directed(node, petgraph::Direction::Outgoing)
            {
                new_graph.add_edge(
                    id_map[&edge.source()],
                    id_map[&edge.target()],
                    *edge.weight(),
                );
            }
        }

        new_graph
    }

    pub fn display_graph(&self) {
        display_graph(&self.debug_graph(false));
    }

    /// Transfer all external references from one node to another (this may happen because one node is about to be removed / merged into another)
    pub fn move_references(
        id_remap: &mut HashMap<NodeIndex, NodeIndex>,
        no_delete: &mut HashSet<NodeIndex<u32>>,
        to_retrieve: &mut HashSet<NodeIndex<u32>>,
        src: NodeIndex,
        trg: NodeIndex,
    ) {
        // Create remap
        id_remap.insert(src, trg);
        // Transfer no_delete
        if no_delete.remove(&src) {
            no_delete.insert(trg);
        }
        // Transfer to_retrieve
        if to_retrieve.remove(&src) {
            to_retrieve.insert(trg);
        }
    }

    /// Get the sources of a node given it's id
    #[allow(clippy::type_complexity)]
    pub fn get_sources(
        &self,
        node_id: NodeIndex,
    ) -> Vec<(NodeIndex, &(Box<dyn Operator>, Vec<RealDim>))> {
        self.graph
            .edges_directed(node_id, Direction::Incoming)
            .map(|e| e.source())
            .map(|n| (n, self.graph.node_weight(n).unwrap()))
            .collect()
    }

    /// Get the dests of a node given it's id
    #[allow(clippy::type_complexity)]
    pub fn get_dests(
        &self,
        node_id: NodeIndex,
    ) -> Vec<(NodeIndex, &(Box<dyn Operator>, Vec<RealDim>))> {
        self.graph
            .edges_directed(node_id, Direction::Outgoing)
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

    pub fn input(mut self, id: NodeIndex) -> Self {
        self.graph_ref
            .graph
            .add_edge(id, self.new_op_id, self.num_srcs);
        self.num_srcs += 1;
        self
    }
}
