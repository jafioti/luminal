#![allow(clippy::needless_range_loop)]

use crate::{
    graph_tensor::GraphTensor,
    op::{self, Operator},
    optimizer::GraphOptimizer,
    shape::*,
    tensor::Tensor,
};
use std::collections::{HashMap, HashSet};

use itertools::Itertools;
use petgraph::{graph::NodeIndex, stable_graph::StableGraph, visit::EdgeRef, Directed, Direction};

#[derive(Debug, Default)]
pub struct Graph {
    pub(crate) tensors: HashMap<NodeIndex, Tensor>,
    pub(crate) id_remap: HashMap<NodeIndex, NodeIndex>,
    pub(crate) graph: StableGraph<Box<dyn Operator>, u8, Directed, u32>,
    pub(crate) no_delete: HashSet<NodeIndex>,
}

impl Graph {
    pub fn new() -> Graph {
        Graph::default()
    }

    pub fn get_tensor(&mut self, mut id: NodeIndex) -> Option<Tensor> {
        // Walk through remaps
        while let Some(new_id) = self.id_remap.get(&id) {
            id = *new_id;
        }

        self.tensors.remove(&id)
    }

    pub fn new_tensor<S: Shape>(&mut self) -> GraphTensor<S> {
        let tensor = GraphTensor {
            id: self.graph.add_node(Box::new(op::Input)),
            graph_ref: self,
            _phantom: Default::default(),
        };
        self.no_delete.insert(tensor.id);
        tensor
    }

    pub fn set_tensor<S: Shape>(&mut self, graph_tensor: GraphTensor<S>, data: Vec<f32>) {
        self.tensors.insert(
            graph_tensor.id,
            Tensor {
                data,
                shape: ShapeTracker::new(S::realized_shape()),
            },
        );
    }

    /// Run the full suite of optimizations
    pub fn optimize<O: GraphOptimizer>(&mut self, optimizer: O) {
        optimizer.optimize(self);
    }

    /// Execute the graph.
    pub fn execute(&mut self) {
        loop {
            let mut new_tensors = vec![];
            // Find all executable ops
            for (node, srcs) in self
                .graph
                .node_indices()
                .filter_map(|n| {
                    if self.tensors.contains_key(&n) {
                        return None;
                    }
                    let mut data = vec![];
                    for e in self
                        .graph
                        .edges_directed(n, petgraph::Direction::Incoming)
                        .sorted_by_key(|e| e.weight())
                    {
                        if let Some(e) = self.tensors.get(&e.source()) {
                            data.push(e);
                        } else {
                            return None;
                        }
                    }
                    Some((n, data))
                })
                .collect_vec()
            {
                // All sources are ready, execute
                let f = self.graph.node_weight(node).unwrap().process(srcs);
                new_tensors.push((node, f));
            }

            // Check if we can delete the source tensors now
            for node in new_tensors.iter().map(|(t, _)| t) {
                // Check we have incoming edges (don't want to remove the sources)
                for source in self
                    .graph
                    .edges_directed(*node, Direction::Incoming)
                    .map(|e| e.source())
                    .filter(|e| self.graph.edges_directed(*e, Direction::Outgoing).count() == 1)
                    .collect_vec()
                {
                    if !self.no_delete.contains(&source) {
                        // Delete tensor and node
                        self.tensors.remove(&source);
                    }
                }
            }

            if new_tensors.is_empty() {
                break;
            }

            for (k, v) in new_tensors {
                self.tensors.insert(k, v);
            }
        }
    }

    /// Convert to debug-viewable graph
    pub fn debug_graph(
        &self,
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

    /// Transfer all external references from one node to another (this may happen because one node is about to be removed / merged into another)
    pub fn move_references(
        id_remap: &mut HashMap<NodeIndex, NodeIndex>,
        no_delete: &mut HashSet<NodeIndex<u32>>,
        src: NodeIndex,
        trg: NodeIndex,
    ) {
        // Create remap
        id_remap.insert(src, trg);
        // Transfer no_delete
        if no_delete.remove(&src) {
            no_delete.insert(trg);
        }
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
        println!("Error displaying graph: {:?}", e);
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
