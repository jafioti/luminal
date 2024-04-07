#![allow(clippy::needless_range_loop)]

use crate::prelude::*;
use std::{
    io::Write,
    ops::{Deref, DerefMut},
};

use colored::Colorize;
use itertools::Itertools;
use petgraph::{stable_graph::StableGraph, visit::EdgeRef, Direction};
use rustc_hash::{FxHashMap, FxHashSet};

use super::compiler_utils::{ToIds, ToIdsMut};

pub type MainGraph = StableGraph<Box<dyn Operator>, Dependency>;
pub use petgraph::stable_graph::NodeIndex;

#[derive(Debug, Default)]
pub struct Graph {
    /// The store of tensors in the graph. Indexed by node index and output index.
    pub tensors: rustc_hash::FxHashMap<(NodeIndex, u8), Tensor>,
    /// A map of dynamic dimensions to concrete dimension sizes
    pub dyn_map: rustc_hash::FxHashMap<char, usize>,
    /// Edge weights: (Input index, Output index, Input shape)
    pub graph: MainGraph,
    /// Tensors marked in this set will not get deleted when the graph is ran
    pub no_delete: rustc_hash::FxHashSet<NodeIndex>,
    /// Tensors marked in this set need to be retrieved later (mostly for optimizers to insert copy back calls, the graph itself doesn't treat these differently)
    pub to_retrieve: rustc_hash::FxHashMap<NodeIndex, (u8, ShapeTracker)>,
    /// A list of current node to run, source nodes, and view nodes to delete after execution.
    #[allow(clippy::type_complexity)]
    pub(crate) linearized_graph: Option<Vec<(NodeIndex, Vec<((NodeIndex, u8), ShapeTracker)>)>>,
    /// Cached consumers (for execution only)
    consumers_map: Option<FxHashMap<(NodeIndex, u8), usize>>,
}

/// A dependency between two nodes
#[derive(Debug, Clone, Copy)]
#[allow(clippy::large_enum_variant)]
pub enum Dependency {
    /// A data dependency (transferring a tensor from one node to the next)
    Data {
        input_order: u8,
        output_order: u8,
        shape: ShapeTracker,
    },
    /// Explicit dependency for ordering. No tensors are transferred through this dependency
    Schedule,
}

impl Dependency {
    /// Try to extract dependency data
    pub fn as_data(self) -> Option<(u8, u8, ShapeTracker)> {
        if let Self::Data {
            input_order,
            output_order,
            shape,
        } = self
        {
            Some((input_order, output_order, shape))
        } else {
            None
        }
    }

    /// Is this a schedule dependency?
    pub fn is_schedule(&self) -> bool {
        matches!(self, Self::Schedule)
    }
}

impl Graph {
    /// Create a new graph
    pub fn new() -> Graph {
        Graph::default()
    }

    /// Try to remove the tensor data from the graph
    pub fn get_tensor(&mut self, id: NodeIndex, ind: u8) -> Option<Tensor> {
        self.tensors.remove(&(id, ind))
    }

    /// Try to get the tensor data in the graph
    pub fn get_tensor_ref(&self, id: NodeIndex, ind: u8) -> Option<&Tensor> {
        self.tensors.get(&(id, ind))
    }

    /// Delete the tensor data from the graph
    pub fn drop_tensors<T: ToIds>(&mut self, tensors: T) {
        for id in tensors.to_ids() {
            self.tensors.remove(&(id, 0));
        }
    }

    /// Mark tensors to be kept
    pub fn keep_tensors<T: ToIds>(&mut self, tensors: T) {
        for id in tensors.to_ids() {
            self.no_delete.insert(id);
        }
    }

    /// Set a tensor's data
    pub fn set_tensor(&mut self, id: NodeIndex, ind: u8, tensor: Tensor) {
        self.tensors.insert((id, ind), tensor);
    }

    /// Set a dynamic dimension
    pub fn set_dyn_dim(&mut self, dimension: char, val: usize) {
        self.dyn_map.insert(dimension, val);
    }

    /// Create a new tensor with shape S
    pub fn tensor<S: Shape>(&mut self) -> GraphTensor<S> {
        self.named_tensor("Tensor")
    }

    /// Create a new tensor with shape S and a name. This name will show up on the graph when displayed
    pub fn named_tensor<S: Shape>(&mut self, name: &str) -> GraphTensor<S> {
        GraphTensor {
            id: self.graph.add_node(Box::new(Function(
                format!("{name} Load"),
                Box::new(|_| panic!("You must set a value for this tensor!")),
            ))),
            graph_ref: self,
            shape: S::to_tracker(),
            _phantom: Default::default(),
        }
    }

    /// Compile the graph using the given compiler
    pub fn compile<T: ToIdsMut, C: Compiler>(&mut self, compiler: C, remap: T) -> C::Output {
        let output = compiler.compile(self, remap);
        self.toposort();
        self.reset();
        output
    }

    /// Refresh the internally sorted graph
    pub(crate) fn toposort(&mut self) {
        self.linearized_graph = Some(
            petgraph::algo::toposort(&self.graph, None)
                .unwrap()
                .into_iter()
                .map(|node| {
                    (
                        node,
                        self.graph
                            .edges_directed(node, Direction::Incoming)
                            .filter_map(|e| e.weight().as_data().map(|i| (e.source(), i)))
                            .sorted_by_key(|(_, (i, _, _))| *i)
                            .map(|(a, (_, b, c))| ((a, b), c))
                            .collect::<Vec<_>>(),
                    )
                })
                .collect(),
        );
        self.create_remaining_consumers_map();
    }

    /// Swap the tensors with these ids
    pub fn swap_tensors<A: Shape, B: Shape>(&mut self, a: GraphTensor<A>, b: GraphTensor<B>) {
        // Swap tensors
        let a_t = self.tensors.remove(&(a.id, 0)); // Assume 0th output for now
        let b_t = self.tensors.remove(&(b.id, 0));
        if let Some(a_t) = a_t {
            self.tensors.insert((b.id, 0), a_t);
        }
        if let Some(b_t) = b_t {
            self.tensors.insert((a.id, 0), b_t);
        }
    }

    /// Refresh the internal remaining consumers map
    fn create_remaining_consumers_map(&mut self) {
        self.consumers_map = Some(
            self.graph
                .node_indices()
                .flat_map(|i| {
                    self.graph
                        .edges_directed(i, Direction::Outgoing)
                        .filter_map(|e| e.weight().as_data().map(|i| (e.source(), i)))
                        .group_by(|(_, (_, i, _))| *i)
                        .into_iter()
                        .map(|(ind, g)| ((i, ind), g.count()))
                        .collect::<Vec<_>>()
                })
                .collect(),
        );
    }

    /// Clear any remaining tensors that may be around from old executions
    pub fn reset(&mut self) {
        self.tensors.retain(|(n, _), _| self.no_delete.contains(n));
    }

    /// Execute the graph.
    pub fn execute(&mut self) {
        // Track the number of views pointing to each tensor so we know when to clear
        if self.linearized_graph.is_none() {
            self.toposort();
        }
        let mut remaining_consumers = self.consumers_map.as_ref().unwrap().clone();
        let mut dim_stack = Vec::new();

        for (node, src_ids) in self.linearized_graph.as_ref().unwrap() {
            if self.tensors.contains_key(&(*node, 0)) {
                continue;
            }

            let mut srcs = Vec::new();
            get_source_tensors(
                &self.no_delete,
                &mut self.tensors,
                src_ids,
                &remaining_consumers,
                &mut srcs,
            );

            // Substitute in the dyn dims
            for (_, st) in srcs.iter_mut() {
                st.resolve_global_dyn_dims_stack(&self.dyn_map, &mut dim_stack);
            }

            // Execute
            let tensors = self.graph.node_weight_mut(*node).unwrap().process(srcs);
            for (i, tensor) in tensors.into_iter().enumerate() {
                self.tensors.insert((*node, i as u8), tensor);
            }

            // Bookkeep remaining consumers
            for (source, _) in src_ids {
                *remaining_consumers.get_mut(source).unwrap() -= 1;
            }
        }
        self.reset();
    }

    /// Execute the graph without deleting intermediate tensors
    pub fn execute_no_delete(&mut self) {
        // Track the number of views pointing to each tensor so we know when to clear;
        if self.linearized_graph.is_none() {
            self.toposort();
        }
        let mut dim_stack = Vec::new();
        for (node, src_ids) in self.linearized_graph.as_ref().unwrap().iter() {
            if self.tensors.contains_key(&(*node, 0)) {
                continue;
            }
            let mut srcs = src_ids
                .iter()
                .map(|(id, st)| (InputTensor::Borrowed(self.tensors.get(id).unwrap()), *st))
                .collect_vec();

            // Substitute in the dyn dims
            for (_, st) in srcs.iter_mut() {
                st.resolve_global_dyn_dims_stack(&self.dyn_map, &mut dim_stack);
            }

            // All sources are ready, execute
            let tensors = self.graph.node_weight_mut(*node).unwrap().process(srcs);
            for (i, tensor) in tensors.into_iter().enumerate() {
                self.tensors.insert((*node, i as u8), tensor);
            }
        }
    }

    /// Execute the graph with debug prints
    pub fn execute_debug(&mut self) {
        // Track the number of views pointing to each tensor so we know when to clear
        if self.linearized_graph.is_none() {
            self.toposort();
        }
        let mut dim_stack = Vec::new();
        let tensors_ptr = &mut self.tensors as *mut _;
        let mut remaining_consumers = self.consumers_map.as_ref().unwrap().clone();
        let mut op_times = FxHashMap::default();

        println!(
            "{:->2$} Executing {:->2$}",
            "",
            "",
            (term_size::dimensions()
                .unwrap()
                .0
                .saturating_sub(" Executing ".len()))
                / 2
        );
        let start = std::time::Instant::now();
        for (node, src_ids) in self.linearized_graph.as_ref().unwrap().iter() {
            if self.tensors.contains_key(&(*node, 0)) {
                continue;
            }
            let op_name = format!("{:?}", self.graph.node_weight(*node).unwrap());
            print!("{}", op_name.bold().bright_green());

            let mut srcs = Vec::new();
            get_source_tensors(
                &self.no_delete,
                tensors_ptr,
                src_ids,
                &remaining_consumers,
                &mut srcs,
            );

            // Substitute in the dyn dims
            for (_, st) in srcs.iter_mut() {
                st.resolve_global_dyn_dims_stack(&self.dyn_map, &mut dim_stack);
            }

            // All sources are ready
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
            print!("{shapes_string}");
            std::io::stdout().flush().unwrap();
            // Execute
            let now = std::time::Instant::now();
            let tensors = self.graph.node_weight_mut(*node).unwrap().process(srcs);
            let elapsed = now.elapsed();
            println!(
                "{:.>1$}",
                if elapsed.as_secs() > 0 {
                    format!("{:.2}s", elapsed.as_secs_f32())
                } else if elapsed.as_millis() > 0 {
                    format!("{}ms", elapsed.as_millis())
                } else {
                    format!("{}µs", elapsed.as_micros())
                }
                .bold(),
                term_size::dimensions()
                    .unwrap()
                    .0
                    .saturating_sub(op_name.len())
                    .saturating_sub(shapes_string.len()),
            );
            for (i, tensor) in tensors.into_iter().enumerate() {
                self.tensors.insert((*node, i as u8), tensor);
            }
            if let Some(t) = op_times.get_mut(&op_name) {
                *t += elapsed;
            } else {
                op_times.insert(op_name, elapsed);
            }

            // Check if we can delete the source tensors now
            for (source, _) in src_ids {
                *remaining_consumers.get_mut(source).unwrap() -= 1;
            }
        }

        // Print out total times
        println!();
        println!(
            "{:->2$} Total Times {:->2$}",
            "",
            "",
            (term_size::dimensions()
                .unwrap()
                .0
                .saturating_sub(" Total Times ".len()))
                / 2
        );
        for (name, elapsed) in op_times.into_iter().sorted_by(|(_, a), (_, b)| b.cmp(a)) {
            print!("{}", name.bold().bright_green());
            println!(
                "{:.>1$}",
                if elapsed.as_secs() > 0 {
                    format!("{:.2}s", elapsed.as_secs_f32())
                } else if elapsed.as_millis() > 0 {
                    format!("{}ms", elapsed.as_millis())
                } else {
                    format!("{}µs", elapsed.as_micros())
                }
                .bold(),
                term_size::dimensions()
                    .unwrap()
                    .0
                    .saturating_sub(name.len()),
            );
        }
        println!(
            "Total: {}",
            if start.elapsed().as_secs() > 0 {
                format!("{:.2}s", start.elapsed().as_secs_f32())
            } else if start.elapsed().as_millis() > 0 {
                format!("{}ms", start.elapsed().as_millis())
            } else {
                format!("{}µs", start.elapsed().as_micros())
            }
            .bold()
        );
        self.reset();
    }
}

impl Deref for Graph {
    type Target = MainGraph;
    fn deref(&self) -> &Self::Target {
        &self.graph
    }
}

impl DerefMut for Graph {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.graph
    }
}

/// Get source tensor array for a node
fn get_source_tensors(
    no_delete: &FxHashSet<NodeIndex>,
    tensors: *mut FxHashMap<(NodeIndex, u8), Tensor>,
    src_ids: &[((NodeIndex, u8), ShapeTracker)],
    remaining_consumers: &FxHashMap<(NodeIndex, u8), usize>,
    srcs: &mut Vec<(InputTensor, ShapeTracker)>,
) {
    for (id, sh) in src_ids {
        if remaining_consumers[id] == 1 && !no_delete.contains(&id.0) {
            srcs.push((
                InputTensor::Owned(unsafe { tensors.as_mut().unwrap() }.remove(id).unwrap()),
                *sh,
            ));
        } else {
            srcs.push((
                InputTensor::Borrowed(unsafe { tensors.as_ref().unwrap() }.get(id).unwrap()),
                *sh,
            ));
        }
    }
}
