#![allow(clippy::needless_range_loop)]

use crate::{
    compiler_utils::Compiler,
    graph_tensor::GraphTensor,
    op::{self, InputTensor, Operator},
    shape::*,
    tensor::Tensor,
};
use std::{
    collections::{HashMap, HashSet, VecDeque},
    io::Write,
};

use colored::Colorize;
use itertools::Itertools;
use petgraph::{graph::NodeIndex, stable_graph::StableGraph, visit::EdgeRef, Direction};

pub type MainGraph = StableGraph<Box<dyn Operator>, Dependency>;

#[derive(Debug, Default)]
pub struct Graph {
    pub tensors: HashMap<(NodeIndex, u8), Tensor>,
    pub id_remap: HashMap<NodeIndex, NodeIndex>,
    pub dyn_map: HashMap<char, usize>,
    /// Edge weights: (Input index, Output index, Input shape)
    pub graph: MainGraph,
    pub no_delete: HashSet<NodeIndex>,
    /// Mark tensors that need to be retrieved later (mostly for optimizers to insert copy back calls, the graph itself doesn't treat these differently)
    pub to_retrieve: HashSet<NodeIndex>,
    /// A list of current node to run, source nodes, and view nodes to delete after execution.
    #[allow(clippy::type_complexity)]
    pub(crate) linearized_graph: Option<Vec<(NodeIndex, Vec<((NodeIndex, u8), ShapeTracker)>)>>,
}

#[derive(Debug, Clone, Copy)]
#[allow(clippy::large_enum_variant)]
pub enum Dependency {
    /// Actual data dependency
    Data {
        input_order: u8,
        output_order: u8,
        shape: ShapeTracker,
    },
    /// Implicit dependency for ordering
    Schedule,
}

impl Dependency {
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

    pub fn is_schedule(&self) -> bool {
        matches!(self, Self::Schedule)
    }
}

impl Graph {
    pub fn new() -> Graph {
        Graph::default()
    }

    pub fn get_tensor(&mut self, id: NodeIndex, ind: u8) -> Option<Tensor> {
        // Walk through remap
        self.tensors.remove(&(remap_id(id, &self.id_remap), ind))
    }

    pub fn get_tensor_ref(&self, id: NodeIndex, ind: u8) -> Option<&Tensor> {
        // Walk through remap
        self.tensors.get(&(remap_id(id, &self.id_remap), ind))
    }

    pub fn set_tensor(&mut self, id: NodeIndex, ind: u8, tensor: Tensor) {
        self.tensors
            .insert((remap_id(id, &self.id_remap), ind), tensor);
    }

    pub fn set_dyn_dim(&mut self, dim: char, val: usize) {
        self.dyn_map.insert(dim, val);
    }

    pub fn new_tensor<S: Shape>(&mut self, name: &str) -> GraphTensor<S> {
        self.graph.free_node = NodeIndex::end(); // Prevent reuse of deleted indexes (screws up remapping)
        GraphTensor {
            id: self.graph.add_node(Box::new(op::Function(
                format!("{name} Load"),
                Box::new(|_| panic!("You must set a value for this tensor!")),
                std::any::TypeId::of::<Vec<f32>>(),
            ))),
            graph_ref: self,
            shape: S::to_tracker(),
            _phantom: Default::default(),
        }
    }

    /// Compile the graph using the given compiler
    pub fn compile<C: Compiler>(&mut self, compiler: C) {
        compiler.compile(self);
        self.toposort();
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
    }

    /// Clear any remaining tensors that may be around from old executions
    pub fn reset(&mut self) {
        // (This is where we should do the tensor caching!)
        for (t, i) in self.tensors.keys().copied().collect_vec() {
            if !self.no_delete.contains(&t) {
                self.tensors.remove(&(t, i));
            }
        }
    }

    /// Swap the tensors with these ids
    pub fn swap_tensors<A: Shape, B: Shape>(&mut self, a: GraphTensor<A>, b: GraphTensor<B>) {
        // Swap tensors
        let a_t = self.tensors.remove(&(a.id(), 0)); // Assume 0th output for now
        let b_t = self.tensors.remove(&(b.id(), 0));
        if let Some(a_t) = a_t {
            self.tensors.insert((b.id(), 0), a_t);
        }
        if let Some(b_t) = b_t {
            self.tensors.insert((a.id(), 0), b_t);
        }
    }

    fn create_remaining_customers_map(&self) -> HashMap<(NodeIndex, u8), usize> {
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
            .collect()
    }

    /// Execute the graph.
    pub fn execute(&mut self) {
        // Track the number of views pointing to each tensor so we know when to clear
        if self.linearized_graph.is_none() {
            self.toposort();
        }
        let mut remaining_consumers = self.create_remaining_customers_map();

        for (node, src_ids) in self.linearized_graph.as_ref().unwrap() {
            if self.tensors.contains_key(&(*node, 0)) {
                continue;
            }

            let mut srcs = get_source_tensors(
                &self.no_delete,
                &mut self.tensors,
                src_ids,
                &remaining_consumers,
            );

            // Substitute in the dyn dims
            for (_, st) in srcs.iter_mut() {
                *st = st.resolve_global_dyn_dims(&self.dyn_map);
            }

            // Execute
            let tensors = self.graph.node_weight(*node).unwrap().process(srcs);
            for (i, tensor) in tensors.into_iter().enumerate() {
                self.tensors.insert((*node, i as u8), tensor);
            }

            // Check if we can delete the source tensors now
            for (source, _) in src_ids {
                *remaining_consumers.get_mut(source).unwrap() -= 1;
            }
        }
        self.reset();
    }

    /// Execute the graph with debug prints
    pub fn execute_debug(&mut self) {
        // Track the number of views pointing to each tensor so we know when to clear
        if self.linearized_graph.is_none() {
            self.toposort();
        }
        let mut remaining_consumers = self.create_remaining_customers_map();
        let mut op_times = HashMap::new();

        println!(
            "{:->2$} Executing {:->2$}",
            "",
            "",
            (term_size::dimensions().unwrap().0 - " Executing ".len()) / 2
        );
        let start = std::time::Instant::now();
        for (node, src_ids) in self.linearized_graph.as_ref().unwrap().iter() {
            if self.tensors.contains_key(&(*node, 0)) {
                continue;
            }
            let op_name = format!("{:?}", self.graph.node_weight(*node).unwrap());
            print!("{}", op_name.bold().bright_green());

            let mut srcs = get_source_tensors(
                &self.no_delete,
                &mut self.tensors,
                src_ids,
                &remaining_consumers,
            );

            // Substitute in the dyn dims
            for (_, st) in srcs.iter_mut() {
                *st = st.resolve_global_dyn_dims(&self.dyn_map);
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
            let tensors = self.graph.node_weight(*node).unwrap().process(srcs);
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
                term_size::dimensions().unwrap().0 - op_name.len() - shapes_string.len(),
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
            (term_size::dimensions().unwrap().0 - " Total Times ".len()) / 2
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
                term_size::dimensions().unwrap().0 - name.len(),
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

    /// Execute the graph without deleting intermediate tensors
    pub fn execute_no_delete(&mut self) {
        // Track the number of views pointing to each tensor so we know when to clear;
        if self.linearized_graph.is_none() {
            self.toposort();
        }
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
                *st = st.resolve_global_dyn_dims(&self.dyn_map);
            }

            // All sources are ready, execute
            let tensors = self.graph.node_weight(*node).unwrap().process(srcs);
            for (i, tensor) in tensors.into_iter().enumerate() {
                self.tensors.insert((*node, i as u8), tensor);
            }
        }
    }
}

fn get_source_tensors<'a>(
    no_delete: &HashSet<NodeIndex>,
    tensors: &'a mut HashMap<(NodeIndex, u8), Tensor>,
    src_ids: &[((NodeIndex, u8), ShapeTracker)],
    remaining_consumers: &HashMap<(NodeIndex, u8), usize>,
) -> Vec<(InputTensor<'a>, ShapeTracker)> {
    // This needs to be done in a weird way with sperate queues to satisfy the borrow checker (all mutable calls to self.tensors happen before references are taken)
    let mut owned = VecDeque::default();
    let mut refs = VecDeque::default();
    for (i, (id, _)) in src_ids.iter().enumerate() {
        if remaining_consumers[id] == 1 && !no_delete.contains(&id.0) {
            owned.push_back((InputTensor::Owned(tensors.remove(id).unwrap()), i));
        }
    }
    for (i, (id, _)) in src_ids.iter().enumerate() {
        if remaining_consumers[id] != 1 || no_delete.contains(&id.0) {
            refs.push_back((InputTensor::Borrowed(tensors.get(id).unwrap()), i));
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
    srcs
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
