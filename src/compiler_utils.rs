// Lots of utilities used by compilers

use std::{any::TypeId, borrow::Borrow, collections::HashSet, fmt::Debug, sync::Arc};

use colored::Colorize;
use itertools::Itertools;
use petgraph::{
    algo::toposort,
    stable_graph::{EdgeIndex, EdgeReference, StableGraph},
    visit::EdgeRef,
    Directed, Direction,
};
use regex::Regex;
use rustc_hash::{FxHashMap, FxHashSet};
use uuid::Uuid;

use crate::prelude::*;

#[derive(Clone, Debug)]
pub enum GraphTerm {
    GMEM {
        // Signifies global memory
        label: Option<String>,
    },
    LoopIn {
        range: Expression,
        stride: Expression,
    },
    LoopOut {
        range: Expression,
        stride: Expression,
    },
    NewAcc {
        starting_value: String,
    },
    Add,
    Mul,
    Max,
    Exp,
    Recip,
    Sin,
    Neg,
    SMEM,     // Signifies shared memory
    SMEMLoad, // Takes in an smem pointer and a gmem pointer, copies the gmem element to smem and returns the smem pointer
    SMEMRead, // Takes in an smem pointer and an smemload, returns the smem pointer
    Constant {
        val: Expression,
    },
    Sqrt,
    LessThan,
}

impl std::fmt::Display for GraphTerm {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GraphTerm::GMEM { label } => {
                if let Some(label) = label {
                    write!(f, "GMEM({label})")
                } else {
                    write!(f, "GMEM")
                }
            }
            GraphTerm::LoopIn { range, stride } => {
                write!(f, "LoopIn(range: {range:?}, stride: {stride:?})")
            }
            GraphTerm::LoopOut { range, stride } => {
                write!(f, "LoopOut(range: {range:?}, stride: {stride:?})")
            }
            GraphTerm::NewAcc { starting_value } => write!(f, "NewAcc({starting_value})"),
            GraphTerm::Add => write!(f, "Add"),
            GraphTerm::Mul => write!(f, "Mul"),
            GraphTerm::Max => write!(f, "Max"),
            GraphTerm::Exp => write!(f, "Exp"),
            GraphTerm::Recip => write!(f, "Recip"),
            GraphTerm::Sin => write!(f, "Sin"),
            GraphTerm::Neg => write!(f, "Neg"),
            GraphTerm::SMEM => write!(f, "SMEM"),
            GraphTerm::SMEMLoad => write!(f, "SMEMLoad"),
            GraphTerm::SMEMRead => write!(f, "SMEMRead"),
            GraphTerm::Constant { val } => {
                write!(f, "Constant(val: {val:?})")
            }
            GraphTerm::Sqrt => write!(f, "Sqrt"),
            GraphTerm::LessThan => write!(f, "LessThan"),
        }
    }
}

fn calculate_broadcast_shape(shape1: &ShapeTracker, shape2: &ShapeTracker) -> ShapeTracker {
    let dims1 = shape1.dims();
    let dims2 = shape2.dims();

    // Get the maximum number of dimensions
    let max_dims = dims1.len().max(dims2.len());
    let mut result_dims = Vec::with_capacity(max_dims);

    // Pad the shorter shape with 1s at the beginning (numpy-style broadcasting)
    let padded_dims1: Vec<_> = if dims1.len() < max_dims {
        std::iter::repeat(Expression::from(1))
            .take(max_dims - dims1.len())
            .chain(dims1.iter().cloned())
            .collect()
    } else {
        dims1
    };

    let padded_dims2: Vec<_> = if dims2.len() < max_dims {
        std::iter::repeat(Expression::from(1))
            .take(max_dims - dims2.len())
            .chain(dims2.iter().cloned())
            .collect()
    } else {
        dims2
    };

    // Calculate broadcast dimensions
    for i in 0..max_dims {
        let dim1 = &padded_dims1[i];
        let dim2 = &padded_dims2[i];

        // Try to convert to usize for comparison
        let dim1_usize = dim1.to_usize();
        let dim2_usize = dim2.to_usize();

        match (dim1_usize, dim2_usize) {
            (Some(d1), Some(d2)) => {
                if d1 == d2 {
                    result_dims.push(*dim1);
                } else if d1 == 1 {
                    result_dims.push(*dim2);
                } else if d2 == 1 {
                    result_dims.push(*dim1);
                } else {
                    // Broadcasting error - incompatible dimensions
                    panic!("Cannot broadcast shapes: dimension {i} has size {d1} vs {d2}");
                }
            }
            // If we can't convert to usize (dynamic dimensions), we need to handle symbolically
            (None, Some(1)) => {
                result_dims.push(*dim1);
            }
            (Some(1), None) => {
                result_dims.push(*dim2);
            }
            (None, None) => {
                // Both are symbolic - assume they're compatible and take the first one
                // In a more sophisticated implementation, you might want to create a max() expression
                result_dims.push(*dim1);
            }
            _ => {
                // One is symbolic, one is not 1 - assume compatible and take the non-1 dimension
                if dim1_usize == Some(1) {
                    result_dims.push(*dim2);
                } else if dim2_usize == Some(1) {
                    result_dims.push(*dim1);
                } else {
                    // Both are non-1, take the first one and hope for the best
                    result_dims.push(*dim1);
                }
            }
        }
    }

    // Create a new ShapeTracker from the result dimensions
    ShapeTracker::new(result_dims)
}

pub trait ToIdsMut {
    fn to_ids_mut(&mut self) -> Vec<&mut NodeIndex>;
}

pub trait ToIds {
    fn to_ids(&self) -> Vec<NodeIndex>;
}

pub trait ToId {
    fn to_id(&self) -> NodeIndex;
}

impl ToId for GraphTensor {
    fn to_id(&self) -> NodeIndex {
        self.id
    }
}

impl ToId for NodeIndex {
    fn to_id(&self) -> NodeIndex {
        *self
    }
}

impl ToIdsMut for GraphTensor {
    fn to_ids_mut(&mut self) -> Vec<&mut NodeIndex> {
        vec![&mut self.id]
    }
}
impl ToIds for GraphTensor {
    fn to_ids(&self) -> Vec<NodeIndex> {
        vec![self.id]
    }
}
impl<T: ToIdsMut> ToIdsMut for Vec<T> {
    fn to_ids_mut(&mut self) -> Vec<&mut NodeIndex> {
        self.iter_mut().flat_map(|i| i.to_ids_mut()).collect()
    }
}
impl<T: ToIds> ToIds for Vec<T> {
    fn to_ids(&self) -> Vec<NodeIndex> {
        self.iter().flat_map(|i| i.to_ids()).collect()
    }
}
impl<T: ToIdsMut> ToIdsMut for &mut [T] {
    fn to_ids_mut(&mut self) -> Vec<&mut NodeIndex> {
        self.iter_mut().flat_map(|i| i.to_ids_mut()).collect()
    }
}
impl ToIdsMut for &mut Vec<NodeIndex> {
    fn to_ids_mut(&mut self) -> Vec<&mut NodeIndex> {
        self.iter_mut().collect()
    }
}
impl ToIdsMut for &mut [NodeIndex] {
    fn to_ids_mut(&mut self) -> Vec<&mut NodeIndex> {
        self.iter_mut().collect()
    }
}
impl<T: ToIds> ToIds for &mut [T] {
    fn to_ids(&self) -> Vec<NodeIndex> {
        self.iter().flat_map(|i| i.to_ids()).collect()
    }
}

impl<T: ToIdsMut> ToIdsMut for &mut T {
    fn to_ids_mut(&mut self) -> Vec<&mut NodeIndex> {
        (*self).to_ids_mut()
    }
}
impl<T: ToIds> ToIds for &T {
    fn to_ids(&self) -> Vec<NodeIndex> {
        <T as ToIds>::to_ids(*self)
    }
}
impl ToIds for NodeIndex {
    fn to_ids(&self) -> Vec<NodeIndex> {
        vec![*self]
    }
}
impl ToIdsMut for &mut NodeIndex {
    fn to_ids_mut(&mut self) -> Vec<&mut NodeIndex> {
        vec![self]
    }
}
impl ToIdsMut for () {
    fn to_ids_mut(&mut self) -> Vec<&mut NodeIndex> {
        vec![]
    }
}
impl ToIds for () {
    fn to_ids(&self) -> Vec<NodeIndex> {
        vec![]
    }
}

impl<T: ToIds> ToIds for FxHashMap<String, T> {
    fn to_ids(&self) -> Vec<NodeIndex> {
        self.values().flat_map(|i| i.to_ids()).collect()
    }
}

impl ToIds for (NodeIndex, ShapeTracker) {
    fn to_ids(&self) -> Vec<NodeIndex> {
        vec![self.0]
    }
}

impl ToIdsMut for (NodeIndex, ShapeTracker) {
    fn to_ids_mut(&mut self) -> Vec<&mut NodeIndex> {
        vec![&mut self.0]
    }
}

macro_rules! tuple_impls {
    ([$($name:ident),+] , [$($idx:tt),+]) => {
        impl<
        $($name:
            ToIdsMut, )+
        > ToIdsMut for ($($name,)+) {
            fn to_ids_mut(&mut self) -> Vec<&mut NodeIndex> {
                let mut v = vec![];
                $(v.append(&mut self.$idx.to_ids_mut());)+
                v
            }
        }
        impl<
        $($name:
            ToIds, )+
        > ToIds for ($($name,)+) {
            fn to_ids(&self) -> Vec<NodeIndex> {
                let mut v = vec![];
                $(v.append(&mut self.$idx.to_ids());)+
                v
            }
        }
    };
}

tuple_impls!([M1], [0]);
tuple_impls!([M1, M2], [0, 1]);
tuple_impls!([M1, M2, M3], [0, 1, 2]);
tuple_impls!([M1, M2, M3, M4], [0, 1, 2, 3]);
tuple_impls!([M1, M2, M3, M4, M5], [0, 1, 2, 3, 4]);
tuple_impls!([M1, M2, M3, M4, M5, M6], [0, 1, 2, 3, 4, 5]);
tuple_impls!([M1, M2, M3, M4, M5, M6, M7], [0, 1, 2, 3, 4, 5, 6]);
tuple_impls!([M1, M2, M3, M4, M5, M6, M7, M8], [0, 1, 2, 3, 4, 5, 6, 7]);
tuple_impls!(
    [M1, M2, M3, M4, M5, M6, M7, M8, M9],
    [0, 1, 2, 3, 4, 5, 6, 7, 8]
);
tuple_impls!(
    [M1, M2, M3, M4, M5, M6, M7, M8, M9, M10],
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
);

pub trait Compiler {
    type Output;
    /// Run a compilation pass
    fn compile<T: ToIdsMut>(&self, graph: &mut Graph, ids: T) -> Self::Output;
}

impl Compiler for () {
    type Output = ();
    fn compile<T: ToIdsMut>(&self, _: &mut Graph, _: T) {}
}

/// Wrap this around a compiler to rerun the compiler until it doesn't change the graph anymore
#[derive(Debug)]
pub struct Looped<C: Compiler + Debug>(C);

impl<C: Compiler + Debug> Compiler for Looped<C> {
    type Output = ();
    fn compile<T: ToIdsMut>(&self, graph: &mut Graph, mut remap: T) {
        let mut linearized = None;
        loop {
            self.0.compile(graph, &mut remap);
            graph.toposort();
            if linearized == graph.linearized_graph {
                break;
            }
            linearized.clone_from(&graph.linearized_graph);
        }
    }
}

impl<C: Default + Compiler + Debug> Default for Looped<C> {
    fn default() -> Self {
        Self(C::default())
    }
}

/// Wrap this around a compiler to measure the time it takes to compile
#[derive(Debug)]
pub struct Timed<C: Compiler + Debug>(pub C);

impl<C: Compiler + Debug> Compiler for Timed<C> {
    type Output = ();
    fn compile<T: ToIdsMut>(&self, graph: &mut Graph, remap: T) {
        let compiler_name = format!("{:?}", self.0).bold();
        println!("Starting {compiler_name}");
        let start = std::time::Instant::now();
        self.0.compile(graph, remap);
        let finished_millis = start.elapsed().as_millis();
        let minutes = finished_millis / 60_000;
        let seconds = (finished_millis % 60_000) / 1000;
        let millis = finished_millis % 1000;
        println!(
            "Finished {compiler_name} in {}",
            if minutes > 0 {
                format!("{minutes}m {seconds}s {millis}ms")
            } else if seconds > 0 {
                format!("{seconds}s {millis}ms")
            } else {
                format!("{millis}ms")
            }
            .bold()
        );
    }
}

impl<C: Default + Compiler + Debug> Default for Timed<C> {
    fn default() -> Self {
        Self(C::default())
    }
}

macro_rules! tuple_impls {
    ([$($name:ident),+] , [$($idx:tt),+]) => {
        impl<
        $($name:
            Compiler, )+
        > Compiler for ($($name,)+) {
            type Output = ( $($name::Output, )+ );
            fn compile<T: ToIdsMut>(&self, graph: &mut Graph, mut remap: T) -> Self::Output {
                ( $(self.$idx.compile(graph, &mut remap), )+ )
            }
        }
    };
}

tuple_impls!([M1], [0]);
tuple_impls!([M1, M2], [0, 1]);
tuple_impls!([M1, M2, M3], [0, 1, 2]);
tuple_impls!([M1, M2, M3, M4], [0, 1, 2, 3]);
tuple_impls!([M1, M2, M3, M4, M5], [0, 1, 2, 3, 4]);
tuple_impls!([M1, M2, M3, M4, M5, M6], [0, 1, 2, 3, 4, 5]);
tuple_impls!([M1, M2, M3, M4, M5, M6, M7], [0, 1, 2, 3, 4, 5, 6]);
tuple_impls!([M1, M2, M3, M4, M5, M6, M7, M8], [0, 1, 2, 3, 4, 5, 6, 7]);
tuple_impls!(
    [M1, M2, M3, M4, M5, M6, M7, M8, M9],
    [0, 1, 2, 3, 4, 5, 6, 7, 8]
);
tuple_impls!(
    [M1, M2, M3, M4, M5, M6, M7, M8, M9, M10],
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
);
tuple_impls!(
    [M1, M2, M3, M4, M5, M6, M7, M8, M9, M10, M11],
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
);
tuple_impls!(
    [M1, M2, M3, M4, M5, M6, M7, M8, M9, M10, M11, M12],
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
);
tuple_impls!(
    [M1, M2, M3, M4, M5, M6, M7, M8, M9, M10, M11, M12, M13],
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
);
tuple_impls!(
    [M1, M2, M3, M4, M5, M6, M7, M8, M9, M10, M11, M12, M13, M14],
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
);
tuple_impls!(
    [M1, M2, M3, M4, M5, M6, M7, M8, M9, M10, M11, M12, M13, M14, M15],
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
);
tuple_impls!(
    [M1, M2, M3, M4, M5, M6, M7, M8, M9, M10, M11, M12, M13, M14, M15, M16],
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
);
tuple_impls!(
    [M1, M2, M3, M4, M5, M6, M7, M8, M9, M10, M11, M12, M13, M14, M15, M16, M17],
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
);
tuple_impls!(
    [M1, M2, M3, M4, M5, M6, M7, M8, M9, M10, M11, M12, M13, M14, M15, M16, M17, M18],
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
);
tuple_impls!(
    [M1, M2, M3, M4, M5, M6, M7, M8, M9, M10, M11, M12, M13, M14, M15, M16, M17, M18, M19],
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
);
tuple_impls!(
    [M1, M2, M3, M4, M5, M6, M7, M8, M9, M10, M11, M12, M13, M14, M15, M16, M17, M18, M19, M20],
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
);

// Helpers

impl Graph {
    /// Add op on the graph, and get back a NewOp
    ///
    /// ```rust
    /// use luminal::prelude::*;
    /// let mut cx = Graph::new();
    /// let a = cx.tensor(3);
    /// let b_id = cx
    ///     .add_op(luminal::op::Mul)
    ///     .input(a.id, 0, a.shape)
    ///     .finish();
    /// let b = GraphTensor::from_id(b_id, a.shape, a.graph());
    /// ```
    pub fn add_op<O: Operator + 'static>(&mut self, op: O) -> NewOp {
        self.linearized_graph = None;
        NewOp {
            new_op_id: self.graph.add_node(Box::new(op)),
            graph_ref: self,
            num_srcs: 0,
        }
    }
    /// Add op on the graph, and get back a NewOp. Just like add_op, except a boxed op is expected.
    pub fn add_boxed_op(&mut self, op: Box<dyn Operator + 'static>) -> NewOp {
        self.linearized_graph = None;
        NewOp {
            new_op_id: self.graph.add_node(op),
            graph_ref: self,
            num_srcs: 0,
        }
    }
    /// Create a schedule dependency between a and b
    pub fn add_schedule_dependency(&mut self, a: NodeIndex, b: NodeIndex) {
        self.graph.add_edge(a, b, Dependency::Schedule);
    }

    /// Run the custom function on a node and get an output
    pub fn node_custom<O: 'static, I: 'static>(
        &mut self,
        node: NodeIndex,
        key: &str,
        input: I,
    ) -> Option<O> {
        let node_weight = self.graph.node_weight_mut(node)?;

        node_weight
            .custom(key, Box::new(input))
            .and_then(|o| o.downcast::<O>().ok().map(|o| *o))
    }

    /// Convert to debug-viewable graph
    fn debug_graph(
        &self,
        show_shapes: bool,
    ) -> (
        StableGraph<String, u8>,
        Vec<EdgeIndex>,
        FxHashMap<NodeIndex, NodeIndex>,
    ) {
        let mut new_graph = StableGraph::default();
        let mut id_map = FxHashMap::default();
        for (id, node) in self.graph.node_indices().zip(self.graph.node_weights()) {
            id_map.insert(id, new_graph.add_node(format!("{node:?} | {}", id.index())));
        }

        let mut schedule_edges = vec![];
        for node in self.graph.node_indices() {
            for edge in self
                .graph
                .edges_directed(node, Direction::Outgoing)
                .sorted_by_key(|e| {
                    if let Some(d) = e.weight().as_data() {
                        d.0
                    } else {
                        0
                    }
                })
            {
                let new_edge = new_graph.add_edge(
                    id_map[&edge.source()],
                    id_map[&edge.target()],
                    if let Some(d) = edge.weight().as_data() {
                        d.0
                    } else {
                        0
                    },
                );
                if edge.weight().is_schedule() {
                    schedule_edges.push(new_edge);
                }
                if show_shapes
                    && new_graph.contains_node(id_map[&edge.target()])
                    && edge
                        .weight()
                        .as_data()
                        .map(|d| !d.2.is_empty())
                        .unwrap_or_default()
                {
                    new_graph
                        .node_weight_mut(id_map[&edge.target()])
                        .unwrap()
                        .push_str(&format!(
                            " | {:?}",
                            edge.weight().as_data().unwrap().2.dims()
                        ));
                }
            }
        }

        (new_graph, schedule_edges, id_map)
    }

    pub fn check_node_type<T: Operator + 'static>(&self, node: NodeIndex) -> bool {
        self.node_weight(node)
            .expect("Node not found in graph!")
            .as_any()
            .is::<T>()
    }

    pub fn display(&self) {
        let (g, e, _) = self.debug_graph(false);
        display_graph(&g, &e, &[]);
    }

    pub fn translate_to_2(&self) -> StableGraph<GraphTerm, u8, Directed> {
        let mut new_graph: StableGraph<GraphTerm, u8, Directed> = StableGraph::new();
        let mut node_mapping: FxHashMap<NodeIndex, NodeIndex> = FxHashMap::default();
        let mut manually_handled_nodes: FxHashSet<NodeIndex> = FxHashSet::default();

        let topo_order =
            toposort(&self.graph, None).expect("Graph has a cycle, cannot be translated");

        for node_idx in topo_order {
            if manually_handled_nodes.contains(&node_idx) {
                continue;
            }

            let node_weight = self.graph.node_weight(node_idx).unwrap();
            let op_name_full = format!("{node_weight:?}");
            let op = op_name_full
                .split('|')
                .next()
                .unwrap_or(&op_name_full)
                .trim();

            let sources = self.get_sources(node_idx);
            let incoming_source_ids: Vec<_> = sources.iter().map(|(id, _, _)| *id).collect();
            let incoming_edges: Vec<_> = sources.iter().map(|(_, _, shape)| *shape).collect();

            manually_handled_nodes.insert(node_idx);

            let new_final_node = match op {
                s if s.ends_with("Load") => new_graph.add_node(GraphTerm::GMEM {
                    label: Some(s.to_string()),
                }),

                "Contiguous" => *node_mapping.get(&incoming_source_ids[0]).unwrap(),

                s if s.starts_with("SumReduce") || s.starts_with("MaxReduce") => {
                    let is_sum = s.starts_with("SumReduce");
                    let (op_term, acc_val) = if is_sum {
                        (GraphTerm::Add, "0.0")
                    } else {
                        (GraphTerm::Max, "-inf")
                    };

                    let dim_str = &s[s.find('(').unwrap() + 1..s.find(')').unwrap()];
                    let reduce_dim_idx = dim_str
                        .parse::<usize>()
                        .expect("Failed to parse reduction dimension");

                    let input_shape = &incoming_edges[0];
                    let mapped_input_node = *node_mapping.get(&incoming_source_ids[0]).unwrap();
                    let input_dims = input_shape.dims();

                    // --- Strides for the main input loop chain (normal traversal) ---
                    let input_strides = calculate_strides(&input_dims, &input_shape.indexes);

                    // --- Strides for the accumulator loop chain ---
                    // These are based on the INPUT dimensions, with 'z' propagation.
                    let acc_strides = calculate_reduction_strides(&input_dims, reduce_dim_idx);

                    // --- Strides for the output loop chain ---
                    // First, define the output dimensions (reduced dim is size 1).
                    let mut output_dims = input_dims.to_vec();
                    if reduce_dim_idx < output_dims.len() {
                        output_dims[reduce_dim_idx] = Expression::from(1);
                    }
                    // Now, calculate strides based on these new OUTPUT dimensions.
                    let loop_out_strides =
                        calculate_reduction_strides(&output_dims, reduce_dim_idx);

                    let acc_node = new_graph.add_node(GraphTerm::NewAcc {
                        starting_value: acc_val.to_string(),
                    });
                    let reduce_op_node = new_graph.add_node(op_term);

                    // Create loop chain for the full-shaped input
                    let mut last_node = mapped_input_node;
                    for i in 0..input_dims.len() {
                        let range = Expression::from(input_dims[i].to_usize().unwrap_or(1) as i32);
                        let loop_in = new_graph.add_node(GraphTerm::LoopIn {
                            range,
                            stride: input_strides[i],
                        });
                        new_graph.add_edge(last_node, loop_in, 0);
                        last_node = loop_in;
                    }
                    new_graph.add_edge(last_node, reduce_op_node, 0);

                    // Create loop chain for the accumulator, iterating over the full input space
                    last_node = acc_node;
                    for i in 0..input_dims.len() {
                        let range = Expression::from(input_dims[i].to_usize().unwrap_or(1) as i32);
                        let loop_in = new_graph.add_node(GraphTerm::LoopIn {
                            range,
                            stride: acc_strides[i],
                        });
                        new_graph.add_edge(last_node, loop_in, 0);
                        last_node = loop_in;
                    }
                    new_graph.add_edge(last_node, reduce_op_node, 1);

                    // Create output loop chain with correct ranges and strides
                    last_node = reduce_op_node;
                    for i in 0..output_dims.len() {
                        let range = Expression::from(output_dims[i].to_usize().unwrap_or(1) as i32);
                        let loop_out = new_graph.add_node(GraphTerm::LoopOut {
                            range,
                            // Use the new strides calculated from the output dimensions
                            stride: loop_out_strides[i],
                        });
                        new_graph.add_edge(last_node, loop_out, 0);
                        last_node = loop_out;
                    }
                    last_node
                }

                s if s.starts_with("Constant") => {
                    let val_str = &s[s.find('(').unwrap() + 1..s.find(')').unwrap()];
                    let value = val_str.parse::<f32>().unwrap_or(0.0);
                    new_graph.add_node(GraphTerm::Constant {
                        val: Expression::from(value as i32),
                    })
                }

                _ => {
                    // Fallback for Unary, Binary, etc.
                    match op {
                        s if s.starts_with("Exp")
                            || matches!(s, "Recip" | "Sin" | "Neg" | "Sqrt") =>
                        {
                            let op_term = if s.starts_with("Exp") {
                                GraphTerm::Exp
                            } else {
                                match s {
                                    "Recip" => GraphTerm::Recip,
                                    "Sin" => GraphTerm::Sin,
                                    "Neg" => GraphTerm::Neg,
                                    "Sqrt" => GraphTerm::Sqrt,
                                    _ => unreachable!(),
                                }
                            };
                            let shape = &incoming_edges[0];
                            let dims = shape.dims();
                            let strides = calculate_strides(&dims, &shape.indexes);
                            let mut last_node = *node_mapping.get(&incoming_source_ids[0]).unwrap();
                            for i in 0..dims.len() {
                                let loop_in = new_graph.add_node(GraphTerm::LoopIn {
                                    range: (dims[i].to_usize().unwrap_or(1) as i32).into(),
                                    stride: strides[i],
                                });
                                new_graph.add_edge(last_node, loop_in, 0);
                                last_node = loop_in;
                            }
                            let op_node = new_graph.add_node(op_term);
                            new_graph.add_edge(last_node, op_node, 0);
                            last_node = op_node;
                            for i in 0..dims.len() {
                                let loop_out = new_graph.add_node(GraphTerm::LoopOut {
                                    range: (dims[i].to_usize().unwrap_or(1) as i32).into(),
                                    stride: strides[i],
                                });
                                new_graph.add_edge(last_node, loop_out, 0);
                                last_node = loop_out;
                            }
                            last_node
                        }
                        "Add" | "Mul" | "Max" | "LessThan" => {
                            let op_term = match op {
                                "Add" => GraphTerm::Add,
                                "Mul" => GraphTerm::Mul,
                                "Max" => GraphTerm::Max,
                                "LessThan" => GraphTerm::LessThan,
                                _ => unreachable!(),
                            };
                            let mut op_inputs = Vec::new();
                            for (i, shape) in incoming_edges.iter().enumerate() {
                                let dims = shape.dims();
                                let strides = calculate_strides(&dims, &shape.indexes);
                                let mut last_node =
                                    *node_mapping.get(&incoming_source_ids[i]).unwrap();
                                for i in 0..dims.len() {
                                    let loop_in = new_graph.add_node(GraphTerm::LoopIn {
                                        range: (dims[i].to_usize().unwrap_or(1) as i32).into(),
                                        stride: strides[i],
                                    });
                                    new_graph.add_edge(last_node, loop_in, 0);
                                    last_node = loop_in;
                                }
                                op_inputs.push(last_node);
                            }
                            let binary_op_node = new_graph.add_node(op_term);
                            for (i, &loop_node) in op_inputs.iter().enumerate() {
                                new_graph.add_edge(loop_node, binary_op_node, i as u8);
                            }
                            let mut last_node = binary_op_node;
                            let mut output_shape = incoming_edges
                                .first()
                                .copied()
                                .unwrap_or_else(|| ShapeTracker::new(Vec::<Expression>::new()));
                            for shape in incoming_edges.iter().skip(1) {
                                output_shape = calculate_broadcast_shape(&output_shape, shape);
                            }
                            let dims = output_shape.dims();
                            let strides = calculate_strides(&dims, &output_shape.indexes);
                            for i in 0..dims.len() {
                                let loop_out = new_graph.add_node(GraphTerm::LoopOut {
                                    range: (dims[i].to_usize().unwrap_or(1) as i32).into(),
                                    stride: strides[i],
                                });
                                new_graph.add_edge(last_node, loop_out, 0);
                                last_node = loop_out;
                            }
                            last_node
                        }
                        _ => new_graph.add_node(GraphTerm::GMEM {
                            label: Some(format!("Unhandled Op: {op}")),
                        }),
                    }
                }
            };
            node_mapping.insert(node_idx, new_final_node);
        }

        let mut graph_to_display: StableGraph<String, u8, Directed> = StableGraph::new();
        let mut display_node_mapping: FxHashMap<NodeIndex, NodeIndex> = FxHashMap::default();

        for node_idx in new_graph.node_indices() {
            if let Some(graph_term) = new_graph.node_weight(node_idx) {
                let display_node = graph_to_display.add_node(graph_term.to_string());
                display_node_mapping.insert(node_idx, display_node);
            }
        }

        for edge in new_graph.edge_indices() {
            if let Some((source, target)) = new_graph.edge_endpoints(edge) {
                if let Some(weight) = new_graph.edge_weight(edge) {
                    if let (Some(&display_source), Some(&display_target)) = (
                        display_node_mapping.get(&source),
                        display_node_mapping.get(&target),
                    ) {
                        graph_to_display.add_edge(display_source, display_target, *weight);
                    }
                }
            }
        }

        display_graph(&graph_to_display, &[], &[]);
        new_graph
    }

    pub fn display_shapes(&self) {
        let (g, e, _) = self.debug_graph(true);
        display_graph(&g, &e, &[]);
    }

    pub fn display_set<T: ToIds>(&self, set: T) {
        let (g, e, id_map) = self.debug_graph(false);
        display_graph(
            &g,
            &e,
            &set.to_ids().iter().map(|i| id_map[i]).collect::<Vec<_>>(),
        );
    }

    /// Remove node if it only has n dests
    pub fn safe_remove_node(&mut self, node: NodeIndex, dests: usize) {
        if self
            .graph
            .edges_directed(node, Direction::Outgoing)
            .filter(|e| !e.weight().is_schedule())
            .count()
            <= dests
        {
            self.graph.remove_node(node);
        }
    }

    /// Get the sources of a node given it's id
    pub fn get_sources(&self, node_id: NodeIndex) -> Vec<(NodeIndex, u8, ShapeTracker)> {
        self.graph
            .edges_directed(node_id, Direction::Incoming)
            .filter_map(|e| e.weight().as_data().map(|i| (e.source(), i)))
            .sorted_by_key(|(_, (i, _, _))| *i)
            .map(|(a, (_, c, b))| (a, c, b))
            .collect()
    }

    /// Get the dests of a node given it's id
    #[allow(clippy::borrowed_box)]
    pub fn get_dests(&self, node_id: NodeIndex) -> Vec<(NodeIndex, &Box<dyn Operator>)> {
        self.graph
            .edges_directed(node_id, Direction::Outgoing)
            .filter_map(|e| e.weight().as_data().map(|i| (e.target(), i)))
            .sorted_by_key(|(_, (i, _, _))| *i)
            .map(|(a, _)| (a, self.graph.node_weight(a).unwrap()))
            .collect()
    }

    pub fn try_get_op<T: Operator + 'static>(&self, node: NodeIndex) -> Option<&T> {
        self.node_weight(node).unwrap().as_any().downcast_ref::<T>()
    }
    pub fn get_op<T: Operator + 'static>(&self, node: NodeIndex) -> &T {
        self.try_get_op(node).unwrap()
    }
    pub fn try_get_op_mut<T: Operator + 'static>(&mut self, node: NodeIndex) -> Option<&mut T> {
        self.node_weight_mut(node)
            .unwrap()
            .as_any_mut()
            .downcast_mut::<T>()
    }
    pub fn get_op_mut<T: Operator + 'static>(&mut self, node: NodeIndex) -> &mut T {
        self.try_get_op_mut(node).unwrap()
    }
}

/// Calculates strides for a reduction accumulator, propagating 'z' symbolically.
/// For a tensor (5,4,3) reduced on dim 1, it produces strides (4*z, z, 1).
fn calculate_reduction_strides(dims: &[Expression], reduce_dim_idx: usize) -> Vec<Expression> {
    let n = dims.len();
    let mut strides = vec![Expression::from(1); n];
    if n == 0 {
        return strides;
    }

    // Phase 1: Handle the reduction axis and all axes to its left.
    // Start with 'z' for the reduction dim, then multiply by dimension sizes moving left.
    let mut current_stride = Expression::from('z');
    for i in (0..=reduce_dim_idx).rev() {
        strides[i] = current_stride;
        if let Some(dim_size) = dims[i].to_usize() {
            current_stride *= dim_size;
        }
    }

    // Phase 2: Handle all axes to the right of the reduction axis (these are standard numeric strides).
    // Start with a stride of 1 for the rightmost dimension.
    let mut current_stride = Expression::from(1);
    for i in (reduce_dim_idx + 1..n).rev() {
        strides[i] = current_stride;
        if let Some(dim_size) = dims[i].to_usize() {
            current_stride *= dim_size;
        }
    }

    strides
}

// Place this helper function back in your file, for example, before `translate_to_2`.
fn calculate_strides(dims: &[Expression], indexes: &[usize]) -> Vec<Expression> {
    if dims.is_empty() || indexes.len() != dims.len() {
        return vec![Expression::from(1); dims.len()];
    }

    // Step 1: Calculate physical strides based on physical dimension order
    let mut physical_strides = vec![Expression::from(1); dims.len()];
    let mut current_stride = Expression::from(1);
    for i in (0..dims.len()).rev() {
        physical_strides[i] = current_stride;
        // Update stride for next dimension (multiply by current dimension size)
        if let Some(d) = dims[i].to_usize() {
            if d > 1 {
                current_stride *= d;
            }
        }
    }

    // Step 2: Reorder physical strides to logical order using indexes
    let mut logical_strides = vec![Expression::from(1); dims.len()];
    for (logical_dim, &physical_dim) in indexes.iter().enumerate() {
        if physical_dim < dims.len() {
            logical_strides[logical_dim] = physical_strides[physical_dim];
        }
    }

    logical_strides
}

/// View a debug graph in the browser
pub fn display_graph(
    graph: &petgraph::stable_graph::StableGraph<String, u8, petgraph::Directed, u32>,
    schedule_edges: &[EdgeIndex],
    mark_nodes: &[NodeIndex],
) {
    let mut graph_string =
        petgraph::dot::Dot::with_config(&graph, &[petgraph::dot::Config::EdgeIndexLabel])
            .to_string();
    let re = Regex::new(r#"label\s*=\s*"\d+""#).unwrap();
    for e in schedule_edges {
        graph_string =
            graph_string.replace(&format!("label = \"{}\"", e.index()), "color=\"green\"");
    }
    graph_string = re.replace_all(&graph_string, "").to_string();
    for n in mark_nodes {
        graph_string = graph_string.replace(
            &format!("    {} [ label =", n.index()),
            &format!(
                "    {} [ style=\"filled\" fillcolor=\"yellow\" label =",
                n.index()
            ),
        );
    }

    let url = format!(
        "https://dreampuf.github.io/GraphvizOnline/#{}",
        urlencoding::encode(&graph_string)
    );
    if let Err(e) = webbrowser::open(&url) {
        panic!("Error displaying graph: {e:?}");
    }
}

pub struct NewOp<'a> {
    new_op_id: NodeIndex,
    graph_ref: &'a mut Graph,
    num_srcs: u8,
}

impl NewOp<'_> {
    pub fn finish(self) -> NodeIndex {
        self.new_op_id
    }

    pub fn input(mut self, id: NodeIndex, from_output: u8, shape: ShapeTracker) -> Self {
        self.graph_ref.graph.add_edge(
            id,
            self.new_op_id,
            Dependency::Data {
                input_order: self.num_srcs,
                output_order: from_output,
                shape,
            },
        );
        self.num_srcs += 1;
        self
    }
}

/// Transfer all external references from one node to another (this may happen because one node is about to be removed / merged into another)
pub fn remap<T: ToIdsMut>(from: NodeIndex, to: NodeIndex, mut ids: T, graph: &mut Graph) {
    for id in ids.to_ids_mut() {
        if *id == from {
            *id = to;
        }
    }
    // Transfer no_delete
    if graph.no_delete.remove(&from) {
        graph.no_delete.insert(to);
    }
    // Transfer to_retrieve
    if let Some(w) = graph.to_retrieve.remove(&from) {
        graph.to_retrieve.insert(to, w);
    }
}

pub fn move_outgoing_edge<N, E: Clone>(
    from: NodeIndex,
    to: NodeIndex,
    graph: &mut StableGraph<N, E>,
) {
    // Carry over outgoing edges from node to other_node
    for (weight, target) in graph
        .edges_directed(from, petgraph::Direction::Outgoing)
        .map(|e| (e.weight().clone(), e.target()))
        .collect::<Vec<_>>()
    {
        graph.add_edge(to, target, weight);
    }
}

pub fn move_incoming_edge<N, E: Clone>(
    from: NodeIndex,
    to: NodeIndex,
    graph: &mut StableGraph<N, E>,
) {
    // Carry over incoming edges from node to other_node
    for (weight, source) in graph
        .edges_directed(from, petgraph::Direction::Incoming)
        .map(|e| (e.weight().clone(), e.source()))
        .collect::<Vec<_>>()
    {
        graph.add_edge(source, to, weight);
    }
}

pub struct GraphSearch {
    current: FxHashMap<Uuid, NodeIndex>,
    selector: StableGraph<(Uuid, SelectOp), Option<u8>>,
    graph: *mut Graph,
    to_return: Vec<FxHashMap<NodeIndex, NodeIndex>>,
    returned_anchors: HashSet<NodeIndex>,
    anchor: NodeIndex,
    pub matched: bool,
}

impl GraphSearch {
    pub fn next_match(&mut self) -> bool {
        // Look through graph for pattern from selector
        let graph = unsafe { self.graph.as_mut().unwrap() };

        if self.to_return.is_empty() {
            // Replenish to_return
            let (_, select_op) = self.selector.node_weight(self.anchor).unwrap();
            for node in graph.graph.node_indices().collect::<Vec<_>>() {
                if !self.returned_anchors.contains(&node)
                    && test_node(select_op, &mut graph.graph, node)
                {
                    // Backtrack to check if this is a match
                    if let Some(mapping) =
                        backtrack_match(self.anchor, &self.selector, node, &mut graph.graph)
                    {
                        self.to_return.push(mapping);
                    }
                }
            }
        }
        if let Some(mapping) = self.to_return.pop() {
            self.returned_anchors.insert(mapping[&self.anchor]);
            self.current = mapping
                .into_iter()
                .map(|(k, v)| (self.selector.node_weight(k).unwrap().0, v))
                .collect();
            self.matched = true;
            return true;
        }
        self.matched = false;
        false
    }

    pub fn clear_cached_results(&mut self) {
        self.to_return.clear();
    }
    pub fn reset(&mut self) {
        self.clear_cached_results();
        self.returned_anchors.clear();
    }
    pub fn check_no_delete(&self, exclude: &[Uuid]) -> bool {
        let graph = unsafe { self.graph.as_ref().unwrap() };
        self.current
            .iter()
            .filter(|(k, _)| !exclude.contains(k))
            .any(|(_, v)| graph.no_delete.contains(v))
    }
    pub fn get<T: Borrow<SelectGraph>>(&self, node: T) -> NodeIndex {
        *self.current.get(&node.borrow().id).unwrap()
    }
    pub fn try_delete(&self) {
        let graph = unsafe { self.graph.as_mut().unwrap() };
        for node in toposort(&self.selector, None).unwrap().into_iter().rev() {
            let id = self.selector.node_weight(node).unwrap().0;
            let node = self.current[&id];
            graph.safe_remove_node(node, 0);
        }
    }
}

fn backtrack_match(
    pattern_root: NodeIndex,
    pattern_graph: &StableGraph<(Uuid, SelectOp), Option<u8>>,
    main_root: NodeIndex,
    main_graph: &mut StorageGraph,
) -> Option<FxHashMap<NodeIndex, NodeIndex>> {
    fn get_parents<N, E>(
        graph: &petgraph::stable_graph::StableGraph<N, E>,
        node_index: NodeIndex,
        edge_filter: fn(&EdgeReference<'_, E>) -> bool,
    ) -> Vec<NodeIndex> {
        graph
            .edges_directed(node_index, Direction::Incoming)
            .filter(edge_filter)
            .map(|e| e.source())
            .collect()
    }

    if !test_node(
        &pattern_graph.node_weight(pattern_root).unwrap().1,
        main_graph,
        main_root,
    ) {
        return None;
    }

    let mut mapping = FxHashMap::default();
    mapping.insert(pattern_root, main_root);
    let main_parents = get_parents(main_graph, main_root, |e| !e.weight().is_schedule());
    'pattern_loop: for pattern_parent in get_parents(pattern_graph, pattern_root, |_| true) {
        for parent in main_parents.iter() {
            if mapping.values().any(|&v| v == *parent) {
                // This main node was used already, skip it
                continue;
            }
            if let Some(new_mapping) =
                backtrack_match(pattern_parent, pattern_graph, *parent, main_graph)
            {
                mapping.extend(new_mapping.into_iter());
                continue 'pattern_loop;
            }
        }
        return None;
    }
    Some(mapping)
}

fn test_node(
    SelectOp {
        type_id,
        check,
        shape,
        fake,
    }: &SelectOp,
    graph: &mut StorageGraph,
    graph_node: NodeIndex,
) -> bool {
    let input_shapes = graph
        .edges_directed(graph_node, petgraph::Direction::Incoming)
        .filter_map(|e| e.weight().as_data())
        .sorted_by_key(|e| e.0)
        .map(|e| e.2)
        .collect::<Vec<_>>();
    let current_weight = graph.node_weight_mut(graph_node).unwrap();
    // Test type
    if let Some(ty) = type_id {
        if current_weight.as_any().type_id() != *ty {
            return false;
        }
    }

    // Test shape
    if let Some(shape) = shape {
        let mut shape_map = FxHashMap::default();
        if shape.len() != input_shapes.len() {
            return false;
        }
        for (a_sh, b_sh) in shape.iter().zip(input_shapes.iter()) {
            if a_sh.len() != b_sh.dims.len() {
                return false;
            }
            for (a, b) in a_sh.iter().zip(b_sh.dims().into_iter()) {
                match a.to_usize() {
                    Some(n) => {
                        if b.to_usize().map(|i| i != n).unwrap_or(true) {
                            return false;
                        }
                    }
                    None => {
                        let c = a
                            .to_symbols()
                            .pop()
                            .expect("Selector dimension must be either a symbol or number");
                        if let Some(expected) = shape_map.get(&c) {
                            if b != *expected {
                                return false;
                            }
                        } else {
                            shape_map.insert(c, b);
                        }
                    }
                }
            }
        }
    }
    // Test fakes
    if let Some(fakes) = fake {
        for (a_sh, b_sh) in fakes.iter().zip(input_shapes.iter()) {
            for (a, b) in a_sh.iter().zip(b_sh.indexes.iter().map(|i| b_sh.fake[*i])) {
                if let Some(a) = a {
                    if *a != b {
                        return false;
                    }
                }
            }
        }
    }

    // Run check
    if let Some(check) = check {
        if !check(current_weight.as_mut(), &input_shapes) {
            return false;
        }
    }
    true
}

#[derive(Default, Clone)]
pub struct SelectOp {
    /// Type constraint
    pub type_id: Option<TypeId>,
    /// Check constraint
    #[allow(clippy::type_complexity)]
    pub check: Option<Arc<Box<dyn Fn(&mut dyn Operator, &[ShapeTracker]) -> bool>>>,
    /// Shape constraint
    pub shape: Option<Vec<Vec<Expression>>>,
    /// Fake constraint
    pub fake: Option<Vec<Vec<Option<bool>>>>,
}

#[macro_export]
macro_rules! select_ty {
    ($t: ty) => {
        SelectOp::new().ty::<$t>()
    };
}

impl SelectOp {
    pub fn new() -> Self {
        Self::default()
    }
}

pub fn check_no_delete(graph: &Graph, nodes: &[NodeIndex]) -> bool {
    nodes.iter().any(|n| graph.no_delete.contains(n))
}

#[derive(Clone)]
pub struct SelectGraph {
    /// The full selection graph
    graph: StableGraph<(Uuid, SelectOp), Option<u8>>,
    /// The current id for the current node in the graph
    pub id: Uuid,
    /// The index of the current node in the graph
    node: NodeIndex,
}

impl SelectGraph {
    fn new(restrictions: SelectOp) -> Self {
        let mut graph = StableGraph::new();
        let id = Uuid::new_v4();
        Self {
            node: graph.add_node((id, restrictions)),
            graph,
            id,
        }
    }

    /// Constrain the op to a type
    pub fn ty<O: Operator + 'static>(&mut self) {
        self.graph.node_weight_mut(self.node).unwrap().1.type_id = Some(TypeId::of::<O>());
    }
    /// Constrain the op to a type
    pub fn type_id(&mut self, type_id: TypeId) {
        self.graph.node_weight_mut(self.node).unwrap().1.type_id = Some(type_id);
    }
    /// Constrain the op to a checking function
    pub fn check<F: Fn(&mut dyn Operator, &[ShapeTracker]) -> bool + 'static>(&mut self, check: F) {
        self.graph.node_weight_mut(self.node).unwrap().1.check = Some(Arc::new(Box::new(check)));
    }
    /// Constrain the op to input shapes
    pub fn shapes<E: Into<Expression>, V: Into<Vec<E>>, S: Into<Vec<V>>>(&mut self, shapes: S) {
        self.graph.node_weight_mut(self.node).unwrap().1.shape = Some(
            shapes
                .into()
                .into_iter()
                .map(|i| i.into().into_iter().map(|i| i.into()).collect())
                .collect(),
        );
    }
    /// Constrain the op to input shape fakes
    pub fn fakes<V: Into<Vec<Option<bool>>>, S: Into<Vec<V>>>(&mut self, fakes: S) {
        self.graph.node_weight_mut(self.node).unwrap().1.fake =
            Some(fakes.into().into_iter().map(|i| i.into()).collect());
    }

    pub fn connect(mut self, b: Self) -> Self {
        // Add b graph to a graph
        let mut node_map = FxHashMap::default();
        let mut a_nodes = FxHashMap::default();
        for node in self.graph.node_indices() {
            let id = self.graph.node_weight(node).unwrap().0;
            a_nodes.insert(id, node);
        }
        for node in b.graph.node_indices() {
            let id = b.graph.node_weight(node).unwrap().0;
            let new_node = a_nodes.get(&id).copied().unwrap_or_else(|| {
                self.graph
                    .add_node(b.graph.node_weight(node).unwrap().clone())
            });
            node_map.insert(node, new_node);
        }
        for edge in b.graph.edge_indices() {
            let (src, trg) = b.graph.edge_endpoints(edge).unwrap();
            self.graph.add_edge(
                node_map[&src],
                node_map[&trg],
                *b.graph.edge_weight(edge).unwrap(),
            );
        }
        self.graph.add_edge(self.node, node_map[&b.node], None);
        self.node = node_map[&b.node];
        self.id = b.id;
        self
    }

    pub fn search(self, graph: &mut Graph) -> GraphSearch {
        let anchor = *toposort(&self.graph, None).unwrap().last().unwrap();
        GraphSearch {
            current: FxHashMap::default(),
            to_return: vec![],
            selector: self.graph,
            graph,
            returned_anchors: HashSet::new(),
            anchor,
            matched: false,
        }
    }
}

pub fn op<T: Operator + 'static>() -> SelectGraph {
    let mut s = SelectOp::new();
    s.type_id = Some(TypeId::of::<T>());
    SelectGraph::new(s)
}

pub fn node() -> SelectGraph {
    SelectGraph::new(SelectOp::new())
}

pub fn unary<T: Operator + 'static>(node: SelectGraph) -> SelectGraph {
    node.connect(op::<T>())
}

pub fn binary<T: Operator + 'static>(a: SelectGraph, b: SelectGraph) -> SelectGraph {
    b.connect(a.connect(op::<T>()))
}

/// Whether or not to do debug prints (env var DEBUG=1)
pub fn debug() -> bool {
    std::env::var("DEBUG")
        .unwrap_or_default()
        .parse::<i32>()
        .map(|i| i == 1)
        .unwrap_or_default()
}

// Add the following test module to the end of your file.

#[cfg(test)]
mod tests {
    use super::*;

    // Helper to find a single node of a specific type. Panics if not found or multiple found.
    fn find_unique_node(
        graph: &StableGraph<GraphTerm, u8, Directed>,
        predicate: impl Fn(&GraphTerm) -> bool,
    ) -> NodeIndex {
        let nodes: Vec<_> = graph
            .node_indices()
            .filter(|&n| predicate(graph.node_weight(n).unwrap()))
            .collect();
        assert_eq!(
            nodes.len(),
            1,
            "Expected 1 matching node, found {}",
            nodes.len()
        );
        nodes[0]
    }

    // Helper to count nodes of a specific type.
    fn count_nodes(
        graph: &StableGraph<GraphTerm, u8, Directed>,
        predicate: impl Fn(&GraphTerm) -> bool,
    ) -> usize {
        graph
            .node_indices()
            .filter(|&n| predicate(graph.node_weight(n).unwrap()))
            .count()
    }

    fn get_loop_chain_strides(
        graph: &StableGraph<GraphTerm, u8, Directed>,
        start_node: NodeIndex,
        direction: Direction,
    ) -> Vec<Expression> {
        let mut strides = vec![];
        let mut current_node = start_node;
        loop {
            // Add the stride of the current node if it's a loop
            if let Some(stride) = match graph.node_weight(current_node).unwrap() {
                GraphTerm::LoopIn { stride, .. } => Some(*stride),
                GraphTerm::LoopOut { stride, .. } => Some(*stride),
                _ => None,
            } {
                strides.push(stride);
            } else {
                // Traversal has reached a non-loop node, which terminates the chain.
                break;
            }

            // Find the next node in the chain
            let neighbors: Vec<_> = graph
                .edges_directed(current_node, direction)
                .map(|e| match direction {
                    Direction::Incoming => e.source(),
                    Direction::Outgoing => e.target(),
                })
                .collect();

            if neighbors.len() != 1 {
                break;
            }
            current_node = neighbors[0];
        }

        if direction == Direction::Incoming {
            strides.reverse();
        }

        strides
    }

    fn get_loop_chain_ranges(
        graph: &StableGraph<GraphTerm, u8, Directed>,
        mut current_node: NodeIndex,
        direction: Direction,
    ) -> Vec<Expression> {
        let mut ranges = vec![];
        loop {
            // Add the range of the current node if it's a loop
            if let Some(range) = match graph.node_weight(current_node).unwrap() {
                GraphTerm::LoopIn { range, .. } => Some(*range),
                GraphTerm::LoopOut { range, .. } => Some(*range),
                _ => None,
            } {
                ranges.push(range);
            } else {
                // Traversal has reached a non-loop node, which terminates the chain.
                break;
            }

            // Find the next node in the chain
            let neighbors: Vec<_> = graph
                .edges_directed(current_node, direction)
                .map(|e| match direction {
                    Direction::Incoming => e.source(),
                    Direction::Outgoing => e.target(),
                })
                .collect();

            // If there's not exactly one connection, the chain ends here
            if neighbors.len() != 1 {
                break;
            }
            current_node = neighbors[0];
        }

        // Incoming traversal builds the list in reverse order (from op to source), so fix it
        if direction == Direction::Incoming {
            ranges.reverse();
        }

        ranges
    }

    #[test]
    fn test_calculate_strides_contiguous() {
        // Test a standard, row-major tensor shape
        let _cx = Graph::new(); // to initialize thread-local storage
        let dims = vec![10.into(), 5.into(), 2.into()];
        let indexes = vec![0, 1, 2]; // Contiguous, no permutation
        let strides = calculate_strides(&dims, &indexes);

        // Expected strides:
        // Dim 0 (size 10): Stride should be 5 * 2 = 10
        // Dim 1 (size 5): Stride should be 2
        // Dim 2 (size 2): Stride should be 1
        let expected_strides: Vec<Expression> = vec![10.into(), 2.into(), 1.into()];

        assert_eq!(strides, expected_strides);
    }

    #[test]
    fn test_calculate_strides_permuted() {
        // Test a permuted tensor shape. Logical shape is [2, 10, 5]
        let _cx = Graph::new(); // to initialize thread-local storage
        let dims = vec![10.into(), 5.into(), 2.into()]; // Physical shape
        let indexes = vec![2, 0, 1]; // Permutation: (d0, d1, d2) -> (d2, d0, d1)

        let strides = calculate_strides(&dims, &indexes);

        // Physical strides (based on dims [10, 5, 2]) are [10, 2, 1]
        // The function should reorder these based on `indexes`.
        // Expected logical strides:
        // Logical Dim 0 (Physical Dim 2): Stride is physical_strides[2] = 1
        // Logical Dim 1 (Physical Dim 0): Stride is physical_strides[0] = 10
        // Logical Dim 2 (Physical Dim 1): Stride is physical_strides[1] = 2
        let expected_strides: Vec<Expression> = vec![1.into(), 10.into(), 2.into()];

        assert_eq!(strides, expected_strides);
    }

    // ensures our loopins and loopouts are done correctly
    #[test]
    fn test_simple_add_translation() {
        // Setup Graph 1
        let mut cx = Graph::new();
        // The actual values set don't affect the translated structure, only the shape does.
        let a = cx.tensor(4).set(vec![1., 2., 3., 4.]);
        let b = cx.tensor(4).set(vec![5., 6., 7., 8.]);
        let c = a + b;
        c.retrieve(); // Mark as output

        // Translate
        let graph2 = cx.translate_to_2();

        // Assertions on Graph 2
        assert_eq!(
            count_nodes(&graph2, |n| matches!(n, GraphTerm::GMEM { .. })),
            2
        );
        assert_eq!(count_nodes(&graph2, |n| matches!(n, GraphTerm::Add)), 1);
        // 2 inputs * 1 dim/input = 2 LoopIn's
        assert_eq!(
            count_nodes(&graph2, |n| matches!(n, GraphTerm::LoopIn { .. })),
            2
        );
        // 1 output * 1 dim/output = 1 LoopOut
        assert_eq!(
            count_nodes(&graph2, |n| matches!(n, GraphTerm::LoopOut { .. })),
            1
        );

        // Check structure
        let add_node = find_unique_node(&graph2, |n| matches!(n, GraphTerm::Add));

        // Check Add inputs
        let add_parents: Vec<_> = graph2
            .edges_directed(add_node, Direction::Incoming)
            .map(|e| e.source())
            .collect();
        assert_eq!(add_parents.len(), 2);
        for parent_idx in add_parents {
            let parent_node = graph2.node_weight(parent_idx).unwrap();
            assert!(matches!(parent_node, GraphTerm::LoopIn { .. }));
        }

        // Check Add output
        let add_children: Vec<_> = graph2
            .edges_directed(add_node, Direction::Outgoing)
            .map(|e| e.target())
            .collect();
        assert_eq!(add_children.len(), 1);
        let child_node = graph2.node_weight(add_children[0]).unwrap();
        assert!(matches!(child_node, GraphTerm::LoopOut { .. }));
    }

    #[test]
    fn test_unary_op_translation() {
        // Setup Graph 1
        let mut cx = Graph::new();
        let a = cx.tensor((2, 3)).set(vec![1., 2., 3., 4., 5., 6.]);
        let b = a.exp2();
        b.retrieve();

        // Translate
        let graph2 = cx.translate_to_2();

        // Assertions
        assert_eq!(
            count_nodes(&graph2, |n| matches!(n, GraphTerm::GMEM { .. })),
            1
        );
        assert_eq!(count_nodes(&graph2, |n| matches!(n, GraphTerm::Exp)), 1);
        // 1 input * 2 dims = 2 LoopIn's
        assert_eq!(
            count_nodes(&graph2, |n| matches!(n, GraphTerm::LoopIn { .. })),
            2
        );
        // 1 output * 2 dims = 2 LoopOut's
        assert_eq!(
            count_nodes(&graph2, |n| matches!(n, GraphTerm::LoopOut { .. })),
            2
        );

        let exp_node = find_unique_node(&graph2, |n| matches!(n, GraphTerm::Exp));
        let gmem_node = find_unique_node(&graph2, |n| matches!(n, GraphTerm::GMEM { .. }));

        // Check that a path from GMEM to Exp exists
        assert!(
            petgraph::algo::has_path_connecting(&graph2, gmem_node, exp_node, None),
            "Path from GMEM to Exp could not be found"
        );
    }

    #[test]
    fn test_contiguous_elimination_translation() {
        // Setup Graph 1
        let mut cx = Graph::new();
        let a = cx.tensor(4).set(vec![1., 2., 3., 4.]);
        let b = a.contiguous(); // This op should be eliminated
        let c = b.exp2();
        c.retrieve();

        // Translate
        let graph2 = cx.translate_to_2();

        // The 'Contiguous' op in Graph 1 should not produce any node in Graph 2.
        // It has a fallback to an ERROR GMEM node, so we check for that.
        assert_eq!(
            count_nodes(&graph2, |n| {
                if let GraphTerm::GMEM { label } = n {
                    label.as_deref() == Some("ERROR")
                } else {
                    false
                }
            }),
            0
        );

        // Check that GMEM is connected to the Exp chain
        let gmem_node = find_unique_node(&graph2, |n| matches!(n, GraphTerm::GMEM { .. }));
        let exp_node = find_unique_node(&graph2, |n| matches!(n, GraphTerm::Exp));

        // Very basic path check: is there a path from GMEM to Exp?
        assert!(
            petgraph::algo::has_path_connecting(&graph2, gmem_node, exp_node, None),
            "Path from GMEM to Exp could not be found"
        );

        // More specific: the child of GMEM should be a LoopIn
        let gmem_children: Vec<_> = graph2
            .edges_directed(gmem_node, Direction::Outgoing)
            .map(|e| e.target())
            .collect();
        assert_eq!(gmem_children.len(), 1);
        assert!(matches!(
            graph2.node_weight(gmem_children[0]).unwrap(),
            GraphTerm::LoopIn { .. }
        ));
    }

    #[test]
    fn test_reduction_loop_in_ranges_match() {
        // 1. Setup: Create a 3D tensor and reduce it on the middle dimension.
        let mut cx = Graph::new();
        let a = cx.tensor((2, 5, 4)); // Shape (2, 5, 4)
        let b = a.sum(1); // Reduce dim 1, output shape should be (2, 1, 4)
        b.retrieve();

        // 2. Translate
        let graph2 = cx.translate_to_2();

        // 3. Assert: The LoopIn chains for both inputs to the Add node must have the same ranges,
        // matching the *original* tensor shape.
        let add_node = find_unique_node(&graph2, |n| matches!(n, GraphTerm::Add));
        let parents: Vec<_> = graph2
            .edges_directed(add_node, Direction::Incoming)
            .map(|e| e.source())
            .collect();
        assert_eq!(parents.len(), 2, "Add node should have two parents");

        // The parents are the last `LoopIn` nodes of their respective chains.
        let input_loop_ranges = get_loop_chain_ranges(&graph2, parents[0], Direction::Incoming);
        let acc_loop_ranges = get_loop_chain_ranges(&graph2, parents[1], Direction::Incoming);

        let expected_ranges: Vec<Expression> = vec![2.into(), 5.into(), 4.into()];

        assert_eq!(
            input_loop_ranges, expected_ranges,
            "Input data loop ranges are incorrect"
        );
        assert_eq!(
            acc_loop_ranges, expected_ranges,
            "Accumulator loop ranges are incorrect"
        );
    }

    #[test]
    fn test_reduction_loop_out_ranges_adjusted() {
        // 1. Setup: Create a 3D tensor and reduce it on the middle dimension.
        let mut cx = Graph::new();
        let a = cx.tensor((2, 5, 4)); // Shape (2, 5, 4)
        let b = a.sum(1); // Reduce dim 1, output shape is (2, 1, 4)
        b.retrieve();

        // 2. Translate
        let graph2 = cx.translate_to_2();

        // 3. Assert: The LoopOut chain must have ranges matching the *new, reduced* tensor shape.
        let add_node = find_unique_node(&graph2, |n| matches!(n, GraphTerm::Add));
        let children: Vec<_> = graph2
            .edges_directed(add_node, Direction::Outgoing)
            .map(|e| e.target())
            .collect();
        assert_eq!(children.len(), 1, "Add node should have one child");

        // The child is the first `LoopOut` node of the output chain.
        let output_loop_ranges = get_loop_chain_ranges(&graph2, children[0], Direction::Outgoing);

        let expected_ranges: Vec<Expression> = vec![2.into(), 1.into(), 4.into()];

        assert_eq!(
            output_loop_ranges, expected_ranges,
            "Output loop ranges are incorrect and were not adjusted for reduction"
        );
    }

    #[test]
    fn test_reduction_acc_input_strides_correct() {
        // 1. Arrange: Create a 3D tensor and reduce it on the middle dimension.
        let mut cx = Graph::new();
        let a = cx.tensor((5, 4, 3)); // Shape (5, 4, 3)
        let b = a.sum(1); // Reduce dim 1
        b.retrieve();

        // 2. Act: Translate the graph.
        let graph2 = cx.translate_to_2();

        // 3. Assert: The accumulator's LoopIn chain should have strides based on the *original*
        // input shape (5, 4, 3), with 'z' propagating correctly.
        let _add_node = find_unique_node(&graph2, |n| matches!(n, GraphTerm::Add));
        let new_acc_node = find_unique_node(&graph2, |n| matches!(n, GraphTerm::NewAcc { .. }));

        // Find the start of the accumulator's loop chain (the child of NewAcc)
        let acc_loop_start = graph2
            .edges_directed(new_acc_node, Direction::Outgoing)
            .next()
            .unwrap()
            .target();
        let acc_loop_strides = get_loop_chain_strides(&graph2, acc_loop_start, Direction::Outgoing);

        // Expected strides for reducing (5, 4, 3) on dim 1:
        // dim 2 (size 3): Stride is 1
        // dim 1 (size 4, reduced): Stride is z
        // dim 0 (size 5): Stride is z * 4
        // The `calculate_reduction_strides` function produces [(z*4), z, 1].
        let z = Expression::from('z');
        let expected_strides: Vec<Expression> = vec![z * 4, z, 1.into()];

        assert_eq!(
            acc_loop_strides, expected_strides,
            "Accumulator loop strides are incorrect"
        );
    }

    #[test]
    fn test_reduction_output_strides_correct() {
        // 1. Arrange: Create a 3D tensor and reduce it on the middle dimension.
        let mut cx = Graph::new();
        let a = cx.tensor((5, 4, 3)); // Shape (5, 4, 3)
        let b = a.sum(1); // Reduce dim 1, output shape is (5, 1, 3)
        b.retrieve();

        // 2. Act: Translate the graph.
        let graph2 = cx.translate_to_2();

        // 3. Assert: The LoopOut chain must have strides calculated from the *final, smaller*
        // output shape (5, 1, 3).
        let add_node = find_unique_node(&graph2, |n| matches!(n, GraphTerm::Add));

        // The child of the Add op is the start of the LoopOut chain.
        let output_loop_start = graph2
            .edges_directed(add_node, Direction::Outgoing)
            .next()
            .unwrap()
            .target();
        let output_loop_strides =
            get_loop_chain_strides(&graph2, output_loop_start, Direction::Outgoing);

        // Expected strides for output shape (5, 1, 3) reduced on dim 1:
        // dim 2 (size 3): Stride is 1
        // dim 1 (size 1, reduced): Stride is z
        // dim 0 (size 5): Stride is z * 1
        // The `calculate_reduction_strides` function produces [(z*1), z, 1].
        let z = Expression::from('z');
        let expected_strides: Vec<Expression> = vec![
            z, // This is z * 1
            z,
            1.into(),
        ];

        assert_eq!(
            output_loop_strides, expected_strides,
            "Output loop strides are incorrect"
        );
    }
}
