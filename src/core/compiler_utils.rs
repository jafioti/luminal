// Lots of utilities used by compilers

use std::{
    any::{Any, TypeId},
    collections::{HashSet, VecDeque},
    fmt::Debug,
};

use colored::Colorize;
use itertools::Itertools;
use petgraph::{
    algo::toposort,
    stable_graph::{EdgeIndex, NodeIndex, StableGraph},
    visit::EdgeRef,
    Direction,
};
use regex::Regex;
use rustc_hash::{FxHashMap, FxHashSet};

use crate::{
    graph::Graph,
    op::Operator,
    prelude::{Dependency, MainGraph, Shape, ShapeTracker},
};

use super::{graph_tensor::GraphTensor, shape::symbolic::Expression};

pub trait ToIdsMut {
    fn to_ids_mut(&mut self) -> Vec<&mut NodeIndex>;
}

pub trait ToIds {
    fn to_ids(&self) -> Vec<NodeIndex>;
}

impl<S: Shape> ToIdsMut for GraphTensor<S> {
    fn to_ids_mut(&mut self) -> Vec<&mut NodeIndex> {
        vec![&mut self.id]
    }
}
impl<S: Shape> ToIds for GraphTensor<S> {
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
    /// Run a compilation pass
    fn compile<T: ToIdsMut>(&self, graph: &mut Graph, remap: T);
}

impl Compiler for () {
    fn compile<T: ToIdsMut>(&self, _: &mut Graph, _: T) {}
}

/// Wrap this around a compiler to rerun the compiler until it doesn't change the graph anymore
pub struct Looped<C: Compiler + Debug>(C);

impl<C: Compiler + Debug> Compiler for Looped<C> {
    fn compile<T: ToIdsMut>(&self, graph: &mut Graph, mut remap: T) {
        graph.toposort();
        let mut linearized = graph.linearized_graph.clone();
        loop {
            self.0.compile(graph, &mut remap);
            graph.toposort();
            if linearized == graph.linearized_graph {
                break;
            }
            linearized = graph.linearized_graph.clone();
        }
    }
}

impl<C: Default + Compiler + Debug> Default for Looped<C> {
    fn default() -> Self {
        Self(C::default())
    }
}

/// Wrap this around a compiler to measure the time it takes to compile
pub struct Timed<C: Compiler + Debug>(C);

impl<C: Compiler + Debug> Compiler for Timed<C> {
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
            fn compile<T: ToIdsMut>(&self, graph: &mut Graph, mut remap: T) {
                $(self.$idx.compile(graph, &mut remap);)+
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

// Helpers

impl Graph {
    /// Add op on the graph, and get back a NewOp
    ///
    /// ```rust
    /// use luminal::prelude::*;
    /// let mut cx = Graph::new();
    /// let a = cx.tensor::<R1<3>>();
    /// let b_id = cx
    ///     .add_op(luminal::op::Mul)
    ///     .input(a.id, 0, a.shape)
    ///     .finish();
    /// let b = GraphTensor::<R1<3>>::from_id(b_id, a.shape, a.graph());
    /// ```
    pub fn add_op<O: Operator + 'static>(&mut self, op: O) -> NewOp {
        NewOp {
            new_op_id: self.graph.add_node(Box::new(op)),
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
        let Some(node_weight) = self.graph.node_weight_mut(node) else {
            return None;
        };

        node_weight
            .custom(key, Box::new(input))
            .and_then(|o| o.downcast::<O>().ok().map(|o| *o))
    }

    /// Convert to debug-viewable graph
    pub fn debug_graph(
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
            id_map.insert(id, new_graph.add_node(format!("{node:?}")));
        }

        let mut schedule_edges = vec![];
        for node in self.graph.node_indices() {
            new_graph
                .node_weight_mut(id_map[&node])
                .unwrap()
                .push_str(&node.index().to_string());
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
                        .map(|d| !d.2.shape().is_empty())
                        .unwrap_or_default()
                {
                    new_graph
                        .node_weight_mut(id_map[&edge.target()])
                        .unwrap()
                        .push_str(&format!(
                            " | {:?}",
                            edge.weight().as_data().unwrap().2.shape()
                        ));
                }
            }
        }

        (new_graph, schedule_edges, id_map)
    }

    pub fn display(&self) {
        let (g, e, _) = self.debug_graph(false);
        display_graph(&g, &e, &[]);
    }

    pub fn display_shapes(&self) {
        let (g, e, _) = self.debug_graph(true);
        display_graph(&g, &e, &[]);
    }

    pub fn display_set(&self, set: &[NodeIndex]) {
        let (g, e, id_map) = self.debug_graph(false);
        display_graph(&g, &e, &set.iter().map(|i| id_map[i]).collect::<Vec<_>>());
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
    #[allow(clippy::type_complexity, clippy::borrowed_box)]
    pub fn get_sources(&self, node_id: NodeIndex) -> Vec<(NodeIndex, u8, ShapeTracker)> {
        self.graph
            .edges_directed(node_id, Direction::Incoming)
            .filter_map(|e| e.weight().as_data().map(|i| (e.source(), i)))
            .sorted_by_key(|(_, (i, _, _))| *i)
            .map(|(a, (_, c, b))| (a, c, b))
            .collect()
    }

    /// Get the dests of a node given it's id
    #[allow(clippy::type_complexity, clippy::borrowed_box)]
    pub fn get_dests(&self, node_id: NodeIndex) -> Vec<(NodeIndex, &Box<dyn Operator>)> {
        self.graph
            .edges_directed(node_id, Direction::Outgoing)
            .filter_map(|e| e.weight().as_data().map(|i| (e.target(), i)))
            .sorted_by_key(|(_, (i, _, _))| *i)
            .map(|(a, _)| (a, self.graph.node_weight(a).unwrap()))
            .collect()
    }
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
        let mut id_map = FxHashMap::default(); // We track the node id remapping here so they don't overlap
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
pub fn move_references<T: ToIdsMut>(
    mut ids: T,
    no_delete: &mut FxHashSet<NodeIndex<u32>>,
    to_retrieve: &mut FxHashSet<NodeIndex<u32>>,
    src: NodeIndex,
    trg: NodeIndex,
) {
    for id in ids.to_ids_mut() {
        if *id == src {
            *id = trg;
        }
    }
    // Transfer no_delete
    if no_delete.remove(&src) {
        no_delete.insert(trg);
    }
    // Transfer to_retrieve
    if to_retrieve.remove(&src) {
        to_retrieve.insert(trg);
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

pub trait TraitObjEq {
    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;
    fn is_equal(&self, _: &dyn Operator) -> bool;
}

// Implement TraitObjEq for all 'static types implementing PartialEq.
impl<S: 'static + PartialEq> TraitObjEq for S {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn is_equal(&self, other: &dyn Operator) -> bool {
        // Do a type-safe casting. If the types are different,
        // return false, otherwise test the values for equality.
        other
            .as_any()
            .downcast_ref::<S>()
            .map(|a| self == a)
            .unwrap_or_default()
    }
}

type SelectionGraph = petgraph::Graph<SelectOp, Option<u8>>;

pub struct GraphSearch {
    selector: SelectionGraph,
    graph: *mut Graph,
    to_return: Vec<FxHashMap<NodeIndex, NodeIndex>>,
    returned_anchors: HashSet<NodeIndex>,
    anchor: NodeIndex,
}

impl GraphSearch {
    pub fn next_match(&mut self) -> bool {
        // Look through graph for pattern from selector
        let graph = unsafe { self.graph.as_mut().unwrap() };

        if self.to_return.is_empty() {
            // Replenish to_return
            let select_op = self.selector.node_weight(self.anchor).unwrap();
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
            // Apply pattern to ptrs
            for (selector_node, ptr) in self.selector.node_indices().flat_map(|n| {
                self.selector
                    .node_weight(n)
                    .unwrap()
                    .pointers
                    .iter()
                    .map(move |i| (n, *i))
            }) {
                unsafe {
                    *ptr = mapping[&selector_node];
                }
            }
            self.returned_anchors.insert(mapping[&self.anchor]);
            return true;
        }
        false
    }

    pub fn clear_cached_results(&mut self) {
        self.to_return.clear();
    }
    pub fn reset(&mut self) {
        self.clear_cached_results();
        self.returned_anchors.clear();
    }
}

/// TODO: This should return **all** possible matches stemming from these roots, not just the first
fn backtrack_match(
    pattern_root: NodeIndex,
    pattern_graph: &SelectionGraph,
    main_root: NodeIndex,
    main_graph: &mut MainGraph,
) -> Option<FxHashMap<NodeIndex, NodeIndex>> {
    let mut matches = FxHashMap::default();
    let mut stack = VecDeque::new();
    matches.insert(pattern_root, main_root);
    stack.push_back((pattern_root, main_root));

    while let Some((pattern_node, main_node)) = stack.pop_back() {
        let pattern_parents =
            pattern_graph.neighbors_directed(pattern_node, petgraph::Direction::Incoming);
        let main_parents = main_graph
            .edges_directed(main_node, petgraph::Direction::Incoming)
            .filter(|e| !e.weight().is_schedule())
            .map(|e| e.source())
            .collect_vec();

        'pattern_loop: for pattern_parent in pattern_parents {
            for main_parent in &main_parents {
                if matches.values().any(|&v| v == *main_parent) {
                    continue;
                }
                if test_node(
                    pattern_graph.node_weight(pattern_parent).unwrap(),
                    main_graph,
                    *main_parent,
                ) {
                    matches.insert(pattern_parent, *main_parent);
                    stack.push_back((pattern_parent, *main_parent));
                    continue 'pattern_loop;
                }
            }
            // If we reach here, no match was found for this pattern parent
            return None;
        }
    }

    Some(matches)
}

fn test_node(
    SelectOp {
        type_id,
        check,
        shape,
        fake,
        pointers: _,
    }: &SelectOp,
    graph: &mut MainGraph,
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
            for (a, b) in a_sh.iter().zip(b_sh.shape().iter()) {
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
                            if b != expected {
                                return false;
                            }
                        } else {
                            shape_map.insert(c, b.clone());
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
    type_id: Option<TypeId>,
    /// Check constraint
    #[allow(clippy::type_complexity)]
    check: Option<fn(&mut dyn Operator, &[ShapeTracker]) -> bool>,
    /// Shape constraint
    shape: Option<Vec<Vec<Expression>>>,
    /// Fake constraint
    fake: Option<Vec<Vec<Option<bool>>>>,
    /// Pointers
    pointers: Vec<*mut NodeIndex>,
}

#[macro_export]
macro_rules! constant_select_op {
    ($i: expr, $t: tt) => {
        SelectOp::new().check(|o, _| {
            if let Some(c) = o.as_any().downcast_ref::<MetalConstant<$t>>() {
                if let op::ConstantValue::Float(f) = c.0 {
                    (f - $i).abs() < 0.0001
                } else {
                    false
                }
            } else {
                false
            }
        })
    };
}

impl SelectOp {
    pub fn new() -> Self {
        Self::default()
    }

    /// Constrain the op to a type
    pub fn ty<O: Operator + 'static>(mut self) -> Self {
        self.type_id = Some(TypeId::of::<O>());
        self
    }
    /// Constrain the op to a type
    pub fn type_id(mut self, type_id: TypeId) -> Self {
        self.type_id = Some(type_id);
        self
    }
    /// Constrain the op to a checking function
    pub fn check(mut self, check: fn(&mut dyn Operator, &[ShapeTracker]) -> bool) -> Self {
        self.check = Some(check);
        self
    }
    /// Constrain the op to input shapes
    pub fn shapes<S: Into<Vec<Vec<Expression>>>>(mut self, shapes: S) -> Self {
        self.shape = Some(shapes.into());
        self
    }
    /// Constrain the op to input shape fakes
    pub fn fakes<S: Into<Vec<Vec<Option<bool>>>>>(mut self, fakes: S) -> Self {
        self.fake = Some(fakes.into());
        self
    }
    /// Register a pointer to set if the op is matched
    pub fn ptr(mut self, ptr: *mut NodeIndex) -> Self {
        self.pointers.push(ptr);
        self
    }

    /// Connect this op with another op
    pub fn edge<T: Into<SelectEdge>>(self, node: T) -> SelectEdge {
        SelectEdge::new(self, node)
    }
}

#[derive(Clone)]
pub struct SelectEdge {
    main_node: NodeIndex,
    graph: SelectionGraph,
}

impl From<SelectOp> for SelectEdge {
    fn from(value: SelectOp) -> Self {
        let mut graph = SelectionGraph::default();
        let main_node = graph.add_node(value);
        Self { main_node, graph }
    }
}

impl SelectEdge {
    fn internal_new<A: Into<SelectEdge>, B: Into<SelectEdge>>(a: A, out: Option<u8>, b: B) -> Self {
        let mut a = a.into();
        let b = b.into();
        // Add b graph to a graph
        let mut node_map = FxHashMap::default();
        for node in b.graph.node_indices() {
            let new_node = a.graph.add_node(b.graph.node_weight(node).unwrap().clone());
            node_map.insert(node, new_node);
        }
        for edge in b.graph.edge_indices() {
            let (src, trg) = b.graph.edge_endpoints(edge).unwrap();
            a.graph.add_edge(
                node_map[&src],
                node_map[&trg],
                *b.graph.edge_weight(edge).unwrap(),
            );
        }
        a.graph.add_edge(a.main_node, node_map[&b.main_node], out);
        a.main_node = node_map[&b.main_node];
        a
    }

    pub fn new<A: Into<SelectEdge>, B: Into<SelectEdge>>(a: A, b: B) -> Self {
        Self::internal_new(a, None, b)
    }

    pub fn new_with_output<A: Into<SelectEdge>, B: Into<SelectEdge>>(a: A, out: u8, b: B) -> Self {
        Self::internal_new(a, Some(out), b)
    }

    pub fn search(self, graph: &mut Graph) -> GraphSearch {
        let anchor = *toposort(&self.graph, None).unwrap().last().unwrap();
        GraphSearch {
            to_return: vec![],
            selector: self.graph,
            graph,
            returned_anchors: HashSet::new(),
            anchor,
        }
    }

    /// Connect this op with another op
    pub fn edge<T: Into<SelectEdge>>(self, node: T) -> Self {
        Self::new(self, node)
    }
}

pub fn check_no_delete(graph: &Graph, nodes: &[NodeIndex]) -> bool {
    nodes.iter().any(|n| graph.no_delete.contains(n))
}

#[cfg(test)]
mod tests {
    use petgraph::adj::NodeIndex;

    use crate::{
        op::{Exp2, Log2, Mul, SumReduce},
        prelude::Graph,
    };

    use super::*;

    #[test]
    fn test_graph_selector() {
        let mut cx = Graph::default();
        // Exp -> Log or Log -> Exp
        let (mut exp, mut log) = (NodeIndex::default(), NodeIndex::default());
        let (exp_select, log_select) = (
            SelectOp::new().ty::<Exp2>().ptr(&mut exp),
            SelectOp::new().ty::<Log2>().ptr(&mut log),
        );
        let selector1 = log_select.clone().edge(exp_select.clone());
        let selector2 = exp_select.edge(log_select);

        assert!(!selector1.search(&mut cx).next_match() && !selector2.search(&mut cx).next_match());

        // Matmul
        let s = SelectOp::new()
            .ty::<Mul>()
            .edge(SelectOp::new().ty::<SumReduce>());

        assert!(!s.search(&mut cx).next_match());
    }
}
