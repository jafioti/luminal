use std::{
    any::{Any, TypeId},
    collections::{hash_map::Entry, BTreeMap, BTreeSet, HashMap, HashSet, VecDeque},
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

use crate::{
    graph::Graph,
    op::Operator,
    prelude::{remap_id, Dependency, MainGraph, ShapeTracker},
};

use super::shape::symbolic::Expression;

pub trait Compiler {
    /// Run a compilation pass
    fn compile(&self, graph: &mut Graph);
}

impl Compiler for () {
    fn compile(&self, _: &mut Graph) {}
}

/// Wrap this around a compiler to measure the time it takes to compile
pub struct TimedCompiler<C: Compiler + Debug>(C);

impl<C: Compiler + Debug> Compiler for TimedCompiler<C> {
    fn compile(&self, graph: &mut Graph) {
        let compiler_name = format!("{:?}", self.0).bold();
        println!("Starting {compiler_name}");
        let start = std::time::Instant::now();
        self.0.compile(graph);
        println!(
            "Finished {compiler_name} in {}",
            format!("{:?}ms", start.elapsed().as_millis()).bold(),
        );
    }
}

impl<C: Default + Compiler + Debug> Default for TimedCompiler<C> {
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
            fn compile(&self, graph: &mut Graph) {
                $(self.$idx.compile(graph);)+
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
    pub fn add_op<O: Operator + 'static>(&mut self, op: O) -> NewOp {
        self.graph.free_node = NodeIndex::end(); // Prevent reuse of deleted indexes (screws up remapping)
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

    /// Convert to debug-viewable graph
    pub fn debug_graph(
        &self,
        show_shapes: bool,
    ) -> (
        StableGraph<String, u8>,
        Vec<EdgeIndex>,
        HashMap<NodeIndex, NodeIndex>,
    ) {
        let mut new_graph = StableGraph::default();
        let mut id_map = HashMap::new();
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

    pub fn display_group(&self, group: &[NodeIndex]) {
        let (g, e, id_map) = self.debug_graph(false);
        display_graph(&g, &e, &group.iter().map(|i| id_map[i]).collect::<Vec<_>>());
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

    pub fn input(mut self, id: NodeIndex, from_output: u8, shape: ShapeTracker) -> Self {
        self.graph_ref.graph.add_edge(
            remap_id(id, &self.graph_ref.id_remap),
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
pub fn move_references(
    id_remap: &mut HashMap<NodeIndex, NodeIndex>,
    no_delete: &mut HashSet<NodeIndex<u32>>,
    to_retrieve: &mut HashSet<NodeIndex<u32>>,
    src: NodeIndex,
    trg: NodeIndex,
) {
    // Only remap if it isn't already remapped, otherwise it would invalidate the past remappig
    if let Entry::Vacant(e) = id_remap.entry(src) {
        // Create remap
        e.insert(trg);
        // Transfer no_delete
        if no_delete.remove(&src) {
            no_delete.insert(trg);
        }
        // Transfer to_retrieve
        if to_retrieve.remove(&src) {
            to_retrieve.insert(trg);
        }
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
    fn is_equal(&self, _: &dyn TraitObjEq) -> bool;
}

// Implement TraitObjEq for all 'static types implementing PartialEq.
impl<S: 'static + PartialEq> TraitObjEq for S {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn is_equal(&self, other: &dyn TraitObjEq) -> bool {
        // Do a type-safe casting. If the types are different,
        // return false, otherwise test the values for equality.
        other
            .as_any()
            .downcast_ref::<S>()
            .map_or(false, |a| self == a)
    }
}

type SelectionGraph = petgraph::Graph<SelectOp, Option<u8>>;

pub struct GraphSearch {
    selector: SelectionGraph,
    graph: *const Graph,
    already_returned: HashSet<BTreeMap<NodeIndex, NodeIndex>>,
}

impl Iterator for GraphSearch {
    type Item = ();

    fn next(&mut self) -> Option<Self::Item> {
        // Look through graph for pattern from selector
        let graph = unsafe { self.graph.as_ref().unwrap() };

        if let Some(mapping) = match_nodes(&graph.graph, &self.selector, &self.already_returned) {
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
            self.already_returned.insert(mapping);
            return Some(());
        }
        None
    }
}

fn match_nodes(
    g0: &MainGraph,
    g1: &SelectionGraph,
    already_returned: &HashSet<BTreeMap<NodeIndex, NodeIndex>>,
) -> Option<BTreeMap<NodeIndex, NodeIndex>> {
    let Ok(topo_order) = toposort(g1, None) else {
        return None;
    };
    let g1_nodes = g1.node_count();
    let predecessors_map = g0
        .node_indices()
        .map(|node| {
            (
                node,
                g0.neighbors_directed(node, Direction::Incoming)
                    .collect::<BTreeSet<_>>(),
            )
        })
        .collect::<HashMap<_, _>>();
    let mut stack = VecDeque::new();
    stack.push_back((0, BTreeMap::new(), HashSet::<NodeIndex>::default()));
    while let Some((current_index, mut mapping, mut visited)) = stack.pop_back() {
        if mapping.len() == g1_nodes && !already_returned.contains(&mapping) {
            return Some(mapping);
        }
        if current_index >= topo_order.len() {
            continue;
        }
        let node_g1 = topo_order[current_index];
        for node_g0 in g0.node_indices() {
            if !visited.contains(&node_g0) {
                // Get predecesors in g0
                let predecessors = &predecessors_map[&node_g0];

                // Check if all g1 predecessors are present in g0
                let predecessors_match =
                    g1.neighbors_directed(node_g1, Direction::Incoming)
                        .all(|pred_g1| {
                            mapping
                                .get(&pred_g1)
                                .map(|i| predecessors.contains(i))
                                .unwrap_or_default()
                        });

                // If so, record this node as visited, map it and take a step down the toposort
                if predecessors_match && test_node(g1.node_weight(node_g1).unwrap(), g0, node_g0) {
                    visited.insert(node_g0);
                    mapping.insert(node_g1, node_g0);

                    stack.push_back((current_index + 1, mapping.clone(), visited.clone()));

                    // Couldnt find a downstream match, unmark and unmap this node
                    visited.remove(&node_g0);
                    mapping.remove(&node_g1);
                }
            }
        }
    }

    None
}

fn test_node(
    SelectOp {
        type_id,
        check,
        shape,
        fake,
        pointers: _,
    }: &SelectOp,
    graph: &MainGraph,
    graph_node: NodeIndex,
) -> bool {
    let current_weight = graph.node_weight(graph_node).unwrap();
    // Test type
    if let Some(ty) = type_id {
        if current_weight.as_any().type_id() != *ty {
            return false;
        }
    }
    let input_shapes = graph
        .edges_directed(graph_node, petgraph::Direction::Incoming)
        .filter_map(|e| e.weight().as_data())
        .sorted_by_key(|e| e.0)
        .map(|e| e.2)
        .collect::<Vec<_>>();

    // Test shape
    if let Some(shape) = shape {
        let mut shape_map = HashMap::new();
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
        if !check(current_weight.as_ref(), &input_shapes) {
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
    check: Option<fn(&dyn Operator, &[ShapeTracker]) -> bool>,
    /// Shape constraint
    shape: Option<Vec<Vec<Expression>>>,
    /// Fake constraint       
    fake: Option<Vec<Vec<Option<bool>>>>,
    /// Pointers      
    pointers: Vec<*mut NodeIndex>,
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
    pub fn check(mut self, check: fn(&dyn Operator, &[ShapeTracker]) -> bool) -> Self {
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
        let mut node_map = HashMap::new();
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

    pub fn search(self, graph: &Graph) -> GraphSearch {
        GraphSearch {
            selector: self.graph,
            graph,
            already_returned: HashSet::new(),
        }
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
        let cx = Graph::default();
        // Exp -> Log or Log -> Exp
        let (mut exp, mut log) = (NodeIndex::default(), NodeIndex::default());
        let selector1 = SelectEdge::new(
            SelectOp::new().ty::<Log2>().ptr(&mut log),
            SelectOp::new().ty::<Exp2>().ptr(&mut exp),
        );
        let selector2 = SelectEdge::new(
            SelectOp::new().ty::<Exp2>().ptr(&mut exp),
            SelectOp::new().ty::<Log2>().ptr(&mut log),
        );

        assert_eq!(
            selector1.search(&cx).chain(selector2.search(&cx)).next(),
            None
        );

        // Matmul
        let s = SelectEdge::new(
            SelectOp::new().ty::<Mul>(),
            SelectOp::new().ty::<SumReduce>(),
        );

        assert_eq!(s.search(&cx).next(), None);
    }
}
