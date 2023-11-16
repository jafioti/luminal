use std::{
    any::{Any, TypeId},
    collections::{hash_map::Entry, HashMap, HashSet},
    fmt::Debug,
    sync::{Arc, Mutex},
};

use itertools::Itertools;
use petgraph::{
    algo::toposort,
    stable_graph::{EdgeIndex, NodeIndex, StableGraph},
    visit::EdgeRef,
    Directed, Direction,
};
use regex::Regex;

use crate::{
    graph::Graph,
    op::Operator,
    prelude::{remap_id, Dependency, Dim, MainGraph, ShapeTracker},
};

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
        println!("Starting {:?}", self.0);
        let start = std::time::Instant::now();
        self.0.compile(graph);
        println!("Finished {:?} in {}ms", self.0, start.elapsed().as_millis());
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
    ) -> (StableGraph<String, u8, Directed, u32>, Vec<EdgeIndex>) {
        let mut new_graph = StableGraph::default();
        let mut id_map = HashMap::new();
        let op_regex = Regex::new(r"(?s)\{.*|\(.*").unwrap();
        for (id, node) in self.graph.node_indices().zip(self.graph.node_weights()) {
            id_map.insert(
                id,
                new_graph.add_node(op_regex.replace_all(&format!("{node:?}"), "").to_string()),
            );
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
                    && !edge.weight().as_data().unwrap().2.shape().is_empty()
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

        (new_graph, schedule_edges)
    }

    pub fn display(&self) {
        let (g, e) = self.debug_graph(false);
        display_graph(&g, &e);
    }

    pub fn display_shapes(&self) {
        let (g, e) = self.debug_graph(true);
        display_graph(&g, &e);
    }

    /// Get the sources of a node given it's id
    #[allow(clippy::type_complexity, clippy::borrowed_box)]
    pub fn get_sources(&self, node_id: NodeIndex) -> Vec<(NodeIndex, ShapeTracker)> {
        self.graph
            .edges_directed(node_id, Direction::Incoming)
            .filter_map(|e| e.weight().as_data().map(|i| (e.source(), i)))
            .sorted_by_key(|(_, (i, _, _))| *i)
            .map(|(a, (_, _, b))| (a, b))
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

type SelectorWeight = (
    Option<TypeId>,                                     // Type constraint
    Option<fn(&dyn Operator, &[ShapeTracker]) -> bool>, // Check constraint
    Option<Vec<Vec<Dim>>>,                              // Shape constraint
    Option<Vec<Vec<bool>>>,                             // Fake constraint
    Vec<*mut NodeIndex>,                                // Pointers
);

type SelectionGraph = petgraph::Graph<SelectorWeight, Option<u8>>;

// Graph Selector
#[derive(Default, Clone)]
pub struct GraphSelector {
    graph: Arc<Mutex<SelectionGraph>>,
}

// pub struct GraphSearch {
//     selector: GraphSelector,
//     already_returned: HashSet<NodeIndex>,
//     graph: *const Graph,
// }

// impl Iterator for GraphSearch {
//     type Item = ();

//     fn next(&mut self) -> Option<Self::Item> {
//         // Look through graph for pattern from selector
//         let Some(select_start) = self.selector.graph.lock().unwrap().node_indices().next() else {
//             return None;
//         };
//         let graph = unsafe { self.graph.as_ref().unwrap() };
//         let mut used = HashSet::new();
//         let mut selector_used = HashSet::new();
//         for node in graph.graph.node_indices() {
//             if self.already_returned.contains(&node) {
//                 continue;
//             }
//             if search(
//                 select_start,
//                 &self.selector.graph.lock().unwrap(),
//                 node,
//                 &graph.graph,
//                 &mut used,
//                 &mut selector_used,
//             ) {
//                 self.already_returned.insert(node);
//                 return Some(());
//             }
//             used.clear();
//         }
//         None
//     }
// }

pub struct GraphSearch {
    selector: GraphSelector,
    graph: *const Graph,
    already_returned: HashSet<Vec<(NodeIndex, NodeIndex)>>,
}

impl Iterator for GraphSearch {
    type Item = ();

    fn next(&mut self) -> Option<Self::Item> {
        // Look through graph for pattern from selector
        let graph = unsafe { self.graph.as_ref().unwrap() };
        let mut mapping = HashMap::new();
        let mut visited = HashSet::new();
        let selector_graph = self.selector.graph.lock().unwrap();

        if let Some(map) = match_nodes(
            &graph.graph,
            &selector_graph,
            &mut mapping,
            &mut visited,
            &self.already_returned,
        ) {
            self.already_returned.insert(
                map.iter()
                    .sorted_by_key(|(i, _)| **i)
                    .map(|(a, b)| (*a, *b))
                    .collect::<Vec<_>>(),
            );
            // Apply pattern to ptrs
            for (selector_node, ptr) in selector_graph.node_indices().flat_map(|n| {
                selector_graph
                    .node_weight(n)
                    .unwrap()
                    .4
                    .iter()
                    .map(move |i| (n, *i))
            }) {
                unsafe {
                    *ptr = map[&selector_node];
                }
            }
            return Some(());
        }
        None
    }
}

fn match_nodes(
    g0: &MainGraph,
    g1: &SelectionGraph,
    mapping: &mut HashMap<NodeIndex, NodeIndex>,
    visited: &mut HashSet<NodeIndex>,
    already_returned: &HashSet<Vec<(NodeIndex, NodeIndex)>>,
) -> Option<HashMap<NodeIndex, NodeIndex>> {
    let mut predecessors_cache: HashMap<NodeIndex, Vec<NodeIndex>> = HashMap::new();

    if let Ok(topo_order) = toposort(g1, None) {
        match_nodes_topo(
            g0,
            g1,
            &topo_order,
            mapping,
            visited,
            &mut predecessors_cache,
            0,
            already_returned,
        )
    } else {
        None
    }
}

#[allow(clippy::too_many_arguments)]
fn match_nodes_topo(
    g0: &MainGraph,
    g1: &SelectionGraph,
    topo_order: &[NodeIndex],
    mapping: &mut HashMap<NodeIndex, NodeIndex>,
    visited: &mut HashSet<NodeIndex>,
    predecessors_cache: &mut HashMap<NodeIndex, Vec<NodeIndex>>,
    current_index: usize,
    already_returned: &HashSet<Vec<(NodeIndex, NodeIndex)>>,
) -> Option<HashMap<NodeIndex, NodeIndex>> {
    if mapping.len() == g1.node_count() {
        let key = mapping
            .iter()
            .sorted_by_key(|(i, _)| **i)
            .map(|(a, b)| (*a, *b))
            .collect::<Vec<_>>();
        if !already_returned.contains(&key) {
            return Some(mapping.clone());
        }
        return None;
    }

    if current_index >= topo_order.len() {
        return None;
    }

    let node_g1 = topo_order[current_index];

    for node_g0 in g0.node_indices() {
        if !visited.contains(&node_g0) && test_node(g1.node_weight(node_g1).unwrap(), g0, node_g0) {
            let predecessors = predecessors_cache.entry(node_g0).or_insert_with(|| {
                g0.neighbors_directed(node_g0, Direction::Incoming)
                    .collect()
            });

            let predecessors_match =
                g1.neighbors_directed(node_g1, Direction::Incoming)
                    .all(|pred_g1| {
                        if let Some(&pred_g0) = mapping.get(&pred_g1) {
                            predecessors.iter().any(|p| p == &pred_g0)
                        } else {
                            false
                        }
                    });

            if predecessors_match {
                visited.insert(node_g0);
                mapping.insert(node_g1, node_g0);

                if let Some(m) = match_nodes_topo(
                    g0,
                    g1,
                    topo_order,
                    mapping,
                    visited,
                    predecessors_cache,
                    current_index + 1,
                    already_returned,
                ) {
                    return Some(m);
                }

                visited.remove(&node_g0);
                mapping.remove(&node_g1);
            }
        }
    }
    None
}

fn test_node(
    (type_id, check, shape, fakes, _): &SelectorWeight,
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
                match a {
                    Dim::Known(n) => {
                        if *b != Dim::Known(*n) {
                            return false;
                        }
                    }
                    Dim::Unknown(c) => {
                        if let Some(expected) = shape_map.get(c) {
                            if b != expected {
                                return false;
                            }
                        } else {
                            shape_map.insert(*c, *b);
                        }
                    }
                }
            }
        }
    }
    // Test fakes
    if let Some(fakes) = fakes {
        for (a_sh, b_sh) in fakes.iter().zip(input_shapes.iter()) {
            for (a, b) in a_sh.iter().zip(b_sh.indexes.iter().map(|i| b_sh.fake[*i])) {
                if *a != b {
                    return false;
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

impl GraphSelector {
    /// Create a new op to select
    pub fn op(&self) -> OpSelector {
        let mut m_self = self.graph.lock().unwrap();
        let id = m_self.add_node((None, None, None, None, vec![]));
        OpSelector { graph: self, id }
    }

    /// Specify an edge between ops
    pub fn edge(&self, o1: OpSelector, o2: OpSelector) -> OpSelector {
        let mut m_self = self.graph.lock().unwrap();
        m_self.add_edge(o1.id, o2.id, None);
        o2
    }

    /// Specify an edge between ops with a specified output
    pub fn edge_output(&self, o1: OpSelector, output: u8, o2: OpSelector) -> OpSelector {
        let mut m_self = self.graph.lock().unwrap();
        m_self.add_edge(o1.id, o2.id, Some(output));
        o2
    }

    /// Start searching a graph
    pub fn search(self, graph: &Graph) -> GraphSearch {
        GraphSearch {
            selector: self,
            already_returned: HashSet::default(),
            graph,
        }
    }
}

#[derive(Clone, Copy)]
pub struct OpSelector {
    graph: *const GraphSelector,
    id: NodeIndex,
}

impl OpSelector {
    /// Constrain the op to a type
    pub fn ty<O: Operator + 'static>(self) -> OpSelector {
        let graph = unsafe { self.graph.as_ref().unwrap() };
        let mut m_graph = graph.graph.lock().unwrap();
        m_graph.node_weight_mut(self.id).unwrap().0 = Some(TypeId::of::<O>());
        self
    }
    /// Constrain the op to a type
    pub fn type_id(self, type_id: TypeId) -> OpSelector {
        let graph = unsafe { self.graph.as_ref().unwrap() };
        let mut m_graph = graph.graph.lock().unwrap();
        m_graph.node_weight_mut(self.id).unwrap().0 = Some(type_id);
        self
    }
    /// Constrain the op to a checking function
    pub fn check(self, check: fn(&dyn Operator, &[ShapeTracker]) -> bool) -> Self {
        let graph = unsafe { self.graph.as_ref().unwrap() };
        let mut m_graph = graph.graph.lock().unwrap();
        m_graph.node_weight_mut(self.id).unwrap().1 = Some(check);
        self
    }
    /// Constrain the op to input shapes
    pub fn shapes<S: Into<Vec<Vec<Dim>>>>(self, shapes: S) -> Self {
        let graph = unsafe { self.graph.as_ref().unwrap() };
        let mut m_graph = graph.graph.lock().unwrap();
        m_graph.node_weight_mut(self.id).unwrap().2 = Some(shapes.into());
        self
    }
    /// Constrain the op to input shape fakes
    pub fn fakes<S: Into<Vec<Vec<bool>>>>(self, fakes: S) -> Self {
        let graph = unsafe { self.graph.as_ref().unwrap() };
        let mut m_graph = graph.graph.lock().unwrap();
        m_graph.node_weight_mut(self.id).unwrap().3 = Some(fakes.into());
        self
    }
    /// Register a pointer to set if the op is matched
    pub fn ptr(self, ptr: *mut NodeIndex) -> Self {
        let graph = unsafe { self.graph.as_ref().unwrap() };
        let mut m_graph = graph.graph.lock().unwrap();
        m_graph.node_weight_mut(self.id).unwrap().4.push(ptr);
        self
    }
}

#[cfg(test)]
mod tests {
    use petgraph::adj::NodeIndex;

    use crate::{
        op::{Exp2, Log2, Mul, SumReduce},
        prelude::Graph,
    };

    use super::GraphSelector;

    #[test]
    fn test_graph_selector() {
        let cx = Graph::default();
        // Exp -> Log or Log -> Exp
        let (mut exp, mut log) = (NodeIndex::default(), NodeIndex::default());
        let selector1 = GraphSelector::default();
        selector1.edge(
            selector1.op().ty::<Log2>().ptr(&mut log),
            selector1.op().ty::<Exp2>().ptr(&mut exp),
        );
        let selector2 = GraphSelector::default();
        selector2.edge(
            selector2.op().ty::<Exp2>().ptr(&mut exp),
            selector2.op().ty::<Log2>().ptr(&mut log),
        );

        assert_eq!(
            selector1.search(&cx).chain(selector2.search(&cx)).next(),
            None
        );

        // Matmul
        let s = GraphSelector::default();
        s.edge(s.op().ty::<Mul>(), s.op().ty::<SumReduce>());

        assert_eq!(s.search(&cx).next(), None);
    }
}
