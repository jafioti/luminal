use std::collections::{HashMap, HashSet};

use egglog::Term;
use egraph_serialize::{ClassId, EGraph, NodeId};
use itertools::Itertools;
use petgraph::{Directed, Direction, graph::NodeIndex, prelude::StableGraph};
use regex::Regex;
use rustc_hash::FxHashMap;

use crate::{GraphTerm, Kernel, extract::ExtractionResult, symbolic::Expression};

pub trait TermToString {
    fn term_to_string(&self) -> String;
}

pub trait EdgeToString {
    fn edge_to_string(&self) -> String;
}

impl EdgeToString for u8 {
    fn edge_to_string(&self) -> String {
        self.to_string()
    }
}

impl EdgeToString for () {
    fn edge_to_string(&self) -> String {
        "".to_string()
    }
}

impl EdgeToString for (u8, u8) {
    fn edge_to_string(&self) -> String {
        format!("{}, {}", self.0, self.1)
    }
}

impl TermToString for Term {
    fn term_to_string(&self) -> String {
        match self {
            Term::App(a, _) => a.to_string(),
            Term::Lit(l) => l.to_string(),
            Term::Var(v) => v.to_string(),
        }
    }
}

impl TermToString for usize {
    fn term_to_string(&self) -> String {
        self.to_string()
    }
}

impl TermToString for String {
    fn term_to_string(&self) -> String {
        self.clone()
    }
}

impl TermToString for (Term, usize) {
    fn term_to_string(&self) -> String {
        let s = match &self.0 {
            Term::App(a, _) => a.to_string(),
            Term::Lit(l) => l.to_string(),
            Term::Var(v) => v.to_string(),
        };
        format!("{s}[{}]", self.1)
    }
}

impl TermToString for Kernel {
    fn term_to_string(&self) -> String {
        if self.code == "Inputs" || self.code == "Outputs" {
            return self.code.clone();
        } else {
            format!(
                "Kernel ({}, {}, {}), ({}, {}, {}) -> {:?}",
                self.grid.0,
                self.grid.1,
                self.grid.2,
                self.threadblock.0,
                self.threadblock.1,
                self.threadblock.2,
                self.outputs
            )
        }
    }
}

impl TermToString for GraphTerm {
    fn term_to_string(&self) -> String {
        match self {
            GraphTerm::Add => "Add".to_string(),
            GraphTerm::Mul => "Mul".to_string(),
            GraphTerm::Max => "Max".to_string(),
            GraphTerm::Exp => "Exp".to_string(),
            GraphTerm::Sin => "Sin".to_string(),
            GraphTerm::Recip => "Recip".to_string(),
            GraphTerm::Neg => "Neg".to_string(),
            GraphTerm::NewAcc { starting_value } => format!("NewAcc({starting_value})"),
            GraphTerm::LoopIn { range, stride } => format!("LoopIn ({range}; {stride})"),
            GraphTerm::LoopOut { range, stride } => format!("LoopOut ({range}; {stride})"),
            GraphTerm::GMEM { label } => {
                if let Some(label) = label {
                    format!("GMEM ({label})")
                } else {
                    "GMEM".to_string()
                }
            }
            GraphTerm::SMEM => "SMEM".to_string(),
            GraphTerm::SMEMLoad => "SMEMLoad".to_string(),
            GraphTerm::SMEMRead => "SMEMRead".to_string(),
        }
    }
}

impl TermToString for (GraphTerm, usize) {
    fn term_to_string(&self) -> String {
        format!("{} [{}]", self.0.term_to_string(), self.1)
    }
}

impl TermToString for (GraphTerm, Vec<Expression>, Vec<usize>) {
    fn term_to_string(&self) -> String {
        format!("{} {:?} {{{:?}}}", self.0.term_to_string(), self.1, self.2)
    }
}

impl TermToString for (GraphTerm, Vec<String>, Vec<usize>) {
    fn term_to_string(&self) -> String {
        format!(
            "{} [{}] {{{:?}}}",
            self.0.term_to_string(),
            self.1.len(),
            self.2
        )
    }
}

/// View a debug graph in the browser
pub fn display_graph<G: TermToString, E: EdgeToString>(
    graph: &petgraph::stable_graph::StableGraph<G, E, petgraph::Directed, u32>,
    mark_nodes: &[(NodeIndex, String)],
) {
    let mut new_graph = StableGraph::new();
    let mut map = HashMap::new();
    for node in graph.node_indices() {
        map.insert(
            node,
            new_graph.add_node(graph.node_weight(node).unwrap().term_to_string()),
        );
    }
    for edge in graph.edge_indices() {
        let weight = graph.edge_weight(edge).unwrap();
        let (src, dest) = graph.edge_endpoints(edge).unwrap();
        new_graph.add_edge(map[&src], map[&dest], weight.edge_to_string());
    }
    let mut graph_string =
        petgraph::dot::Dot::with_config(&new_graph, &[petgraph::dot::Config::EdgeIndexLabel])
            .to_string();
    let re = Regex::new(r#"label\s*=\s*"\d+""#).unwrap();
    graph_string = re.replace_all(&graph_string, "").to_string();
    for (n, color) in mark_nodes {
        graph_string = graph_string.replace(
            &format!("    {} [ label =", n.index()),
            &format!(
                "    {} [ style=\"filled\" fillcolor=\"{color}\" label =",
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

pub fn validate_graph(graph: &StableGraph<(GraphTerm, usize), u8, Directed>) {
    // walk the graph and make sure loopins -> next loop level (or loopout) and prev loop (or loopin) -> loopout
    for node in graph.node_indices() {
        let (curr_term, curr_level) = graph.node_weight(node).unwrap();
        if matches!(curr_term, GraphTerm::LoopIn { .. }) {
            // All loopins must have outputs that are one level more, unless they are loopouts
            for new_node in graph.neighbors_directed(node, Direction::Outgoing) {
                let (new_term, new_level) = graph.node_weight(new_node).unwrap();
                if !matches!(new_term, GraphTerm::LoopOut { .. }) {
                    if *new_level != *curr_level + 1 {
                        display_graph(
                            graph,
                            &[
                                (node, "yellow".to_string()),
                                (new_node, "yellow".to_string()),
                            ],
                        );
                        panic!("incorrect levels");
                    }
                }
            }
        } else if matches!(curr_term, GraphTerm::LoopOut { .. }) {
            // All loopouts must have inputs that are one level more, unless they are loopins
            for new_node in graph.neighbors_directed(node, Direction::Incoming) {
                let (new_term, new_level) = graph.node_weight(new_node).unwrap();
                if !matches!(new_term, GraphTerm::LoopIn { .. }) {
                    if *new_level != *curr_level + 1 {
                        display_graph(
                            graph,
                            &[
                                (node, "yellow".to_string()),
                                (new_node, "yellow".to_string()),
                            ],
                        );
                        panic!("incorrect levels");
                    }
                }
            }
        } else {
            for new_node in graph.neighbors_directed(node, Direction::Incoming) {
                let (new_term, new_level) = graph.node_weight(new_node).unwrap();
                if !matches!(
                    new_term,
                    GraphTerm::LoopIn { .. } | GraphTerm::LoopOut { .. }
                ) {
                    if *new_level != *curr_level {
                        display_graph(
                            graph,
                            &[
                                (node, "yellow".to_string()),
                                (new_node, "yellow".to_string()),
                            ],
                        );
                        panic!("incorrect levels");
                    }
                }
            }

            if graph
                .neighbors_directed(node, Direction::Incoming)
                .next()
                .is_none()
                && !matches!(
                    graph.node_weight(node).unwrap().0,
                    GraphTerm::NewAcc { .. } | GraphTerm::SMEM
                )
            {
                if *curr_level != 0 {
                    display_graph(graph, &[(node, "yellow".to_string())]);
                    println!("Inputs must have level 0, found {curr_level}");
                }
            }
        }
    }
}

pub fn extraction_to_petgraph<'a>(
    egraph: &'a EGraph,
    extraction: &'a ExtractionResult<'a>,
) -> StableGraph<String, (), Directed> {
    let mut map = HashMap::<&NodeId, NodeIndex>::default();
    let mut dfs = egraph
        .root_eclasses
        .iter()
        .map(|c| extraction.choices[c])
        .collect_vec();
    let mut graph = StableGraph::default();
    for root in &dfs {
        let new_node = graph.add_node(egraph.nodes[*root].op.clone());
        map.insert(*root, new_node);
    }
    let mut finished = HashSet::<&NodeId>::new();

    while let Some(node) = dfs.pop() {
        if finished.contains(node) {
            continue;
        }
        // Create children
        let g_node = map[node];
        for child in &egraph.nodes[node].children {
            let g_child = match map.get(child) {
                Some(c) => *c,
                None => {
                    let new_node = graph.add_node(egraph.nodes[child].op.clone());
                    map.insert(child, new_node);
                    dfs.push(child);
                    new_node
                }
            };
            graph.add_edge(g_node, g_child, ());
        }
        finished.insert(node);
    }
    graph
}
