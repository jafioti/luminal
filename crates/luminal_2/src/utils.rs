#![allow(unused)]

use std::collections::HashMap;

use egglog::{EGraph, Error, Term, prelude::exprs::var};
use luminal::{
    prelude::{
        NodeIndex,
        petgraph::{
            Directed, Direction,
            algo::toposort,
            dot::{Config, Dot},
            prelude::StableGraph,
            visit::Topo,
        },
    },
    shape::Expression,
};
use regex::Regex;

pub fn unary(
    a: NodeIndex,
    term: GraphTerm,
    graph: &mut StableGraph<GraphTerm, (), Directed>,
) -> NodeIndex {
    let tmp = graph.add_node(term);
    graph.add_edge(a, tmp, ());
    tmp
}

pub fn binary(
    a: NodeIndex,
    b: NodeIndex,
    term: GraphTerm,
    graph: &mut StableGraph<GraphTerm, (), Directed>,
) -> NodeIndex {
    let tmp = graph.add_node(term);
    graph.add_edge(a, tmp, ());
    graph.add_edge(b, tmp, ());
    tmp
}

pub fn loop_in(
    node: NodeIndex,
    range: impl Into<Expression>,
    stride: impl Into<Expression>,
    marker: impl ToString,
    graph: &mut StableGraph<GraphTerm, (), Directed>,
) -> NodeIndex {
    unary(
        node,
        GraphTerm::LoopIn {
            range: range.into(),
            stride: stride.into(),
            marker: marker.to_string(),
        },
        graph,
    )
}

pub fn loop_out(
    node: NodeIndex,
    range: impl Into<Expression>,
    stride: impl Into<Expression>,
    marker: impl ToString,
    graph: &mut StableGraph<GraphTerm, (), Directed>,
) -> NodeIndex {
    unary(
        node,
        GraphTerm::LoopOut {
            range: range.into(),
            stride: stride.into(),
            marker: marker.to_string(),
        },
        graph,
    )
}

pub fn pad_in(
    mut node: NodeIndex,
    graph: &mut StableGraph<GraphTerm, (), Directed>,
    levels: usize,
) -> NodeIndex {
    for i in 0..levels {
        node = loop_in(node, 1, 0, "pad".to_string(), graph);
    }
    node
}

pub fn pad_out(
    mut node: NodeIndex,
    graph: &mut StableGraph<GraphTerm, (), Directed>,
    levels: usize,
) -> NodeIndex {
    for i in (0..levels).rev() {
        node = loop_out(node, 1, 0, "pad".to_string(), graph);
    }
    node
}

use crate::{GraphTerm, Kernel};

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
            GraphTerm::Exp2 => "Exp".to_string(),
            GraphTerm::Log2 => "Log2".to_string(),
            GraphTerm::Sin => "Sin".to_string(),
            GraphTerm::Recip => "Recip".to_string(),
            GraphTerm::Neg => "Neg".to_string(),
            GraphTerm::Sqrt => "Sqrt".to_string(),
            GraphTerm::Mod => "Mod".to_string(),
            GraphTerm::LessThan => "LessThan".to_string(),
            GraphTerm::LoopIn {
                range,
                stride,
                marker,
            } => format!("LoopIn ({range}; {stride}; -{marker}-)"),
            GraphTerm::LoopOut {
                range,
                stride,
                marker,
            } => format!("LoopOut ({range}; {stride}; -{marker}-)"),
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
    graph: &StableGraph<G, E, Directed, u32>,
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
    let mut graph_string = Dot::with_config(&new_graph, &[Config::EdgeIndexLabel]).to_string();
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

pub fn validate_graph(graph: &StableGraph<(GraphTerm, usize), (), Directed>) {
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
                && !matches!(graph.node_weight(node).unwrap().0, GraphTerm::SMEM)
            {
                if *curr_level != 0 {
                    display_graph(graph, &[(node, "yellow".to_string())]);
                    panic!("Inputs must have level 0, found {curr_level}");
                }
            }
        }
    }
}

pub fn build_search_space(
    graph: &StableGraph<GraphTerm, (), Directed>,
    iters: usize,
    remove_tiling: bool,
) -> egraph_serialize::EGraph {
    let (rendered, root) = render_egglog(graph);
    if option_env!("PRINT_KERNELS").is_some() {
        println!("{rendered}");
    }
    let code = include_str!("code.lisp");

    let mut final_code = code
        .replace("{code}", &rendered)
        .replace("{iters}", &iters.to_string());
    if remove_tiling {
        final_code = final_code.replace(
            r#"(rewrite
	(LoopOut ?body (Loop ?loop (MNum ?range)) ?stride)
	(LoopOut
		(LoopOut
			(TileLoop ?body ?loop)
			(Loop (+ ?loop "_tile") (MNum 8))
			?stride
		)
		(Loop ?loop (MNum (/ ?range 8)))
		(MReplace ?stride (MVar "z") (MMul (MVar "z") (MNum 8)))
	)
	:when ((> ?range 8) (= (% ?range 8) 0))
)"#,
            "",
        ); // tiling rule is causing an egglog bug!
    }
    let (_egglog_messages, serialized) = run_egglog_program(&final_code, &root).unwrap();
    println!("Done");
    serialized
}

fn render_egglog(graph: &StableGraph<GraphTerm, (), Directed>) -> (String, String) {
    // 1.  Topo-order so operands are rendered before users
    let mut topo = Topo::new(&graph);

    // 2.  Map <node-id> → <egglog var name>
    let mut names: HashMap<NodeIndex, String> = HashMap::new();
    let mut next_id = 0usize;
    let mut out = String::new();

    // helper to fetch operand text (there are up-edges from user → operand)
    let operand = |n: NodeIndex,
                   names: &HashMap<NodeIndex, String>,
                   g: &StableGraph<GraphTerm, (), Directed>|
     -> Vec<String> {
        g.neighbors_directed(n, Direction::Incoming)
            .map(|child| names[&child].clone())
            .collect()
    };

    while let Some(n) = topo.next(&graph) {
        let var = format!("t{next_id}");
        next_id += 1;
        let code = match &graph[n] {
            GraphTerm::GMEM { label } => {
                format!("(GMEM \"{}\")", label.clone().unwrap_or_default())
            }
            GraphTerm::SMEM => "(SMEM)".into(),

            GraphTerm::LoopIn {
                range,
                stride,
                marker,
            } => {
                let [ref src] = operand(n, &names, &graph)[..] else {
                    panic!("LoopIn expects 1 child");
                };
                format!(
                    "(LoopIn {src} (Loop \"{marker}\" {}) {})",
                    range.to_egglog(),
                    stride.to_egglog()
                )
            }
            GraphTerm::LoopOut {
                range,
                stride,
                marker,
            } => {
                let [ref body] = operand(n, &names, &graph)[..] else {
                    panic!("LoopOut expects 1 child");
                };
                format!(
                    "(LoopOut {body} (Loop \"{marker}\" {}) {})",
                    range.to_egglog(),
                    stride.to_egglog()
                )
            }

            GraphTerm::Add
            | GraphTerm::Mul
            | GraphTerm::Max
            | GraphTerm::Exp2
            | GraphTerm::Log2
            | GraphTerm::Mod
            | GraphTerm::LessThan
            | GraphTerm::Recip
            | GraphTerm::Sin
            | GraphTerm::Neg
            | GraphTerm::Sqrt
            | GraphTerm::SMEMLoad
            | GraphTerm::SMEMRead => {
                let mut ops = operand(n, &names, &graph);
                let op = match &graph[n] {
                    GraphTerm::Add => "Add",
                    GraphTerm::Mul => "Mul",
                    GraphTerm::Max => "Max",
                    GraphTerm::Exp2 => "Exp2",
                    GraphTerm::Log2 => "Log2",
                    GraphTerm::Recip => "Recip",
                    GraphTerm::Sin => "Sin",
                    GraphTerm::Neg => "Neg",
                    GraphTerm::Sqrt => "Sqrt",
                    GraphTerm::Mod => "Mod",
                    GraphTerm::LessThan => "LessThan",
                    GraphTerm::SMEMLoad => "SMEMLoad",
                    GraphTerm::SMEMRead => "SMEMRead",
                    _ => unreachable!(),
                };
                if ops.len() == 1 {
                    format!("({op} {})", ops.pop().unwrap())
                } else {
                    format!("({op} {})", ops.join(" "))
                }
            }
        };

        out.push_str(&format!("(let {var} {code})\n"));
        names.insert(n, var);
    }

    let root = graph
        .node_indices()
        .find(|&idx| {
            graph
                .neighbors_directed(idx, Direction::Outgoing)
                .next()
                .is_none()
        })
        .and_then(|idx| names.get(&idx))
        .cloned()
        .unwrap_or_else(|| "t0".into());
    (out, root)
}

/// Runs an Egglog program from a string and returns its output messages.
fn run_egglog_program(
    code: &str,
    root: &str,
) -> Result<(Vec<String>, egraph_serialize::EGraph), Error> {
    // Create a fresh EGraph with all the defaults
    let mut egraph = EGraph::default();
    egraph.enable_messages();
    let commands = egraph.parser.get_program_from_string(None, code)?;
    let msgs = egraph.run_program(commands)?;
    if option_env!("PRINT_EGGLOG")
        .map(|s| s.parse::<i32>().map(|i| i == 1).unwrap_or_default())
        .unwrap_or_default()
    {
        println!("Run Report:  {}", egraph.get_run_report().as_ref().unwrap());
    }
    let (sort, value) = egraph.eval_expr(&egglog::var!(root))?;
    // let (_petgraph, _root_idx) = dag_to_petgraph(&termdag, termdag.lookup(&root));
    let s = egraph.serialize(egglog::SerializeConfig {
        root_eclasses: vec![(sort, value)],
        ..Default::default()
    });
    if option_env!("PRINT_EGGLOG")
        .map(|s| s.parse::<i32>().map(|i| i == 1).unwrap_or_default())
        .unwrap_or_default()
    {
        println!(
            "Nodes: {} Roots: {} Class Data: {}",
            s.nodes.len(),
            s.root_eclasses.len(),
            s.class_data.len()
        );
    }
    Ok((msgs, s))
}

pub fn print_kernels(kernels: &StableGraph<Kernel, (usize, usize), Directed>) {
    println!("Kernels: {}", kernels.node_count() - 2);
    for (i, node) in toposort(&kernels, None).unwrap().into_iter().enumerate() {
        let Kernel {
            code,
            grid,
            threadblock,
            smem,
            outputs,
        } = kernels.node_weight(node).unwrap();
        if !code.starts_with("Inputs") && code != "Outputs" {
            println!("Kernel {i} Grid: {grid:?} Threadblock: {threadblock:?} Smem: {smem}");
            println!("{code}");
            println!("Outputs: {:?}", outputs);
        }
    }
}
