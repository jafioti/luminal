// TODO
// unit test complex codegen with correctness checks
// get profiling working
// get brute force extraction working

// conceptual sketch of codegen process
//
// egglog ir -> termdag_to_petgraph() -> term_graph
//
//
// fn codegen(term_graph) {
// 		let kernels = split_kernels(term_graph);
// 		for (kernel_graph, inputs, outputs, smem_buffers) in kernels {
// 			let kernel_lines = make_kernel(kernel_graph, inputs, outputs, smem_buffers);
// 			let kernel = format!("...", kernel_lines.join("\n"));
//		}
// }
//
// fn split_kernels(term_graph) {
// 		let metadata_graph = make_metadata_graph(term_graph); // add in per-node metadata like loop level and kernel indexes
// 		get_loop_levels(&mut metadata_graph);
// 		get_kernel_indexes(&mut metadata_graph);
// 		let kernel_graphs = split_into_kernel_graphs(metadata_graph);
// 		record_smem_buffers(&mut kernel_graphs);
// 		return kernel_graphs;
// }

mod codegen;
mod extract;
mod run;
mod symbolic;
mod utils;

#[cfg(test)]
mod tests;

use colored::Colorize;
use egglog::{EGraph, Error, var};
use itertools::Itertools;
use petgraph::{Directed, graph::NodeIndex, prelude::StableGraph, visit::Topo};
use rand::{Rng, rng};
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, fmt::Debug};
use utils::*;

use crate::{
    extract::search,
    symbolic::{Expression, Term, expression_cleanup},
};

#[derive(Clone, PartialEq, Eq)]
enum GPUArch {
    CUDA,
    Metal(HashMap<usize, &'static str>),
}

impl GPUArch {
    fn metal_buffer_type(&self, var: usize) -> &'static str {
        match self {
            Self::Metal(m) => m.get(&var).copied().unwrap_or(""),
            _ => "",
        }
    }

    fn add_metal_buffer_type(&mut self, var: usize, buf_type: &'static str) {
        if let Self::Metal(m) = self {
            m.insert(var, buf_type);
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Hash)]
struct Kernel {
    code: String,
    // launch params
    grid: (Expression, Expression, Expression),
    threadblock: (Expression, Expression, Expression),
    smem: Expression, // sizes of required shared memory buffers
    outputs: Vec<Expression>,
}

#[derive(Clone)]
enum GMEMBuffer {
    PrevKernel { kernel: usize, output: usize },
    Input { index: usize, label: Option<String> },
}

#[derive(Clone, Debug, Serialize)]
enum GraphTerm {
    GMEM {
        // Signifies global memory
        label: Option<String>,
    },
    LoopIn {
        range: Expression,
        stride: Expression,
        marker: String,
    },
    LoopOut {
        range: Expression,
        stride: Expression,
        marker: String,
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
}

fn main() {
    // let (graph, root) = kernels::make_tiled_matmul_basic(4096, 4096, 4096);
    // let kernels = codegen(graph, root, GPUArch::Metal(HashMap::new()));
    // let mut rng = rng();
    // let a = (0..(4096 * 4096)).map(|_| rng.random()).collect_vec();
    // let b = (0..(4096 * 4096)).map(|_| rng.random()).collect_vec();
    // let mut avgs = vec![];
    // for _ in 0..5 {
    //     let start = std::time::Instant::now();
    //     run_graph(vec![a.clone(), b.clone()], &kernels);
    //     avgs.push(start.elapsed().as_millis());
    // }
    // println!("naive {}ms", avgs.into_iter().sum::<u128>() / 10);

    // let (graph, root) = kernels::make_tiled_matmul(4096, 4096, 4096);
    // let kernels = codegen(graph, root, GPUArch::Metal(HashMap::new()));
    // let mut avgs = vec![];
    // for _ in 0..5 {
    //     let start = std::time::Instant::now();
    //     run_graph(vec![a.clone(), b.clone()], &kernels);
    //     avgs.push(start.elapsed().as_millis());
    // }
    // println!("tiled {}ms", avgs.into_iter().sum::<u128>() / 10);
    // expression_cleanup();
    let start = std::time::Instant::now();
    let (g, _) = make_sum_reduce();
    let (rendered, root) = render_egglog(g);
    let code = include_str!("code.lisp");
    println!("{rendered}");
    let final_code = code.replace("{code}", &rendered);
    match run_egglog_program(&final_code, &root) {
        Ok((_egglog_messages, serialized)) => {
            println!(
                "Search space built in {}",
                format!("{}ms", start.elapsed().as_millis()).bold()
            );
            let mut rng = rng();
            search(
                &serialized,
                &[
                    (0..8 * 16).map(|_| rng.random()).collect_vec(),
                    (0..16 * 32).map(|_| rng.random()).collect_vec(),
                    vec![0.0],
                ],
            );
        }
        Err(e) => println!("{e}"),
    }
    expression_cleanup();
}

fn render_egglog(graph: StableGraph<GraphTerm, (), Directed>) -> (String, String) {
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
        g.neighbors_directed(n, petgraph::Incoming)
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
            | GraphTerm::Exp
            | GraphTerm::Recip
            | GraphTerm::Sin
            | GraphTerm::Neg
            | GraphTerm::SMEMLoad
            | GraphTerm::SMEMRead => {
                let mut ops = operand(n, &names, &graph);
                let op = match &graph[n] {
                    GraphTerm::Add => "Add",
                    GraphTerm::Mul => "Mul",
                    GraphTerm::Max => "Max",
                    GraphTerm::Exp => "Exp",
                    GraphTerm::Recip => "Recip",
                    GraphTerm::Sin => "Sin",
                    GraphTerm::Neg => "Neg",
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
                .neighbors_directed(idx, petgraph::Outgoing)
                .next()
                .is_none()
        })
        .and_then(|idx| names.get(&idx))
        .cloned()
        .unwrap_or_else(|| "t0".into());
    (out, root)
}

fn make_sum_reduce() -> (StableGraph<GraphTerm, (), Directed>, NodeIndex) {
    let (m, k, n) = (8, 16, 32);
    let mut graph = StableGraph::new();

    let mut a = graph.add_node(GraphTerm::GMEM {
        label: Some("A".to_string()),
    });
    a = pad_in(a, &mut graph, 2);
    a = loop_in(a, m, Expression::from('z') * k, 'm', &mut graph);
    a = loop_in(a, n, 0, 'n', &mut graph);
    a = loop_in(a, k, 'z', 'k', &mut graph);

    let mut b = graph.add_node(GraphTerm::GMEM {
        label: Some("B".to_string()),
    });
    b = pad_in(b, &mut graph, 2);
    b = loop_in(b, m, 0, 'm', &mut graph);
    b = loop_in(b, n, 'z', 'n', &mut graph);
    b = loop_in(b, k, Expression::from('z') * n, 'k', &mut graph);

    let mut acc = graph.add_node(GraphTerm::GMEM {
        label: Some("Acc".to_string()),
    });
    acc = pad_in(acc, &mut graph, 2);
    acc = loop_in(acc, m, Expression::from('z') * n, 'm', &mut graph);
    acc = loop_in(acc, n, 'z', 'n', &mut graph);
    acc = loop_in(acc, k, Term::Acc('a'), 'k', &mut graph);

    let mut out = binary(
        binary(a, b, GraphTerm::Mul, &mut graph),
        acc,
        GraphTerm::Add,
        &mut graph,
    );

    out = loop_out(out, k, Term::Acc('a'), 'k', &mut graph);
    out = loop_out(out, n, 'z', 'n', &mut graph);
    out = loop_out(out, m, Expression::from('z') * n, 'm', &mut graph);
    out = pad_out(out, &mut graph, 2);
    (graph, out)
}

/// Runs an Egglog program from a string and returns its output messages.
pub fn run_egglog_program(
    code: &str,
    root: &str,
) -> Result<(Vec<String>, egraph_serialize::EGraph), Error> {
    // Create a fresh EGraph with all the defaults
    let mut egraph = EGraph::default();
    egraph.enable_messages();
    let commands = egraph.parser.get_program_from_string(None, code)?;
    let msgs = egraph.run_program(commands)?;
    if option_env!("PRINT_KERNELS")
        .map(|s| s.parse::<i32>().map(|i| i == 1).unwrap_or_default())
        .unwrap_or_default()
    {
        println!("Run Report:  {}", egraph.get_run_report().as_ref().unwrap());
    }
    let (sort, value) = egraph.eval_expr(&var!(root))?;
    // let (_petgraph, _root_idx) = dag_to_petgraph(&termdag, termdag.lookup(&root));
    let s = egraph.serialize(egglog::SerializeConfig {
        root_eclasses: vec![(sort, value)],
        ..Default::default()
    });
    if option_env!("PRINT_KERNELS")
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
