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
use rand::{Rng, rng};
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, fmt::Debug};

use crate::{
    extract::search,
    symbolic::{Expression, expression_cleanup},
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

#[derive(Clone, Copy)]
enum GMEMBuffer {
    PrevKernel { kernel: usize, output: usize },
    Input(usize),
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
    match run_egglog_program(include_str!("code.lisp")) {
        Ok((_egglog_messages, serialized)) => {
            println!(
                "Search space built in {}",
                format!("{}ms", start.elapsed().as_millis()).bold()
            );
            let mut rng = rng();
            search(
                &serialized,
                &[
                    (0..10 * 5).map(|_| rng.random()).collect_vec(),
                    (0..10 * 5).map(|_| rng.random()).collect_vec(),
                ],
            );
        }
        Err(e) => println!("{e}"),
    }
    expression_cleanup();
}

/// Runs an Egglog program from a string and returns its output messages.
pub fn run_egglog_program(code: &str) -> Result<(Vec<String>, egraph_serialize::EGraph), Error> {
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
    let (sort, value) = egraph.eval_expr(&var!("full"))?;
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
