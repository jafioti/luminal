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
use petgraph::{Directed, graph::NodeIndex, prelude::StableGraph, visit::Topo};
use serde::Serialize;
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
    Input { label: Option<String> },
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
    let (g, _) = make_gelu(64);
    // let (g, _) = make_square_matmul();
    // let (g, _) = make_gelu(64);
    let (rendered, root) = render_egglog(g);
    let code = include_str!("code.lisp");
    println!("{rendered}");
    let final_code = code.replace("{code}", &rendered);
    let (_egglog_messages, serialized) = run_egglog_program(&final_code, &root).unwrap();
    println!(
        "Search space built in {}",
        format!("{}ms", start.elapsed().as_millis()).bold()
    );
    let q = vec![
        [-1.1258, -1.1524, -0.2506, -0.4339, 0.5988],
        [-1.5551, -0.3414, 1.8530, 0.4681, -0.1577],
        [1.4437, 0.2660, 1.3894, 1.5863, 0.9463],
        [-0.8437, 0.9318, 1.2590, 2.0050, 0.0537],
    ]
    .into_flattened();
    let k = vec![
        [0.4397, 0.1124, 0.6408, 0.4412, 0.2055],
        [-0.4503, -0.5731, -0.5554, 0.5943, 1.5419],
        [0.5073, -0.5910, -1.3253, 0.1886, -0.0691],
        [-0.4949, -1.4959, -0.1938, 0.4455, 1.3253],
    ]
    .into_flattened();
    let v = vec![
        [1.5091, 2.0820, 1.7067, 2.3804, 1.9415],
        [0.7915, -0.0203, -0.4372, 1.6459, -1.3602],
        [0.3446, 0.5199, -0.3656, -1.3024, 0.0994],
        [0.4418, 0.2469, 0.0769, 0.3380, 0.4544],
    ]
    .into_flattened();
    search(
        &serialized,
        &[
            ("A", q),
            ("1.702", vec![1.702]),
            ("1.0", vec![1.0]),
            // ("Q", q),
            // ("K", k),
            // ("V", v),
            // ("DOT_ACC", vec![0.0]),
            // ("EXP_SUM_ACC", vec![0.0]),
            // ("MAX_ACC", vec![0.0]),
            // ("OUTPUT_ACC", vec![0.0]),
        ],
    );
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

fn fusion_test() -> (StableGraph<GraphTerm, (), Directed>, NodeIndex) {
    let mut graph = StableGraph::new();
    let mut a = graph.add_node(GraphTerm::GMEM {
        label: Some("A".to_string()),
    });
    a = loop_in(a, 10, 'z', 'a', &mut graph);
    a = unary(a, GraphTerm::Exp, &mut graph);
    a = loop_out(a, 10, 'z', 'a', &mut graph);
    a = loop_in(a, 10, 'z', 'a', &mut graph);
    a = unary(a, GraphTerm::Sin, &mut graph);
    a = loop_out(a, 10, 'z', 'a', &mut graph);
    (graph, a)
}

fn make_naive_attention() -> (StableGraph<GraphTerm, (), Directed>, NodeIndex) {
    let mut graph = StableGraph::new();

    let n_qkv = 4;
    let d = 5;

    // dot products ---------------------------------------------------------
    let mut q = graph.add_node(GraphTerm::GMEM {
        label: Some("Q".to_string()),
    });
    q = loop_in(q, n_qkv, Expression::from('z') * d, 'q', &mut graph);
    q = loop_in(q, n_qkv, 0, 'k', &mut graph);
    q = pad_in(q, &mut graph, 2);
    q = loop_in(q, d, 'z', "dot", &mut graph);
    let mut k = graph.add_node(GraphTerm::GMEM {
        label: Some("K".to_string()),
    });
    k = loop_in(k, n_qkv, 0, 'q', &mut graph);
    k = loop_in(k, n_qkv, Expression::from('z') * d, 'k', &mut graph);
    k = pad_in(k, &mut graph, 2);
    k = loop_in(k, d, 'z', "dot", &mut graph);
    let mut dot_acc = graph.add_node(GraphTerm::GMEM {
        label: Some("DOT_ACC".to_string()),
    });
    dot_acc = loop_in(dot_acc, n_qkv, 0, 'q', &mut graph);
    dot_acc = loop_in(dot_acc, n_qkv, 0, 'k', &mut graph);
    dot_acc = pad_in(dot_acc, &mut graph, 2); // CHANGED depth 3 → 2
    dot_acc = loop_in(dot_acc, d, Term::Acc('d'), "dot", &mut graph);
    let mut dots = binary(
        binary(q, k, GraphTerm::Mul, &mut graph),
        dot_acc,
        GraphTerm::Add,
        &mut graph,
    );
    dots = loop_out(dots, d, Term::Acc('d'), "dot", &mut graph);
    dots = pad_out(dots, &mut graph, 2); // depth matches pads above
    dots = loop_out(dots, n_qkv, 'z', 'k', &mut graph);
    dots = loop_out(dots, n_qkv, Expression::from('z') * n_qkv, 'q', &mut graph);

    // max ---------------------------------------------------------
    let mut dots_in = loop_in(dots, n_qkv, Expression::from('z') * n_qkv, 'q', &mut graph);
    dots_in = pad_in(dots_in, &mut graph, 3);
    dots_in = loop_in(dots_in, n_qkv, 'z', 'k', &mut graph);
    let mut max_acc = graph.add_node(GraphTerm::GMEM {
        label: Some("MAX_ACC".to_string()),
    });
    max_acc = loop_in(max_acc, n_qkv, 0, 'q', &mut graph);
    max_acc = pad_in(max_acc, &mut graph, 3);
    max_acc = loop_in(max_acc, n_qkv, Term::Acc('m'), "k", &mut graph);
    let mut max = binary(max_acc, dots_in, GraphTerm::Max, &mut graph);
    max = loop_out(max, n_qkv, Term::Acc('m'), 'k', &mut graph);
    max = pad_out(max, &mut graph, 3);
    max = loop_out(max, n_qkv, 'z', 'q', &mut graph);

    // neg max
    let max_in = loop_in(max, n_qkv, 'z', 'q', &mut graph);
    let mut neg_max = unary(max_in, GraphTerm::Neg, &mut graph);
    neg_max = loop_out(neg_max, n_qkv, 'z', 'q', &mut graph);
    let mut neg_max_in = loop_in(neg_max, n_qkv, 'z', 'q', &mut graph);
    neg_max_in = loop_in(neg_max_in, n_qkv, 0, "k", &mut graph);
    // sub
    let mut t_dots_in = loop_in(dots, n_qkv, Expression::from('z') * n_qkv, 'q', &mut graph);
    t_dots_in = loop_in(t_dots_in, n_qkv, 'z', 'k', &mut graph);
    let mut normed_dots_in = binary(t_dots_in, neg_max_in, GraphTerm::Add, &mut graph);
    normed_dots_in = loop_out(normed_dots_in, n_qkv, 'z', 'k', &mut graph);
    normed_dots_in = loop_out(
        normed_dots_in,
        n_qkv,
        Expression::from('z') * n_qkv,
        'q',
        &mut graph,
    );
    normed_dots_in = loop_in(
        normed_dots_in,
        n_qkv,
        Expression::from('z') * n_qkv,
        'q',
        &mut graph,
    );
    normed_dots_in = pad_in(normed_dots_in, &mut graph, 3);
    normed_dots_in = loop_in(normed_dots_in, n_qkv, 'z', 'k', &mut graph);

    // exp sum ------------------------------------------------------
    let mut exp_sum_acc = graph.add_node(GraphTerm::GMEM {
        label: Some("EXP_SUM_ACC".to_string()),
    });
    exp_sum_acc = loop_in(exp_sum_acc, n_qkv, 0, 'q', &mut graph);
    exp_sum_acc = pad_in(exp_sum_acc, &mut graph, 3);
    exp_sum_acc = loop_in(exp_sum_acc, n_qkv, Term::Acc('e'), "k", &mut graph);
    let mut exp_sum = binary(
        unary(normed_dots_in, GraphTerm::Exp, &mut graph),
        exp_sum_acc,
        GraphTerm::Add,
        &mut graph,
    );
    exp_sum = loop_out(exp_sum, n_qkv, Term::Acc('e'), 'k', &mut graph);
    exp_sum = pad_out(exp_sum, &mut graph, 3);
    exp_sum = loop_out(exp_sum, n_qkv, 'z', 'q', &mut graph);

    // final scores -------------------------------------------------
    let mut exp_sum_in = loop_in(exp_sum, n_qkv, 'z', 'q', &mut graph);
    exp_sum_in = pad_in(exp_sum_in, &mut graph, 3);
    exp_sum_in = loop_in(exp_sum_in, n_qkv, 0, 'k', &mut graph);
    let mut final_scores = binary(
        unary(normed_dots_in, GraphTerm::Exp, &mut graph),
        unary(exp_sum_in, GraphTerm::Recip, &mut graph),
        GraphTerm::Mul,
        &mut graph,
    );
    final_scores = loop_out(final_scores, n_qkv, 'z', 'k', &mut graph);
    final_scores = pad_out(final_scores, &mut graph, 3);
    final_scores = loop_out(
        final_scores,
        n_qkv,
        Expression::from('z') * n_qkv,
        'q',
        &mut graph,
    );

    // output --------------------------------------------------------
    let mut final_scores_in = loop_in(
        final_scores,
        n_qkv,
        Expression::from('z') * n_qkv,
        'q',
        &mut graph,
    );
    final_scores_in = pad_in(final_scores_in, &mut graph, 3);
    final_scores_in = loop_in(final_scores_in, n_qkv, 'z', 'k', &mut graph);
    final_scores_in = loop_in(final_scores_in, d, 0, 'd', &mut graph);
    let mut v = graph.add_node(GraphTerm::GMEM {
        label: Some("V".to_string()),
    });
    v = loop_in(v, n_qkv, 0, 'q', &mut graph);
    v = pad_in(v, &mut graph, 3);
    v = loop_in(v, n_qkv, Expression::from('z') * d, 'k', &mut graph);
    v = loop_in(v, d, 'z', "dot", &mut graph);

    let mut output_acc = graph.add_node(GraphTerm::GMEM {
        label: Some("OUTPUT_ACC".to_string()),
    });
    output_acc = loop_in(output_acc, n_qkv, 0, 'q', &mut graph);
    output_acc = pad_in(output_acc, &mut graph, 3);
    output_acc = loop_in(output_acc, n_qkv, Term::Acc('o'), "k", &mut graph);
    output_acc = loop_in(output_acc, d, 'z', "d", &mut graph);

    let mut output = binary(
        binary(v, final_scores_in, GraphTerm::Mul, &mut graph),
        output_acc,
        GraphTerm::Add,
        &mut graph,
    );
    output = loop_out(output, d, 'z', 'd', &mut graph);
    output = loop_out(output, n_qkv, Term::Acc('o'), 'k', &mut graph);
    output = pad_out(output, &mut graph, 3);
    output = loop_out(output, n_qkv, Expression::from('z') * d, 'q', &mut graph);

    (graph, output)
}

//sigmoid approximation = x * sigmoid (-1.702 * x)
fn make_gelu(size: usize) -> (StableGraph<GraphTerm, (), Directed>, NodeIndex) {
    let mut graph = StableGraph::new();

    // Input tensor
    let mut input = graph.add_node(GraphTerm::GMEM {
        label: Some("A".to_string()),
    });
    input = loop_in(input, size, 'z', 'i', &mut graph);

    let mut constant = graph.add_node(GraphTerm::GMEM {
        label: Some("1.702".to_string()),
    });
    constant = loop_in(constant, size, 0, 'i', &mut graph);

    // First loop: Calculate 1.702 * x
    let scaled_input = binary(input, constant, GraphTerm::Mul, &mut graph);
    let scaled_input = loop_out(scaled_input, size, 'z', 'i', &mut graph);

    // Second loop: Calculate -1.702 * x for exp(-1.702 * x)
    let scaled_input_looped = loop_in(scaled_input, size, 'z', 'i', &mut graph);
    let neg_scaled = unary(scaled_input_looped, GraphTerm::Neg, &mut graph);
    let neg_scaled = loop_out(neg_scaled, size, 'z', 'i', &mut graph);

    // Third loop: Calculate exp(-1.702 * x)
    let neg_scaled_looped = loop_in(neg_scaled, size, 'z', 'i', &mut graph);
    let exp_neg_scaled = unary(neg_scaled_looped, GraphTerm::Exp, &mut graph);
    let exp_neg_scaled = loop_out(exp_neg_scaled, size, 'z', 'i', &mut graph);

    // Fourth loop: Calculate 1 + exp(-1.702 * x)
    let mut one = graph.add_node(GraphTerm::GMEM {
        label: Some("1.0".to_string()),
    });
    one = loop_in(one, size, 0, 'i', &mut graph);

    let exp_neg_scaled_looped = loop_in(exp_neg_scaled, size, 'z', 'i', &mut graph);
    let one_plus_exp = binary(one, exp_neg_scaled_looped, GraphTerm::Add, &mut graph);
    let one_plus_exp = loop_out(one_plus_exp, size, 'z', 'i', &mut graph);

    // Fifth loop: Calculate sigmoid = 1 / (1 + exp(-1.702 * x))
    let one_plus_exp_looped = loop_in(one_plus_exp, size, 'z', 'i', &mut graph);
    let sigmoid = unary(one_plus_exp_looped, GraphTerm::Recip, &mut graph);
    let sigmoid = loop_out(sigmoid, size, 'z', 'i', &mut graph);

    // Sixth loop: Calculate GELU = x * sigmoid(1.702 * x)
    let sigmoid_final = loop_in(sigmoid, size, 'z', 'i', &mut graph);
    let mut output = binary(input, sigmoid_final, GraphTerm::Mul, &mut graph);
    output = loop_out(output, size, 'z', 'i', &mut graph);
    (graph, output)
}

fn make_nonsquare_matmul() -> (StableGraph<GraphTerm, (), Directed>, NodeIndex) {
    let (m, k, n) = (64, 64, 64);
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

fn make_transposed_matmul() -> (StableGraph<GraphTerm, (), Directed>, NodeIndex) {
    let (m, k, n) = (64, 32, 128);
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
    b = loop_in(b, n, Expression::from('z') * n, 'n', &mut graph);
    b = loop_in(b, k, 'z', 'k', &mut graph);

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

fn make_square_matmul() -> (StableGraph<GraphTerm, (), Directed>, NodeIndex) {
    let (m, k, n) = (64, 64, 64);
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
