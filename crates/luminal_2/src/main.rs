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
    Sqrt,
    SMEM,     // Signifies shared memory
    SMEMLoad, // Takes in an smem pointer and a gmem pointer, copies the gmem element to smem and returns the smem pointer
    SMEMRead, // Takes in an smem pointer and an smemload, returns the smem pointer
}

fn main() {
    // make_square_matmul();
    // make_nonsquare_matmul();
    // make_gelu();
    // make_single_head_attention();
    // make_multi_head_attention();
    // make_softmax();
    make_layernorm();
    // fusion_test();
    // let (g, _) = make_square_matmul();
    // let (g, _) = make_gelu(64);

    expression_cleanup();
}

fn fusion_test() {
    let mut graph = StableGraph::new();
    let mut a = graph.add_node(GraphTerm::GMEM {
        label: Some("A".to_string()),
    });
    a = loop_in(a, 10, 'z', 'a', &mut graph);
    a = unary(a, GraphTerm::Exp, &mut graph);
    a = loop_out(a, 10, 'z', 'a', &mut graph);
    a = loop_in(a, 10, 'z', 'a', &mut graph);
    a = unary(a, GraphTerm::Sin, &mut graph);
    loop_out(a, 10, 'z', 'a', &mut graph);

    let egraph = build_search_space(&graph, 2, false);
    let mut rng = rng();
    search(
        &egraph,
        &[("A", (0..10).map(|_| rng.random()).collect_vec())],
    );
}

fn make_single_head_attention() {
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
    loop_out(output, n_qkv, Expression::from('z') * d, 'q', &mut graph);

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
    let egraph = build_search_space(&graph, 1, false);
    search(
        &egraph,
        &[
            ("Q", q),
            ("K", k),
            ("V", v),
            ("DOT_ACC", vec![0.0]),
            ("MAX_ACC", vec![f32::NEG_INFINITY]),
            ("EXP_SUM_ACC", vec![0.0]),
            ("OUTPUT_ACC", vec![0.0]),
        ],
    );
}

fn make_multi_head_attention() {
    let mut graph = StableGraph::new();

    let heads = 3;
    let n_qkv = 4;
    let d = 5;

    // dot products ---------------------------------------------------------
    let mut q = graph.add_node(GraphTerm::GMEM {
        label: Some("Q".to_string()),
    });
    q = loop_in(q, heads, Expression::from('z') * d * n_qkv, 'h', &mut graph);
    q = loop_in(q, n_qkv, Expression::from('z') * d, 'q', &mut graph);
    q = loop_in(q, n_qkv, 0, 'k', &mut graph);
    q = pad_in(q, &mut graph, 2);
    q = loop_in(q, d, 'z', "dot", &mut graph);
    let mut k = graph.add_node(GraphTerm::GMEM {
        label: Some("K".to_string()),
    });
    k = loop_in(k, heads, Expression::from('z') * d * n_qkv, 'h', &mut graph);
    k = loop_in(k, n_qkv, 0, 'q', &mut graph);
    k = loop_in(k, n_qkv, Expression::from('z') * d, 'k', &mut graph);
    k = pad_in(k, &mut graph, 2);
    k = loop_in(k, d, 'z', "dot", &mut graph);
    let mut dot_acc = graph.add_node(GraphTerm::GMEM {
        label: Some("DOT_ACC".to_string()),
    });
    dot_acc = loop_in(dot_acc, heads, 0, 'h', &mut graph);
    dot_acc = loop_in(dot_acc, n_qkv, 0, 'q', &mut graph);
    dot_acc = loop_in(dot_acc, n_qkv, 0, 'k', &mut graph);
    dot_acc = pad_in(dot_acc, &mut graph, 2);
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
    dots = loop_out(
        dots,
        heads,
        Expression::from('z') * n_qkv * n_qkv,
        'h',
        &mut graph,
    );

    // max ---------------------------------------------------------
    let mut dots_in = loop_in(
        dots,
        heads,
        Expression::from('z') * n_qkv * n_qkv,
        'h',
        &mut graph,
    );
    dots_in = loop_in(
        dots_in,
        n_qkv,
        Expression::from('z') * n_qkv,
        'q',
        &mut graph,
    );
    dots_in = pad_in(dots_in, &mut graph, 3);
    dots_in = loop_in(dots_in, n_qkv, 'z', 'k', &mut graph);
    let mut max_acc = graph.add_node(GraphTerm::GMEM {
        label: Some("MAX_ACC".to_string()),
    });
    max_acc = loop_in(max_acc, heads, 0, 'h', &mut graph);
    max_acc = loop_in(max_acc, n_qkv, 0, 'q', &mut graph);
    max_acc = pad_in(max_acc, &mut graph, 3);
    max_acc = loop_in(max_acc, n_qkv, Term::Acc('m'), "k", &mut graph);
    let mut max = binary(max_acc, dots_in, GraphTerm::Max, &mut graph);
    max = loop_out(max, n_qkv, Term::Acc('m'), 'k', &mut graph);
    max = pad_out(max, &mut graph, 3);
    max = loop_out(max, n_qkv, 'z', 'q', &mut graph);
    max = loop_out(max, heads, Expression::from('z') * n_qkv, 'h', &mut graph);

    // neg max
    let mut max_in = loop_in(max, heads, Expression::from('z') * n_qkv, 'h', &mut graph);
    max_in = loop_in(max_in, n_qkv, 'z', 'q', &mut graph);
    let mut neg_max = unary(max_in, GraphTerm::Neg, &mut graph);
    neg_max = loop_out(neg_max, n_qkv, 'z', 'q', &mut graph);
    neg_max = loop_out(
        neg_max,
        heads,
        Expression::from('z') * n_qkv,
        'h',
        &mut graph,
    );
    let mut neg_max_in = loop_in(
        neg_max,
        heads,
        Expression::from('z') * n_qkv,
        'h',
        &mut graph,
    );
    neg_max_in = loop_in(neg_max_in, n_qkv, 'z', 'q', &mut graph);
    neg_max_in = loop_in(neg_max_in, n_qkv, 0, "k", &mut graph);
    // sub
    let mut t_dots_in = loop_in(
        dots,
        heads,
        Expression::from('z') * n_qkv * n_qkv,
        'h',
        &mut graph,
    );
    t_dots_in = loop_in(
        t_dots_in,
        n_qkv,
        Expression::from('z') * n_qkv,
        'q',
        &mut graph,
    );
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
    normed_dots_in = loop_out(
        normed_dots_in,
        heads,
        Expression::from('z') * n_qkv * n_qkv,
        'h',
        &mut graph,
    );
    normed_dots_in = loop_in(
        normed_dots_in,
        heads,
        Expression::from('z') * n_qkv * n_qkv,
        'h',
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
    exp_sum_acc = loop_in(exp_sum_acc, heads, 0, 'h', &mut graph);
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
    exp_sum = loop_out(
        exp_sum,
        heads,
        Expression::from('z') * n_qkv,
        'h',
        &mut graph,
    );

    // final scores -------------------------------------------------
    let mut exp_sum_in = loop_in(
        exp_sum,
        heads,
        Expression::from('z') * n_qkv,
        'h',
        &mut graph,
    );
    exp_sum_in = loop_in(exp_sum_in, n_qkv, 'z', 'q', &mut graph);
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
    final_scores = loop_out(
        final_scores,
        heads,
        Expression::from('z') * n_qkv * n_qkv,
        'h',
        &mut graph,
    );

    // output --------------------------------------------------------
    let mut final_scores_in = loop_in(
        final_scores,
        heads,
        Expression::from('z') * n_qkv * n_qkv,
        'h',
        &mut graph,
    );
    final_scores_in = loop_in(
        final_scores_in,
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
    v = loop_in(v, n_qkv, Expression::from('z') * d * n_qkv, 'h', &mut graph);
    v = loop_in(v, n_qkv, 0, 'q', &mut graph);
    v = pad_in(v, &mut graph, 3);
    v = loop_in(v, n_qkv, Expression::from('z') * d, 'k', &mut graph);
    v = loop_in(v, d, 'z', "dot", &mut graph);

    let mut output_acc = graph.add_node(GraphTerm::GMEM {
        label: Some("OUTPUT_ACC".to_string()),
    });
    output_acc = loop_in(output_acc, heads, 0, 'h', &mut graph);
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
    loop_out(
        output,
        heads,
        Expression::from('z') * d * n_qkv,
        'h',
        &mut graph,
    );

    let q = vec![
        [-1.1258, -1.1524, -0.2506, -0.4339, 0.5988],
        [-1.5551, -0.3414, 1.8530, 0.4681, -0.1577],
        [1.4437, 0.2660, 1.3894, 1.5863, 0.9463],
        [-0.8437, 0.9318, 1.2590, 2.0050, 0.0537],
    ]
    .into_flattened()
    .repeat(3);
    let k = vec![
        [0.4397, 0.1124, 0.6408, 0.4412, 0.2055],
        [-0.4503, -0.5731, -0.5554, 0.5943, 1.5419],
        [0.5073, -0.5910, -1.3253, 0.1886, -0.0691],
        [-0.4949, -1.4959, -0.1938, 0.4455, 1.3253],
    ]
    .into_flattened()
    .repeat(3);
    let v = vec![
        [1.5091, 2.0820, 1.7067, 2.3804, 1.9415],
        [0.7915, -0.0203, -0.4372, 1.6459, -1.3602],
        [0.3446, 0.5199, -0.3656, -1.3024, 0.0994],
        [0.4418, 0.2469, 0.0769, 0.3380, 0.4544],
    ]
    .into_flattened()
    .repeat(3);
    let egraph = build_search_space(&graph, 2, false);
    search(
        &egraph,
        &[
            ("Q", q),
            ("K", k),
            ("V", v),
            ("DOT_ACC", vec![0.0]),
            ("MAX_ACC", vec![f32::NEG_INFINITY]),
            ("EXP_SUM_ACC", vec![0.0]),
            ("OUTPUT_ACC", vec![0.0]),
        ],
    );
}

fn make_softmax() {
    let size = 64;
    let mut graph = StableGraph::new();

    // === Stage 1: Max Reduction ('j' loop) ===
    // This computes the scalar max(X).
    let mut max_input = graph.add_node(GraphTerm::GMEM {
        label: Some("X".to_string()),
    });
    max_input = pad_in(max_input, &mut graph, 4);
    max_input = loop_in(max_input, size, 'z', 'j', &mut graph);

    let mut max_acc = graph.add_node(GraphTerm::GMEM {
        label: Some("max_val".to_string()),
    });
    max_acc = pad_in(max_acc, &mut graph, 4);
    max_acc = loop_in(max_acc, size, Term::Acc('m'), 'j', &mut graph);
    let max_op = binary(max_input, max_acc, GraphTerm::Max, &mut graph);
    let max_reduction_scalar = loop_out(max_op, size, Term::Acc('m'), 'j', &mut graph);

    // === Stage 2: Negate the max value (map with 'j' loop) ===
    // This creates a vector where each element is -max(X).
    let neg_input = loop_in(max_reduction_scalar, size, 'z', 'j', &mut graph);
    let neg_op = unary(neg_input, GraphTerm::Neg, &mut graph);
    let neg_max_vector = loop_out(neg_op, size, 'z', 'j', &mut graph);

    // === Stage 3: Numerator Calculation ('i' loop) ===
    // This calculates the vector exp(X - max(X)).
    let mut num_x_stream = graph.add_node(GraphTerm::GMEM {
        label: Some("X".to_string()),
    });
    num_x_stream = pad_in(num_x_stream, &mut graph, 4);
    num_x_stream = loop_in(num_x_stream, size, 'z', 'i', &mut graph);
    // Note: The loop marker 'i' is used here, as this is part of the final output stream.
    let num_neg_max_stream = loop_in(neg_max_vector.clone(), size, 'z', 'i', &mut graph);
    let num_add_op = binary(num_x_stream, num_neg_max_stream, GraphTerm::Add, &mut graph);
    // The graph shows a staged calculation for the exponent, which we replicate.
    let intermediate_num_out = loop_out(num_add_op, size, 'z', 'i', &mut graph);
    let intermediate_num_in = loop_in(intermediate_num_out, size, 'z', 'i', &mut graph);
    let num_exp_op = unary(intermediate_num_in, GraphTerm::Exp, &mut graph);
    let numerator_vector = loop_out(num_exp_op, size, 'z', 'i', &mut graph);

    // === Stage 4: Denominator Calculation ('k' loop) ===
    // This calculates the scalar sum(exp(X - max(X))).
    let mut den_x_stream = graph.add_node(GraphTerm::GMEM {
        label: Some("X".to_string()),
    });
    den_x_stream = pad_in(den_x_stream, &mut graph, 4);
    den_x_stream = loop_in(den_x_stream, size, 'z', 'k', &mut graph);

    let den_neg_max_stream = loop_in(neg_max_vector, size, 'z', 'k', &mut graph);
    let mut den_add_op = binary(den_x_stream, den_neg_max_stream, GraphTerm::Add, &mut graph);
    den_add_op = loop_out(den_add_op, size, 'z', 'k', &mut graph);
    let mut den_exp_op = loop_in(den_add_op, size, 'z', 'k', &mut graph);
    den_exp_op = unary(den_exp_op, GraphTerm::Exp, &mut graph);
    den_exp_op = loop_out(den_exp_op, size, 'z', 'k', &mut graph);
    let mut sum_acc = graph.add_node(GraphTerm::GMEM {
        label: Some("sum_val".to_string()),
    });
    sum_acc = pad_in(sum_acc, &mut graph, 4);
    sum_acc = loop_in(sum_acc, size, Term::Acc('s'), 'k', &mut graph);
    den_exp_op = loop_in(den_exp_op, size, 'z', 'k', &mut graph);
    let sum_op = binary(den_exp_op, sum_acc, GraphTerm::Add, &mut graph);
    let denominator_scalar = loop_out(sum_op, size, Term::Acc('s'), 'k', &mut graph);

    // === Stage 5: Final Calculation ('i' loop) ===
    // This calculates Numerator * (1 / Denominator).
    // First, calculate the reciprocal of the denominator scalar.
    let inv_den_in = loop_in(denominator_scalar, size, 'z', 'i', &mut graph);
    let recip_op = unary(inv_den_in, GraphTerm::Recip, &mut graph);
    let inv_denominator_vector = loop_out(recip_op, size, 'z', 'i', &mut graph);

    // Finally, multiply the numerator vector with the inverted denominator vector.
    let final_num_in = loop_in(numerator_vector, size, 'z', 'i', &mut graph);
    let final_inv_den_in = loop_in(inv_denominator_vector, size, 'z', 'i', &mut graph);
    let mul_op = binary(final_num_in, final_inv_den_in, GraphTerm::Mul, &mut graph);
    let output = loop_out(mul_op, size, 'z', 'i', &mut graph);
    pad_out(output, &mut graph, 4);

    let egraph = build_search_space(&graph, 10, true);
    let mut rng = rng();
    search(
        &egraph,
        &[
            ("X", (0..size).map(|_| rng.random()).collect_vec()),
            ("max_val", vec![f32::NEG_INFINITY]),
            ("sum_val", vec![0.0]),
        ],
    );
}

fn make_layernorm() {
    let (n, d) = (128, 768);
    let mut graph = StableGraph::new();

    // Input tensor X of shape [N, D]
    let mut x = graph.add_node(GraphTerm::GMEM {
        label: Some("X".to_string()),
    });
    x = pad_in(x, &mut graph, 2);

    // Scale parameter Gamma of shape [D]
    let mut gamma = graph.add_node(GraphTerm::GMEM {
        label: Some("Gamma".to_string()),
    });
    gamma = pad_in(gamma, &mut graph, 1);

    // Shift parameter Beta of shape [D]
    let mut beta = graph.add_node(GraphTerm::GMEM {
        label: Some("Beta".to_string()),
    });
    beta = pad_in(beta, &mut graph, 1);

    // Constant 1.0/D for mean calculation
    let inv_d = graph.add_node(GraphTerm::GMEM {
        label: Some("inv_D".to_string()),
    });

    // Small constant for numerical stability
    let epsilon = graph.add_node(GraphTerm::GMEM {
        label: Some("Epsilon".to_string()),
    });

    // Loop over batch dimension N
    let x_batch = loop_in(x, n, Expression::from('z') * d, 'n', &mut graph);

    // First pass: Calculate mean for each row
    // Sum over D dimension
    let x_sum_inner = loop_in(x_batch, d, 'z', 'd', &mut graph);
    let zero_acc = graph.add_node(GraphTerm::GMEM {
        label: Some("0".to_string()),
    });
    let zero_acc = loop_in(zero_acc, d, Term::Acc('a'), 'd', &mut graph);

    let sum_result = binary(x_sum_inner, zero_acc, GraphTerm::Add, &mut graph);
    let sum_result = loop_out(sum_result, d, Term::Acc('a'), 'd', &mut graph);

    // Calculate mean = sum * (1/D)
    let inv_d_broadcast = loop_in(inv_d, n, 0, 'n', &mut graph);
    let mean_n = binary(sum_result, inv_d_broadcast, GraphTerm::Mul, &mut graph);

    // Second pass: Calculate variance
    // Need to compute sum of squares
    let x_sq_inner = loop_in(x_batch, d, 'z', 'd', &mut graph);
    let x_squared = binary(x_sq_inner, x_sq_inner, GraphTerm::Mul, &mut graph);

    let zero_acc_sq = graph.add_node(GraphTerm::GMEM {
        label: Some("0".to_string()),
    });
    let zero_acc_sq = loop_in(zero_acc_sq, d, Term::Acc('a'), 'd', &mut graph);

    let sum_sq_result = binary(x_squared, zero_acc_sq, GraphTerm::Add, &mut graph);
    let sum_sq_result = loop_out(sum_sq_result, d, Term::Acc('a'), 'd', &mut graph);

    // Calculate mean of squares = sum_sq * (1/D)
    let mean_sq = binary(sum_sq_result, inv_d_broadcast, GraphTerm::Mul, &mut graph);

    // Variance = mean(x^2) - mean(x)^2
    let mean_n_sq = binary(mean_n, mean_n, GraphTerm::Mul, &mut graph);
    let neg_mean_sq = unary(mean_n_sq, GraphTerm::Neg, &mut graph);
    let variance = binary(mean_sq, neg_mean_sq, GraphTerm::Add, &mut graph);

    // Third pass: Apply normalization
    let x_norm_inner = loop_in(x_batch, d, 'z', 'd', &mut graph);
    let gamma_inner = loop_in(gamma, d, 'z', 'd', &mut graph);
    let beta_inner = loop_in(beta, d, 'z', 'd', &mut graph);

    // Broadcast mean and variance for elementwise operations
    let mean_broadcast = loop_in(mean_n, d, 0, 'd', &mut graph);
    let variance_broadcast = loop_in(variance, d, 0, 'd', &mut graph);
    let epsilon_broadcast = loop_in(epsilon, d, 0, 'd', &mut graph);

    // (x - mean)
    let neg_mean = unary(mean_broadcast, GraphTerm::Neg, &mut graph);
    let x_centered = binary(x_norm_inner, neg_mean, GraphTerm::Add, &mut graph);

    // sqrt(variance + epsilon)
    let var_plus_eps = binary(
        variance_broadcast,
        epsilon_broadcast,
        GraphTerm::Add,
        &mut graph,
    );
    let sqrt_var_eps = unary(var_plus_eps, GraphTerm::Sqrt, &mut graph);
    let inv_sqrt_var_eps = unary(sqrt_var_eps, GraphTerm::Recip, &mut graph);

    // (x - mean) / sqrt(variance + epsilon)
    let normalized = binary(x_centered, inv_sqrt_var_eps, GraphTerm::Mul, &mut graph);

    // * gamma
    let scaled = binary(normalized, gamma_inner, GraphTerm::Mul, &mut graph);

    // + beta
    let mut output = binary(scaled, beta_inner, GraphTerm::Add, &mut graph);

    // Loop out operations
    output = loop_out(output, d, 'z', 'd', &mut graph);
    output = loop_out(output, n, Expression::from('z') * d, 'n', &mut graph);
    pad_out(output, &mut graph, 2);

    let egraph = build_search_space(&graph, 10, true);
    let mut rng = rng();
    search(
        &egraph,
        &[
            ("A", (0..n * d).map(|_| rng.random()).collect_vec()),
            ("Gamma", (0..d).map(|_| rng.random()).collect_vec()),
            ("Beta", (0..d).map(|_| rng.random()).collect_vec()),
            ("inv_d", vec![1.0 / d as f32]),
            ("Epsilon", vec![1e-5]),
            ("0", vec![0.0]),
        ],
    );
}

//sigmoid approximation = x * sigmoid (-1.702 * x)
fn make_gelu() {
    let size = 64;
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
    let output = binary(input, sigmoid_final, GraphTerm::Mul, &mut graph);
    loop_out(output, size, 'z', 'i', &mut graph);

    let egraph = build_search_space(&graph, 10, true);
    let mut rng = rng();
    search(
        &egraph,
        &[
            ("A", (0..size).map(|_| rng.random()).collect_vec()),
            ("1.702", vec![1.702]),
            ("1.0", vec![1.0]),
        ],
    );
}

fn make_nonsquare_matmul() {
    let (m, k, n) = (32, 16, 64);
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
    pad_out(out, &mut graph, 2);

    let egraph = build_search_space(&graph, 10, false);
    let mut rng = rng();
    search(
        &egraph,
        &[
            ("A", (0..(m * k)).map(|_| rng.random()).collect_vec()),
            ("B", (0..(k * n)).map(|_| rng.random()).collect_vec()),
            ("Acc", vec![0.0]),
        ],
    );
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

fn make_square_matmul() {
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
    pad_out(out, &mut graph, 2);

    let egraph = build_search_space(&graph, 10, false);
    let mut rng = rng();
    search(
        &egraph,
        &[
            ("A", (0..(m * k)).map(|_| rng.random()).collect_vec()),
            ("B", (0..(k * n)).map(|_| rng.random()).collect_vec()),
            ("Acc", vec![0.0]),
        ],
    );
}
