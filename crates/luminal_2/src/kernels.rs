use petgraph::{Directed, graph::NodeIndex, prelude::StableGraph};

use crate::GraphTerm;

fn unary(
    a: NodeIndex,
    term: GraphTerm,
    graph: &mut StableGraph<GraphTerm, u8, Directed>,
) -> NodeIndex {
    let tmp = graph.add_node(term);
    graph.add_edge(a, tmp, 0);
    tmp
}

fn binary(
    a: NodeIndex,
    b: NodeIndex,
    term: GraphTerm,
    graph: &mut StableGraph<GraphTerm, u8, Directed>,
) -> NodeIndex {
    let tmp = graph.add_node(term);
    graph.add_edge(a, tmp, 0);
    graph.add_edge(b, tmp, 0);
    tmp
}

fn loop_in(
    node: NodeIndex,
    range: impl ToString,
    stride: impl ToString,
    graph: &mut StableGraph<GraphTerm, u8, Directed>,
) -> NodeIndex {
    unary(
        node,
        GraphTerm::LoopIn {
            range: range.to_string(),
            stride: stride.to_string(),
        },
        graph,
    )
}

fn loop_out(
    node: NodeIndex,
    range: impl ToString,
    stride: impl ToString,
    graph: &mut StableGraph<GraphTerm, u8, Directed>,
) -> NodeIndex {
    unary(
        node,
        GraphTerm::LoopOut {
            range: range.to_string(),
            stride: stride.to_string(),
        },
        graph,
    )
}

fn pad_in(
    mut node: NodeIndex,
    graph: &mut StableGraph<GraphTerm, u8, Directed>,
    levels: usize,
) -> NodeIndex {
    for _ in 0..levels {
        node = loop_in(node, "1", "0", graph);
    }
    node
}

fn pad_out(
    mut node: NodeIndex,
    graph: &mut StableGraph<GraphTerm, u8, Directed>,
    levels: usize,
) -> NodeIndex {
    for _ in 0..levels {
        node = loop_out(node, "1", "0", graph);
    }
    node
}

pub fn make_complex_kernel() -> (StableGraph<GraphTerm, u8, Directed>, NodeIndex) {
    let mut graph = StableGraph::new();
    let a = graph.add_node(GraphTerm::GMEM {
        label: Some("A".to_string()),
    });
    let in_a = loop_in(
        loop_in(pad_in(a, &mut graph, 6), "50", "z * 5", &mut graph),
        "5",
        "z",
        &mut graph,
    );

    // Acc
    let slin0 = graph.add_node(GraphTerm::GMEM {
        label: Some("acc".to_string()),
    });
    let in_exp_acc = loop_in(
        loop_in(pad_in(slin0, &mut graph, 6), "50", "z", &mut graph),
        "5",
        "Accz",
        &mut graph,
    );

    // Exp-acc
    let exp = unary(in_a, GraphTerm::Exp, &mut graph);
    let add_acc = binary(exp, in_exp_acc, GraphTerm::Add, &mut graph);
    let add_acc_out = loop_out(add_acc, "5", "Accz", &mut graph);

    // Sin
    let sin = unary(add_acc_out, GraphTerm::Sin, &mut graph);

    // Other input
    let b = graph.add_node(GraphTerm::GMEM {
        label: Some("B".to_string()),
    });
    let in_b = loop_in(pad_in(b, &mut graph, 6), "50", "z", &mut graph);

    // Mul
    let mul = binary(sin, in_b, GraphTerm::Mul, &mut graph);

    let mul_out = loop_out(mul, "50", "z", &mut graph);

    let mut out = pad_out(mul_out, &mut graph, 6);
    out = unary(out, GraphTerm::GMEM { label: None }, &mut graph);
    (graph, out)
}

pub fn make_sum_reduce() -> (StableGraph<GraphTerm, u8, Directed>, NodeIndex) {
    let mut graph = StableGraph::new();
    let mut a = graph.add_node(GraphTerm::GMEM {
        label: Some("A".to_string()),
    });
    a = pad_in(a, &mut graph, 6);
    a = loop_in(a, "5", "z", &mut graph);

    let mut acc = graph.add_node(GraphTerm::NewAcc {
        starting_value: "0.0".to_string(),
    });
    acc = loop_in(acc, "5", "Accz", &mut graph);

    let mut out = binary(a, acc, GraphTerm::Add, &mut graph);
    out = loop_out(out, "5", "Accz", &mut graph);
    out = pad_out(out, &mut graph, 6);
    out = unary(out, GraphTerm::GMEM { label: None }, &mut graph);
    (graph, out)
}

pub fn make_matmul() -> (StableGraph<GraphTerm, u8, Directed>, NodeIndex) {
    let mut graph = StableGraph::new();
    let mut a = graph.add_node(GraphTerm::GMEM {
        label: Some("A".to_string()),
    });
    a = loop_in(a, "M", "z * K", &mut graph);
    a = loop_in(a, "N", "0", &mut graph);
    a = pad_in(a, &mut graph, 4);
    a = loop_in(a, "K", "z", &mut graph);

    let mut b = graph.add_node(GraphTerm::GMEM {
        label: Some("B".to_string()),
    });
    b = loop_in(b, "M", "0", &mut graph);
    b = loop_in(b, "N", "z", &mut graph);
    b = pad_in(b, &mut graph, 4);
    b = loop_in(b, "K", "z * N", &mut graph);

    let mut acc = graph.add_node(GraphTerm::NewAcc {
        starting_value: "0.0".to_string(),
    });
    acc = loop_in(acc, "K", "Acca", &mut graph);

    let mut out = binary(
        binary(a, b, GraphTerm::Mul, &mut graph),
        acc,
        GraphTerm::Add,
        &mut graph,
    );

    out = loop_out(out, "K", "Acca", &mut graph);
    out = pad_out(out, &mut graph, 4);
    out = loop_out(out, "N", "z", &mut graph);
    out = loop_out(out, "M", "z * N", &mut graph);
    out = unary(out, GraphTerm::GMEM { label: None }, &mut graph);
    (graph, out)
}

pub fn make_tiled_matmul_basic() -> (StableGraph<GraphTerm, u8, Directed>, NodeIndex) {
    let mut graph = StableGraph::new();
    let mut a = graph.add_node(GraphTerm::GMEM {
        label: Some("A".to_string()),
    });
    a = loop_in(a, "M / 8", "z * K * 8", &mut graph);
    a = loop_in(a, "N / 8", "0", &mut graph);
    a = pad_in(a, &mut graph, 1);
    a = loop_in(a, "8", "z * K", &mut graph);
    a = loop_in(a, "8", "0", &mut graph);
    a = pad_in(a, &mut graph, 1);
    a = loop_in(a, "K / 8", "z * 8", &mut graph);
    a = loop_in(a, "8", "z", &mut graph);

    let mut b = graph.add_node(GraphTerm::GMEM {
        label: Some("B".to_string()),
    });
    b = loop_in(b, "M / 8", "0", &mut graph);
    b = loop_in(b, "N / 8", "z * 8", &mut graph);
    b = pad_in(b, &mut graph, 1);
    b = loop_in(b, "8", "0", &mut graph);
    b = loop_in(b, "8", "z", &mut graph);
    b = pad_in(b, &mut graph, 1);
    b = loop_in(b, "K / 8", "z * N * 8", &mut graph);
    b = loop_in(b, "8", "z * N", &mut graph);

    let mut acc = graph.add_node(GraphTerm::NewAcc {
        starting_value: "0.0".to_string(),
    });
    acc = loop_in(acc, "K / 8", "Acca", &mut graph);
    acc = loop_in(acc, "8", "Acca", &mut graph);

    let mut out = binary(
        binary(a, b, GraphTerm::Mul, &mut graph),
        acc,
        GraphTerm::Add,
        &mut graph,
    );

    out = loop_out(out, "K / 8", "Acca", &mut graph);
    out = loop_out(out, "8", "Acca", &mut graph);
    out = pad_out(out, &mut graph, 1);
    out = loop_out(out, "8", "z", &mut graph);
    out = loop_out(out, "8", "z * N", &mut graph);
    out = pad_out(out, &mut graph, 1);
    out = loop_out(out, "N / 8", "z * 8", &mut graph);
    out = loop_out(out, "M / 8", "z * N * 8", &mut graph);
    out = unary(out, GraphTerm::GMEM { label: None }, &mut graph);
    (graph, out)
}

pub fn make_tiled_matmul() -> (StableGraph<GraphTerm, u8, Directed>, NodeIndex) {
    let mut graph = StableGraph::new();
    let mut a = graph.add_node(GraphTerm::GMEM {
        label: Some("A".to_string()),
    });
    a = loop_in(a, "M / 8", "z * K * 8", &mut graph);
    a = loop_in(a, "N / 8", "0", &mut graph);
    a = pad_in(a, &mut graph, 1);
    a = loop_in(a, "8", "z * K", &mut graph);
    a = loop_in(a, "8", "0", &mut graph);
    a = pad_in(a, &mut graph, 1);
    a = loop_in(a, "K / 8", "z * 8", &mut graph);
    a = unary(
        a,
        GraphTerm::ZeroStrideLoad {
            range: "8".to_string(),
            stride: "z".to_string(),
        },
        &mut graph,
    );
    a = loop_in(a, "8", "z", &mut graph);

    let mut b = graph.add_node(GraphTerm::GMEM {
        label: Some("B".to_string()),
    });
    b = loop_in(b, "M / 8", "0", &mut graph);
    b = loop_in(b, "N / 8", "z * 8", &mut graph);
    b = pad_in(b, &mut graph, 1);
    b = loop_in(b, "8", "0", &mut graph);
    b = loop_in(b, "8", "z", &mut graph);
    b = pad_in(b, &mut graph, 1);
    b = loop_in(b, "K / 8", "z * N * 8", &mut graph);
    b = unary(
        b,
        GraphTerm::ZeroStrideLoad {
            range: "8".to_string(),
            stride: "z".to_string(),
        },
        &mut graph,
    );
    b = loop_in(b, "8", "z * N", &mut graph);

    let mut acc = graph.add_node(GraphTerm::NewAcc {
        starting_value: "0.0".to_string(),
    });
    acc = loop_in(acc, "K / 8", "Acca", &mut graph);
    acc = loop_in(acc, "8", "Accb", &mut graph);

    let mut out = binary(
        binary(a, b, GraphTerm::Mul, &mut graph),
        acc,
        GraphTerm::Add,
        &mut graph,
    );

    out = loop_out(out, "8", "Accb", &mut graph);
    out = loop_out(out, "K / 8", "Acca", &mut graph);
    out = pad_out(out, &mut graph, 1);
    out = loop_out(out, "8", "z", &mut graph);
    out = loop_out(out, "8", "z * N", &mut graph);
    out = pad_out(out, &mut graph, 1);
    out = loop_out(out, "N / 8", "z * 8", &mut graph);
    out = loop_out(out, "M / 8", "z * N * 8", &mut graph);
    out = unary(out, GraphTerm::GMEM { label: None }, &mut graph);
    (graph, out)
}

pub fn make_naive_attention() -> (StableGraph<GraphTerm, u8, Directed>, NodeIndex) {
    let mut graph = StableGraph::new();

    // inputs
    let mut q = graph.add_node(GraphTerm::GMEM {
        label: Some("Q".to_string()),
    });
    q = loop_in(q, "4096", "z * 64", &mut graph);
    q = loop_in(q, "4096", "0", &mut graph);
    q = pad_in(q, &mut graph, 4);
    q = loop_in(q, "64", "z", &mut graph);
    let mut k = graph.add_node(GraphTerm::GMEM {
        label: Some("K".to_string()),
    });
    k = loop_in(k, "4096", "0", &mut graph);
    k = loop_in(k, "4096", "z * 64", &mut graph);
    k = pad_in(k, &mut graph, 4);
    k = loop_in(k, "64", "z", &mut graph);
    let mut v = graph.add_node(GraphTerm::GMEM {
        label: Some("V".to_string()),
    });
    v = loop_in(v, "4096", "0", &mut graph);
    v = pad_in(v, &mut graph, 5);
    v = loop_in(v, "4096", "z * 64", &mut graph);
    v = loop_in(v, "64", "z", &mut graph);

    // accumulators
    let mut dot_acc = graph.add_node(GraphTerm::NewAcc {
        starting_value: "0.0".to_string(),
    });
    dot_acc = loop_in(dot_acc, "64", "AccDot", &mut graph);
    let mut score_max_acc = graph.add_node(GraphTerm::NewAcc {
        starting_value: "-INFINITY".to_string(),
    });
    score_max_acc = loop_in(score_max_acc, "4096", "AccScoreMax", &mut graph);
    let mut exp_sum_acc = graph.add_node(GraphTerm::NewAcc {
        starting_value: "0.0".to_string(),
    });
    exp_sum_acc = loop_in(exp_sum_acc, "4096", "AccExpSum", &mut graph);
    let mut output_acc = graph.add_node(GraphTerm::NewAcc {
        starting_value: "0.0".to_string(),
    });
    output_acc = loop_in(output_acc, "4096", "AccOutput", &mut graph);
    output_acc = loop_in(output_acc, "64", "z", &mut graph);

    // get dot products
    let mut dots = binary(
        binary(q, k, GraphTerm::Mul, &mut graph),
        dot_acc,
        GraphTerm::Add,
        &mut graph,
    );
    dots = loop_out(dots, "64", "AccDot", &mut graph);
    dots = pad_out(dots, &mut graph, 4);
    dots = loop_out(dots, "4096", "z", &mut graph);
    dots = loop_out(dots, "4096", "z * 4096", &mut graph);

    // get max
    let mut dots_in = loop_in(dots, "4096", "z * 4096", &mut graph);
    dots_in = pad_in(dots_in, &mut graph, 5);
    dots_in = loop_in(dots_in, "4096", "z", &mut graph);
    let mut max = binary(score_max_acc, dots_in, GraphTerm::Max, &mut graph);
    max = loop_out(max, "4096", "AccScoreMax", &mut graph);
    max = pad_out(max, &mut graph, 5);
    max = loop_out(max, "4096", "z", &mut graph);

    // get exp sum
    let mut max_in = loop_in(max, "4096", "z", &mut graph);
    max_in = pad_in(max_in, &mut graph, 5);
    max_in = loop_in(max_in, "4096", "0", &mut graph);
    let mut exp_sum = binary(
        unary(
            binary(
                dots_in,
                unary(max_in, GraphTerm::Neg, &mut graph),
                GraphTerm::Add,
                &mut graph,
            ),
            GraphTerm::Exp,
            &mut graph,
        ),
        exp_sum_acc,
        GraphTerm::Add,
        &mut graph,
    );
    exp_sum = loop_out(exp_sum, "4096", "AccExpSum", &mut graph);
    exp_sum = pad_out(exp_sum, &mut graph, 5);
    exp_sum = loop_out(exp_sum, "4096", "z", &mut graph);

    // get final scores
    let mut exp_sum_in = loop_in(exp_sum, "4096", "z", &mut graph);
    exp_sum_in = pad_in(exp_sum_in, &mut graph, 5);
    exp_sum_in = loop_in(exp_sum_in, "4096", "0", &mut graph);
    let mut final_scores = binary(
        unary(
            binary(
                dots_in,
                unary(max_in, GraphTerm::Neg, &mut graph),
                GraphTerm::Add,
                &mut graph,
            ),
            GraphTerm::Exp,
            &mut graph,
        ),
        unary(exp_sum_in, GraphTerm::Recip, &mut graph),
        GraphTerm::Mul,
        &mut graph,
    );
    final_scores = loop_out(final_scores, "4096", "z", &mut graph);
    final_scores = pad_out(final_scores, &mut graph, 5);
    final_scores = loop_out(final_scores, "4096", "z * 4096", &mut graph);

    // get output
    let mut final_scores_in = loop_in(final_scores, "4096", "z * 4096", &mut graph);
    final_scores_in = pad_in(final_scores_in, &mut graph, 5);
    final_scores_in = loop_in(final_scores_in, "4096", "z", &mut graph);
    final_scores_in = loop_in(final_scores_in, "64", "0", &mut graph);
    let mut output = binary(
        binary(v, final_scores_in, GraphTerm::Mul, &mut graph),
        output_acc,
        GraphTerm::Add,
        &mut graph,
    );
    output = loop_out(output, "64", "z", &mut graph);
    output = loop_out(output, "4096", "AccOutput", &mut graph);
    output = pad_out(output, &mut graph, 5);
    output = loop_out(output, "4096", "z * 64", &mut graph);
    output = unary(output, GraphTerm::GMEM { label: None }, &mut graph);
    (graph, output)
}

pub fn make_flash_attention() -> (StableGraph<GraphTerm, u8, Directed>, NodeIndex) {
    let mut graph = StableGraph::new();

    // inputs
    let mut q = graph.add_node(GraphTerm::GMEM {
        label: Some("Q".to_string()),
    });
    q = loop_in(q, "4096", "z * 64", &mut graph);
    q = pad_in(q, &mut graph, 5);
    q = loop_in(q, "4096", "0", &mut graph);
    q = loop_in(q, "64", "z", &mut graph);
    let mut k = graph.add_node(GraphTerm::GMEM {
        label: Some("K".to_string()),
    });
    k = loop_in(k, "4096", "0", &mut graph);
    k = pad_in(k, &mut graph, 5);
    k = loop_in(k, "4096", "z * 64", &mut graph);
    k = loop_in(k, "64", "z", &mut graph);
    let mut v = graph.add_node(GraphTerm::GMEM {
        label: Some("V".to_string()),
    });
    v = loop_in(v, "4096", "0", &mut graph);
    v = pad_in(v, &mut graph, 5);
    v = loop_in(v, "4096", "z * 64", &mut graph);
    v = loop_in(v, "64", "z", &mut graph);

    // accumulators
    let mut dot_acc = graph.add_node(GraphTerm::NewAcc {
        starting_value: "0.0".to_string(),
    });
    dot_acc = loop_in(dot_acc, "64", "AccDot", &mut graph);
    let mut score_max_acc = graph.add_node(GraphTerm::NewAcc {
        starting_value: "-INFINITY".to_string(),
    });
    score_max_acc = loop_in(score_max_acc, "4096", "AccScoreMax", &mut graph);
    let mut exp_sum_acc = graph.add_node(GraphTerm::NewAcc {
        starting_value: "0.0".to_string(),
    });
    exp_sum_acc = loop_in(exp_sum_acc, "4096", "AccExpSum", &mut graph);
    let mut output_acc = graph.add_node(GraphTerm::NewAcc {
        starting_value: "0.0".to_string(),
    });
    output_acc = loop_in(output_acc, "4096", "AccOutput", &mut graph);
    output_acc = loop_in(output_acc, "64", "z", &mut graph);

    // get dot products
    let dots = loop_out(
        binary(
            binary(q, k, GraphTerm::Mul, &mut graph),
            dot_acc,
            GraphTerm::Add,
            &mut graph,
        ),
        "64",
        "AccDot",
        &mut graph,
    );
    let new_max = binary(score_max_acc, dots, GraphTerm::Max, &mut graph);
    loop_out(new_max, "4096", "AccScoreMax", &mut graph); // This is needed so we know to feed new max back in for score max acc

    let rescale = unary(
        binary(
            score_max_acc,
            unary(new_max, GraphTerm::Neg, &mut graph),
            GraphTerm::Add,
            &mut graph,
        ),
        GraphTerm::Exp,
        &mut graph,
    );
    let weight = unary(
        binary(
            dots,
            unary(new_max, GraphTerm::Neg, &mut graph),
            GraphTerm::Add,
            &mut graph,
        ),
        GraphTerm::Exp,
        &mut graph,
    );
    let exp_sum_new = binary(
        binary(exp_sum_acc, rescale, GraphTerm::Mul, &mut graph),
        weight,
        GraphTerm::Add,
        &mut graph,
    );
    let weight_b = loop_in(weight, "64", "0", &mut graph);
    let rescale_b = loop_in(rescale, "64", "0", &mut graph);
    let mut partial_output = binary(
        binary(output_acc, rescale_b, GraphTerm::Mul, &mut graph),
        binary(weight_b, v, GraphTerm::Mul, &mut graph),
        GraphTerm::Add,
        &mut graph,
    );
    partial_output = loop_out(partial_output, "64", "z", &mut graph);
    partial_output = loop_out(partial_output, "4096", "AccOutput", &mut graph);
    partial_output = loop_in(partial_output, "64", "z", &mut graph);
    let exp_sum = loop_out(exp_sum_new, "4096", "AccExpSum", &mut graph);
    let exp_sum_b = loop_in(exp_sum, "64", "0", &mut graph);
    let mut output = binary(
        partial_output,
        unary(exp_sum_b, GraphTerm::Recip, &mut graph),
        GraphTerm::Mul,
        &mut graph,
    );
    output = loop_out(output, "64", "z", &mut graph);
    output = pad_out(output, &mut graph, 5);
    output = loop_out(output, "4096", "z * 64", &mut graph);
    output = unary(output, GraphTerm::GMEM { label: None }, &mut graph);
    (graph, output)
}
