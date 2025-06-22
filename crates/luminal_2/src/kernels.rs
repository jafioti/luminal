use petgraph::{Directed, graph::NodeIndex, prelude::StableGraph};

use crate::{
    GraphTerm,
    symbolic::{Expression, Term},
};

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
    range: impl Into<Expression>,
    stride: impl Into<Expression>,
    graph: &mut StableGraph<GraphTerm, u8, Directed>,
) -> NodeIndex {
    unary(
        node,
        GraphTerm::LoopIn {
            range: range.into(),
            stride: stride.into(),
        },
        graph,
    )
}

fn loop_out(
    node: NodeIndex,
    range: impl Into<Expression>,
    stride: impl Into<Expression>,
    graph: &mut StableGraph<GraphTerm, u8, Directed>,
) -> NodeIndex {
    unary(
        node,
        GraphTerm::LoopOut {
            range: range.into(),
            stride: stride.into(),
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
        node = loop_in(node, 1, 0, graph);
    }
    node
}

fn pad_out(
    mut node: NodeIndex,
    graph: &mut StableGraph<GraphTerm, u8, Directed>,
    levels: usize,
) -> NodeIndex {
    for _ in 0..levels {
        node = loop_out(node, 1, 0, graph);
    }
    node
}

pub fn make_complex_kernel() -> (StableGraph<GraphTerm, u8, Directed>, NodeIndex) {
    let mut graph = StableGraph::new();
    let a = graph.add_node(GraphTerm::GMEM {
        label: Some("A".to_string()),
    });
    let in_a = loop_in(
        loop_in(
            pad_in(a, &mut graph, 6),
            50,
            Expression::from('z') * 5,
            &mut graph,
        ),
        5,
        'z',
        &mut graph,
    );

    // Acc
    let slin0 = graph.add_node(GraphTerm::GMEM {
        label: Some("acc".to_string()),
    });
    let in_exp_acc = loop_in(
        loop_in(pad_in(slin0, &mut graph, 6), 50, 'z', &mut graph),
        5,
        Term::Acc("Accz"),
        &mut graph,
    );

    // Exp-acc
    let exp = unary(in_a, GraphTerm::Exp, &mut graph);
    let add_acc = binary(exp, in_exp_acc, GraphTerm::Add, &mut graph);
    let add_acc_out = loop_out(add_acc, 5, Term::Acc("Accz"), &mut graph);

    // Sin
    let sin = unary(add_acc_out, GraphTerm::Sin, &mut graph);

    // Other input
    let b = graph.add_node(GraphTerm::GMEM {
        label: Some("B".to_string()),
    });
    let in_b = loop_in(pad_in(b, &mut graph, 6), 50, 'z', &mut graph);

    // Mul
    let mul = binary(sin, in_b, GraphTerm::Mul, &mut graph);

    let mul_out = loop_out(mul, 50, 'z', &mut graph);

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
    a = loop_in(a, 5, 'z', &mut graph);

    let mut acc = graph.add_node(GraphTerm::NewAcc {
        starting_value: "0.0".to_string(),
    });
    acc = loop_in(acc, 5, Term::Acc("Accz"), &mut graph);

    let mut out = binary(a, acc, GraphTerm::Add, &mut graph);
    out = loop_out(out, 5, Term::Acc("Accz"), &mut graph);
    out = pad_out(out, &mut graph, 6);
    out = unary(out, GraphTerm::GMEM { label: None }, &mut graph);
    (graph, out)
}

pub fn make_matmul() -> (StableGraph<GraphTerm, u8, Directed>, NodeIndex) {
    let mut graph = StableGraph::new();
    let mut a = graph.add_node(GraphTerm::GMEM {
        label: Some("A".to_string()),
    });
    a = loop_in(a, 'M', Expression::from('z') * 'K', &mut graph);
    a = loop_in(a, 'N', 0, &mut graph);
    a = pad_in(a, &mut graph, 4);
    a = loop_in(a, 'K', 'z', &mut graph);

    let mut b = graph.add_node(GraphTerm::GMEM {
        label: Some("B".to_string()),
    });
    b = loop_in(b, 'M', 0, &mut graph);
    b = loop_in(b, 'N', 'z', &mut graph);
    b = pad_in(b, &mut graph, 4);
    b = loop_in(b, 'K', Expression::from('z') * 'N', &mut graph);

    let mut acc = graph.add_node(GraphTerm::NewAcc {
        starting_value: "0.0".to_string(),
    });
    acc = loop_in(acc, 'K', Term::Acc("Acca"), &mut graph);

    let mut out = binary(
        binary(a, b, GraphTerm::Mul, &mut graph),
        acc,
        GraphTerm::Add,
        &mut graph,
    );

    out = loop_out(out, 'K', Term::Acc("Acca"), &mut graph);
    out = pad_out(out, &mut graph, 4);
    out = loop_out(out, 'N', 'z', &mut graph);
    out = loop_out(out, 'M', Expression::from('z') * 'N', &mut graph);
    out = unary(out, GraphTerm::GMEM { label: None }, &mut graph);
    (graph, out)
}

pub fn make_tiled_matmul_basic() -> (StableGraph<GraphTerm, u8, Directed>, NodeIndex) {
    let mut graph = StableGraph::new();
    let mut a = graph.add_node(GraphTerm::GMEM {
        label: Some("A".to_string()),
    });
    a = loop_in(
        a,
        Expression::from('M') / 8,
        Expression::from('z') * 'K' * 8,
        &mut graph,
    );
    a = loop_in(a, Expression::from('N') / 8, 0, &mut graph);
    a = pad_in(a, &mut graph, 1);
    a = loop_in(a, 8, Expression::from('z') * 'K', &mut graph);
    a = loop_in(a, 8, 0, &mut graph);
    a = pad_in(a, &mut graph, 1);
    a = loop_in(
        a,
        Expression::from('K') / 8,
        Expression::from('z') * 8,
        &mut graph,
    );
    a = loop_in(a, 8, 'z', &mut graph);

    let mut b = graph.add_node(GraphTerm::GMEM {
        label: Some("B".to_string()),
    });
    b = loop_in(b, Expression::from('M') / 8, 0, &mut graph);
    b = loop_in(
        b,
        Expression::from('N') / 8,
        Expression::from('z') * 8,
        &mut graph,
    );
    b = pad_in(b, &mut graph, 1);
    b = loop_in(b, 8, 0, &mut graph);
    b = loop_in(b, 8, 'z', &mut graph);
    b = pad_in(b, &mut graph, 1);
    b = loop_in(
        b,
        Expression::from('K') / 8,
        Expression::from('z') * 'N' * 8,
        &mut graph,
    );
    b = loop_in(b, 8, Expression::from('z') * 'N', &mut graph);

    let mut acc = graph.add_node(GraphTerm::NewAcc {
        starting_value: "0.0".to_string(),
    });
    acc = loop_in(
        acc,
        Expression::from('K') / 8,
        Term::Acc("Acca"),
        &mut graph,
    );
    acc = loop_in(acc, 8, Term::Acc("Acca"), &mut graph);

    let mut out = binary(
        binary(a, b, GraphTerm::Mul, &mut graph),
        acc,
        GraphTerm::Add,
        &mut graph,
    );

    out = loop_out(
        out,
        Expression::from('K') / 8,
        Term::Acc("Acca"),
        &mut graph,
    );
    out = loop_out(out, 8, Term::Acc("Acca"), &mut graph);
    out = pad_out(out, &mut graph, 1);
    out = loop_out(out, 8, 'z', &mut graph);
    out = loop_out(out, 8, Expression::from('z') * 'N', &mut graph);
    out = pad_out(out, &mut graph, 1);
    out = loop_out(
        out,
        Expression::from('N') / 8,
        Expression::from('z') * 8,
        &mut graph,
    );
    out = loop_out(
        out,
        Expression::from('M') / 8,
        Expression::from('z') * 'N' * 8,
        &mut graph,
    );
    out = unary(out, GraphTerm::GMEM { label: None }, &mut graph);
    (graph, out)
}

pub fn make_tiled_matmul() -> (StableGraph<GraphTerm, u8, Directed>, NodeIndex) {
    let mut graph = StableGraph::new();
    let mut a = graph.add_node(GraphTerm::GMEM {
        label: Some("A".to_string()),
    });
    a = loop_in(
        a,
        Expression::from('M') / 8,
        Expression::from('z') * 'K' * 8,
        &mut graph,
    );
    a = loop_in(a, Expression::from('N') / 8, 0, &mut graph);
    a = pad_in(a, &mut graph, 1);
    a = loop_in(a, 8, Expression::from('z') * 'K', &mut graph);
    a = loop_in(a, 8, 0, &mut graph);
    a = pad_in(a, &mut graph, 1);
    a = loop_in(
        a,
        Expression::from('K') / 8,
        Expression::from('z') * 8,
        &mut graph,
    );
    a = unary(
        a,
        GraphTerm::ZeroStrideLoad {
            range: 8.into(),
            stride: 'z'.into(),
        },
        &mut graph,
    );
    a = loop_in(a, 8, 'z', &mut graph);

    let mut b = graph.add_node(GraphTerm::GMEM {
        label: Some("B".to_string()),
    });
    b = loop_in(b, Expression::from('M') / 8, 0, &mut graph);
    b = loop_in(
        b,
        Expression::from('N') / 8,
        Expression::from('z') * 8,
        &mut graph,
    );
    b = pad_in(b, &mut graph, 1);
    b = loop_in(b, 8, 0, &mut graph);
    b = loop_in(b, 8, 'z', &mut graph);
    b = pad_in(b, &mut graph, 1);
    b = loop_in(
        b,
        Expression::from('K') / 8,
        Expression::from('z') * 'N' * 8,
        &mut graph,
    );
    b = unary(
        b,
        GraphTerm::ZeroStrideLoad {
            range: 8.into(),
            stride: 'z'.into(),
        },
        &mut graph,
    );
    b = loop_in(b, 8, Expression::from('z') * 'N', &mut graph);

    let mut acc = graph.add_node(GraphTerm::NewAcc {
        starting_value: "0.0".to_string(),
    });
    acc = loop_in(
        acc,
        Expression::from('K') / 8,
        Term::Acc("Acca"),
        &mut graph,
    );
    acc = loop_in(acc, 8, Term::Acc("Accb"), &mut graph);

    let mut out = binary(
        binary(a, b, GraphTerm::Mul, &mut graph),
        acc,
        GraphTerm::Add,
        &mut graph,
    );

    out = loop_out(out, 8, Term::Acc("Accb"), &mut graph);
    out = loop_out(
        out,
        Expression::from('K') / 8,
        Term::Acc("Acca"),
        &mut graph,
    );
    out = pad_out(out, &mut graph, 1);
    out = loop_out(out, 8, 'z', &mut graph);
    out = loop_out(out, 8, Expression::from('z') * 'N', &mut graph);
    out = pad_out(out, &mut graph, 1);
    out = loop_out(
        out,
        Expression::from('N') / 8,
        Expression::from('z') * 8,
        &mut graph,
    );
    out = loop_out(
        out,
        Expression::from('M') / 8,
        Expression::from('z') * 'N' * 8,
        &mut graph,
    );
    out = unary(out, GraphTerm::GMEM { label: None }, &mut graph);
    (graph, out)
}

pub fn make_naive_attention() -> (StableGraph<GraphTerm, u8, Directed>, NodeIndex) {
    let mut graph = StableGraph::new();

    let n_qkv = 4;
    let d = 5;

    // inputs
    let mut q = graph.add_node(GraphTerm::GMEM {
        label: Some("Q".to_string()),
    });
    q = loop_in(q, n_qkv, Expression::from('z') * d, &mut graph);
    q = loop_in(q, n_qkv, 0, &mut graph);
    q = pad_in(q, &mut graph, 4);
    q = loop_in(q, d, 'z', &mut graph);
    let mut k = graph.add_node(GraphTerm::GMEM {
        label: Some("K".to_string()),
    });
    k = loop_in(k, n_qkv, 0, &mut graph);
    k = loop_in(k, n_qkv, Expression::from('z') * d, &mut graph);
    k = pad_in(k, &mut graph, 4);
    k = loop_in(k, d, 'z', &mut graph);
    let mut v = graph.add_node(GraphTerm::GMEM {
        label: Some("V".to_string()),
    });
    v = loop_in(v, n_qkv, 0, &mut graph);
    v = pad_in(v, &mut graph, 5);
    v = loop_in(v, n_qkv, Expression::from('z') * d, &mut graph);
    v = loop_in(v, d, 'z', &mut graph);

    // accumulators
    let mut dot_acc = graph.add_node(GraphTerm::NewAcc {
        starting_value: "0.0".to_string(),
    });
    dot_acc = loop_in(dot_acc, d, Term::Acc("AccDot"), &mut graph);
    let mut score_max_acc = graph.add_node(GraphTerm::NewAcc {
        starting_value: "-INFINITY".to_string(),
    });
    score_max_acc = loop_in(score_max_acc, n_qkv, Term::Acc("AccScoreMax"), &mut graph);
    let mut exp_sum_acc = graph.add_node(GraphTerm::NewAcc {
        starting_value: "0.0".to_string(),
    });
    exp_sum_acc = loop_in(exp_sum_acc, n_qkv, Term::Acc("AccExpSum"), &mut graph);
    let mut output_acc = graph.add_node(GraphTerm::NewAcc {
        starting_value: "0.0".to_string(),
    });
    output_acc = loop_in(output_acc, n_qkv, Term::Acc("AccOutput"), &mut graph);
    output_acc = loop_in(output_acc, d, 'z', &mut graph);

    // get dot products
    let mut dots = binary(
        binary(q, k, GraphTerm::Mul, &mut graph),
        dot_acc,
        GraphTerm::Add,
        &mut graph,
    );
    dots = loop_out(dots, d, Term::Acc("AccDot"), &mut graph);
    dots = pad_out(dots, &mut graph, 4);
    dots = loop_out(dots, n_qkv, 'z', &mut graph);
    dots = loop_out(dots, n_qkv, Expression::from('z') * n_qkv, &mut graph);

    // get max
    let mut dots_in = loop_in(dots, n_qkv, Expression::from('z') * n_qkv, &mut graph);
    dots_in = pad_in(dots_in, &mut graph, 5);
    dots_in = loop_in(dots_in, n_qkv, 'z', &mut graph);
    let mut max = binary(score_max_acc, dots_in, GraphTerm::Max, &mut graph);
    max = loop_out(max, n_qkv, Term::Acc("AccScoreMax"), &mut graph);
    max = pad_out(max, &mut graph, 5);
    max = loop_out(max, n_qkv, 'z', &mut graph);

    // get exp sum
    let mut max_in = loop_in(max, n_qkv, 'z', &mut graph);
    max_in = pad_in(max_in, &mut graph, 5);
    max_in = loop_in(max_in, n_qkv, 0, &mut graph);
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
    exp_sum = loop_out(exp_sum, n_qkv, Term::Acc("AccExpSum"), &mut graph);
    exp_sum = pad_out(exp_sum, &mut graph, 5);
    exp_sum = loop_out(exp_sum, n_qkv, 'z', &mut graph);

    // get final scores
    let mut exp_sum_in = loop_in(exp_sum, n_qkv, 'z', &mut graph);
    exp_sum_in = pad_in(exp_sum_in, &mut graph, 5);
    exp_sum_in = loop_in(exp_sum_in, n_qkv, 0, &mut graph);
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
    final_scores = loop_out(final_scores, n_qkv, 'z', &mut graph);
    final_scores = pad_out(final_scores, &mut graph, 5);
    final_scores = loop_out(
        final_scores,
        n_qkv,
        Expression::from('z') * n_qkv,
        &mut graph,
    );

    // get output
    let mut final_scores_in = loop_in(
        final_scores,
        n_qkv,
        Expression::from('z') * n_qkv,
        &mut graph,
    );
    final_scores_in = pad_in(final_scores_in, &mut graph, 5);
    final_scores_in = loop_in(final_scores_in, n_qkv, 'z', &mut graph);
    final_scores_in = loop_in(final_scores_in, d, 0, &mut graph);
    let mut output = binary(
        binary(v, final_scores_in, GraphTerm::Mul, &mut graph),
        output_acc,
        GraphTerm::Add,
        &mut graph,
    );
    output = loop_out(output, d, 'z', &mut graph);
    output = loop_out(output, n_qkv, Term::Acc("AccOutput"), &mut graph);
    output = pad_out(output, &mut graph, 5);
    output = loop_out(output, n_qkv, Expression::from('z') * d, &mut graph);
    output = unary(output, GraphTerm::GMEM { label: None }, &mut graph);
    (graph, output)
}

pub fn make_flash_attention() -> (StableGraph<GraphTerm, u8, Directed>, NodeIndex) {
    let mut graph = StableGraph::new();

    let n_qkv = 4;
    let d = 5;

    // inputs
    let mut q = graph.add_node(GraphTerm::GMEM {
        label: Some("Q".to_string()),
    });
    q = loop_in(q, n_qkv, Expression::from('z') * d, &mut graph);
    q = pad_in(q, &mut graph, 5);
    q = loop_in(q, n_qkv, 0, &mut graph);
    q = loop_in(q, d, 'z', &mut graph);
    let mut k = graph.add_node(GraphTerm::GMEM {
        label: Some("K".to_string()),
    });
    k = loop_in(k, n_qkv, 0, &mut graph);
    k = pad_in(k, &mut graph, 5);
    k = loop_in(k, n_qkv, Expression::from('z') * d, &mut graph);
    k = loop_in(k, d, 'z', &mut graph);
    let mut v = graph.add_node(GraphTerm::GMEM {
        label: Some("V".to_string()),
    });
    v = loop_in(v, n_qkv, 0, &mut graph);
    v = pad_in(v, &mut graph, 5);
    v = loop_in(v, n_qkv, Expression::from('z') * d, &mut graph);
    v = loop_in(v, d, 'z', &mut graph);

    // accumulators
    let mut dot_acc = graph.add_node(GraphTerm::NewAcc {
        starting_value: "0.0".to_string(),
    });
    dot_acc = loop_in(dot_acc, d, Term::Acc("AccDot"), &mut graph);
    let mut score_max_acc = graph.add_node(GraphTerm::NewAcc {
        starting_value: "-INFINITY".to_string(),
    });
    score_max_acc = loop_in(score_max_acc, n_qkv, Term::Acc("AccScoreMax"), &mut graph);
    let mut exp_sum_acc = graph.add_node(GraphTerm::NewAcc {
        starting_value: "0.0".to_string(),
    });
    exp_sum_acc = loop_in(exp_sum_acc, n_qkv, Term::Acc("AccExpSum"), &mut graph);
    let mut output_acc = graph.add_node(GraphTerm::NewAcc {
        starting_value: "0.0".to_string(),
    });
    output_acc = loop_in(output_acc, n_qkv, Term::Acc("AccOutput"), &mut graph);
    output_acc = loop_in(output_acc, d, 'z', &mut graph);

    // get dot products
    let dots = loop_out(
        binary(
            binary(q, k, GraphTerm::Mul, &mut graph),
            dot_acc,
            GraphTerm::Add,
            &mut graph,
        ),
        d,
        Term::Acc("AccDot"),
        &mut graph,
    );
    let new_max = binary(score_max_acc, dots, GraphTerm::Max, &mut graph);
    loop_out(new_max, n_qkv, Term::Acc("AccScoreMax"), &mut graph); // This is needed so we know to feed new max back in for score max acc

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
    let weight_b = loop_in(weight, d, 0, &mut graph);
    let rescale_b = loop_in(rescale, d, 0, &mut graph);
    let mut partial_output = binary(
        binary(output_acc, rescale_b, GraphTerm::Mul, &mut graph),
        binary(weight_b, v, GraphTerm::Mul, &mut graph),
        GraphTerm::Add,
        &mut graph,
    );
    partial_output = loop_out(partial_output, d, 'z', &mut graph);
    partial_output = loop_out(partial_output, n_qkv, Term::Acc("AccOutput"), &mut graph);
    partial_output = loop_in(partial_output, d, 'z', &mut graph);
    let exp_sum = loop_out(exp_sum_new, n_qkv, Term::Acc("AccExpSum"), &mut graph);
    let exp_sum_b = loop_in(exp_sum, d, 0, &mut graph);
    let mut output = binary(
        partial_output,
        unary(exp_sum_b, GraphTerm::Recip, &mut graph),
        GraphTerm::Mul,
        &mut graph,
    );
    output = loop_out(output, d, 'z', &mut graph);
    output = pad_out(output, &mut graph, 5);
    output = loop_out(output, n_qkv, Expression::from('z') * d, &mut graph);
    output = unary(output, GraphTerm::GMEM { label: None }, &mut graph);
    (graph, output)
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use crate::{GPUArch, codegen, display_graph, run_graph, symbolic::expression_cleanup};

    use super::*;

    #[test]
    fn test_sum_reduce() {
        let (graph, root) = make_sum_reduce();
        let kernels = codegen(graph, root, GPUArch::Metal(HashMap::new()));
        let input = vec![0., 1., 2., 3., 4.];
        let outputs = run_graph(vec![input], &kernels);
        assert_eq!(outputs[0], vec![10.0]);
        expression_cleanup();
    }

    #[test]
    fn test_flash_attention() {
        let (graph, root) = make_flash_attention();
        let kernels = codegen(graph, root, GPUArch::Metal(HashMap::new()));
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
        let outputs = run_graph(vec![q, k, v], &kernels).pop().unwrap();
        let pt_output = vec![
            [0.5441, 0.2194, -0.0533, 0.6271, -0.0108],
            [0.8770, 0.8527, 0.5614, 1.2643, 0.6696],
            [1.2617, 1.5422, 1.1725, 1.9658, 1.2628],
            [1.2003, 1.3576, 0.9888, 1.9141, 0.9768],
        ]
        .into_flattened();
        for (a, b) in outputs.into_iter().zip(pt_output) {
            assert!((a - b).abs() < 1e-3);
        }
        expression_cleanup();
    }

    #[test]
    fn test_naive_attention() {
        let (graph, root) = make_naive_attention();
        let kernels = codegen(graph, root, GPUArch::Metal(HashMap::new()));
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
        let outputs = run_graph(vec![q, k, v], &kernels).pop().unwrap();
        let pt_output = vec![
            [0.5441, 0.2194, -0.0533, 0.6271, -0.0108],
            [0.8770, 0.8527, 0.5614, 1.2643, 0.6696],
            [1.2617, 1.5422, 1.1725, 1.9658, 1.2628],
            [1.2003, 1.3576, 0.9888, 1.9141, 0.9768],
        ]
        .into_flattened();
        for (a, b) in outputs.into_iter().zip(pt_output) {
            assert!((a - b).abs() < 1e-3);
        }
        expression_cleanup();
    }
}
