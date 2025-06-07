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
    let a = graph.add_node(GraphTerm::Tensor {
        name: "A".to_string(),
    });
    let in_a = loop_in(
        loop_in(pad_in(a, &mut graph, 6), "50", "z * 5", &mut graph),
        "5",
        "z",
        &mut graph,
    );

    // Acc
    let slin0 = graph.add_node(GraphTerm::Tensor {
        name: "acc".to_string(),
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
    let b = graph.add_node(GraphTerm::Tensor {
        name: "B".to_string(),
    });
    let in_b = loop_in(pad_in(b, &mut graph, 6), "50", "z", &mut graph);

    // Mul
    let mul = binary(sin, in_b, GraphTerm::Mul, &mut graph);

    let mul_out = loop_out(mul, "50", "z", &mut graph);

    let out = pad_out(mul_out, &mut graph, 6);
    (graph, out)
}

pub fn make_sum_reduce() -> (StableGraph<GraphTerm, u8, Directed>, NodeIndex) {
    let mut graph = StableGraph::new();
    let lin0 = graph.add_node(GraphTerm::Tensor {
        name: "A".to_string(),
    });
    let lin5 = pad_in(lin0, &mut graph, 6);
    let lin6 = graph.add_node(GraphTerm::LoopIn {
        range: "5".to_string(),
        stride: "z".to_string(),
    });
    graph.add_edge(lin5, lin6, 0);

    let slin0 = graph.add_node(GraphTerm::Tensor {
        name: "acc".to_string(),
    });
    let slin5 = pad_in(slin0, &mut graph, 6);
    let slin6 = graph.add_node(GraphTerm::LoopIn {
        range: "5".to_string(),
        stride: "Accz".to_string(),
    });
    graph.add_edge(slin5, slin6, 0);

    let add = graph.add_node(GraphTerm::Add);
    graph.add_edge(lin6, add, 0);
    graph.add_edge(slin6, add, 0);

    let acc = graph.add_node(GraphTerm::LoopOut {
        range: "5".to_string(),
        stride: "Accz".to_string(),
    });
    graph.add_edge(add, acc, 0);
    let olin5 = pad_out(acc, &mut graph, 6);

    (graph, olin5)
}

pub fn make_matmul() -> (StableGraph<GraphTerm, u8, Directed>, NodeIndex) {
    let mut graph = StableGraph::new();
    let mut a = graph.add_node(GraphTerm::Tensor {
        name: "A".to_string(),
    });
    a = loop_in(a, "M", "z * K", &mut graph);
    a = loop_in(a, "N", "0", &mut graph);
    a = pad_in(a, &mut graph, 4);
    a = loop_in(a, "K", "z", &mut graph);

    let mut b = graph.add_node(GraphTerm::Tensor {
        name: "B".to_string(),
    });
    b = loop_in(b, "M", "0", &mut graph);
    b = loop_in(b, "N", "z", &mut graph);
    b = pad_in(b, &mut graph, 4);
    b = loop_in(b, "K", "z * N", &mut graph);

    let mut acc = graph.add_node(GraphTerm::Tensor {
        name: "acc".to_string(),
    });
    acc = loop_in(acc, "M", "0", &mut graph);
    acc = loop_in(acc, "N", "0", &mut graph);
    acc = pad_in(acc, &mut graph, 4);
    acc = loop_in(acc, "K", "Acca", &mut graph);

    let mul = graph.add_node(GraphTerm::Mul);
    graph.add_edge(a, mul, 0);
    graph.add_edge(b, mul, 0);

    let add = graph.add_node(GraphTerm::Add);
    graph.add_edge(mul, add, 0);
    graph.add_edge(acc, add, 0);

    let mut out = loop_out(add, "K", "Acca", &mut graph);
    out = pad_out(out, &mut graph, 4);
    out = loop_out(out, "N", "z", &mut graph);
    out = loop_out(out, "M", "z * N", &mut graph);

    (graph, out)
}

pub fn make_tiled_matmul_basic() -> (StableGraph<GraphTerm, u8, Directed>, NodeIndex) {
    let mut graph = StableGraph::new();
    let mut a = graph.add_node(GraphTerm::Tensor {
        name: "A".to_string(),
    });
    a = loop_in(a, "M / 8", "z * K * 8", &mut graph);
    a = loop_in(a, "N / 8", "0", &mut graph);
    a = pad_in(a, &mut graph, 1);
    a = loop_in(a, "8", "z * K", &mut graph);
    a = loop_in(a, "8", "0", &mut graph);
    a = pad_in(a, &mut graph, 1);
    a = loop_in(a, "K / 8", "z * 8", &mut graph);
    a = loop_in(a, "8", "z", &mut graph);

    let mut b = graph.add_node(GraphTerm::Tensor {
        name: "B".to_string(),
    });
    b = loop_in(b, "M / 8", "0", &mut graph);
    b = loop_in(b, "N / 8", "z * 8", &mut graph);
    b = pad_in(b, &mut graph, 1);
    b = loop_in(b, "8", "0", &mut graph);
    b = loop_in(b, "8", "z", &mut graph);
    b = pad_in(b, &mut graph, 1);
    b = loop_in(b, "K / 8", "z * N * 8", &mut graph);
    b = loop_in(b, "8", "z * N", &mut graph);

    let mut acc = graph.add_node(GraphTerm::Tensor {
        name: "acc".to_string(),
    });
    acc = loop_in(acc, "M / 8", "0", &mut graph);
    acc = loop_in(acc, "N / 8", "0", &mut graph);
    acc = pad_in(acc, &mut graph, 1);
    acc = loop_in(acc, "8", "0", &mut graph);
    acc = loop_in(acc, "8", "0", &mut graph);
    acc = pad_in(acc, &mut graph, 1);
    acc = loop_in(acc, "K / 8", "Acca", &mut graph);
    acc = loop_in(acc, "8", "Acca", &mut graph);

    let mul = graph.add_node(GraphTerm::Mul);
    graph.add_edge(a, mul, 0);
    graph.add_edge(b, mul, 0);

    let add = graph.add_node(GraphTerm::Add);
    graph.add_edge(mul, add, 0);
    graph.add_edge(acc, add, 0);

    let mut out = loop_out(add, "K / 8", "Acca", &mut graph);
    out = loop_out(out, "8", "Acca", &mut graph);
    out = pad_out(out, &mut graph, 1);
    out = loop_out(out, "8", "z", &mut graph);
    out = loop_out(out, "8", "z * N", &mut graph);
    out = pad_out(out, &mut graph, 1);
    out = loop_out(out, "N / 8", "z * 8", &mut graph);
    out = loop_out(out, "M / 8", "z * N * 8", &mut graph);

    (graph, out)
}

pub fn make_tiled_matmul() -> (StableGraph<GraphTerm, u8, Directed>, NodeIndex) {
    let mut graph = StableGraph::new();
    let mut a = graph.add_node(GraphTerm::Tensor {
        name: "A".to_string(),
    });
    a = loop_in(a, "M / 8", "z * K * 8", &mut graph);
    a = loop_in(a, "N / 8", "0", &mut graph);
    a = pad_in(a, &mut graph, 1);
    a = loop_in(a, "8", "z * K", &mut graph);
    a = loop_in(a, "8", "z", &mut graph);
    a = pad_in(a, &mut graph, 1);
    a = loop_in(a, "K / 8", "z * 8", &mut graph);

    let mut b = graph.add_node(GraphTerm::Tensor {
        name: "B".to_string(),
    });
    b = loop_in(b, "M / 8", "0", &mut graph);
    b = loop_in(b, "N / 8", "z * 8", &mut graph);
    b = pad_in(b, &mut graph, 1);
    b = loop_in(b, "8", "z * N", &mut graph);
    b = loop_in(b, "8", "z", &mut graph);
    b = pad_in(b, &mut graph, 1);
    b = loop_in(b, "K / 8", "z * N * 8", &mut graph);

    let mut acc = graph.add_node(GraphTerm::Tensor {
        name: "acc".to_string(),
    });
    acc = loop_in(acc, "M / 8", "0", &mut graph);
    acc = loop_in(acc, "N / 8", "0", &mut graph);
    acc = pad_in(acc, &mut graph, 1);
    acc = loop_in(acc, "8", "0", &mut graph);
    acc = loop_in(acc, "8", "0", &mut graph);
    acc = pad_in(acc, &mut graph, 1);
    acc = loop_in(acc, "K / 8", "Acca", &mut graph);
    acc = loop_in(acc, "8", "Accb", &mut graph);

    let mut smem_a = graph.add_node(GraphTerm::Smem);
    smem_a = loop_in(smem_a, "M / 8", "0", &mut graph);
    smem_a = loop_in(smem_a, "N / 8", "0", &mut graph);
    smem_a = pad_in(smem_a, &mut graph, 1);
    smem_a = loop_in(smem_a, "8", "z * 8", &mut graph);
    smem_a = loop_in(smem_a, "8", "z", &mut graph);
    smem_a = pad_in(smem_a, &mut graph, 1);
    smem_a = loop_in(smem_a, "K / 8", "0", &mut graph);

    smem_a = binary(smem_a, a, GraphTerm::SmemCopy, &mut graph);
    smem_a = loop_in(smem_a, "8", "z", &mut graph);

    let mut smem_b = graph.add_node(GraphTerm::Smem);
    smem_b = loop_in(smem_b, "M / 8", "0", &mut graph);
    smem_b = loop_in(smem_b, "N / 8", "0", &mut graph);
    smem_b = pad_in(smem_b, &mut graph, 1);
    smem_b = loop_in(smem_b, "8", "z * 8", &mut graph);
    smem_b = loop_in(smem_b, "8", "z", &mut graph);
    smem_b = pad_in(smem_b, &mut graph, 1);
    smem_b = loop_in(smem_b, "K / 8", "0", &mut graph);

    smem_b = binary(smem_b, b, GraphTerm::SmemCopy, &mut graph);
    smem_b = loop_in(smem_b, "8", "z * 8", &mut graph);

    let mul = binary(smem_a, smem_b, GraphTerm::Mul, &mut graph);
    let add = binary(mul, acc, GraphTerm::Add, &mut graph);

    let mut out = loop_out(add, "8", "Accb", &mut graph);
    out = loop_out(out, "K / 8", "Acca", &mut graph);
    out = pad_out(out, &mut graph, 1);
    out = loop_out(out, "8", "z", &mut graph);
    out = loop_out(out, "8", "z * N", &mut graph);
    out = pad_out(out, &mut graph, 1);
    out = loop_out(out, "N / 8", "z * 8", &mut graph);
    out = loop_out(out, "M / 8", "z * N * 8", &mut graph);

    (graph, out)
}
