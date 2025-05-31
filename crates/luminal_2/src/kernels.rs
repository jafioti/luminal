use petgraph::{Directed, graph::NodeIndex, prelude::StableGraph};

use crate::GraphTerm;

fn loop_in(
    node: NodeIndex,
    range: impl ToString,
    stride: impl ToString,
    graph: &mut StableGraph<GraphTerm, u8, Directed>,
) -> NodeIndex {
    let tmp = graph.add_node(GraphTerm::LoopIn {
        range: range.to_string(),
        stride: stride.to_string(),
    });
    graph.add_edge(node, tmp, 0);
    tmp
}

fn loop_out(
    node: NodeIndex,
    range: impl ToString,
    stride: impl ToString,
    graph: &mut StableGraph<GraphTerm, u8, Directed>,
) -> NodeIndex {
    let tmp = graph.add_node(GraphTerm::LoopOut {
        range: range.to_string(),
        stride: stride.to_string(),
    });
    graph.add_edge(node, tmp, 0);
    tmp
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
    let exp = graph.add_node(GraphTerm::Exp);
    graph.add_edge(in_a, exp, 0);
    let add_acc = graph.add_node(GraphTerm::Add);
    graph.add_edge(exp, add_acc, 0);
    graph.add_edge(in_exp_acc, add_acc, 0);
    let add_acc_out = loop_out(add_acc, "5", "Accz", &mut graph);

    // Sin
    let sin = graph.add_node(GraphTerm::Sin);
    graph.add_edge(add_acc_out, sin, 0);

    // Other input
    let b = graph.add_node(GraphTerm::Tensor {
        name: "B".to_string(),
    });
    let in_b = loop_in(pad_in(b, &mut graph, 6), "50", "z", &mut graph);

    // Mul
    let mul = graph.add_node(GraphTerm::Mul);
    graph.add_edge(sin, mul, 0);
    graph.add_edge(in_b, mul, 0);

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
