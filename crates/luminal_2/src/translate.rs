use luminal::prelude::{
    petgraph::{Directed, algo::toposort, prelude::StableGraph},
    *,
};
use rustc_hash::FxHashMap;

use crate::{
    GraphTerm,
    codegen::{GRID_DIMS, THREADBLOCK_DIMS},
};

pub fn translate_graph(
    graph: &Graph,
) -> (
    StableGraph<GraphTerm, (), Directed>,
    FxHashMap<NodeIndex, NodeIndex>,
    Vec<(NodeIndex, f32, Expression)>,
) {
    let mut new_graph = StableGraph::new();
    let mut node_mapping = FxHashMap::default();
    let mut old_to_new_mapping = FxHashMap::default();
    let mut accumulators = vec![];
    for node in toposort(&graph.graph, None).unwrap() {
        let node_weight = graph.node_weight(node).unwrap();
        let op_name_full = format!("{node_weight:?}");
        let op = op_name_full
            .split('|')
            .next()
            .unwrap_or(&op_name_full)
            .trim();
        let mut sources = graph.get_sources(node);
        match op {
            "Sqrt" | "Exp2" | "Log2" | "Sin" | "Contiguous" | "Recip" => {
                // walk through input ranges and strides, making new loopins as we go
                let (source, output_index, shape) = sources.pop().unwrap();
                let mut new_source = node_mapping[&(source, output_index)];
                for (i, (range, stride)) in
                    shape.dims().into_iter().zip(shape.strides()).enumerate()
                {
                    let loopin = new_graph.add_node(GraphTerm::LoopIn {
                        range,
                        stride: Expression::from('z') * stride,
                        marker: i.to_string(),
                    });
                    new_graph.add_edge(new_source, loopin, ());
                    new_source = loopin;
                }
                let mut op = if op == "Contiguous" {
                    new_source
                } else {
                    let r = new_graph.add_node(match op {
                        "Sqrt" => GraphTerm::Sqrt,
                        "Exp2" => GraphTerm::Exp2,
                        "Log2" => GraphTerm::Log2,
                        "Sin" => GraphTerm::Sin,
                        "Recip" => GraphTerm::Recip,
                        _ => unreachable!(),
                    });
                    new_graph.add_edge(new_source, r, ());
                    r
                };
                // walk through output and place loopouts
                for (i, (stride, range)) in shape
                    .dims()
                    .into_iter()
                    .rev()
                    .scan(Expression::from(1), |i, s| {
                        let r = *i;
                        *i *= s;
                        Some(r)
                    })
                    .zip(shape.dims())
                    .enumerate()
                {
                    let loopout = new_graph.add_node(GraphTerm::LoopOut {
                        range,
                        stride: Expression::from('z') * stride,
                        marker: (shape.dims().len() - i - 1).to_string(),
                    });
                    new_graph.add_edge(op, loopout, ());
                    op = loopout;
                }
                node_mapping.insert((node, 0), op);
            }
            "Add" | "Mul" | "Mod" | "LessThan" => {
                // walk through input ranges and strides, making new loopins as we go
                let (source_a, output_index_a, shape_a) = sources.pop().unwrap();
                let mut new_source_a = node_mapping[&(source_a, output_index_a)];
                for (i, (range, stride)) in shape_a
                    .dims()
                    .into_iter()
                    .zip(shape_a.strides())
                    .enumerate()
                {
                    let loopin = new_graph.add_node(GraphTerm::LoopIn {
                        range,
                        stride: Expression::from('z') * stride,
                        marker: i.to_string(),
                    });
                    new_graph.add_edge(new_source_a, loopin, ());
                    new_source_a = loopin;
                }
                let (source_b, output_index_b, shape_b) = sources.pop().unwrap();
                let mut new_source_b = node_mapping[&(source_b, output_index_b)];
                for (i, (range, stride)) in shape_b
                    .dims()
                    .into_iter()
                    .zip(shape_b.strides())
                    .enumerate()
                {
                    let loopin = new_graph.add_node(GraphTerm::LoopIn {
                        range,
                        stride: Expression::from('z') * stride,
                        marker: i.to_string(),
                    });
                    new_graph.add_edge(new_source_b, loopin, ());
                    new_source_b = loopin;
                }
                let mut op = new_graph.add_node(match op {
                    "Add" => GraphTerm::Add,
                    "Mul" => GraphTerm::Mul,
                    "Mod" => GraphTerm::Mod,
                    "LessThan" => GraphTerm::LessThan,
                    _ => unreachable!(),
                });
                new_graph.add_edge(new_source_a, op, ());
                new_graph.add_edge(new_source_b, op, ());
                // walk through output and place loopouts
                for (i, (stride, range)) in shape_a
                    .dims()
                    .into_iter()
                    .rev()
                    .scan(Expression::from(1), |i, s| {
                        let r = *i;
                        *i *= s;
                        Some(r)
                    })
                    .zip(shape_a.dims().into_iter().rev())
                    .enumerate()
                {
                    let loopout = new_graph.add_node(GraphTerm::LoopOut {
                        range,
                        stride: Expression::from('z') * stride,
                        marker: (shape_a.dims().len() - i - 1).to_string(),
                    });
                    new_graph.add_edge(op, loopout, ());
                    op = loopout;
                }
                node_mapping.insert((node, 0), op);
            }
            s if s.starts_with("SumReduce") || s.starts_with("MaxReduce") => {
                // make accumulator
                let (start_val, term, reduce_dim) = match op {
                    s if s.starts_with("SumReduce") => {
                        (0.0, GraphTerm::Add, graph.get_op::<SumReduce>(node).0)
                    }
                    s if s.starts_with("MaxReduce") => (
                        f32::NEG_INFINITY,
                        GraphTerm::Max,
                        graph.get_op::<MaxReduce>(node).0,
                    ),
                    _ => unreachable!(),
                };
                let mut acc = new_graph.add_node(GraphTerm::GMEM {
                    label: Some(format!("acc_{}", accumulators.len())),
                });
                let orig_acc = acc;
                // walk through input ranges and strides, making new loopins as we go
                let (source, output_index, shape) = sources.pop().unwrap();
                let mut new_source = node_mapping[&(source, output_index)];
                let mut rm_strides = shape
                    .dims()
                    .into_iter()
                    .rev()
                    .scan(Expression::from(1), |i, s| {
                        let r = *i;
                        *i *= s;
                        Some(r)
                    })
                    .collect::<Vec<_>>();
                rm_strides.reverse();
                let mut acc_size = Expression::from(1);
                for (i, ((range, stride), acc_stride)) in shape
                    .dims()
                    .into_iter()
                    .zip(shape.strides())
                    .zip(rm_strides)
                    .enumerate()
                {
                    if i == reduce_dim {
                        for z in i..THREADBLOCK_DIMS + GRID_DIMS {
                            let loopin = new_graph.add_node(GraphTerm::LoopIn {
                                range: 1.into(),
                                stride: 0.into(),
                                marker: format!("pad{z}"),
                            });
                            new_graph.add_edge(new_source, loopin, ());
                            new_source = loopin;
                            let new_acc = new_graph.add_node(GraphTerm::LoopIn {
                                range: 1.into(),
                                stride: 0.into(),
                                marker: format!("pad{z}"),
                            });
                            new_graph.add_edge(acc, new_acc, ());
                            acc = new_acc;
                        }
                    }
                    let loopin = new_graph.add_node(GraphTerm::LoopIn {
                        range,
                        stride: Expression::from('z') * stride,
                        marker: i.to_string(),
                    });
                    new_graph.add_edge(new_source, loopin, ());
                    new_source = loopin;
                    let stride = if i == reduce_dim {
                        Expression::from(Term::Acc('a'))
                    } else if i > reduce_dim {
                        acc_size = acc_size.max(acc_stride * range);
                        Expression::from('z') * acc_stride
                    } else {
                        Expression::from(0)
                    };
                    let new_acc = new_graph.add_node(GraphTerm::LoopIn {
                        range,
                        stride,
                        marker: i.to_string(),
                    });
                    new_graph.add_edge(acc, new_acc, ());
                    acc = new_acc;
                }
                accumulators.push((orig_acc, start_val, acc_size));
                // Insert op
                let mut op = new_graph.add_node(term);
                new_graph.add_edge(new_source, op, ());
                new_graph.add_edge(acc, op, ());
                // walk through output and place loopouts
                for ((stride, i), range) in shape
                    .dims()
                    .into_iter()
                    .enumerate()
                    .rev()
                    .scan(Expression::from(1), |i, (ind, range)| {
                        if ind == reduce_dim {
                            Some((Expression::from(Term::Acc('a')), ind))
                        } else {
                            let r = *i;
                            *i *= range;
                            Some((r, ind))
                        }
                    })
                    .zip(shape.dims().into_iter().rev())
                {
                    let loopout = new_graph.add_node(GraphTerm::LoopOut {
                        range,
                        stride: if i == reduce_dim {
                            stride
                        } else {
                            Expression::from('z') * stride
                        },
                        marker: i.to_string(),
                    });
                    new_graph.add_edge(op, loopout, ());
                    op = loopout;
                    if i == reduce_dim {
                        for z in (i..THREADBLOCK_DIMS + GRID_DIMS).rev() {
                            let loopout = new_graph.add_node(GraphTerm::LoopOut {
                                range: 1.into(),
                                stride: 0.into(),
                                marker: format!("pad{z}"),
                            });
                            new_graph.add_edge(op, loopout, ());
                            op = loopout;
                        }
                    }
                }
                node_mapping.insert((node, 0), op);
            }
            s if s.starts_with("Constant(") => {
                // Fake constant support by providing it as an "accumulator" initialization from gmem
                let value = if s.contains("p") {
                    0.0
                } else {
                    s.replace("Constant(", "")
                        .replace(")", "")
                        .parse::<f32>()
                        .unwrap()
                };
                println!("CONSTANT: {:?}", value);
                let new = new_graph.add_node(GraphTerm::GMEM {
                    label: Some(op.to_string()),
                });
                node_mapping.insert((node, 0), new);
                accumulators.push((new, value, Expression::from(1)));
            }
            _ => {
                assert!(op.contains("Load"));
                // Assume a load
                let new = new_graph.add_node(GraphTerm::GMEM {
                    label: Some(op.to_string()),
                });
                node_mapping.insert((node, 0), new);
                old_to_new_mapping.insert(node, new);
            }
        }
    }

    (new_graph, old_to_new_mapping, accumulators)
}
