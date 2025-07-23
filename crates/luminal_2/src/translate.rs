use luminal::prelude::{
    petgraph::{Directed, algo::toposort, prelude::StableGraph},
    *,
};
use rustc_hash::FxHashMap;

use crate::{
    CompatKernel, Diff, GraphTerm, Kernel,
    codegen::{GRID_DIMS, THREADBLOCK_DIMS},
    utils::loop_in,
};

pub enum InitData {
    Expr(Expression),
    Data(Vec<f32>),
}

pub fn translate_graph(
    graph: &Graph,
) -> (
    StableGraph<GraphTerm, (), Directed>,
    FxHashMap<NodeIndex, NodeIndex>,
    Vec<(NodeIndex, InitData)>,
) {
    let mut new_graph = StableGraph::new();
    let mut node_mapping = FxHashMap::default();
    let mut old_to_new_mapping = FxHashMap::default();
    let mut inits = vec![];
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
                let (source, output_index, mut shape) = sources.pop().unwrap();
                let mut new_source = node_mapping[&(source, output_index)];
                new_source = scope_in(new_source, shape, None, &mut new_graph, &mut inits);
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
                shape = shape.contiguous();
                for (i, (stride, range)) in shape
                    .dims()
                    .into_iter()
                    .rev()
                    .scan(Expression::from(1), |i, s| {
                        let r = *i;
                        *i *= s;
                        Some(r)
                    })
                    .zip(shape.dims().into_iter().rev())
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
                old_to_new_mapping.insert(node, op);
                node_mapping.insert((node, 0), op);
            }
            "Add" | "Mul" | "Mod" | "LessThan" => {
                // walk through input ranges and strides, making new loopins as we go
                let (source_a, output_index_a, mut shape_a) = sources.pop().unwrap();
                let mut new_source_a = node_mapping[&(source_a, output_index_a)];
                new_source_a = scope_in(new_source_a, shape_a, None, &mut new_graph, &mut inits);
                let (source_b, output_index_b, shape_b) = sources.pop().unwrap();
                let mut new_source_b = node_mapping[&(source_b, output_index_b)];
                new_source_b = scope_in(new_source_b, shape_b, None, &mut new_graph, &mut inits);
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
                shape_a = shape_a.contiguous();
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
                old_to_new_mapping.insert(node, op);
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
                    label: Some(format!("acc_{}", inits.len())),
                });
                let orig_acc = acc;
                // walk through input ranges and strides, making new loopins as we go
                let (source, output_index, mut shape) = sources.pop().unwrap();
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
                new_source = scope_in(
                    new_source,
                    shape,
                    Some(reduce_dim),
                    &mut new_graph,
                    &mut inits,
                );
                for (i, (range, acc_stride)) in shape.dims().into_iter().zip(rm_strides).enumerate()
                {
                    if i == reduce_dim {
                        for z in i..THREADBLOCK_DIMS + GRID_DIMS {
                            let new_acc = new_graph.add_node(GraphTerm::LoopIn {
                                range: 1.into(),
                                stride: 0.into(),
                                marker: format!("pad{z}"),
                            });
                            new_graph.add_edge(acc, new_acc, ());
                            acc = new_acc;
                        }
                    }
                    let stride = if i == reduce_dim {
                        Expression::from(Term::Acc('a'))
                    } else if i > reduce_dim {
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
                inits.push((orig_acc, InitData::Data(vec![start_val])));
                // Insert op
                let mut op = new_graph.add_node(term);
                new_graph.add_edge(new_source, op, ());
                new_graph.add_edge(acc, op, ());
                // walk through output and place loopouts
                shape = shape.contiguous();
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
                old_to_new_mapping.insert(node, op);
                node_mapping.insert((node, 0), op);
            }
            _ => {
                if let Some(constant) = node_weight.as_any().downcast_ref::<Constant>() {
                    // Init constants in GMEM
                    let new = new_graph.add_node(GraphTerm::GMEM {
                        label: Some(op.to_string()),
                    });
                    old_to_new_mapping.insert(node, new);
                    node_mapping.insert((node, 0), new);
                    inits.push((
                        new,
                        match constant.0 {
                            ConstantValue::Expression(e) => InitData::Expr(e),
                            ConstantValue::Float(f) => InitData::Data(vec![f]),
                        },
                    ));
                } else if let Some(kernel) = node_weight.as_any().downcast_ref::<CompatKernel>() {
                    // Add a custom kernel
                    let custom = new_graph.add_node(GraphTerm::Custom(kernel.0.clone()));
                    for (source, ind, _) in sources {
                        let new_source = node_mapping[&(source, ind)];
                        new_graph.add_edge(new_source, custom, ());
                    }
                    node_mapping.insert((node, 0), custom);
                    old_to_new_mapping.insert(node, custom);
                } else if let Some(diff) = node_weight.as_any().downcast_ref::<Diff>() {
                    // Add a custom kernel
                    let custom = new_graph.add_node(GraphTerm::Diff(diff.name.clone()));
                    for (source, ind, _) in sources {
                        let new_source = node_mapping[&(source, ind)];
                        new_graph.add_edge(new_source, custom, ());
                    }
                    node_mapping.insert((node, 0), custom);
                    old_to_new_mapping.insert(node, custom);
                } else {
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
    }

    // // Add gmems for to_retrieve
    // for (t, _) in &graph.to_retrieve {
    //     let gmem = new_graph.add_node(GraphTerm::GMEM {
    //         label: Some("Output".to_string()),
    //     });
    //     new_graph.add_edge(old_to_new_mapping[t], gmem, ());
    //     old_to_new_mapping.insert(*t, gmem);
    // }

    (new_graph, old_to_new_mapping, inits)
}

fn scope_in(
    mut src: NodeIndex,
    shape: ShapeTracker,
    reduce_dim: Option<usize>,
    graph: &mut StableGraph<GraphTerm, (), Directed>,
    inits: &mut Vec<(NodeIndex, InitData)>,
) -> NodeIndex {
    // Loop in through all dimensions, handle padding
    let strides = shape.strides();
    let mut pad_mask = None; // mask for pads
    for i in 0..shape.len() {
        let mut range = shape.dims()[i];
        let mut stride = strides[i] * 'z';
        let (left_pad, right_pad) = shape.padding[shape.indexes[i]];
        let (left_slice, right_slice) = shape.mask[shape.indexes[i]];
        if reduce_dim.map(|d| d == i).unwrap_or_default() {
            assert!(
                left_pad == 0 && right_pad == 0,
                "pad on a reduce dim not implemented!"
            );
            for z in i..THREADBLOCK_DIMS + GRID_DIMS {
                src = loop_in(src, 1, 0, format!("pad{z}"), graph);
            }
            src = loop_in(src, range, stride, i, graph); // Problem: acc stride only is ever 'a', which doesn't work if there is multiple accs!
        } else if left_pad != 0 {
            assert!(right_pad == 0);
            // Pad left
            stride = stride.substitute('z', (Expression::from('z') - left_pad).max(0));
            // Bring in mask
            let mut mask = graph.add_node(GraphTerm::GMEM {
                label: Some("Mask".to_string()),
            });
            inits.push((mask, InitData::Data(vec![0., 1.])));
            // Loop mask in
            for level in 0..i {
                mask = loop_in(mask, shape.dims()[level], 0, level, graph);
            }
            mask = loop_in(mask, range, Expression::from('z').gte(left_pad), i, graph);
            for level in (i + 1)..shape.len() {
                mask = loop_in(mask, shape.dims()[level], 0, level, graph);
            }
            pad_mask = Some(mask);
            src = loop_in(src, range, stride, i, graph);
        } else if right_pad != 0 {
            assert!(left_pad == 0);
            // Pad right
            stride = stride.substitute(
                'z',
                Expression::from('z').min(shape.dims[shape.indexes[i]] - 1),
            );
            // Bring in mask
            let mut mask = graph.add_node(GraphTerm::GMEM {
                label: Some("Mask".to_string()),
            });
            inits.push((mask, InitData::Data(vec![0., 1.])));
            // Loop mask in
            for level in 0..i {
                mask = loop_in(mask, shape.dims()[level], 0, level, graph);
            }
            mask = loop_in(
                mask,
                range,
                Expression::from('z').lt(shape.dims[shape.indexes[i]]),
                i,
                graph,
            );
            for level in (i + 1)..shape.len() {
                mask = loop_in(mask, shape.dims()[level], 0, level, graph);
            }
            pad_mask = Some(mask);
            src = loop_in(src, range, stride, i, graph);
        } else if left_slice != 0 || right_slice != i32::MAX {
            range = (right_slice.min(shape.dims[shape.indexes[i]]) - left_slice).max(0);
            stride = stride.substitute('z', Expression::from('z') + left_slice);
            src = loop_in(src, range, stride, i, graph);
        } else {
            // No pads or reduces
            src = loop_in(src, range, stride, i, graph);
        }
    }
    if let Some(mask) = pad_mask {
        crate::utils::binary(src, mask, GraphTerm::Mul, graph)
    } else {
        src
    }
}
