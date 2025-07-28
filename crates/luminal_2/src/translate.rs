use luminal::prelude::{
    petgraph::{Directed, algo::toposort, prelude::StableGraph},
    *,
};
use rustc_hash::FxHashMap;

use crate::{
    CompatKernel, Diff, GraphTerm,
    codegen::{GRID_DIMS, THREADBLOCK_DIMS},
    utils::{loop_in, loop_out},
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
    let mut simplify_cache = FxHashMap::default();
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
                let new_source = node_mapping[&(source, output_index)];
                let (new_source, ranges) = scope_in(
                    new_source,
                    shape,
                    None,
                    &mut new_graph,
                    &mut inits,
                    &mut simplify_cache,
                );
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
                op = scope_out(op, ranges, false, &mut new_graph);
                old_to_new_mapping.insert(node, op);
                node_mapping.insert((node, 0), op);
            }
            "Add" | "Mul" | "Mod" | "LessThan" => {
                // walk through input ranges and strides, making new loopins as we go
                let (source_a, output_index_a, shape_a) = sources.pop().unwrap();
                let (new_source_a, ranges) = scope_in(
                    node_mapping[&(source_a, output_index_a)],
                    shape_a,
                    None,
                    &mut new_graph,
                    &mut inits,
                    &mut simplify_cache,
                );
                let (source_b, output_index_b, shape_b) = sources.pop().unwrap();
                let (new_source_b, _) = scope_in(
                    node_mapping[&(source_b, output_index_b)],
                    shape_b,
                    None,
                    &mut new_graph,
                    &mut inits,
                    &mut simplify_cache,
                );
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
                op = scope_out(op, ranges, false, &mut new_graph);
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
                let (source, output_index, shape) = sources.pop().unwrap();
                let (new_source, ranges) = scope_in(
                    node_mapping[&(source, output_index)],
                    shape,
                    Some(reduce_dim),
                    &mut new_graph,
                    &mut inits,
                    &mut simplify_cache,
                );
                let mut rm_strides = ranges
                    .iter()
                    .rev()
                    .scan(Expression::from(1), |i, (s, _)| {
                        let r = *i;
                        *i *= s;
                        Some(r)
                    })
                    .collect::<Vec<_>>();
                rm_strides.reverse();
                for (i, ((range, name), acc_stride)) in ranges.iter().zip(rm_strides).enumerate() {
                    let stride = if i == GRID_DIMS + THREADBLOCK_DIMS {
                        Expression::from(Term::Acc('a'))
                    } else if i > GRID_DIMS + THREADBLOCK_DIMS {
                        Expression::from('z') * acc_stride
                    } else {
                        Expression::from(0)
                    };
                    let new_acc = new_graph.add_node(GraphTerm::LoopIn {
                        range: *range,
                        stride,
                        marker: name.to_string(),
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
                op = scope_out(op, ranges, true, &mut new_graph);
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
                    // Add a diff
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

fn smart_loop_in(
    shape: ShapeTracker,
    shape_index: usize,
) -> (Expression, Expression, Option<Expression>) {
    // range, stride, mask stride
    let mut range = shape.dims()[shape_index];
    let mut stride = shape.strides()[shape_index] * 'z';
    let (left_pad, right_pad) = shape.padding[shape.indexes[shape_index]];
    let (left_slice, right_slice) = shape.mask[shape.indexes[shape_index]];
    let mut mask_stride = None;
    if left_pad != 0 {
        // Pad left
        assert!(right_pad == 0);
        stride = stride.substitute('z', (Expression::from('z') - left_pad).max(0));
        mask_stride = Some(Expression::from('z').gte(left_pad));
    } else if right_pad != 0 {
        // Pad right
        assert!(left_pad == 0);
        stride = stride.substitute(
            'z',
            Expression::from('z').min((shape.dims[shape.indexes[shape_index]] - 1).max(0)),
        );
        mask_stride = Some(Expression::from('z').lt(shape.dims[shape.indexes[shape_index]]));
    } else if left_slice != 0 || right_slice != i32::MAX {
        range = (right_slice.min(shape.dims[shape.indexes[shape_index]]) - left_slice).max(0);
        stride = stride.substitute('z', Expression::from('z') + left_slice);
    }
    (range, stride, mask_stride)
}

fn scope_in(
    mut src: NodeIndex,
    shape: ShapeTracker,
    reduce_dim: Option<usize>,
    graph: &mut StableGraph<GraphTerm, (), Directed>,
    inits: &mut Vec<(NodeIndex, InitData)>,
    simplify_cache: &mut FxHashMap<Expression, Expression>,
) -> (NodeIndex, Vec<(Expression, String)>) {
    let mut masks = vec![];
    let mut ranges = vec![];

    // Go through all dims up to the reduce and put in first grid dimension (super inefficient!)
    let mut range = Expression::from(1);
    let mut stride = Expression::from(0);
    let n_squeezed_dims = reduce_dim.unwrap_or(shape.len());
    for i in 0..n_squeezed_dims {
        let (curr_range, curr_stride, mask_stride) = smart_loop_in(shape, i);
        let element_size = shape
            .dims()
            .iter()
            .take(n_squeezed_dims)
            .skip(i + 1)
            .copied()
            .product::<Expression>()
            .max(1);
        if let Some(mask_stride) = mask_stride {
            // Bring in mask
            let mask = graph.add_node(GraphTerm::GMEM {
                label: Some("Mask".to_string()),
            });
            inits.push((mask, InitData::Data(vec![0., 1.])));
            // Loop mask in
            let mut mask_range = curr_range;
            for level in 0..i {
                mask_range *= shape.dims()[level];
            }
            let mask_stride =
                mask_stride.substitute('z', Expression::from('z') / element_size % curr_range);
            for level in (i + 1)..shape.len() {
                mask_range *= shape.dims()[level];
            }
            masks.push(loop_in(
                mask,
                mask_range.simplify_cache(simplify_cache),
                mask_stride.simplify_cache(simplify_cache),
                0,
                graph,
            ));
        };
        range *= curr_range;
        stride += if i == shape.len() - 1 {
            curr_stride.substitute('z', Expression::from('z') % curr_range)
        } else {
            curr_stride.substitute('z', (Expression::from('z') / element_size) % curr_range)
        };
    }
    ranges.push((range, "0".to_string()));
    src = loop_in(
        src,
        range.simplify_cache(simplify_cache),
        stride.simplify_cache(simplify_cache),
        0,
        graph,
    );

    if let Some(reduce_dim) = reduce_dim {
        // Go through rest of grid and threadblock as pads
        for i in 1..(GRID_DIMS + THREADBLOCK_DIMS) {
            ranges.push((Expression::from(1), format!("-pad{i}-")));
            src = loop_in(src, 1, 0, format!("-pad{i}-"), graph);
            for mask in &mut masks {
                *mask = loop_in(*mask, 1, 0, format!("-pad{i}-"), graph);
            }
        }
        // Do reduction
        let (left_pad, right_pad) = shape.padding[shape.indexes[reduce_dim]];
        let (left_slice, right_slice) = shape.mask[shape.indexes[reduce_dim]];
        assert!(left_pad == 0 && right_pad == 0 && left_slice == 0 && right_slice == i32::MAX);
        ranges.push((shape.dims()[reduce_dim], reduce_dim.to_string()));
        src = loop_in(
            src,
            shape.dims()[reduce_dim],
            shape.strides()[reduce_dim] * 'z',
            reduce_dim,
            graph,
        );
        for mask in &mut masks {
            *mask = loop_in(
                *mask,
                shape.dims()[reduce_dim],
                0,
                THREADBLOCK_DIMS + GRID_DIMS,
                graph,
            );
        }
    }
    // Go through thread dims
    let mut thread_range = Expression::from(1);
    let mut thread_stride = Expression::from(0);
    let mut thread_dims = false;
    for i in reduce_dim.map(|r| r + 1).unwrap_or(shape.len())..shape.len() {
        thread_dims = true;
        let (curr_range, curr_stride, mask_stride) = smart_loop_in(shape, i);
        assert!(mask_stride.is_none());
        let element_size = shape
            .dims()
            .iter()
            .skip(i + 1)
            .copied()
            .product::<Expression>()
            .max(1);
        thread_range *= curr_range;
        thread_stride += if i == shape.len() - 1 {
            curr_stride.substitute('z', Expression::from('z') % curr_range)
        } else {
            curr_stride.substitute('z', (Expression::from('z') / element_size) % curr_range)
        };
    }
    if thread_dims {
        ranges.push((thread_range, 0.to_string()));
        src = loop_in(
            src,
            thread_range.simplify(),
            thread_stride.simplify(),
            0,
            graph,
        );
        for mask in &mut masks {
            *mask = loop_in(
                *mask,
                thread_range.simplify(),
                0,
                GRID_DIMS + THREADBLOCK_DIMS,
                graph,
            );
        }
    }
    // Multiply masks
    for mask in masks {
        src = crate::utils::binary(src, mask, GraphTerm::Mul, graph);
    }
    (src, ranges)
}

fn scope_out(
    mut src: NodeIndex,
    ranges: Vec<(Expression, String)>,
    reduce: bool,
    graph: &mut StableGraph<GraphTerm, (), Directed>,
) -> NodeIndex {
    for (stride, range, loop_name) in ranges.into_iter().enumerate().rev().scan(
        Expression::from('z'),
        |i, (ind, (range, loop_name))| {
            if reduce && ind == THREADBLOCK_DIMS + GRID_DIMS {
                Some((Expression::from(Term::Acc('a')), range, loop_name)) // THIS IS WRONG, IT SHOULD MIRROR THE INCOMING ACC AND BE RANDOMLY GENERATED
            } else {
                let r = *i;
                *i *= range;
                Some((r, range, loop_name))
            }
        },
    ) {
        src = loop_out(src, range, stride, loop_name, graph);
    }
    src
}
