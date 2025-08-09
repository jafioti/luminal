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

#[derive(Debug)]
pub enum InitData {
    Expr(Expression),
    Data(Vec<f32>),
}

pub type SubGraph = StableGraph<GraphTerm, (), Directed>;
pub type MetaGraph = StableGraph<SubGraph, (NodeIndex, NodeIndex), Directed>;

pub fn translate_graph_meta(
    graph: &Graph,
) -> (
    MetaGraph,
    FxHashMap<NodeIndex, (NodeIndex /*meta*/, NodeIndex /*inner*/)>,
    FxHashMap<NodeIndex /*meta*/, Vec<(NodeIndex /*inner*/, InitData)>>,
) {
    let mut meta: MetaGraph = MetaGraph::new();

    // --- current slice state ---
    let mut g: SubGraph = SubGraph::new();
    let mut node_map: FxHashMap<(NodeIndex, usize), NodeIndex> = FxHashMap::default();
    let mut old_to_new: FxHashMap<NodeIndex, NodeIndex> = FxHashMap::default();
    let mut inits: Vec<(NodeIndex, InitData)> = vec![];
    let mut simplify_cache = FxHashMap::default();

    // meta-level outputs
    let mut global_map: FxHashMap<NodeIndex, (NodeIndex, NodeIndex)> = FxHashMap::default();
    let mut inits_by_meta: FxHashMap<NodeIndex, Vec<(NodeIndex, InitData)>> = FxHashMap::default();

    // For the CURRENT slice being built, remember meta-edge stubs that must be wired
    // when this slice is finalized: (src_meta, src_inner_out, dst_inner_placeholder)
    let mut pending_in_edges: Vec<(NodeIndex, NodeIndex, NodeIndex)> = vec![];

    // finalize current slice into the meta-graph and return its meta node
    let finalize_current_slice =
        |meta: &mut MetaGraph,
         g: &mut SubGraph,
         old_to_new: &mut FxHashMap<NodeIndex, NodeIndex>,
         inits: &mut Vec<(NodeIndex, InitData)>,
         pending_in_edges: &mut Vec<(NodeIndex, NodeIndex, NodeIndex)>,
         global_map: &mut FxHashMap<NodeIndex, (NodeIndex, NodeIndex)>,
         inits_by_meta: &mut FxHashMap<NodeIndex, Vec<(NodeIndex, InitData)>>|
         -> NodeIndex {
            // move inner graph into meta node
            let meta_node = meta.add_node(std::mem::take(g));

            // wire all deferred inbound edges to this meta node
            for (src_meta, src_inner, dst_inner) in pending_in_edges.drain(..) {
                meta.add_edge(src_meta, meta_node, (src_inner, dst_inner));
            }

            // publish inits and global mapping for nodes in the slice we just sealed
            inits_by_meta.insert(meta_node, std::mem::take(inits));
            for (old, inner) in old_to_new.drain() {
                global_map.insert(old, (meta_node, inner));
            }

            // clear per-slice maps (node_map cleared by caller if needed)
            meta_node
        };

    for old_node in toposort(&graph.graph, None).unwrap() {
        let node_weight = graph.node_weight(old_node).unwrap();
        let op_name_full = format!("{node_weight:?}");
        let op = op_name_full
            .split('|')
            .next()
            .unwrap_or(&op_name_full)
            .trim();
        let mut sources = graph.get_sources(old_node);
        match op {
            // ---- GRAPH BREAK ----
            "GraphBreak" => {
                // 1) identify producer's inner node (no scope_in/out needed)
                let (src_old, out_idx, _shape_unused) = sources.pop().unwrap();
                let producer_inner_out = node_map[&(src_old, out_idx as usize)];

                // 2) seal current slice into meta
                let producer_meta = finalize_current_slice(
                    &mut meta,
                    &mut g,
                    &mut old_to_new,
                    &mut inits,
                    &mut pending_in_edges,
                    &mut global_map,
                    &mut inits_by_meta,
                );

                // 3) start fresh slice; create placeholder fed by the producer
                node_map.clear();
                simplify_cache = FxHashMap::default();

                let placeholder = g.add_node(GraphTerm::GMEM {
                    label: format!("break_{}", old_node.index()),
                });

                // defer wiring: (producer_meta, producer_inner_out) -> (this_meta (TBD), placeholder)
                pending_in_edges.push((producer_meta, producer_inner_out, placeholder));

                // map Break node to the placeholder (passthrough)
                node_map.insert((old_node, 0), placeholder);
                old_to_new.insert(old_node, placeholder);
            }

            // ---- UNARY ELEMENTWISE ----
            "Sqrt" | "Exp2" | "Log2" | "Sin" | "Contiguous" | "Recip" => {
                let (s0, i0, shape0) = sources.pop().unwrap();
                let base0 = node_map[&(s0, i0 as usize)];
                let (base0, ranges) =
                    scope_in(base0, shape0, None, &mut g, &mut inits, &mut simplify_cache);

                let mut out = if op == "Contiguous" {
                    base0
                } else {
                    let r = g.add_node(match op {
                        "Sqrt" => GraphTerm::Sqrt,
                        "Exp2" => GraphTerm::Exp2,
                        "Log2" => GraphTerm::Log2,
                        "Sin" => GraphTerm::Sin,
                        "Recip" => GraphTerm::Recip,
                        _ => unreachable!(),
                    });
                    g.add_edge(base0, r, ());
                    r
                };
                out = scope_out(out, ranges, false, &mut g);
                old_to_new.insert(old_node, out);
                node_map.insert((old_node, 0), out);
            }

            // ---- BINARY ELEMENTWISE ----
            "Add" | "Mul" | "Mod" | "LessThan" => {
                let (sa, ia, shape_a) = sources.pop().unwrap();
                let (sb, ib, shape_b) = sources.pop().unwrap();
                let (ain, ranges) = scope_in(
                    node_map[&(sa, ia as usize)],
                    shape_a,
                    None,
                    &mut g,
                    &mut inits,
                    &mut simplify_cache,
                );
                let (bin, _) = scope_in(
                    node_map[&(sb, ib as usize)],
                    shape_b,
                    None,
                    &mut g,
                    &mut inits,
                    &mut simplify_cache,
                );

                let mut opn = g.add_node(match op {
                    "Add" => GraphTerm::Add,
                    "Mul" => GraphTerm::Mul,
                    "Mod" => GraphTerm::Mod,
                    "LessThan" => GraphTerm::LessThan,
                    _ => unreachable!(),
                });
                g.add_edge(ain, opn, ());
                g.add_edge(bin, opn, ());
                opn = scope_out(opn, ranges, false, &mut g);
                old_to_new.insert(old_node, opn);
                node_map.insert((old_node, 0), opn);
            }

            // ---- REDUCTIONS ----
            s if s.starts_with("SumReduce") || s.starts_with("MaxReduce") => {
                let (start_val, term, reduce_dim) = match op {
                    s if s.starts_with("SumReduce") => {
                        (0.0, GraphTerm::Add, graph.get_op::<SumReduce>(old_node).0)
                    }
                    s if s.starts_with("MaxReduce") => (
                        f32::NEG_INFINITY,
                        GraphTerm::Max,
                        graph.get_op::<MaxReduce>(old_node).0,
                    ),
                    _ => unreachable!(),
                };

                let mut acc = g.add_node(GraphTerm::GMEM {
                    label: format!("acc_{}", inits.len()),
                });
                let orig_acc = acc;

                let (source, output_index, shape) = sources.pop().unwrap();
                let (new_source, ranges) = scope_in(
                    node_map[&(source, output_index as usize)],
                    shape,
                    Some(reduce_dim),
                    &mut g,
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
                    let new_acc = g.add_node(GraphTerm::LoopIn {
                        range: *range,
                        stride,
                        marker: name.to_string(),
                    });
                    g.add_edge(acc, new_acc, ());
                    acc = new_acc;
                }
                inits.push((orig_acc, InitData::Data(vec![start_val])));

                let mut opn = g.add_node(term);
                g.add_edge(new_source, opn, ());
                g.add_edge(acc, opn, ());
                opn = scope_out(opn, ranges, true, &mut g);
                old_to_new.insert(old_node, opn);
                node_map.insert((old_node, 0), opn);
            }

            // ---- CONSTANTS / CUSTOM / DIFF / LOADS ----
            _ => {
                if let Some(constant) = node_weight.as_any().downcast_ref::<Constant>() {
                    let newn = g.add_node(GraphTerm::GMEM {
                        label: op.to_string(),
                    });
                    old_to_new.insert(old_node, newn);
                    node_map.insert((old_node, 0), newn);
                    inits.push((
                        newn,
                        match constant.0 {
                            ConstantValue::Expression(e) => InitData::Expr(e),
                            ConstantValue::Float(f) => InitData::Data(vec![f]),
                        },
                    ));
                } else if let Some(kernel) = node_weight.as_any().downcast_ref::<CompatKernel>() {
                    let custom = g.add_node(GraphTerm::Custom(kernel.0.clone()));
                    for (source, ind, _) in sources {
                        g.add_edge(node_map[&(source, ind as usize)], custom, ());
                    }
                    for i in 0..kernel.0.outputs.len() {
                        node_map.insert((old_node, i), custom);
                    }
                    old_to_new.insert(old_node, custom);
                } else if let Some(diff) = node_weight.as_any().downcast_ref::<Diff>() {
                    let custom = g.add_node(GraphTerm::Diff(diff.name.clone()));
                    for (source, ind, _) in sources {
                        g.add_edge(node_map[&(source, ind as usize)], custom, ());
                    }
                    node_map.insert((old_node, 0), custom);
                    old_to_new.insert(old_node, custom);
                } else {
                    // Assume a load
                    let newn = g.add_node(GraphTerm::GMEM {
                        label: op.to_string(),
                    });
                    node_map.insert((old_node, 0), newn);
                    old_to_new.insert(old_node, newn);
                }
            }
        }
    }

    // seal trailing slice if non-empty or if it has pending inbound edges
    if g.node_count() > 0 || !pending_in_edges.is_empty() {
        finalize_current_slice(
            &mut meta,
            &mut g,
            &mut old_to_new,
            &mut inits,
            &mut pending_in_edges,
            &mut global_map,
            &mut inits_by_meta,
        );
    }

    (meta, global_map, inits_by_meta)
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
                label: "Mask".to_string(),
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
            thread_range.simplify_cache(simplify_cache),
            thread_stride.simplify_cache(simplify_cache),
            0,
            graph,
        );
        for mask in &mut masks {
            *mask = loop_in(
                *mask,
                thread_range.simplify_cache(simplify_cache),
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
