use itertools::Itertools;
use petgraph::{
    Directed, Direction, algo::toposort, graph::NodeIndex, prelude::StableGraph, visit::EdgeRef,
};
use std::{
    collections::{BTreeSet, HashMap, HashSet},
    iter::repeat,
};

use crate::{
    GMEMBuffer, GPUArch, GraphTerm, Kernel,
    symbolic::Expression,
    utils::{display_graph, validate_graph},
};

pub fn codegen(
    graph: StableGraph<GraphTerm, u8, Directed>,
    root: NodeIndex,
    mut arch: GPUArch,
) -> Option<StableGraph<Kernel, (u8, u8), Directed>> {
    let (kernels, root_kernel) = split_kernels(graph, root);
    // Create kernel meta graph to toposort
    let mut meta_graph = StableGraph::new();
    for _ in 0..kernels.len() {
        meta_graph.add_node(Kernel::default());
    }
    let global_input = meta_graph.add_node(Kernel {
        code: "Inputs".to_string(),
        ..Default::default()
    });
    let global_output = meta_graph.add_node(Kernel {
        code: "Outputs".to_string(),
        ..Default::default()
    });
    for (n_kernel, (_, inputs, _, _)) in kernels.iter().enumerate() {
        for (n_input, (input_kernel, _)) in inputs.into_iter().enumerate() {
            match input_kernel {
                GMEMBuffer::PrevKernel { kernel, output } => meta_graph.add_edge(
                    NodeIndex::new(*kernel),
                    NodeIndex::new(n_kernel),
                    (*output as u8, n_input as u8),
                ),
                GMEMBuffer::Input(g_inp) => meta_graph.add_edge(
                    global_input,
                    NodeIndex::new(n_kernel),
                    (*g_inp as u8, n_input as u8),
                ),
            };
        }
    }
    meta_graph.add_edge(NodeIndex::new(root_kernel), global_output, (0, 0));
    for node in toposort(&meta_graph, None).unwrap() {
        if kernels.len() <= node.index() {
            continue; // Either input node or output node
        }
        let (kernel_graph, inputs, outputs, smem_buffers) = kernels[node.index()].clone();
        validate_graph(&kernel_graph);
        // display_graph(&kernel_graph, &[]);
        let mut node_to_var = inputs
            .iter()
            .map(|(_, n)| *n)
            .chain(outputs.iter().map(|(_, i)| *i))
            .chain(smem_buffers.iter().map(|(_, i, _)| *i))
            .enumerate()
            .map(|(v, n)| (n, (v, true, None)))
            .collect::<HashMap<_, _>>();
        for (_, (n, _, _)) in &node_to_var {
            arch.add_metal_buffer_type(*n, "device ");
        }
        for (n, _, _) in &smem_buffers {
            arch.add_metal_buffer_type(*n, "threadgroup ");
        }
        let mut loop_levels = vec![];
        let kernel = make_kernel(
            &kernel_graph,
            kernel_graph.node_indices().collect(),
            &mut node_to_var,
            &mut (inputs.len() + outputs.len()),
            &mut loop_levels,
            &mut HashMap::new(),
            &smem_buffers
                .iter()
                .map(|(ind, node, _)| (*node, *ind))
                .collect(),
            0,
            &mut arch,
        )?;
        let grid = loop_levels
            .clone()
            .into_iter()
            .chain(repeat(1.into()))
            .take(3)
            .collect_vec();
        let threadblock = loop_levels
            .into_iter()
            .skip(3)
            .chain(repeat(1.into()))
            .take(3)
            .collect_vec();
        let kernel_lines = kernel.into_iter().map(|s| format!("\t{s}")).join("\n");
        let kernel = match &arch {
            GPUArch::CUDA => {
                let inputs = inputs
                    .into_iter()
                    .map(|(_, a)| a)
                    .chain(outputs.iter().map(|(_, i)| *i))
                    .map(|a| format!("float* {}", var_to_char(node_to_var[&a].0)))
                    .join(", ");
                let smem_setup = if smem_buffers.is_empty() {
                    "".to_string()
                } else {
                    format!(
                        "\textern __shared__ float sm[];\n{}",
                        smem_buffers
                            .iter()
                            .scan("".to_string(), |prev_buffers, (n, _, size)| {
                                let r =
                                    format!("\tfloat* {} = sm{prev_buffers};\n", var_to_char(*n));
                                prev_buffers.push_str(&format!(" + {size}"));
                                Some(r)
                            })
                            .join("")
                    )
                };
                format!(
                    "extern \"C\" __global__ void kernel{}({inputs}) {{
{smem_setup}{kernel_lines}
}}",
                    node.index()
                )
            }
            GPUArch::Metal { .. } => {
                let inputs = inputs
                    .into_iter()
                    .map(|(_, a)| a)
                    .chain(outputs.iter().map(|(_, i)| *i))
                    .enumerate()
                    .map(|(i, a)| {
                        format!(
                            "device float* {} [[buffer({i})]]",
                            var_to_char(node_to_var[&a].0)
                        )
                    })
                    .join(",\n\t");
                let (smem_setup, smem_input) = if smem_buffers.is_empty() {
                    ("".to_string(), "".to_string())
                } else {
                    (
                        smem_buffers
                            .iter()
                            .scan("".to_string(), |prev_buffers, (n, _, size)| {
                                let r = format!(
                                    "\tthreadgroup float* {} = sm{prev_buffers};\n",
                                    var_to_char(*n)
                                );
                                prev_buffers.push_str(&format!(" + {size}"));
                                Some(r)
                            })
                            .join(""),
                        ", threadgroup float* sm [[threadgroup(0)]]".to_string(),
                    )
                };
                format!(
                    "#include <metal_stdlib>
using namespace metal;
kernel void kernel{}(
	uint3 blockIdx [[threadgroup_position_in_grid]],
	uint3 threadIdx [[thread_position_in_threadgroup]],
	{inputs}{smem_input}
) {{
{smem_setup}{kernel_lines}
}}",
                    node.index(),
                )
            }
        };
        *meta_graph.node_weight_mut(node).unwrap() = Kernel {
            code: kernel,
            grid: (grid[0], grid[1], grid[2]),
            threadblock: (threadblock[0], threadblock[1], threadblock[2]),
            smem: smem_buffers.into_iter().map(|(_, _, a)| a).sum(),
            outputs: outputs.into_iter().map(|(o, _)| o).collect(),
        };
    }
    Some(meta_graph)
}

fn var_to_char(var: usize) -> String {
    if var > 25 {
        format!("{}{}", var_to_char(var / 26), var_to_char(var % 26))
    } else {
        ((var + 97) as u8 as char).to_string()
    }
}

fn make_kernel(
    kernel_graph: &StableGraph<(GraphTerm, usize), u8, Directed>,
    include_nodes: HashSet<NodeIndex>,
    node_to_var: &mut HashMap<NodeIndex, (usize, bool, Option<Expression>)>,
    prev_max_var: &mut usize, // contains the char last used
    loop_levels: &mut Vec<Expression>,
    loop_indexes: &mut HashMap<NodeIndex, usize>,
    smem_buffers: &HashMap<NodeIndex, usize>,
    current_loop_level: usize,
    arch: &mut GPUArch,
) -> Option<Vec<String>> {
    let mut kernel_lines = vec![];
    let spacing = (0..current_loop_level.saturating_sub(6))
        .map(|_| "\t")
        .join("");

    // Walk through the toposorted nodes
    for node in toposort_subset(kernel_graph, &include_nodes) {
        if node_to_var.contains_key(&node)
            || kernel_graph.node_weight(node).unwrap().1 != current_loop_level
        {
            continue;
        }
        let (term, loop_level) = kernel_graph.node_weight(node).unwrap();
        match term {
            GraphTerm::LoopIn { range, stride } => {
                // go through graph to find inputs and outputs
                let mut loop_inputs = HashSet::new();
                loop_inputs.insert((node, stride));
                let mut loop_outputs = HashSet::new();
                let mut dfs = kernel_graph
                    .neighbors_directed(node, Direction::Outgoing)
                    .collect_vec();
                let mut body_nodes = HashSet::new();
                body_nodes.insert(node);
                while let Some(n) = dfs.pop() {
                    if body_nodes.contains(&n) {
                        continue;
                    }
                    body_nodes.insert(n);
                    if let (GraphTerm::LoopOut { stride, .. }, level) =
                        &kernel_graph.node_weight(n).unwrap()
                    {
                        if *level == *loop_level {
                            loop_outputs.insert((n, stride));
                            continue;
                        }
                    } else if let (GraphTerm::LoopIn { stride, .. }, level) =
                        &kernel_graph.node_weight(n).unwrap()
                    {
                        if *level == *loop_level {
                            loop_inputs.insert((n, stride));
                            continue;
                        }
                    }
                    body_nodes.insert(node);
                    dfs.extend(kernel_graph.neighbors_undirected(n));
                }

                if let Some((_, _)) = loop_inputs.iter().find(|(i, _)| {
                    kernel_graph
                        .neighbors_directed(*i, Direction::Incoming)
                        .next()
                        .map(|n| !node_to_var.contains_key(&n))
                        .unwrap_or_default()
                }) {
                    // Not all inputs to this loop are ready
                    continue;
                }
                for (i, _) in loop_inputs.iter().chain(&loop_outputs) {
                    body_nodes.remove(i);
                }
                // display_graph(
                //     &kernel_graph,
                //     &body_nodes
                //         .iter()
                //         .copied()
                //         .map(|i| (i, "yellow".to_string()))
                //         .chain(loop_inputs.iter().map(|(i, _)| (*i, "green".to_string())))
                //         .chain(loop_outputs.iter().map(|(i, _)| (*i, "red".to_string())))
                //         .collect_vec(),
                // );

                let inner_spacing = if current_loop_level < 6 {
                    "".to_string()
                } else {
                    format!("{spacing}\t")
                };

                // Make new loop
                if *loop_level < 6 {
                    if current_loop_level >= loop_levels.len() {
                        loop_levels.push(*range);
                    }
                    if loop_inputs
                        .iter()
                        .chain(&loop_outputs)
                        .any(|(_, st)| **st != 0)
                    {
                        *prev_max_var += 1;
                        kernel_lines.push(format!(
                            "int loop_{} = {};",
                            var_to_char(*prev_max_var),
                            [
                                "blockIdx.x",
                                "blockIdx.y",
                                "blockIdx.z",
                                "threadIdx.x",
                                "threadIdx.y",
                                "threadIdx.z"
                            ][*loop_level]
                        ));
                    }
                } else {
                    *prev_max_var += 1;
                    let loop_var = var_to_char(*prev_max_var);
                    kernel_lines.push(format!("{spacing}for (int loop_{loop_var} = 0; loop_{loop_var} < {range}; loop_{loop_var} += 1) {{"));
                };
                let loop_var = var_to_char(*prev_max_var);
                let loop_var_int = *prev_max_var;

                // Move input pointers (allocate new variables)
                let mut new_vars = vec![];
                for (input, stride) in &loop_inputs {
                    let src = kernel_graph
                        .neighbors_directed(*input, Direction::Incoming)
                        .next()
                        .unwrap();
                    let (real_input, is_ptr, real_size) = node_to_var[&src].clone();
                    if stride.is_acc() {
                        assert!(
                            *loop_level > 5,
                            "No accumulations allowed on grid or threadblock levels!"
                        );
                        new_vars.push(real_input);
                        node_to_var.insert(*input, (real_input, is_ptr, real_size));
                    } else {
                        if **stride != 0 && *range != 1 {
                            if !is_ptr {
                                return None; // We can't handle thread-level buffers yet!
                            }
                            *prev_max_var += 1;
                            arch.add_metal_buffer_type(
                                *prev_max_var,
                                arch.metal_buffer_type(real_input),
                            );
                            kernel_lines.push(format!(
                                "{inner_spacing}{}float* {} = {} + {};",
                                arch.metal_buffer_type(*prev_max_var),
                                var_to_char(*prev_max_var),
                                var_to_char(real_input),
                                stride.to_string().replace('z', &format!("loop_{loop_var}"))
                            ));
                            new_vars.push(*prev_max_var);
                            node_to_var.insert(*input, (*prev_max_var, is_ptr, None));
                        } else {
                            new_vars.push(real_input);
                            node_to_var.insert(*input, (real_input, is_ptr, real_size));
                        }
                    }
                }
                // Move output pointers (allocate new variables)
                let mut new_output_vars = vec![];
                for (output, stride) in &loop_outputs {
                    if stride.is_acc() {
                        assert!(
                            *loop_level > 5,
                            "No accumulations allowed on grid or threadblock levels!"
                        );
                        // Re-use accumulator
                        let (input, _) = loop_inputs
                            .iter()
                            .find(|(_, s)| **s == **stride)
                            .unwrap_or_else(|| {
                                display_graph(
                                    &kernel_graph,
                                    &loop_inputs
                                        .iter()
                                        .map(|(i, _)| (*i, "yellow".to_string()))
                                        .collect_vec(),
                                );
                                panic!("Cannot find accumulator {stride}");
                            });
                        let (input_var, is_ptr, input_size) = node_to_var[input].clone();
                        new_output_vars.push(input_var);
                        node_to_var.insert(*output, (input_var, is_ptr, input_size));
                        continue;
                    }
                    let dest = kernel_graph
                        .neighbors_directed(*output, Direction::Outgoing)
                        .next()
                        .unwrap();
                    if let Some((real_output, is_ptr, real_size)) = node_to_var.get(&dest).copied()
                    {
                        if **stride != 0 && *range != 1 {
                            assert!(is_ptr, "Only pointers can be offset!");
                            *prev_max_var += 1;
                            arch.add_metal_buffer_type(
                                *prev_max_var,
                                arch.metal_buffer_type(real_output),
                            );
                            kernel_lines.push(format!(
                                "{inner_spacing}{}float* {} = {} + {};",
                                arch.metal_buffer_type(*prev_max_var),
                                var_to_char(*prev_max_var),
                                var_to_char(real_output),
                                stride.to_string().replace('z', &format!("loop_{loop_var}"))
                            ));
                            new_output_vars.push(*prev_max_var);
                            node_to_var.insert(*output, (*prev_max_var, is_ptr, None));
                        } else {
                            new_output_vars.push(real_output);
                            node_to_var.insert(*output, (real_output, is_ptr, real_size));
                        }
                    } else {
                        // Handle the case where the dest is not the real loop output
                    }
                }
                for (inp, _) in &loop_inputs {
                    loop_indexes.insert(*inp, loop_var_int);
                }
                let loop_body = make_kernel(
                    kernel_graph,
                    body_nodes,
                    node_to_var,
                    prev_max_var,
                    loop_levels,
                    loop_indexes,
                    smem_buffers,
                    current_loop_level + 1,
                    arch,
                )?;

                if loop_body.is_empty() && loop_inputs.len() == 1 && loop_outputs.len() == 1 {
                    // Need to copy inputs to outputs
                    let (src, src_ptr, _) = node_to_var[&loop_inputs.iter().next().unwrap().0];
                    let (dest, _, _) = node_to_var[&loop_outputs.iter().next().unwrap().0];
                    kernel_lines.push(format!(
                        "{inner_spacing}*{} = {}{};",
                        var_to_char(dest),
                        if src_ptr { "*" } else { "" },
                        var_to_char(src)
                    ));
                }
                kernel_lines.extend(loop_body);

                // Set outputs if nessecary
                for (output, _) in loop_outputs {
                    let body_out = kernel_graph
                        .neighbors_directed(output, Direction::Incoming)
                        .next()
                        .unwrap();
                    let is_acc = if let GraphTerm::LoopOut { stride, .. } =
                        &kernel_graph.node_weight(body_out).unwrap().0
                    {
                        stride.is_acc()
                    } else {
                        false
                    };
                    let (body_out, body_out_ptr, body_out_size) = node_to_var[&body_out];
                    if let Some((output, output_ptr, _)) = node_to_var.get(&output).copied() {
                        if output != body_out && (!body_out_ptr || is_acc) {
                            if let Some(size) = &body_out_size {
                                if *size == 1 {
                                    kernel_lines.push(format!(
                                        "{inner_spacing}{}{} = {}{};",
                                        if output_ptr { "*" } else { "" },
                                        var_to_char(output),
                                        if body_out_ptr { "*" } else { "" },
                                        var_to_char(body_out),
                                    ));
                                } else {
                                    // Save size numbers
                                    kernel_lines.push(format!(
                                    "{inner_spacing}for (int save = 0; save < {size}; save++) {{",
                                ));
                                    assert!(
                                        output_ptr && body_out_ptr,
                                        "Both src and dest must be pointers when saving a block"
                                    );
                                    kernel_lines.push(format!(
                                        "{inner_spacing}\t{}[save] = {}[save];",
                                        var_to_char(output),
                                        var_to_char(body_out),
                                    ));
                                    kernel_lines.push(format!("{inner_spacing}}}"));
                                }
                            } else {
                                kernel_lines.push(format!(
                                    "{inner_spacing}{}{} = {}{};",
                                    if output_ptr { "*" } else { "" },
                                    var_to_char(output),
                                    if body_out_ptr { "*" } else { "" },
                                    var_to_char(body_out),
                                ));
                            }
                        }
                    } else {
                        assert!(body_out_size.is_none());
                        node_to_var.insert(output, (body_out, false, None));
                    }
                }

                if *loop_level >= 6 {
                    kernel_lines.push(format!("{spacing}}}"));
                }
            }
            GraphTerm::NewAcc { starting_value } => {
                // Go through all loopins after this to figure out how many accumulators we need
                let mut curr = node;
                let mut size = Expression::from(1);
                loop {
                    curr = kernel_graph
                        .neighbors_directed(curr, Direction::Outgoing)
                        .next()
                        .unwrap();
                    if let GraphTerm::LoopIn { range, stride } =
                        &kernel_graph.node_weight(curr).unwrap().0
                    {
                        if !stride.is_acc() && *stride != 0 {
                            size *= range;
                        }
                    } else {
                        break;
                    }
                }
                // Create accumulator
                *prev_max_var += 1;
                arch.add_metal_buffer_type(*prev_max_var, "thread ");
                kernel_lines.push(format!(
                    "{spacing}{}float {}[{size}] = {{{starting_value}}};",
                    arch.metal_buffer_type(*prev_max_var),
                    var_to_char(*prev_max_var)
                ));
                node_to_var.insert(node, (*prev_max_var, true, Some(size)));
            }
            GraphTerm::LoopOut { range, stride } => {
                panic!("found loopout range: {range} stride: {stride}")
            }
            GraphTerm::SMEMLoad | GraphTerm::SMEMRead => {
                // Find the gmem input and smem input
                let inputs = kernel_graph
                    .neighbors_directed(node, Direction::Incoming)
                    .collect_vec();
                assert_eq!(inputs.len(), 2);
                let search_for_smem = |mut node| {
                    loop {
                        let Some(prev) = kernel_graph
                            .neighbors_directed(node, Direction::Incoming)
                            .next()
                        else {
                            return false;
                        };
                        match kernel_graph.node_weight(prev).unwrap().0 {
                            GraphTerm::LoopIn { .. } => node = prev,
                            GraphTerm::SMEM => return true,
                            _ => return false,
                        };
                    }
                };
                let (smem, gmem) = if search_for_smem(inputs[0]) {
                    (inputs[0], inputs[1])
                } else {
                    assert!(search_for_smem(inputs[1])); // ensure the other input is an smem ptr
                    (inputs[1], inputs[0])
                };
                let (gmem, gmem_ptr, _) = node_to_var[&gmem];
                let (smem, smem_ptr, _) = node_to_var[&smem];
                assert!(smem_ptr);
                let sync_barrier = match arch {
                    GPUArch::CUDA => "__syncthreads()",
                    GPUArch::Metal(_) => "threadgroup_barrier(mem_flags::mem_threadgroup)",
                };
                match term {
                    GraphTerm::SMEMLoad => {
                        kernel_lines.push(format!("{spacing}{sync_barrier};"));
                        kernel_lines.push(format!(
                            "{spacing}*{} = {}{};",
                            var_to_char(smem),
                            if gmem_ptr { "*" } else { "" },
                            var_to_char(gmem),
                        ));
                        node_to_var.insert(node, (smem, true, None));
                    }
                    GraphTerm::SMEMRead => {
                        // gmem ptr isn't actually gmem, it should be pointing to the smem copy
                        kernel_lines.push(format!("{spacing}{sync_barrier};"));
                        node_to_var.insert(node, (smem, true, None));
                    }
                    _ => panic!(),
                }
            }
            GraphTerm::GMEM { .. } => {}
            GraphTerm::SMEM => {}
            GraphTerm::Sin | GraphTerm::Exp | GraphTerm::Neg | GraphTerm::Recip => {
                *prev_max_var += 1;
                let (src, src_ptr, _) = node_to_var[&kernel_graph
                    .neighbors_directed(node, Direction::Incoming)
                    .next()
                    .unwrap()];
                node_to_var.insert(node, (*prev_max_var, false, None));
                let inp = format!("{}{}", if src_ptr { "*" } else { "" }, var_to_char(src));
                let expr = match term {
                    GraphTerm::Sin => format!("sin({inp})"),
                    GraphTerm::Exp => format!("exp({inp})"),
                    GraphTerm::Neg => format!("-{inp}"),
                    GraphTerm::Recip => format!("1.0 / {inp}"),
                    _ => panic!(),
                };
                kernel_lines.push(format!(
                    "{spacing}float {} = {expr};",
                    var_to_char(*prev_max_var),
                ));
            }
            GraphTerm::Mul | GraphTerm::Add | GraphTerm::Max => {
                *prev_max_var += 1;
                let mut srcs = kernel_graph.neighbors_directed(node, Direction::Incoming);
                let (src_a, src_a_ptr, _) = node_to_var[&srcs.next().unwrap()];
                let (src_b, src_b_ptr, _) = node_to_var[&srcs.next().unwrap()];
                node_to_var.insert(node, (*prev_max_var, false, None));
                let inp_a = format!("{}{}", if src_a_ptr { "*" } else { "" }, var_to_char(src_a));
                let inp_b = format!("{}{}", if src_b_ptr { "*" } else { "" }, var_to_char(src_b));
                let expr = match &term {
                    GraphTerm::Add => format!("{inp_a} + {inp_b}"),
                    GraphTerm::Mul => format!("{inp_a} * {inp_b}"),
                    GraphTerm::Max => format!(
                        "{}({inp_a}, {inp_b})",
                        if matches!(arch, GPUArch::Metal(_)) {
                            "fmax"
                        } else {
                            "__max"
                        }
                    ),
                    _ => panic!(),
                };
                kernel_lines.push(format!(
                    "{spacing}float {} = {expr};",
                    var_to_char(*prev_max_var)
                ));
            }
        }
    }
    Some(kernel_lines)
}

/// Toposort a subset of a graph
fn toposort_subset<N, E>(
    graph: &StableGraph<N, E, Directed>,
    subset: &HashSet<NodeIndex>,
) -> Vec<NodeIndex> {
    // in-degree restricted to `subset`
    let mut indeg: HashMap<NodeIndex, usize> = subset.iter().map(|&n| (n, 0)).collect();
    for &n in subset {
        for succ in graph.neighbors_directed(n, Direction::Outgoing) {
            if subset.contains(&succ) {
                *indeg.get_mut(&succ).unwrap() += 1;
            }
        }
    }

    // ready set: smallest index first for deterministic order
    let mut ready: BTreeSet<NodeIndex> = indeg
        .iter()
        .filter_map(|(&n, &d)| (d == 0).then_some(n))
        .collect();

    let mut order = Vec::with_capacity(subset.len());
    while let Some(&n) = ready.iter().next() {
        ready.remove(&n);
        order.push(n);

        for succ in graph.neighbors_directed(n, Direction::Outgoing) {
            if let Some(d) = indeg.get_mut(&succ) {
                *d -= 1;
                if *d == 0 {
                    ready.insert(succ);
                }
            }
        }
    }
    order
}

/// add kernel dimensions so that all loop-to-loop dependencies are between seperate kernels or on the threadblock / thread levels
fn split_kernels(
    graph: StableGraph<GraphTerm, u8, Directed>,
    mut root: NodeIndex,
) -> (
    Vec<(
        StableGraph<(GraphTerm, usize), u8, Directed>,
        Vec<(GMEMBuffer, NodeIndex)>, // (src buffer, current graph node)
        Vec<(Expression, NodeIndex)>, // output node
        Vec<(usize, NodeIndex, Expression)>, // (shared memory buffer name, node, buffer size)
    )>,
    usize, // Root kernel
) {
    // Mark level of ops in graph
    let mut marked_graph = StableGraph::new();
    let mut map = HashMap::<NodeIndex, NodeIndex>::default();
    // Add nodes
    for node in graph.node_indices() {
        let new_node = marked_graph.add_node((
            graph.node_weight(node).unwrap().clone(),
            vec![],
            if node == root { vec![0] } else { vec![] },
        ));
        map.insert(node, new_node);
    }
    // Add edges
    for edge in graph.edge_indices() {
        let (start, end) = graph.edge_endpoints(edge).unwrap();
        marked_graph.add_edge(map[&start], map[&end], *graph.edge_weight(edge).unwrap());
    }
    root = map[&root];

    let mut dfs = marked_graph.neighbors_undirected(root).collect_vec();
    let mut seen = HashSet::new();
    seen.insert(root);
    while let Some(n) = dfs.pop() {
        if seen.contains(&n) {
            continue;
        }
        seen.insert(n);
        let curr_term = marked_graph.node_weight(n).unwrap().0.clone();
        if let Some(outgoing_neighbor) = marked_graph
            .neighbors_directed(n, Direction::Outgoing)
            .find(|n| seen.contains(n))
        {
            // Base level off outgoing neighbor
            let (neighbor_weight, mut neighbor_levels, _) =
                marked_graph.node_weight(outgoing_neighbor).unwrap().clone();
            if let GraphTerm::LoopOut { range, .. } = neighbor_weight {
                neighbor_levels.push(range);
            }
            if matches!(curr_term, GraphTerm::LoopIn { .. }) {
                neighbor_levels.pop().unwrap();
            }
            marked_graph.node_weight_mut(n).unwrap().1 = neighbor_levels;
        } else if let Some(incoming_neighbor) = marked_graph
            .neighbors_directed(n, Direction::Incoming)
            .find(|n| seen.contains(n))
        {
            // Base level off incoming neighbor
            let (neighbor_weight, mut neighbor_levels, _) =
                marked_graph.node_weight(incoming_neighbor).unwrap().clone();
            if let GraphTerm::LoopIn { range, .. } = neighbor_weight {
                neighbor_levels.push(range);
            }
            if matches!(curr_term, GraphTerm::LoopOut { .. }) {
                neighbor_levels.pop().unwrap();
            }
            marked_graph.node_weight_mut(n).unwrap().1 = neighbor_levels;
        } else {
            display_graph(&marked_graph, &[(n, "yellow".to_string())]);
            panic!("No seen neighbors when building loop levels!");
        }
        dfs.extend(marked_graph.neighbors_undirected(n));
    }

    // Assign kernel numbers
    let mut dfs_stack = vec![root];
    let mut n_kernels = 1;
    while let Some(node) = dfs_stack.pop() {
        let (term, curr_level, curr_kernel) = marked_graph.node_weight(node).unwrap().clone();
        for source in marked_graph
            .edges_directed(node, Direction::Incoming)
            .map(|e| e.source())
            .collect_vec()
        {
            let (src_term, src_level, src_kernel) = marked_graph.node_weight_mut(source).unwrap();
            if matches!(term, GraphTerm::LoopIn { .. })
                && matches!(src_term, GraphTerm::LoopOut { .. })
                && curr_level.len() < 6
            {
                let max_kernel = *curr_kernel.iter().max().unwrap();
                n_kernels = n_kernels.max(max_kernel + 2);
                *src_kernel = vec![max_kernel + 1];
            } else {
                for k in &curr_kernel {
                    if !src_kernel.contains(k) {
                        src_kernel.push(*k);
                    }
                }
            }
            if !matches!(term, GraphTerm::LoopIn { .. } | GraphTerm::LoopOut { .. })
                && !matches!(
                    src_term,
                    GraphTerm::LoopIn { .. } | GraphTerm::LoopOut { .. }
                )
            {
                assert_eq!(curr_level, *src_level); // Edges need to go between the same levels if neither op is a LoopIn or LoopOut
            }
            dfs_stack.push(source);
        }
    }
    // Run forward from all inputs to catch any nodes we didn't hit already
    dfs_stack = marked_graph
        .node_indices()
        .filter(|n| {
            marked_graph
                .neighbors_directed(*n, Direction::Incoming)
                .next()
                .is_none()
        })
        .collect_vec();
    while let Some(node) = dfs_stack.pop() {
        dfs_stack.extend(marked_graph.neighbors_directed(node, Direction::Outgoing));
        let (term, curr_level, mut curr_kernel) = marked_graph.node_weight(node).unwrap().clone();
        if !curr_kernel.is_empty() {
            continue;
        }
        for source in marked_graph
            .edges_directed(node, Direction::Incoming)
            .map(|e| e.source())
            .collect_vec()
        {
            let (src_term, src_level, src_kernel) = marked_graph.node_weight(source).unwrap();
            if matches!(term, GraphTerm::LoopIn { .. })
                && matches!(src_term, GraphTerm::LoopOut { .. })
                && curr_level.len() < 6
            {
                let max_kernel = *src_kernel.iter().max().unwrap();
                n_kernels = n_kernels.max(max_kernel + 2);
                curr_kernel = vec![max_kernel + 1];
            } else {
                for k in src_kernel {
                    if !curr_kernel.contains(k) {
                        curr_kernel.push(*k);
                    }
                }
            }
            if !matches!(term, GraphTerm::LoopIn { .. } | GraphTerm::LoopOut { .. })
                && !matches!(
                    src_term,
                    GraphTerm::LoopIn { .. } | GraphTerm::LoopOut { .. }
                )
            {
                assert_eq!(curr_level, *src_level); // Edges need to go between the same levels if neither op is a LoopIn or LoopOut
            }
        }
        marked_graph.node_weight_mut(node).unwrap().2 = curr_kernel;
    }
    // Prune out unnessecary kernel members
    let mut dfs_stack = marked_graph
        .neighbors_directed(root, Direction::Incoming)
        .collect_vec();
    while let Some(node) = dfs_stack.pop() {
        dfs_stack.extend(marked_graph.neighbors_directed(node, Direction::Incoming));
        let (term, curr_level, mut curr_kernel) = marked_graph.node_weight(node).unwrap().clone();
        let split_cond = curr_level.len() < 6 && matches!(term, GraphTerm::LoopOut { .. });
        curr_kernel.retain(|k| {
            marked_graph
                .neighbors_directed(node, Direction::Outgoing)
                .any(|n| {
                    (split_cond
                        && matches!(
                            marked_graph.node_weight(n).unwrap().0,
                            GraphTerm::LoopIn { .. }
                        ))
                        || marked_graph.node_weight(n).unwrap().2.contains(k)
                })
        });
        marked_graph.node_weight_mut(node).unwrap().2 = curr_kernel;
    }

    // Add kernel barriers
    for edge in marked_graph.edge_indices().collect_vec() {
        let (mut src, mut dest) = marked_graph.edge_endpoints(edge).unwrap();
        let (dest_term, dest_level, dest_kernel) = marked_graph.node_weight(dest).unwrap().clone();
        let (src_term, _, src_kernel) = marked_graph.node_weight(src).unwrap().clone();
        if dest_level.len() > 0 && dest_kernel.iter().any(|i| !src_kernel.contains(i)) {
            // Put a barrier here
            let GraphTerm::LoopOut {
                range: src_range,
                stride: src_stride,
            } = src_term
            else {
                panic!("{:?}", src_term);
            };
            let GraphTerm::LoopIn {
                range: dest_range,
                stride: dest_stride,
            } = dest_term
            else {
                panic!("{:?}", dest_term);
            };
            marked_graph.remove_edge(edge);
            for i in (0..dest_level.len()).rev() {
                let new_src = marked_graph.add_node((
                    GraphTerm::LoopOut {
                        range: dest_level[i].clone(),
                        stride: src_stride * src_range,
                    },
                    dest_level[..i].to_vec(),
                    src_kernel.clone(),
                ));
                marked_graph.add_edge(src, new_src, 0);
                src = new_src;
                let new_dest = marked_graph.add_node((
                    GraphTerm::LoopIn {
                        range: dest_level[i].clone(),
                        stride: dest_stride * dest_range,
                    },
                    dest_level[..i].to_vec(),
                    dest_kernel.clone(),
                ));
                marked_graph.add_edge(new_dest, dest, 0);
                dest = new_dest;
            }
            marked_graph.add_edge(src, dest, 0);
        }
    }

    // Place nodes in kernel graphs
    let mut kernel_graphs = (0..n_kernels)
        .map(|_| (StableGraph::new(), vec![], vec![], vec![]))
        .collect_vec();
    let mut node_maps = (0..n_kernels).map(|_| HashMap::new()).collect_vec();
    for node in marked_graph.node_indices() {
        let (term, loop_level, kernels) = marked_graph.node_weight(node).unwrap();
        for kernel in kernels {
            let (kernel_graph, _, outputs, _) = &mut kernel_graphs[*kernel];
            node_maps[*kernel].insert(
                node,
                kernel_graph.add_node((term.clone(), loop_level.len())),
            );
            if node == root {
                outputs.push((Expression::default(), node_maps[*kernel][&node]));
            }
        }
    }
    // Go through graph inputs and add them to the kernel hashmap as inputs
    for (n, input) in marked_graph
        .node_indices()
        // Must not have any input nodes
        .filter(|n| {
            marked_graph
                .neighbors_directed(*n, Direction::Incoming)
                .next()
                .is_none()
        })
        // Must not be a NewAcc
        .filter(|n| {
            !matches!(
                marked_graph.node_weight(*n).unwrap().0,
                GraphTerm::NewAcc { .. } | GraphTerm::SMEM
            )
        })
        .enumerate()
    {
        let (_, _, kernels) = marked_graph.node_weight(input).unwrap();
        for kernel in kernels {
            kernel_graphs[*kernel]
                .1
                .push((GMEMBuffer::Input(n), node_maps[*kernel][&input]));
        }
    }
    // Mark cross kernel dependencies in the kernel inputs / outputs
    for edge in marked_graph.edge_indices() {
        let (start, end) = marked_graph.edge_endpoints(edge).unwrap();
        let weight = marked_graph.edge_weight(edge).unwrap();
        let (_, _, start_kernels) = marked_graph.node_weight(start).unwrap();
        let (_, _, end_kernels) = marked_graph.node_weight(end).unwrap();
        for end_kernel in end_kernels {
            if !start_kernels.contains(end_kernel) {
                // start kernel must be outputted
                let start_kernel = start_kernels.first().unwrap();
                // Start and end are in different kernels
                let src_kernel_node = node_maps[*start_kernel][&start];
                let dest_kernel_node = node_maps[*end_kernel][&end];

                // make sure we're outputting the src node from the src graph
                let src_output_index = if let Some(p) = kernel_graphs[*start_kernel]
                    .2
                    .iter()
                    .position(|(_, s)| *s == src_kernel_node)
                {
                    p
                } else {
                    kernel_graphs[*start_kernel]
                        .2
                        .push((Expression::default(), src_kernel_node));
                    kernel_graphs[*start_kernel].2.len() - 1
                };
                // add input to dest
                kernel_graphs[*end_kernel].1.push((
                    GMEMBuffer::PrevKernel {
                        kernel: *start_kernel,
                        output: src_output_index,
                    },
                    dest_kernel_node,
                ));
            } else {
                // end kernel is same as start kernel
                kernel_graphs[*end_kernel].0.add_edge(
                    node_maps[*end_kernel][&start],
                    node_maps[*end_kernel][&end],
                    *weight,
                );
            }
        }
    }

    // Go through SMEM buffers
    for (kernel_graph, inputs, outputs, smem_buffers) in &mut kernel_graphs {
        // Get all ZeroStrideLoad ones at thread level 0..3
        for node in kernel_graph.node_indices() {
            if let (GraphTerm::SMEM, _) = kernel_graph.node_weight(node).unwrap() {
                // Walk forward until load to find the size
                for mut curr in kernel_graph.neighbors_directed(node, Direction::Outgoing) {
                    let mut size = Expression::from(1);
                    let mut load = false;
                    loop {
                        match kernel_graph.node_weight(curr).unwrap().0 {
                            GraphTerm::LoopIn { range, stride } => {
                                if stride != 0 {
                                    size *= range;
                                }
                                curr = kernel_graph
                                    .neighbors_directed(curr, Direction::Outgoing)
                                    .next()
                                    .unwrap();
                            }
                            GraphTerm::SMEMLoad => {
                                load = true;
                                break;
                            }
                            _ => break,
                        }
                    }
                    if load {
                        let buf_index = inputs.len() + outputs.len() + smem_buffers.len();
                        smem_buffers.push((buf_index, node, size));
                        break;
                    }
                }
            }
        }

        // Ensure GMEM is placed on the graph
        for (_, input) in inputs {
            if !matches!(
                kernel_graph.node_weight(*input).unwrap().0,
                GraphTerm::GMEM { .. }
            ) {
                let new_input = kernel_graph.add_node((GraphTerm::GMEM { label: None }, 0));
                kernel_graph.add_edge(new_input, *input, 0);
                *input = new_input;
            }
        }
        for (size, output) in outputs {
            if !matches!(
                kernel_graph.node_weight(*output).unwrap().0,
                GraphTerm::GMEM { .. }
            ) {
                let new_output = kernel_graph.add_node((GraphTerm::GMEM { label: None }, 0));
                kernel_graph.add_edge(*output, new_output, 0);
                *output = new_output;
            }
            // Loop back through all loopouts to find the size of the output
            let mut curr = *output;
            let mut new_size = vec![Expression::from(1)];
            loop {
                let term = &kernel_graph.node_weight(curr).unwrap().0;
                if let GraphTerm::LoopOut { range, stride } = term {
                    if !stride.is_acc() && *stride != 0 {
                        new_size.push(*range);
                    }
                } else if !matches!(term, GraphTerm::GMEM { .. }) {
                    break;
                }
                curr = kernel_graph
                    .neighbors_directed(curr, Direction::Incoming)
                    .next()
                    .unwrap();
            }
            *size = new_size.into_iter().product();
        }
    }
    let root_kernel = marked_graph.node_weight(root).unwrap().2[0];
    (kernel_graphs, root_kernel)
}
