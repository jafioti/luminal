// TODO
// get complex codegen working
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

mod kernels;

use colored::Colorize;
use egglog::{EGraph, Error, Term, TermDag, TermId, ast::Literal, var};
use itertools::Itertools;
use petgraph::{
    Directed, Direction,
    algo::toposort,
    graph::NodeIndex,
    prelude::StableGraph,
    visit::{EdgeRef, NodeRef},
};
use regex::Regex;
use std::{
    collections::{BTreeSet, HashMap, HashSet, VecDeque},
    fmt::Debug,
    iter::repeat,
};

#[cfg(target_os = "macos")]
const PRELUDE: &str = "
#include <metal_stdlib>
using namespace metal;

float mul(float a, float b) {
	return a * b;
}
";

#[cfg(target_os = "linux")]
const PRELUDE: &str = "
#include \"cuda_fp16.h\"
__device__ float mul(float a, float b) {
	return a * b;
}
";

#[derive(Clone, Debug)]
enum GraphTerm {
    GMEM { label: Option<String> },
    LoopIn { range: String, stride: String },
    LoopOut { range: String, stride: String },
    NewAcc { starting_value: String },
    Add,
    Mul,
    Max,
    Exp,
    Recip,
    Sin,
    Neg,
    ZeroStrideLoad { range: String, stride: String },
}

pub trait TermToString {
    fn term_to_string(&self) -> String;
}

impl TermToString for Term {
    fn term_to_string(&self) -> String {
        match self {
            Term::App(a, _) => a.to_string(),
            Term::Lit(l) => l.to_string(),
            Term::Var(v) => v.to_string(),
        }
    }
}

impl TermToString for usize {
    fn term_to_string(&self) -> String {
        self.to_string()
    }
}

impl TermToString for (Term, usize) {
    fn term_to_string(&self) -> String {
        let s = match &self.0 {
            Term::App(a, _) => a.to_string(),
            Term::Lit(l) => l.to_string(),
            Term::Var(v) => v.to_string(),
        };
        format!("{s}[{}]", self.1)
    }
}

impl TermToString for GraphTerm {
    fn term_to_string(&self) -> String {
        match self {
            GraphTerm::Add => "Add".to_string(),
            GraphTerm::Mul => "Mul".to_string(),
            GraphTerm::Max => "Max".to_string(),
            GraphTerm::Exp => "Exp".to_string(),
            GraphTerm::Sin => "Sin".to_string(),
            GraphTerm::Recip => "Recip".to_string(),
            GraphTerm::Neg => "Neg".to_string(),
            GraphTerm::NewAcc { starting_value } => format!("NewAcc({starting_value})"),
            GraphTerm::LoopIn { range, stride } => format!("LoopIn ({range}; {stride})"),
            GraphTerm::LoopOut { range, stride } => format!("LoopOut ({range}; {stride})"),
            GraphTerm::GMEM { label } => {
                if let Some(label) = label {
                    format!("GMEM ({label})")
                } else {
                    "GMEM".to_string()
                }
            }
            GraphTerm::ZeroStrideLoad { range, stride } => {
                format!("ZeroStrideLoad({range}; {stride})")
            }
        }
    }
}

impl TermToString for (GraphTerm, usize) {
    fn term_to_string(&self) -> String {
        format!("{} [{}]", self.0.term_to_string(), self.1)
    }
}

impl TermToString for (GraphTerm, Vec<String>, usize) {
    fn term_to_string(&self) -> String {
        format!(
            "{} [{}] {{{}}}",
            self.0.term_to_string(),
            self.1.len(),
            self.2
        )
    }
}

impl TermToString for (GraphTerm, Vec<String>, Vec<usize>) {
    fn term_to_string(&self) -> String {
        format!(
            "{} [{}] {{{:?}}}",
            self.0.term_to_string(),
            self.1.len(),
            self.2
        )
    }
}

fn main() {
    let (g, r) = kernels::make_flash_attention();
    codegen(g, r);
    // match run_egglog_program(include_str!("code.lisp")) {
    //     Ok((s, _serialized, termdag, root)) => {
    //         if s.is_empty() {
    //             println!("{}", "Success!".bright_green().bold())
    //         } else {
    //             println!("{}", format!("Success: {s:?}").bright_green().bold())
    //         }
    //         // let (graph, root) = dag_to_petgraph(&termdag, termdag.lookup(&root));

    //     }
    //     Err(e) => println!("{e}"),
    // }
}

fn validate_graph(graph: &StableGraph<(GraphTerm, usize), u8, Directed>) {
    // walk the graph and make sure loopins -> next loop level (or loopout) and prev loop (or loopin) -> loopout
    for node in graph.node_indices() {
        let (curr_term, curr_level) = graph.node_weight(node).unwrap();
        if matches!(curr_term, GraphTerm::LoopIn { .. }) {
            // All loopins must have outputs that are one level more, unless they are loopouts
            for new_node in graph.neighbors_directed(node, Direction::Outgoing) {
                let (new_term, new_level) = graph.node_weight(new_node).unwrap();
                if !matches!(new_term, GraphTerm::LoopOut { .. }) {
                    if *new_level != *curr_level + 1 {
                        display_graph(
                            graph,
                            &[
                                (node, "yellow".to_string()),
                                (new_node, "yellow".to_string()),
                            ],
                        );
                        panic!("incorrect levels");
                    }
                }
            }
        } else if matches!(curr_term, GraphTerm::LoopOut { .. }) {
            // All loopouts must have inputs that are one level more, unless they are loopins
            for new_node in graph.neighbors_directed(node, Direction::Incoming) {
                let (new_term, new_level) = graph.node_weight(new_node).unwrap();
                if !matches!(new_term, GraphTerm::LoopIn { .. }) {
                    if *new_level != *curr_level + 1 {
                        display_graph(
                            graph,
                            &[
                                (node, "yellow".to_string()),
                                (new_node, "yellow".to_string()),
                            ],
                        );
                        panic!("incorrect levels");
                    }
                }
            }
        } else {
            for new_node in graph.neighbors_directed(node, Direction::Incoming) {
                let (new_term, new_level) = graph.node_weight(new_node).unwrap();
                if !matches!(
                    new_term,
                    GraphTerm::LoopIn { .. } | GraphTerm::LoopOut { .. }
                ) {
                    if *new_level != *curr_level {
                        display_graph(
                            graph,
                            &[
                                (node, "yellow".to_string()),
                                (new_node, "yellow".to_string()),
                            ],
                        );
                        panic!("incorrect levels");
                    }
                }
            }

            if graph
                .neighbors_directed(node, Direction::Incoming)
                .next()
                .is_none()
                && !matches!(graph.node_weight(node).unwrap().0, GraphTerm::NewAcc { .. })
            {
                if *curr_level != 0 {
                    display_graph(graph, &[(node, "yellow".to_string())]);
                    println!("Inputs must have level 0, found {curr_level}");
                }
            }
        }
    }
}

fn codegen(graph: StableGraph<GraphTerm, u8, Directed>, root: NodeIndex) -> (String, TermId) {
    let (kernels, root_kernel) = split_kernels(graph, root);
    println!("Kernels: {} Root Kernel: {root_kernel}", kernels.len());
    // Create kernel meta graph to toposort
    let mut meta_graph = StableGraph::<usize, u8, Directed>::default();
    for i in 0..kernels.len() {
        meta_graph.add_node(i);
    }
    for (n_kernel, (_, inputs, _, _)) in kernels.iter().enumerate() {
        for (input_kernel, _, _) in inputs {
            if let Some(input_kernel) = input_kernel {
                meta_graph.add_edge(NodeIndex::new(*input_kernel), NodeIndex::new(n_kernel), 0);
            }
        }
    }
    for node in toposort(&meta_graph, None).unwrap() {
        let kernel_index = *meta_graph.node_weight(node).unwrap();
        let (kernel_graph, inputs, outputs, smem_buffers) = kernels[kernel_index].clone();
        println!("KERNEL {kernel_index}");
        validate_graph(&kernel_graph);
        let inputs: Vec<_> = inputs
            .into_iter()
            .enumerate()
            .map(|(a, (_, _, b))| (a, b))
            .collect();
        let outputs: Vec<_> = outputs
            .into_iter()
            .enumerate()
            .map(|(i, a)| (inputs.len() + i, a))
            .collect();
        let mut loop_levels = vec![];
        let mut node_to_var = inputs
            .iter()
            .map(|(v, n)| (*n, *v))
            .chain(outputs.iter().map(|(v, n)| (*n, *v)))
            .map(|(k, v)| (k, (v, true, None)))
            .collect::<HashMap<_, _>>();

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
        )
        .unwrap();
        let grid = loop_levels
            .clone()
            .into_iter()
            .chain(repeat("1".to_string()))
            .take(3)
            .collect_vec();
        let threadblock = loop_levels
            .into_iter()
            .skip(3)
            .chain(repeat("1".to_string()))
            .take(3)
            .collect_vec();
        let kernel = format!(
            "extern \"C\" __global__ void kernel{kernel_index}({}) {{
{}{}{}
}}",
            inputs
                .into_iter()
                .map(|(a, _)| format!("float* {}", var_to_char(a))) // put const in here for the future
                .chain(
                    outputs
                        .into_iter()
                        .map(|(a, _)| format!("float* {}", var_to_char(a)))
                )
                .join(", "),
            if smem_buffers.is_empty() {
                ""
            } else {
                "\textern __shared__ float sm[];\n"
            },
            smem_buffers
                .into_iter()
                .scan("".to_string(), |prev_buffers, (n, _, size)| {
                    let r = format!("\tfloat* {} = sm{};\n", var_to_char(n), prev_buffers);
                    prev_buffers.push_str(&format!(" + {size}"));
                    Some(r)
                })
                .join(""),
            kernel.into_iter().map(|s| format!("\t{s}")).join("\n")
        );
        println!("Grid: {grid:?} Threadblock: {threadblock:?}");
        println!("{kernel}");
    }
    ("".to_string(), 0)
}

fn make_kernel(
    kernel_graph: &StableGraph<(GraphTerm, usize), u8, Directed>,
    include_nodes: HashSet<NodeIndex>,
    node_to_var: &mut HashMap<NodeIndex, (usize, bool, Option<String>)>,
    prev_max_var: &mut usize, // contains the char last used
    loop_levels: &mut Vec<String>,
    loop_indexes: &mut HashMap<NodeIndex, usize>,
    smem_buffers: &HashMap<NodeIndex, usize>,
    current_loop_level: usize,
) -> Option<Vec<String>> {
    let mut kernel_lines = vec![];
    // toposort down the graph
    let toposorted = toposort_subset(kernel_graph, &include_nodes);
    for node in toposorted
        .into_iter()
        .filter(|n| kernel_graph.node_weight(*n).unwrap().1 == current_loop_level)
    {
        if node_to_var.contains_key(&node) {
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
                    if let Some(inp) = kernel_graph
                        .neighbors_directed(*i, Direction::Incoming)
                        .next()
                    {
                        !node_to_var.contains_key(&inp) // no recorded key
                    } else {
                        false
                    }
                }) {
                    continue;
                }
                for i in &loop_inputs {
                    body_nodes.remove(&i.0);
                }
                for i in &loop_outputs {
                    body_nodes.remove(&i.0);
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

                // Make new loop
                if *loop_level < 6 {
                    if loop_inputs.iter().any(|(_, st)| **st != "0")
                        || loop_outputs.iter().any(|(_, st)| **st != "0")
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
                    kernel_lines.push(format!("{}for (int loop_{loop_var} = 0; loop_{loop_var} < {range}; loop_{loop_var} += 1) {{", (0..loop_level.saturating_sub(6)).map(|_| "\t").join("")));
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
                    if stride.contains("Acc") {
                        assert!(
                            *loop_level > 5,
                            "No accumulations allowed on grid or threadblock levels!"
                        );
                        new_vars.push(real_input);
                        node_to_var.insert(*input, (real_input, is_ptr, real_size));
                    } else {
                        if *stride != "0" {
                            assert!(is_ptr);
                            *prev_max_var += 1;
                            kernel_lines.push(format!(
                                "{}float* {} = {} + {};",
                                (0..(*loop_level + 1).saturating_sub(6))
                                    .map(|_| "\t")
                                    .join(""),
                                var_to_char(*prev_max_var),
                                var_to_char(real_input),
                                stride.replace('z', &format!("loop_{loop_var}"))
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
                    if stride.contains("Acc") {
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
                    let (real_output, is_ptr, real_size) = node_to_var[&dest].clone();
                    if *stride != "0" {
                        assert!(is_ptr, "Only pointers can be offset!");
                        *prev_max_var += 1;
                        kernel_lines.push(format!(
                            "{}float* {} = {} + {};",
                            (0..(*loop_level + 1).saturating_sub(6))
                                .map(|_| "\t")
                                .join(""),
                            var_to_char(*prev_max_var),
                            var_to_char(real_output),
                            stride.replace('z', &format!("loop_{loop_var}"))
                        ));
                        new_output_vars.push(*prev_max_var);
                        node_to_var.insert(*output, (*prev_max_var, is_ptr, None));
                    } else {
                        new_output_vars.push(real_output);
                        node_to_var.insert(*output, (real_output, is_ptr, real_size));
                    }
                }
                loop_levels.push(range.to_string());
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
                )?;

                kernel_lines.extend(loop_body);

                // Set outputs if nessecary
                for (output, _) in loop_outputs {
                    let body_out = kernel_graph
                        .neighbors_directed(output, Direction::Incoming)
                        .next()
                        .unwrap();
                    if node_to_var[&output].0 != node_to_var[&body_out].0
                        && !node_to_var[&body_out].1
                    {
                        if let Some(size) = &node_to_var[&body_out].2 {
                            // Save size numbers
                            kernel_lines.push(format!(
                                "{}for (int save = 0; save < {size}; save++) {{",
                                (0..(*loop_level + 1).saturating_sub(6))
                                    .map(|_| "\t")
                                    .join("")
                            ));
                            assert!(
                                node_to_var[&output].1 && node_to_var[&body_out].1,
                                "Both src and dest must be pointers when saving a block"
                            );
                            kernel_lines.push(format!(
                                "{}{}[save] = {}[save];",
                                (0..(*loop_level + 2).saturating_sub(6))
                                    .map(|_| "\t")
                                    .join(""),
                                var_to_char(node_to_var[&output].0),
                                var_to_char(node_to_var[&body_out].0),
                            ));
                            kernel_lines.push(format!(
                                "{}}}",
                                (0..(*loop_level + 1).saturating_sub(6))
                                    .map(|_| "\t")
                                    .join("")
                            ));
                        } else {
                            kernel_lines.push(format!(
                                "{}{}{} = {}{};",
                                (0..(*loop_level + 1).saturating_sub(6))
                                    .map(|_| "\t")
                                    .join(""),
                                if node_to_var[&output].1 { "*" } else { "" },
                                var_to_char(node_to_var[&output].0),
                                if node_to_var[&body_out].1 { "*" } else { "" },
                                var_to_char(node_to_var[&body_out].0),
                            ));
                        }
                    }
                }

                if *loop_level >= 6 {
                    kernel_lines.push(format!(
                        "{}}}",
                        (0..loop_level.saturating_sub(6)).map(|_| "\t").join("")
                    ));
                }
            }
            GraphTerm::NewAcc { starting_value } => {
                // Go through all loopins after this to figure out how many accumulators we need
                let mut curr = node;
                let mut size = "1".to_string();
                loop {
                    curr = kernel_graph
                        .neighbors_directed(curr, Direction::Outgoing)
                        .next()
                        .unwrap();
                    if let GraphTerm::LoopIn { range, stride } =
                        &kernel_graph.node_weight(curr).unwrap().0
                    {
                        if !stride.contains("Acc") && stride != "0" {
                            size = format!("{size} * {range}");
                        }
                    } else {
                        break;
                    }
                }
                // Create accumulator
                *prev_max_var += 1;
                kernel_lines.push(format!(
                    "{}float {}[{size}] = {{{starting_value}}};",
                    (0..loop_level.saturating_sub(6)).map(|_| "\t").join(""),
                    var_to_char(*prev_max_var)
                ));
                node_to_var.insert(node, (*prev_max_var, true, Some(size)));
            }
            GraphTerm::LoopOut { range, stride } => {
                panic!("found loopout range: {range} stride: {stride}")
            }
            GraphTerm::ZeroStrideLoad { range, stride } => {
                // Search back for last loopin of same range with stride of 0
                let mut curr = node;
                let found;
                let mut offset = vec![];
                loop {
                    let prev = kernel_graph
                        .neighbors_directed(curr, Direction::Incoming)
                        .next()
                        .unwrap();
                    if let (
                        GraphTerm::LoopIn {
                            range: prev_range,
                            stride: prev_stride,
                        },
                        level,
                    ) = kernel_graph.node_weight(prev).unwrap()
                    {
                        if *prev_range == *range && prev_stride == "0" {
                            // Found
                            found = Some((prev, *level));
                            break;
                        }
                        if *level < 6 && *level > 2 && prev_stride != "0" {
                            offset.push(prev_stride.replace(
                                "z",
                                &format!("loop_{}", var_to_char(loop_indexes[&prev])),
                            ));
                        }
                    } else {
                        panic!("found non-loop-in!");
                    }
                    curr = prev;
                }
                let Some((base, level)) = found else {
                    return None;
                };
                let src = kernel_graph
                    .neighbors_directed(node, Direction::Incoming)
                    .next()
                    .unwrap();
                // Depending on the level do different things
                match level {
                    ..3 => return None, // Grid
                    3..6 => {
                        // Threadblock
                        // We need to get the smem pointer to write to and pass along
                        let buffer = smem_buffers[&node];
                        let mut smem_ptr = buffer;
                        if !offset.is_empty() {
                            *prev_max_var += 1;
                            smem_ptr = *prev_max_var;
                            kernel_lines.push(format!(
                                "{}float* {} = {} + {};",
                                (0..loop_level.saturating_sub(6)).map(|_| "\t").join(""),
                                var_to_char(smem_ptr),
                                var_to_char(buffer),
                                offset.iter().join(" + ")
                            ));
                        }
                        if kernel_lines
                            .last()
                            .map(|i| i.contains("__syncthreads()"))
                            .unwrap_or_default()
                        {
                            kernel_lines.pop();
                        } else {
                            kernel_lines.push(format!(
                                "{}__syncthreads();",
                                (0..loop_level.saturating_sub(6)).map(|_| "\t").join("")
                            ));
                        }
                        kernel_lines.push(format!(
                            "{}{}[loop_{}] = *{} + {};",
                            (0..loop_level.saturating_sub(6)).map(|_| "\t").join(""),
                            var_to_char(smem_ptr),
                            var_to_char(loop_indexes[&base]),
                            var_to_char(node_to_var[&src].0),
                            stride.replace(
                                "z",
                                &format!("loop_{}", var_to_char(loop_indexes[&base]))
                            )
                        ));
                        kernel_lines.push(format!(
                            "{}__syncthreads();",
                            (0..loop_level.saturating_sub(6)).map(|_| "\t").join("")
                        ));
                        node_to_var.insert(node, (smem_ptr, false, None));
                    }
                    6.. => {
                        // Thread
                        panic!("Need to set up zerostrideload for register buffers")
                    }
                }
            }
            GraphTerm::GMEM { .. } => {}
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
                    "{}float {} = {expr};",
                    (0..loop_level.saturating_sub(6)).map(|_| "\t").join(""),
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
                    GraphTerm::Max => format!("fmax({inp_a}, {inp_b})"),
                    _ => panic!(),
                };
                kernel_lines.push(format!(
                    "{}float {} = {expr};",
                    (0..loop_level.saturating_sub(6)).map(|_| "\t").join(""),
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

/// Given a loop term in the dag, return the loop's name, range, and stride for this loop in / out
fn get_loop_info(termdag: &TermDag, loop_in_out: TermId) -> (String, String, String) {
    let Term::App(_, children) = termdag.get(loop_in_out) else {
        panic!("Loop in / out is not a loop in / out!");
    };
    let Term::App(_, loop_children) = termdag.get(children[1]) else {
        panic!()
    };
    let Term::Lit(loop_name) = termdag.get(loop_children[0]) else {
        panic!();
    };
    let Term::App(mnum, ch) = termdag.get(loop_children[1]) else {
        panic!("{:?}", termdag.get(loop_children[1]));
    };
    assert_eq!(mnum.as_str(), "MNum");
    let Term::Lit(loop_range) = termdag.get(ch[0]) else {
        panic!()
    };
    let stride = render_math_expr(termdag, children[2]);
    (loop_name.to_string(), loop_range.to_string(), stride)
}

fn render_math_expr(termdag: &TermDag, root: TermId) -> String {
    match termdag.get(root) {
        Term::App(a, ch) => match a.as_str() {
            "MNum" => render_math_expr(termdag, ch[0]),
            "MVar" => render_math_expr(termdag, ch[0]),
            "MAdd" | "MSub" | "MMul" | "MDiv" | "MMod" => format!(
                "{} {} {}",
                render_math_expr(termdag, ch[0]),
                match a.as_str() {
                    "MAdd" => "+",
                    "MSub" => "-",
                    "MMul" => "*",
                    "MDiv" => "/",
                    "MMod" => "%",
                    _ => panic!(),
                },
                render_math_expr(termdag, ch[1])
            ),
            _ => panic!("unhandled math expr"),
        },
        Term::Lit(l) => match l {
            Literal::String(s) => s.as_str().to_string(),
            _ => panic!(),
        },
        Term::Var(_) => panic!("unhandled var"),
    }
}

/// Runs an Egglog program from a string and returns its output messages.
pub fn run_egglog_program(code: &str) -> Result<(Vec<String>, String, TermDag, Term), Error> {
    // Create a fresh EGraph with all the defaults
    let mut egraph = EGraph::default();
    egraph.enable_messages();
    let commands = egraph.parser.get_program_from_string(None, code)?;
    let msgs = egraph.run_program(commands)?;
    println!("Run Report:  {}", egraph.get_run_report().as_ref().unwrap());
    let (sort, value) = egraph.eval_expr(&var!("full"))?;
    let (termdag, root) = egraph.extract_value(&sort, value)?;
    let (_petgraph, _root_idx) = dag_to_petgraph(&termdag, termdag.lookup(&root));
    let s = egraph.serialize(egglog::SerializeConfig {
        root_eclasses: vec![(sort, value)],
        ..Default::default()
    });
    println!(
        "Nodes: {} Roots: {} Class Data: {}",
        s.nodes.len(),
        s.root_eclasses.len(),
        s.class_data.len()
    );
    let json = serde_json::to_string_pretty(&s).unwrap();
    Ok((msgs, json, termdag, root))
}

fn dag_to_petgraph(
    dag: &TermDag,
    root: TermId,
) -> (StableGraph<GraphTerm, u8, Directed>, NodeIndex) {
    let mut graph = StableGraph::new();
    let mut map: HashMap<Term, NodeIndex> = HashMap::new();

    // recursive DFS that interns each term exactly once
    fn intern(
        dag: &TermDag,
        g: &mut StableGraph<GraphTerm, u8, Directed>,
        map: &mut HashMap<Term, NodeIndex>,
        t: Term,
    ) -> Option<NodeIndex> {
        if let Term::App(a, _) = &t {
            if a.as_str() == "Loop" || a.as_str() == "MVar" || a.as_str() == "MNum" {
                return None;
            }
        }
        if let Some(&idx) = map.get(&t) {
            return Some(idx);
        }
        let graphterm = match &t {
            Term::App(a, ch) => match a.as_str() {
                "Exp" => GraphTerm::Exp,
                "Sin" => GraphTerm::Sin,
                "Add" => GraphTerm::Add,
                "LoopIn" => {
                    let (_, range, stride) = get_loop_info(dag, dag.lookup(&t));
                    GraphTerm::LoopIn { range, stride }
                }
                "LoopOut" => {
                    let (_, range, stride) = get_loop_info(dag, dag.lookup(&t));
                    GraphTerm::LoopOut { range, stride }
                }
                "Tensor" => {
                    let Term::Lit(name) = dag.get(ch[0]) else {
                        panic!("invalid tensor")
                    };
                    GraphTerm::GMEM {
                        label: match name {
                            Literal::String(s) => Some(s.as_str().to_string()),
                            _ => panic!(),
                        },
                    }
                }
                "ZeroStrideLoad" => {
                    let (_, range, stride) = get_loop_info(dag, dag.lookup(&t));
                    GraphTerm::ZeroStrideLoad { range, stride }
                }
                _ => panic!("invalid term: {}", a.as_str()),
            },
            Term::Lit(_) | Term::Var(_) => return None,
        };
        let idx = g.add_node(graphterm);
        map.insert(t.clone(), idx);
        if let Term::App(_, children) = &t {
            for (i, child) in children.iter().enumerate() {
                if let Some(c_idx) = intern(dag, g, map, dag.get(*child).clone()) {
                    g.add_edge(c_idx, idx, i as u8);
                }
            }
        }
        Some(idx)
    }

    let root_idx = intern(dag, &mut graph, &mut map, dag.get(root).clone());
    (graph, root_idx.unwrap())
}

/// add kernel dimensions so that all loop-to-loop dependencies are between seperate kernels or on the threadblock / thread levels
fn split_kernels(
    graph: StableGraph<GraphTerm, u8, Directed>,
    mut root: NodeIndex,
) -> (
    Vec<(
        StableGraph<(GraphTerm, usize), u8, Directed>,
        Vec<(Option<usize>, usize, NodeIndex)>, // (src kernel, src kernel output, current graph node)
        Vec<NodeIndex>,                         // output node
        Vec<(usize, NodeIndex, String)>,        // (shared memory buffer name, node, buffer size)
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
    dfs = vec![root];
    seen.clear();
    let mut n_kernels = 1;
    while let Some(node) = dfs.pop() {
        if seen.contains(&node) {
            continue;
        }
        seen.insert(node);
        let (term, curr_level, mut curr_kernel) = marked_graph.node_weight(node).unwrap().clone();
        for source in marked_graph
            .neighbors_undirected(node)
            .filter(|n| seen.contains(n))
            .collect_vec()
        {
            let (src_term, src_level, src_kernel) = marked_graph.node_weight(source).unwrap();
            if matches!(term, GraphTerm::LoopIn { .. })
                && matches!(src_term, GraphTerm::LoopOut { .. })
                && curr_level.len() < 3
            {
                // We have a loopout -> loopin at a grid level, which means we make a new kernel
                let max_kernel = *curr_kernel.iter().max().unwrap();
                n_kernels = n_kernels.max(max_kernel + 2);
                curr_kernel = vec![max_kernel + 1];
                break;
            } else {
                // We are in the same kernel(s) as the source
                for k in src_kernel {
                    if !curr_kernel.contains(k) {
                        curr_kernel.push(*k);
                    }
                }
            }
            // Correctness check
            if !matches!(term, GraphTerm::LoopIn { .. } | GraphTerm::LoopOut { .. })
                && !matches!(
                    src_term,
                    GraphTerm::LoopIn { .. } | GraphTerm::LoopOut { .. }
                )
            {
                assert_eq!(
                    curr_level, *src_level,
                    "Edges need to go between the same levels if src and dest are not LoopIn or LoopOut"
                );
            }
        }
        marked_graph.node_weight_mut(node).unwrap().2 = curr_kernel;
        dfs.extend(marked_graph.neighbors_undirected(node));
    }
    // // Prune out unnessecary kernel members
    // let mut dfs_stack = marked_graph
    //     .node_indices()
    //     .filter(|n| {
    //         marked_graph
    //             .neighbors_directed(*n, Direction::Outgoing)
    //             .next()
    //             .is_none()
    //     })
    //     .collect_vec();
    // while let Some(node) = dfs_stack.pop() {
    //     dfs_stack.extend(marked_graph.neighbors_directed(node, Direction::Incoming));
    //     if marked_graph
    //         .neighbors_directed(node, Direction::Outgoing)
    //         .next()
    //         .is_none()
    //     {
    //         continue;
    //     }
    //     let (term, curr_level, mut curr_kernel) = marked_graph.node_weight(node).unwrap().clone();
    //     let split_cond = curr_level.len() < 3 && matches!(term, GraphTerm::LoopOut { .. });
    //     curr_kernel.retain(|k| {
    //         marked_graph
    //             .neighbors_directed(node, Direction::Outgoing)
    //             .any(|n| {
    //                 (split_cond
    //                     && matches!(
    //                         marked_graph.node_weight(n).unwrap().0,
    //                         GraphTerm::LoopIn { .. }
    //                     ))
    //                     || marked_graph.node_weight(n).unwrap().2.contains(k)
    //             })
    //     });
    //     marked_graph.node_weight_mut(node).unwrap().2 = curr_kernel;
    // }

    // Add kernel barriers
    for edge in marked_graph.edge_indices().collect_vec() {
        let (source, dest) = marked_graph.edge_endpoints(edge).unwrap();
        let (_, dest_level, dest_kernel) = marked_graph.node_weight(dest).unwrap().clone();
        let (_, _, src_kernel) = marked_graph.node_weight(source).unwrap().clone();
        if dest_level.len() > 0 && dest_kernel.iter().any(|i| !src_kernel.contains(i)) {
            // Put a barrier here
            let (mut src, mut dest) = (dest, source);
            for i in (0..dest_level.len()).rev() {
                let new_src = marked_graph.add_node((
                    GraphTerm::LoopOut {
                        range: dest_level[i].clone(),
                        stride: "0".to_string(),
                    },
                    dest_level[..i].to_vec(),
                    src_kernel.clone(),
                ));
                marked_graph.add_edge(src, new_src, 0);
                src = new_src;
                let new_dest = marked_graph.add_node((
                    GraphTerm::LoopIn {
                        range: dest_level[i].clone(),
                        stride: "0".to_string(),
                    },
                    dest_level[..i].to_vec(),
                    dest_kernel.clone(),
                ));
                marked_graph.add_edge(new_dest, dest, 0);
                dest = new_dest;
            }
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
                outputs.push(node_maps[*kernel][&node]);
            }
        }
    }
    // Go through graph inputs and add them to the kernel hashmap as inputs
    for input in marked_graph
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
                GraphTerm::NewAcc { .. }
            )
        })
    {
        let (_, _, kernels) = marked_graph.node_weight(input).unwrap();
        for kernel in kernels {
            kernel_graphs[*kernel]
                .1
                .push((None, 0, node_maps[*kernel][&input]));
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
                    .position(|s| *s == src_kernel_node)
                {
                    p
                } else {
                    kernel_graphs[*start_kernel].2.push(src_kernel_node);
                    kernel_graphs[*start_kernel].2.len() - 1
                };
                // add input to dest
                kernel_graphs[*end_kernel].1.push((
                    Some(*start_kernel),
                    src_output_index,
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
            if let (GraphTerm::ZeroStrideLoad { range, .. }, _) =
                kernel_graph.node_weight(node).unwrap()
            {
                // Walk backwards to find threadblock level loopins and multiply all ranges with non-zero strides
                let mut size = range.to_string();
                let mut curr = node;
                let mut base_level;
                loop {
                    let prev = graph
                        .neighbors_directed(curr, Direction::Incoming)
                        .next()
                        .unwrap();
                    if let (
                        GraphTerm::LoopIn {
                            range: prev_range,
                            stride,
                        },
                        prev_level,
                    ) = kernel_graph.node_weight(prev).unwrap()
                    {
                        base_level = *prev_level;
                        if *prev_level < 3 {
                            break;
                        } else if *prev_level < 6 {
                            if *stride != "0" {
                                size = format!("{size} * {prev_range}");
                            }
                        }
                        // UNCLEAR IF THIS IS CORRECT OR NOT. THIS MULTIPLIES ALL THREADBLOCK DIMS
                    }
                    curr = prev;
                }
                if base_level > 5 {
                    continue;
                }
                let buf_index = inputs.len() + outputs.len() + smem_buffers.len();
                smem_buffers.push((buf_index, node, size));
            }
        }
    }
    let root_kernel = marked_graph.node_weight(root).unwrap().2[0];
    (kernel_graphs, root_kernel)
}

/// View a debug graph in the browser
pub fn display_graph<G: TermToString>(
    graph: &petgraph::stable_graph::StableGraph<G, u8, petgraph::Directed, u32>,
    mark_nodes: &[(NodeIndex, String)],
) {
    let mut new_graph = StableGraph::new();
    let mut map = HashMap::new();
    for node in graph.node_indices() {
        map.insert(
            node,
            new_graph.add_node(graph.node_weight(node).unwrap().term_to_string()),
        );
    }
    for edge in graph.edge_indices() {
        let weight = graph.edge_weight(edge).unwrap();
        let (src, dest) = graph.edge_endpoints(edge).unwrap();
        new_graph.add_edge(map[&src], map[&dest], weight);
    }
    let mut graph_string =
        petgraph::dot::Dot::with_config(&new_graph, &[petgraph::dot::Config::EdgeIndexLabel])
            .to_string();
    let re = Regex::new(r#"label\s*=\s*"\d+""#).unwrap();
    graph_string = re.replace_all(&graph_string, "").to_string();
    for (n, color) in mark_nodes {
        graph_string = graph_string.replace(
            &format!("    {} [ label =", n.index()),
            &format!(
                "    {} [ style=\"filled\" fillcolor=\"{color}\" label =",
                n.index()
            ),
        );
    }

    let url = format!(
        "https://dreampuf.github.io/GraphvizOnline/#{}",
        urlencoding::encode(&graph_string)
    );
    if let Err(e) = webbrowser::open(&url) {
        panic!("Error displaying graph: {:?}", e);
    }
}

fn var_to_char(var: usize) -> String {
    if var > 25 {
        format!("{}{}", var_to_char(var / 26), var_to_char(var % 26))
    } else {
        ((var + 97) as u8 as char).to_string()
    }
}
