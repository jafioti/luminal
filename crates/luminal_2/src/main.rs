// TODO
// unit test complex codegen with correctness checks
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
mod symbolic;

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
    collections::{BTreeSet, HashMap, HashSet},
    fmt::Debug,
    iter::repeat,
};

use crate::symbolic::Expression;

#[derive(Clone, PartialEq, Eq)]
enum GPUArch {
    CUDA,
    Metal(HashMap<usize, &'static str>),
}

impl GPUArch {
    fn metal_buffer_type(&self, var: usize) -> &'static str {
        match self {
            Self::Metal(m) => m.get(&var).copied().unwrap_or(""),
            _ => "",
        }
    }

    fn add_metal_buffer_type(&mut self, var: usize, buf_type: &'static str) {
        if let Self::Metal(m) = self {
            m.insert(var, buf_type);
        }
    }
}

#[derive(Debug, Clone, Default)]
struct Kernel {
    code: String,
    // launch params
    grid: (Expression, Expression, Expression),
    threadblock: (Expression, Expression, Expression),
    smem: Expression, // sizes of required shared memory buffers
    outputs: Vec<Expression>,
}

#[derive(Clone, Copy)]
enum GMEMBuffer {
    PrevKernel { kernel: usize, output: usize },
    Input(usize),
}

#[derive(Clone, Debug)]
enum GraphTerm {
    GMEM {
        label: Option<String>,
    },
    LoopIn {
        range: Expression,
        stride: Expression,
    },
    LoopOut {
        range: Expression,
        stride: Expression,
    },
    NewAcc {
        starting_value: String,
    },
    Add,
    Mul,
    Max,
    Exp,
    Recip,
    Sin,
    Neg,
    ZeroStrideLoad {
        range: Expression,
        stride: Expression,
    },
}

pub trait TermToString {
    fn term_to_string(&self) -> String;
}

pub trait EdgeToString {
    fn edge_to_string(&self) -> String;
}

impl EdgeToString for u8 {
    fn edge_to_string(&self) -> String {
        self.to_string()
    }
}

impl EdgeToString for (u8, u8) {
    fn edge_to_string(&self) -> String {
        format!("{}, {}", self.0, self.1)
    }
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

impl TermToString for Kernel {
    fn term_to_string(&self) -> String {
        if self.code == "Inputs" || self.code == "Outputs" {
            return self.code.clone();
        } else {
            format!(
                "Kernel ({}, {}, {}), ({}, {}, {}) -> {:?}",
                self.grid.0,
                self.grid.1,
                self.grid.2,
                self.threadblock.0,
                self.threadblock.1,
                self.threadblock.2,
                self.outputs
            )
        }
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

impl TermToString for (GraphTerm, Vec<Expression>, Vec<usize>) {
    fn term_to_string(&self) -> String {
        format!(
            "{} [{}] {{{:?}}}",
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
    codegen(g, r, GPUArch::CUDA);
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

fn codegen(
    graph: StableGraph<GraphTerm, u8, Directed>,
    root: NodeIndex,
    mut arch: GPUArch,
) -> StableGraph<Kernel, (u8, u8), Directed> {
    let (kernels, root_kernel) = split_kernels(graph, root);
    println!("Kernels: {} Root Kernel: {root_kernel}", kernels.len());
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
        println!("KERNEL {}", node.index());
        validate_graph(&kernel_graph);
        let mut node_to_var = inputs
            .iter()
            .map(|(_, n)| *n)
            .chain(outputs.iter().map(|(_, i)| *i))
            .enumerate()
            .map(|(v, n)| (n, (v, true, None)))
            .collect::<HashMap<_, _>>();
        for (_, (n, _, _)) in &node_to_var {
            arch.add_metal_buffer_type(*n, "device ");
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
        )
        .unwrap();
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
        println!("Grid: {grid:?} Threadblock: {threadblock:?}");
        println!("{kernel}");
        *meta_graph.node_weight_mut(node).unwrap() = Kernel {
            code: kernel,
            grid: (grid[0], grid[1], grid[2]),
            threadblock: (threadblock[0], threadblock[1], threadblock[2]),
            smem: smem_buffers.into_iter().map(|(_, _, a)| a).product(),
            outputs: outputs.into_iter().map(|(o, _)| o).collect(),
        };
    }
    meta_graph
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
                    loop_levels.push(*range);
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
                        if **stride != 0 {
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
                    let (real_output, is_ptr, real_size) = node_to_var[&dest].clone();
                    if **stride != 0 {
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
                    let (output, output_ptr, _) = &node_to_var[&output];
                    let (body_out, body_out_ptr, body_out_size) = &node_to_var[&body_out];
                    if output != body_out && (!body_out_ptr || is_acc) {
                        if let Some(size) = &body_out_size {
                            if *size == 1 {
                                kernel_lines.push(format!(
                                    "{inner_spacing}{}{} = {}{};",
                                    if *output_ptr { "*" } else { "" },
                                    var_to_char(*output),
                                    if *body_out_ptr { "*" } else { "" },
                                    var_to_char(*body_out),
                                ));
                            } else {
                                // Save size numbers
                                kernel_lines.push(format!(
                                    "{inner_spacing}for (int save = 0; save < {size}; save++) {{",
                                ));
                                assert!(
                                    *output_ptr && *body_out_ptr,
                                    "Both src and dest must be pointers when saving a block"
                                );
                                kernel_lines.push(format!(
                                    "{inner_spacing}\t{}[save] = {}[save];",
                                    var_to_char(*output),
                                    var_to_char(*body_out),
                                ));
                                kernel_lines.push(format!("{inner_spacing}}}"));
                            }
                        } else {
                            kernel_lines.push(format!(
                                "{inner_spacing}{}{} = {}{};",
                                if *output_ptr { "*" } else { "" },
                                var_to_char(*output),
                                if *body_out_ptr { "*" } else { "" },
                                var_to_char(*body_out),
                            ));
                        }
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
                        if *prev_range == *range && *prev_stride == 0 {
                            // Found
                            found = Some((prev, *level));
                            break;
                        }
                        if *level < 6 && *level > 2 && *prev_stride != 0 {
                            offset.push(prev_stride.to_string().replace(
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
                            arch.add_metal_buffer_type(*prev_max_var, "threadgroup ");
                            smem_ptr = *prev_max_var;
                            kernel_lines.push(format!(
                                "{spacing}{}float* {} = {} + {};",
                                arch.metal_buffer_type(*prev_max_var),
                                var_to_char(smem_ptr),
                                var_to_char(buffer),
                                offset.iter().join(" + ")
                            ));
                        }
                        if kernel_lines
                            .get(kernel_lines.len() - 2)
                            .map(|i| i.contains("__syncthreads()"))
                            .unwrap_or_default()
                        {
                            kernel_lines.remove(kernel_lines.len() - 2);
                        } else {
                            kernel_lines.push(format!("{spacing}__syncthreads();"));
                        }
                        kernel_lines.push(format!(
                            "{spacing}{}[loop_{}] = *{} + {};",
                            var_to_char(smem_ptr),
                            var_to_char(loop_indexes[&base]),
                            var_to_char(node_to_var[&src].0),
                            stride.to_string().replace(
                                "z",
                                &format!("loop_{}", var_to_char(loop_indexes[&base]))
                            )
                        ));
                        kernel_lines.push(format!("{spacing}__syncthreads();"));
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
    let (termdag, root, _) = egraph.extract_value(&sort, value)?;
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
                    GraphTerm::LoopIn {
                        range: Expression::from_string(&range),
                        stride: Expression::from_string(&stride),
                    }
                }
                "LoopOut" => {
                    let (_, range, stride) = get_loop_info(dag, dag.lookup(&t));
                    GraphTerm::LoopOut {
                        range: Expression::from_string(&range),
                        stride: Expression::from_string(&stride),
                    }
                }
                "GMEM" => {
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
                    GraphTerm::ZeroStrideLoad {
                        range: Expression::from_string(&range),
                        stride: Expression::from_string(&stride),
                    }
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
                && curr_level.len() < 3
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
                && curr_level.len() < 3
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
        let split_cond = curr_level.len() < 3 && matches!(term, GraphTerm::LoopOut { .. });
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
                        stride: 0.into(),
                    },
                    dest_level[..i].to_vec(),
                    src_kernel.clone(),
                ));
                marked_graph.add_edge(src, new_src, 0);
                src = new_src;
                let new_dest = marked_graph.add_node((
                    GraphTerm::LoopIn {
                        range: dest_level[i].clone(),
                        stride: 0.into(),
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
                GraphTerm::NewAcc { .. }
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
            if let (GraphTerm::ZeroStrideLoad { range, .. }, _) =
                kernel_graph.node_weight(node).unwrap()
            {
                // Walk backwards to find threadblock level loopins and multiply all ranges with non-zero strides
                let mut size = *range;
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
                            if *stride != 0 {
                                size *= prev_range;
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

/// View a debug graph in the browser
pub fn display_graph<G: TermToString, E: EdgeToString>(
    graph: &petgraph::stable_graph::StableGraph<G, E, petgraph::Directed, u32>,
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
        new_graph.add_edge(map[&src], map[&dest], weight.edge_to_string());
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

fn run_graph(inputs: Vec<Vec<f32>>, kernels: &StableGraph<Kernel, (u8, u8)>) -> Vec<Vec<f32>> {
    use metal_rs::{
        CompileOptions, ComputePassDescriptor, ComputePipelineDescriptor, Device,
        MTLResourceOptions, MTLSize,
    };
    let device = Device::system_default().unwrap();
    let queue = device.new_command_queue();
    let command_buffer = queue.new_command_buffer();
    // Allocate input buffers
    let mut buffers = HashMap::new();
    for node in toposort(kernels, None).unwrap() {
        let kernel = kernels.node_weight(node).unwrap();
        if kernel.code == "Inputs" {
            buffers.insert(
                node,
                inputs
                    .iter()
                    .map(|buf| {
                        device.new_buffer_with_data(
                            buf.as_ptr() as *mut _,
                            (buf.len() * std::mem::size_of::<f32>()) as u64,
                            MTLResourceOptions::StorageModeShared,
                        )
                    })
                    .collect_vec(),
            );
        } else if kernel.code == "Outputs" {
            // Run
            command_buffer.commit();
            command_buffer.wait_until_completed();

            // Copy outputs back
            return kernels
                .edges_directed(node, Direction::Incoming)
                .map(|e| &buffers[&e.source()][e.weight().0 as usize])
                .map(|buffer| {
                    let mut curr_data =
                        vec![0.0; buffer.length() as usize / std::mem::size_of::<f32>()];
                    let ptr = buffer.contents() as *mut f32;
                    for (i, d) in curr_data.iter_mut().enumerate() {
                        *d = unsafe { *ptr.add(i) };
                    }
                    curr_data
                })
                .collect();
        } else {
            // allocate output buffers
            let outputs = kernel
                .outputs
                .iter()
                .map(|size| {
                    device.new_buffer(
                        (size.to_usize().unwrap() * std::mem::size_of::<f32>()) as u64,
                        MTLResourceOptions::StorageModeShared,
                    )
                })
                .collect_vec();
            buffers.insert(node, outputs);

            // compile kernel
            let encoder = command_buffer
                .compute_command_encoder_with_descriptor(ComputePassDescriptor::new());
            let options = CompileOptions::new();
            options.set_fast_math_enabled(true);
            let lib = device
                .new_library_with_source(&kernel.code, &options)
                .unwrap();
            let pipeline_state_descriptor = ComputePipelineDescriptor::new();
            pipeline_state_descriptor.set_compute_function(Some(
                &lib.get_function(&format!("kernel{}", node.index()), None)
                    .unwrap(),
            ));
            let pipeline = device
                .new_compute_pipeline_state_with_function(
                    pipeline_state_descriptor.compute_function().unwrap(),
                )
                .unwrap();
            encoder.set_compute_pipeline_state(&pipeline);

            // set inputs
            for (i, (input, input_index)) in kernels
                .edges_directed(node, Direction::Incoming)
                .sorted_by_key(|n| n.weight().1)
                .map(|n| (n.source(), n.weight().0))
                .enumerate()
            {
                encoder.set_buffer(i as u64, Some(&buffers[&input][input_index as usize]), 0);
            }
            // set output
            let n_inputs = kernels.edges_directed(node, Direction::Incoming).count();
            for (i, output) in buffers[&node].iter().enumerate() {
                encoder.set_buffer((i + n_inputs) as u64, Some(output), 0);
            }
            // set smem
            if !kernel.smem.is_empty() {
                encoder.set_threadgroup_memory_length(
                    0,
                    (kernel.smem.to_usize().unwrap() * std::mem::size_of::<f32>()) as u64,
                );
            }

            // Set dispatch
            encoder.dispatch_thread_groups(
                MTLSize::new(
                    kernel.grid.0.to_usize().unwrap() as u64,
                    kernel.grid.1.to_usize().unwrap() as u64,
                    kernel.grid.2.to_usize().unwrap() as u64,
                ),
                MTLSize::new(
                    kernel.threadblock.0.to_usize().unwrap() as u64,
                    kernel.threadblock.1.to_usize().unwrap() as u64,
                    kernel.threadblock.2.to_usize().unwrap() as u64,
                ),
            );
            encoder.end_encoding();
        }
    }
    panic!("No output kernel detected in graph!");
}
