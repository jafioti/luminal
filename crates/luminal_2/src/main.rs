// TODO
// get simple codegen working
// get complex codegen working
// get profiling working
// get brute force extraction working
//
// write rewrite trajectory for optimal simdgroup matmul
// write flash attention in ir
// write rewrite trajector for flash attention
// Put flattened IR into egglog
// If flattened IR doesn't go into egglog, put nested IR into egglog and write flattening function
// write scoring function (profiling based)
// write rewrite rules in egglog
// get optimal matrix multiplication generated automatically
// get flash attention generated automatically
// get one-layer transformer generated
// get full llm generated
// post update and spread the good word

mod kernels;

use colored::Colorize;
use egglog::{EGraph, Error, Term, TermDag, TermId, ast::Literal, var};
use itertools::Itertools;
use petgraph::{
    Directed, Direction,
    graph::NodeIndex,
    prelude::{Dfs, StableGraph},
    visit::{EdgeRef, NodeRef},
};
use regex::Regex;
use std::{
    collections::{HashMap, HashSet, VecDeque},
    fmt::{Debug, Display, format},
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
    Tensor { name: String },
    Smem,
    LoopIn { range: String, stride: String },
    LoopOut { range: String, stride: String },
    Add,
    Mul,
    Exp,
    Sin,
    SmemCopy,
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
            GraphTerm::Exp => "Exp".to_string(),
            GraphTerm::Sin => "Sin".to_string(),
            GraphTerm::LoopIn { range, stride } => format!("LoopIn ({range}; {stride})"),
            GraphTerm::LoopOut { range, stride } => format!("LoopOut ({range}; {stride})"),
            GraphTerm::Tensor { name } => format!("Tensor({name})"),
            GraphTerm::Smem => "SMEM".to_string(),
            GraphTerm::SmemCopy => "SmemCopy".to_string(),
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
        format!("{} [{}]", self.0.term_to_string(), self.1.len())
    }
}

fn main() {
    let (g, r) = kernels::make_tiled_matmul_basic();
    match run_egglog_program(include_str!("code.lisp")) {
        Ok((s, _serialized, termdag, root)) => {
            if s.is_empty() {
                println!("{}", "Success!".bright_green().bold())
            } else {
                println!("{}", format!("Success: {s:?}").bright_green().bold())
            }
            // let (graph, root) = dag_to_petgraph(&termdag, termdag.lookup(&root));
            codegen(g, r);
        }
        Err(e) => println!("{e}"),
    }
}

fn validate_graph(graph: &StableGraph<(GraphTerm, usize), u8, Directed>) {
    // walk the graph and make sure loopins -> next loop level (or loopout) and prev loop (or loopin) -> loopout
    for node in graph.node_indices() {
        let (curr_term, curr_level) = graph.node_weight(node).unwrap();
        if matches!(curr_term, GraphTerm::LoopIn { .. }) {
            for new_node in graph.neighbors_directed(node, Direction::Outgoing) {
                let (new_term, new_level) = graph.node_weight(new_node).unwrap();
                if !matches!(new_term, GraphTerm::LoopOut { .. }) {
                    if *new_level != *curr_level + 1 {
                        display_graph(graph, &[node, new_node]);
                        panic!("incorrect levels");
                    }
                }
            }
        } else if matches!(curr_term, GraphTerm::LoopOut { .. }) {
            for new_node in graph.neighbors_directed(node, Direction::Incoming) {
                let (new_term, new_level) = graph.node_weight(new_node).unwrap();
                if !matches!(new_term, GraphTerm::LoopIn { .. }) {
                    if *new_level != *curr_level + 1 {
                        display_graph(graph, &[node, new_node]);
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
                        display_graph(graph, &[node, new_node]);
                        panic!("incorrect levels");
                    }
                }
            }
        }
    }
}

fn codegen(graph: StableGraph<GraphTerm, u8, Directed>, root: NodeIndex) -> (String, TermId) {
    let (kernels, _root_kernel) = split_kernels(graph, root);

    for (n_kernel, (kernel_graph, inputs, outputs, smem_buffers)) in kernels.into_iter().enumerate()
    {
        validate_graph(&kernel_graph);
        // display_graph(&kernel_graph, &[]);
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
        let kernel = make_kernel(
            &kernel_graph,
            inputs
                .iter()
                .cloned()
                .chain(smem_buffers.iter().map(|(a, b, _)| (*a, *b)))
                .collect(),
            outputs.clone(),
            &mut HashMap::new(),
            &mut (inputs.len() + outputs.len()),
            &mut loop_levels,
        );
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
            "extern \"C\" __global__ void kernel{n_kernel}({}) {{
{}{}{}
}}",
            inputs
                .into_iter()
                .chain(outputs)
                .map(|(a, _)| format!("float* {}", var_to_char(a)))
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
    inputs: Vec<(usize, NodeIndex)>,
    outputs: Vec<(usize, NodeIndex)>,
    node_to_var: &mut HashMap<NodeIndex, (usize, bool)>,
    prev_max_var: &mut usize, // contains the char last used
    loop_levels: &mut Vec<String>,
) -> Vec<String> {
    let mut kernel_lines = vec![];
    let mut loop_body_covered = HashSet::new();
    // toposort down the graph
    let toposorted = partial_toposort(
        kernel_graph,
        &inputs.iter().map(|(_, i)| *i).collect_vec(),
        &outputs.iter().map(|(_, i)| *i).collect_vec(),
    );
    for node in toposorted {
        println!("Node: {:?}", node.index());
        if loop_body_covered.contains(&node) {
            continue;
        }
        let (term, loop_level) = kernel_graph.node_weight(node).unwrap();
        match term {
            GraphTerm::LoopIn { range, .. } => {
                // go through graph to find inputs and outputs
                let mut new_loop_body_covered = HashSet::new();
                let mut loop_outputs = vec![];
                let mut body_outputs = vec![];
                let mut dfs = vec![node];
                while let Some(n) = dfs.pop() {
                    if let GraphTerm::LoopOut { stride, .. } =
                        &kernel_graph.node_weight(n).unwrap().0
                    {
                        if kernel_graph.node_weight(n).unwrap().1 == *loop_level {
                            loop_outputs.push((n, stride));
                            body_outputs.push(
                                kernel_graph
                                    .neighbors_directed(n, Direction::Incoming)
                                    .next()
                                    .unwrap(),
                            );
                        } else {
                            dfs.extend(kernel_graph.neighbors_directed(n, Direction::Outgoing));
                        }
                    } else {
                        dfs.extend(kernel_graph.neighbors_directed(n, Direction::Outgoing));
                    }
                    new_loop_body_covered.insert(n);
                }
                dfs.clear();
                dfs.push(loop_outputs[0].0);
                let mut loop_inputs = vec![];
                let mut body_inputs = vec![];
                while let Some(n) = dfs.pop() {
                    if let GraphTerm::LoopIn { stride, .. } =
                        &kernel_graph.node_weight(n).unwrap().0
                    {
                        if kernel_graph.node_weight(n).unwrap().1 == *loop_level {
                            loop_inputs.push((n, stride));
                            body_inputs.push(
                                kernel_graph
                                    .neighbors_directed(n, Direction::Outgoing)
                                    .next()
                                    .unwrap(),
                            );
                        } else {
                            dfs.extend(kernel_graph.neighbors_directed(n, Direction::Incoming));
                        }
                    } else {
                        dfs.extend(kernel_graph.neighbors_directed(n, Direction::Incoming));
                    }
                    new_loop_body_covered.insert(n);
                }

                if loop_inputs.iter().any(|(i, _)| {
                    if let Some(inp) = kernel_graph
                        .neighbors_directed(*i, Direction::Incoming)
                        .next()
                    {
                        !node_to_var.contains_key(&inp) // no recorded key
                    } else {
                        inputs.iter().all(|(_, a)| *a != *i) // not in input
                    }
                }) {
                    continue;
                }
                loop_body_covered.extend(new_loop_body_covered);

                // Make new loop
                let loop_kernel_line = kernel_lines.len();
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

                // Move input pointers (allocate new variables)
                let mut new_vars = vec![];
                for (input, stride) in &loop_inputs {
                    if stride.contains("Acc") {
                        assert!(
                            *loop_level > 5,
                            "No accumulations allowed on grid or threadblock levels!"
                        );
                        *prev_max_var += 1;
                        // Create accumulator
                        let src = kernel_graph
                            .neighbors_directed(*input, Direction::Incoming)
                            .next()
                            .unwrap();
                        kernel_lines.insert(
                            loop_kernel_line,
                            format!(
                                "{}float {} = {}{};",
                                (0..loop_level.saturating_sub(6)).map(|_| "\t").join(""),
                                var_to_char(*prev_max_var),
                                if node_to_var[&src].1 { "*" } else { "" },
                                var_to_char(node_to_var[&src].0)
                            ),
                        );
                        new_vars.push(*prev_max_var);
                        node_to_var.insert(*input, (*prev_max_var, false));
                        continue;
                    }
                    let real_input = if let Some(n) = kernel_graph
                        .neighbors_directed(*input, Direction::Incoming)
                        .next()
                    {
                        node_to_var[&n].0
                    } else {
                        inputs
                            .iter()
                            .find_map(|(i, n)| if *n == *input { Some(*i) } else { None })
                            .unwrap()
                    };
                    if *stride != "0" {
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
                        node_to_var.insert(*input, (*prev_max_var, true));
                    } else {
                        new_vars.push(real_input);
                        node_to_var.insert(*input, (real_input, true));
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
                        let (input, _) = loop_inputs.iter().find(|(_, s)| **s == **stride).unwrap();
                        let (input_var, _) = node_to_var[input];
                        new_output_vars.push(input_var);
                        node_to_var.insert(*output, (input_var, false));
                        continue;
                    }
                    let real_output = if let Some(v) = node_to_var.get(&output) {
                        v.0
                    } else {
                        outputs
                            .iter()
                            .find_map(|(i, n)| if *n == *output { Some(*i) } else { None })
                            .unwrap()
                    };
                    if *stride != "0" {
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
                        node_to_var.insert(*output, (*prev_max_var, true));
                    } else {
                        new_output_vars.push(real_output);
                        node_to_var.insert(*output, (real_output, true));
                    }
                }
                loop_levels.push(range.to_string());
                let loop_body = make_kernel(
                    kernel_graph,
                    new_vars.into_iter().zip(body_inputs).collect_vec(),
                    new_output_vars
                        .into_iter()
                        .zip(body_outputs.clone())
                        .collect_vec(),
                    node_to_var,
                    prev_max_var,
                    loop_levels,
                );

                kernel_lines.extend(loop_body);

                // Set outputs if nessecary
                for (body_out, (output, _)) in body_outputs.into_iter().zip(loop_outputs) {
                    let save_var = if let GraphTerm::LoopOut { stride, .. } =
                        &kernel_graph.node_weight(body_out).unwrap().0
                    {
                        stride.contains("Acc")
                    } else {
                        true
                    };
                    if save_var {
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

                if *loop_level >= 6 {
                    kernel_lines.push(format!(
                        "{}}}",
                        (0..loop_level.saturating_sub(6)).map(|_| "\t").join("")
                    ));
                }
            }
            GraphTerm::LoopOut { range, stride } => {
                panic!("found loopout range: {range} stride: {stride}")
            }
            GraphTerm::Tensor { .. } | GraphTerm::Smem => {
                node_to_var.insert(
                    node,
                    (
                        inputs
                            .iter()
                            .find(|(_, b)| *b == node)
                            .map(|(a, _)| *a)
                            .unwrap(),
                        true,
                    ),
                );
            }
            GraphTerm::SmemCopy => {
                let srcs = kernel_graph
                    .edges_directed(node, Direction::Incoming)
                    .map(|e| e.id())
                    .sorted()
                    .map(|e| kernel_graph.edge_endpoints(e).unwrap().0)
                    .collect_vec();
                println!("srcs: {:?}", srcs[0]);
                let src_a = node_to_var[&srcs[0]];
                let src_b = node_to_var[&srcs[1]];
                kernel_lines.push(format!(
                    "{}__syncthreads();",
                    (0..loop_level.saturating_sub(6)).map(|_| "\t").join("")
                ));
                kernel_lines.push(format!(
                    "{}*{} = {}{};",
                    (0..loop_level.saturating_sub(6)).map(|_| "\t").join(""),
                    var_to_char(src_a.0),
                    if src_b.1 { "*" } else { "" },
                    var_to_char(src_b.0)
                ));
                kernel_lines.push(format!(
                    "{}__syncthreads();",
                    (0..loop_level.saturating_sub(6)).map(|_| "\t").join("")
                ));
                node_to_var.insert(node, (src_a.0, true));
            }
            GraphTerm::Sin => {
                *prev_max_var += 1;
                let src = node_to_var[&kernel_graph
                    .neighbors_directed(node, Direction::Incoming)
                    .next()
                    .unwrap()];
                node_to_var.insert(node, (*prev_max_var, false));
                kernel_lines.push(format!(
                    "{}float {} = sin({}{});",
                    (0..loop_level.saturating_sub(6)).map(|_| "\t").join(""),
                    var_to_char(*prev_max_var),
                    if src.1 { "*" } else { "" },
                    var_to_char(src.0)
                ));
            }
            GraphTerm::Exp => {
                *prev_max_var += 1;
                let src = node_to_var[&kernel_graph
                    .neighbors_directed(node, Direction::Incoming)
                    .next()
                    .unwrap()];
                node_to_var.insert(node, (*prev_max_var, false));
                kernel_lines.push(format!(
                    "{}float {} = exp({}{});",
                    (0..loop_level.saturating_sub(6)).map(|_| "\t").join(""),
                    var_to_char(*prev_max_var),
                    if src.1 { "*" } else { "" },
                    var_to_char(src.0)
                ));
            }
            GraphTerm::Mul | GraphTerm::Add => {
                *prev_max_var += 1;
                let mut srcs = kernel_graph.neighbors_directed(node, Direction::Incoming);
                let src_a = node_to_var[&srcs.next().unwrap()];
                let src_b = node_to_var[&srcs.next().unwrap()];
                node_to_var.insert(node, (*prev_max_var, false));
                kernel_lines.push(format!(
                    "{}float {} = {}{} {} {}{};",
                    (0..loop_level.saturating_sub(6)).map(|_| "\t").join(""),
                    var_to_char(*prev_max_var),
                    if src_a.1 { "*" } else { "" },
                    var_to_char(src_a.0),
                    match &term {
                        GraphTerm::Add => "+",
                        GraphTerm::Mul => "*",
                        _ => panic!(),
                    },
                    if src_b.1 { "*" } else { "" },
                    var_to_char(src_b.0)
                ));
            }
        }
    }
    kernel_lines
}

fn partial_toposort<N, E>(
    graph: &StableGraph<N, E, Directed>,
    start_nodes: &[NodeIndex],
    end_nodes: &[NodeIndex],
) -> Vec<NodeIndex> {
    // Find valid subgraph nodes
    let reachable_from_start = find_reachable_nodes(graph, start_nodes);
    let can_reach_end = find_nodes_that_can_reach(graph, end_nodes);
    let valid_nodes: HashSet<NodeIndex> = reachable_from_start
        .intersection(&can_reach_end)
        .copied()
        .collect();

    // Calculate in-degrees for valid nodes only
    let mut in_degree: std::collections::HashMap<NodeIndex, usize> =
        valid_nodes.iter().map(|&n| (n, 0)).collect();

    for &node in &valid_nodes {
        for neighbor in graph.neighbors_directed(node, Direction::Outgoing) {
            if valid_nodes.contains(&neighbor) {
                *in_degree.get_mut(&neighbor).unwrap() += 1;
            }
        }
    }

    // Kahn's algorithm for topological sorting
    let mut queue = VecDeque::new();
    let mut result = Vec::new();

    // Start with nodes that have in-degree 0 within our subgraph
    for (&node, &degree) in &in_degree {
        if degree == 0 {
            queue.push_back(node);
        }
    }

    while let Some(current) = queue.pop_front() {
        result.push(current);

        // Reduce in-degree of neighbors
        for neighbor in graph.neighbors_directed(current, Direction::Outgoing) {
            if let Some(degree) = in_degree.get_mut(&neighbor) {
                *degree -= 1;
                if *degree == 0 {
                    queue.push_back(neighbor);
                }
            }
        }
    }

    result
}

fn find_reachable_nodes<N, E>(
    graph: &StableGraph<N, E, Directed>,
    start_nodes: &[NodeIndex],
) -> HashSet<NodeIndex> {
    let mut reachable = HashSet::new();
    let mut queue = VecDeque::new();

    // Add all start nodes
    for &node in start_nodes {
        queue.push_back(node);
        reachable.insert(node);
    }

    // BFS to find all reachable nodes
    while let Some(current) = queue.pop_front() {
        for neighbor in graph.neighbors_directed(current, Direction::Outgoing) {
            if reachable.insert(neighbor) {
                queue.push_back(neighbor);
            }
        }
    }

    reachable
}

fn find_nodes_that_can_reach<N, E>(
    graph: &StableGraph<N, E, Directed>,
    end_nodes: &[NodeIndex],
) -> HashSet<NodeIndex> {
    let mut can_reach = HashSet::new();
    let mut queue = VecDeque::new();

    // Add all end nodes
    for &node in end_nodes {
        queue.push_back(node);
        can_reach.insert(node);
    }

    // Reverse BFS to find all nodes that can reach end nodes
    while let Some(current) = queue.pop_front() {
        for neighbor in graph.neighbors_directed(current, Direction::Incoming) {
            if can_reach.insert(neighbor) {
                queue.push_back(neighbor);
            }
        }
    }

    can_reach
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
                    GraphTerm::Tensor {
                        name: match name {
                            Literal::String(s) => s.as_str().to_string(),
                            _ => panic!(),
                        },
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
        Vec<(Option<usize>, usize, NodeIndex)>, // (src kernel, src kernel output, current graph node)
        Vec<NodeIndex>,                         // output node
        Vec<(usize, NodeIndex, String)>,        // (shared memory buffer name, node, buffer size)
    )>,
    usize, // Root kernel
) {
    // Mark level of ops in graph
    let mut marked_graph = StableGraph::new();
    let mut map = HashMap::<NodeIndex, NodeIndex>::default();
    for node in graph.node_indices() {
        let weight = graph.node_weight(node).unwrap().clone();
        map.insert(node, marked_graph.add_node((weight, vec![], 0_usize)));
    }
    for edge in graph.edge_indices() {
        let (start, end) = graph.edge_endpoints(edge).unwrap();
        let weight = graph.edge_weight(edge).unwrap();
        marked_graph.add_edge(map[&start], map[&end], *weight);
    }
    root = map[&root];

    let mut dfs_nodes = vec![root];
    while let Some(n) = dfs_nodes.pop() {
        let mut max_neighbor = marked_graph
            .neighbors_directed(n, Direction::Outgoing)
            .map(|n| {
                let mut weight = marked_graph.node_weight(n).unwrap().1.clone();
                if let GraphTerm::LoopOut { range, .. } = &marked_graph.node_weight(n).unwrap().0 {
                    weight.push(range.clone())
                }
                weight
            })
            .max_by(|a, b| a.len().cmp(&b.len()))
            .unwrap_or_default();
        if let GraphTerm::LoopIn { .. } = marked_graph.node_weight(n).unwrap().0 {
            max_neighbor.pop();
        }
        marked_graph.node_weight_mut(n).unwrap().1 = max_neighbor;
        dfs_nodes.extend(marked_graph.neighbors_directed(n, Direction::Incoming));
    }

    // Check for groups of kernels
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
            let mut real_kernel = curr_kernel;
            if matches!(term, GraphTerm::LoopIn { .. })
                && matches!(src_term, GraphTerm::LoopOut { .. })
                && curr_level.len() < 3
            {
                real_kernel += 1;
                n_kernels = n_kernels.max(real_kernel + 1);
            } else if !matches!(term, GraphTerm::LoopIn { .. } | GraphTerm::LoopOut { .. })
                && !matches!(
                    src_term,
                    GraphTerm::LoopIn { .. } | GraphTerm::LoopOut { .. }
                )
            {
                assert_eq!(curr_level, *src_level); // Edges need to go between the same levels if neither op is a LoopIn or LoopOut
            }
            if *src_kernel < real_kernel {
                *src_kernel = real_kernel;
                dfs_stack.push(source);
            }
        }
    }

    // Add kernel barriers
    for edge in marked_graph.edge_indices().collect_vec() {
        let (source, dest) = marked_graph.edge_endpoints(edge).unwrap();
        let (_, dest_level, dest_kernel) = marked_graph.node_weight(dest).unwrap().clone();
        let (_, _, src_kernel) = marked_graph.node_weight(source).unwrap().clone();
        if dest_level.len() > 0 && dest_kernel != src_kernel {
            // Put a barrier here
            let (mut src, mut dest) = (dest, source);
            for i in (0..dest_level.len()).rev() {
                let new_src = marked_graph.add_node((
                    GraphTerm::LoopOut {
                        range: dest_level[i].clone(),
                        stride: "0".to_string(),
                    },
                    dest_level[..i].to_vec(),
                    src_kernel,
                ));
                marked_graph.add_edge(src, new_src, 0);
                src = new_src;
                let new_dest = marked_graph.add_node((
                    GraphTerm::LoopIn {
                        range: dest_level[i].clone(),
                        stride: "0".to_string(),
                    },
                    dest_level[..i].to_vec(),
                    dest_kernel,
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
        let (term, loop_level, kernel) = marked_graph.node_weight(node).unwrap();
        let (kernel_graph, _, outputs, _) = &mut kernel_graphs[*kernel];
        node_maps[*kernel].insert(
            node,
            kernel_graph.add_node((term.clone(), loop_level.len())),
        );
        if node == root {
            outputs.push(node_maps[*kernel][&node]);
        }
    }
    // Go through inputs
    for input in marked_graph.node_indices().filter(|n| {
        marked_graph
            .neighbors_directed(*n, Direction::Incoming)
            .next()
            .is_none()
    }) {
        let (t, _, kernel) = marked_graph.node_weight(input).unwrap();
        if !matches!(t, GraphTerm::Smem) {
            kernel_graphs[*kernel].1.push((None, 0, input));
        }
    }
    for edge in marked_graph.edge_indices() {
        let (start, end) = marked_graph.edge_endpoints(edge).unwrap();
        let weight = marked_graph.edge_weight(edge).unwrap();
        let (_, _, start_kernel) = marked_graph.node_weight(start).unwrap();
        let (_, _, end_kernel) = marked_graph.node_weight(end).unwrap();
        if *start_kernel == *end_kernel {
            // Start and end are in the same kernel
            kernel_graphs[*start_kernel].0.add_edge(
                node_maps[*start_kernel][&start],
                node_maps[*start_kernel][&end],
                *weight,
            );
        } else {
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
        }
    }
    // Go through SMEM buffers
    for (kernel_graph, inputs, outputs, smem_buffers) in &mut kernel_graphs {
        for node in kernel_graph.node_indices() {
            if matches!(kernel_graph.node_weight(node).unwrap().0, GraphTerm::Smem) {
                // Walk forward through all loopins to get size of buffer
                let mut curr_node = node;
                let mut buf_size = vec![];
                loop {
                    curr_node = kernel_graph
                        .neighbors_directed(curr_node, Direction::Outgoing)
                        .next()
                        .unwrap();
                    if let GraphTerm::LoopIn { range, stride } =
                        &kernel_graph.node_weight(curr_node).unwrap().0
                    {
                        if stride != "0" {
                            buf_size.push(range);
                        }
                    } else {
                        break;
                    }
                }
                let buf_index = inputs.len() + outputs.len() + smem_buffers.len();
                smem_buffers.push((buf_index, node, buf_size.into_iter().join(" * ")));
            }
        }
    }
    let root_kernel = marked_graph.node_weight(root).unwrap().2;
    (kernel_graphs, root_kernel)
}

/// View a debug graph in the browser
pub fn display_graph<G: TermToString>(
    graph: &petgraph::stable_graph::StableGraph<G, u8, petgraph::Directed, u32>,
    mark_nodes: &[NodeIndex],
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
    for n in mark_nodes {
        graph_string = graph_string.replace(
            &format!("    {} [ label =", n.index()),
            &format!(
                "    {} [ style=\"filled\" fillcolor=\"yellow\" label =",
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

fn var_to_char(var: usize) -> char {
    if var < 26 {
        (var + 97) as u8 as char
    } else if var < 52 {
        (var - 26 + 65) as u8 as char
    } else {
        panic!("Var is too high: {var}");
    }
}
