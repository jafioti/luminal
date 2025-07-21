use std::usize;

use crate::Kernel;
use crate::run::produce_buffer_map;
use crate::utils::{display_graph, print_kernels};
use crate::{GPUArch, GraphTerm, run::run_graph};
use colored::Colorize;
use egraph_serialize::{ClassId, EGraph, NodeId};
use luminal::prelude::NodeIndex;
use luminal::prelude::petgraph::prelude::StableGraph;
use luminal::prelude::petgraph::{Directed, Direction};
use luminal::shape::{Expression, Term};
use rustc_hash::FxHashMap;

const WARMUP_TRIALS: usize = 1;
const TRIALS: usize = 10;
const MAX_SEARCHED_GRAPHS: usize = 600_000;
const MAX_CYCLES: usize = 1;
const INVALID_IR: &[&str] = &[
    "SwapLoops",
    "TileLoop",
    "UnpadLoop",
    "Unary",
    "Binary",
    "MReplace",
];

type Cost = u128; // Execution time in microseconds

pub fn search(
    egraph: &EGraph,
    inputs: &[(NodeIndex, Vec<f32>)],
    arch: GPUArch,
) -> Option<StableGraph<Kernel, (usize, usize)>> {
    fn recurse<'a>(
        egraph: &'a EGraph,
        current_class: &'a ClassId,
        seen: &mut FxHashMap<&'a NodeId, usize>,
    ) -> (Vec<Vec<&'a NodeId>>, usize) {
        let mut trajectories = vec![];
        let mut total_completed = 0;
        'enode_loop: for enode in &egraph.classes()[current_class].nodes {
            if total_completed >= MAX_SEARCHED_GRAPHS {
                break;
            }
            if INVALID_IR.contains(&egraph.nodes[enode].op.as_str())
                || seen.get(&enode).copied().unwrap_or_default() >= MAX_CYCLES
            {
                // Either this is an invalid enode or we've cycled too many times
                continue;
            }
            let mut enode_trajectories = vec![];
            let mut completed = 0;
            for child in &egraph.nodes[enode].children {
                // Ask what's the child's trajectories
                *seen.entry(enode).or_insert(0) += 1;
                let (child_trajectories, child_completed) =
                    recurse(egraph, egraph.nid_to_cid(child), seen);
                *seen.get_mut(&enode).unwrap() -= 1;
                completed = completed
                    .max(child_completed.min(1))
                    .saturating_mul(child_completed);

                // if we are getting no trajectories out of this child, this whole enode is bad
                if child_trajectories.is_empty() {
                    continue 'enode_loop;
                }

                if enode_trajectories.is_empty() {
                    for mut child_trajectory in child_trajectories {
                        child_trajectory.insert(0, enode);
                        enode_trajectories.push(child_trajectory);
                    }
                } else {
                    // Cartisian product the current trajectories with the new trajectories
                    let mut new_enode_trajectories = vec![];
                    for past_trajectory in &enode_trajectories {
                        for new_trajectory in &child_trajectories {
                            new_enode_trajectories
                                .push([past_trajectory.clone(), new_trajectory.clone()].concat());
                        }
                    }
                    if !new_enode_trajectories.is_empty() {
                        enode_trajectories = new_enode_trajectories;
                    }
                }
            }

            if egraph.nodes[enode].children.is_empty() {
                // Leaf node → single-element trajectory
                trajectories.push(vec![enode]);
                completed += 1;
            } else {
                // Add combined trajectories
                trajectories.extend(enode_trajectories);
            }
            total_completed += completed;
        }
        (trajectories, total_completed)
    }

    let (trajectories, _) = recurse(egraph, &egraph.root_eclasses[0], &mut FxHashMap::default());
    // Now we have DFS trajectories
    let mut ref_outputs: Vec<Vec<f32>> = vec![];
    let mut best_time = u128::MAX;
    let mut best_graph = None;
    let mut valid_graphs = 0;
    let total_trajectories = trajectories.len().min(MAX_SEARCHED_GRAPHS);
    let mut min_kernels = usize::MAX;
    'trajectory_loop: for (n, trajectory) in trajectories
        .into_iter()
        .take(MAX_SEARCHED_GRAPHS)
        .enumerate()
    {
        // Build termdag
        let graph = extraction_to_graph(egraph, &trajectory);
        let root = graph.externals(Direction::Outgoing).next().unwrap();
        // display_graph(&graph, &[]);
        let Some(kernels) = crate::codegen::codegen(graph.clone(), vec![root], arch.clone(), 0)
        else {
            continue;
        };
        if kernels.node_count() == 1 {
            display_graph(&graph, &[]);
            panic!();
        }
        // Print kernels
        if option_env!("PRINT_KERNELS").is_some() {
            print_kernels(&kernels);
        }
        match &arch {
            GPUArch::CUDA => {
                valid_graphs += 1;
                if option_env!("PRINT_KERNELS").is_some() {
                    println!(
                        "{}",
                        format!(
                            "Graph {valid_graphs} ({:.1}%) ",
                            (n as f32 / total_trajectories as f32) * 100.0
                        )
                        .bold()
                    );
                }
            }
            GPUArch::Metal(_) => {
                // if kernels.node_count() - 2 > 1 || graph.node_count() >= 27 {
                //     continue;
                // }
                // println!("KERNEL NUMBER {n}");
                // display_graph(&graph, &[]);
                if let Some((us, outs)) = cost(&kernels, inputs) {
                    valid_graphs += 1;
                    if option_env!("PRINT_KERNELS").is_some() {
                        println!(
                            "{}{}",
                            format!(
                                "Graph {valid_graphs} ({:.1}%) ",
                                (n as f32 / total_trajectories as f32) * 100.0
                            )
                            .bold(),
                            format!("{us}µs").bright_green().bold()
                        );
                    }
                    if ref_outputs.is_empty() {
                        ref_outputs = outs;
                    } else {
                        for (a, b) in ref_outputs.iter().zip(&outs) {
                            for (x, y) in a.iter().zip(b) {
                                if (x - y).abs() >= 1e-3 {
                                    if option_env!("PRINT_KERNELS").is_some() {
                                        println!(
                                            "{} {x} != {y}",
                                            "Output Mismatch".on_bright_red()
                                        );
                                    }
                                    continue 'trajectory_loop;
                                }
                            }
                        }
                        if option_env!("PRINT_KERNELS").is_some() {
                            println!("{}", "Outputs Validated".on_bright_green());
                        }
                    }
                    if kernels.node_count() - 2 < min_kernels {
                        min_kernels = kernels.node_count() - 2;
                        best_graph = Some(kernels);
                    }
                }
            }
        }
        // if min_kernels == 1 {
        //     break;
        // }
    }
    println!("MIN: {min_kernels}");
    best_graph
}

pub fn extraction_to_graph(
    egraph: &EGraph,
    trajectory: &[&NodeId],
) -> StableGraph<GraphTerm, (), Directed> {
    let mut g: StableGraph<GraphTerm, (), Directed> = StableGraph::new();

    enum Ret {
        Expr(NodeIndex),
        Math(Expression),
    }

    fn recurse(
        egraph: &EGraph,
        trajectory: &[&NodeId],
        current: &mut usize,
        g: &mut StableGraph<GraphTerm, (), Directed>,
    ) -> Ret {
        let node_choice = trajectory[*current];
        let enode = &egraph.nodes[node_choice];
        // println!("{}", enode.op);
        match enode.op.as_str() {
            "GMEM" => {
                *current += 1;
                Ret::Expr(
                    g.add_node(GraphTerm::GMEM {
                        label: Some(
                            egraph.nodes[&enode.children[0]]
                                .op
                                .replace("Boxed(\"", "")
                                .replace("\")", ""),
                        ),
                    }),
                )
            }
            "SMEM" => Ret::Expr(g.add_node(GraphTerm::SMEM)),

            // LoopIn  = (LoopIn <expr> <LoopType> <Math>)
            "LoopIn" | "LoopOut" => {
                *current += 1;
                let Ret::Expr(child_one) = recurse(egraph, trajectory, current, g) else {
                    panic!()
                };
                *current += 1;
                let Ret::Math(range) = recurse(egraph, trajectory, current, g) else {
                    panic!()
                };
                *current += 1;
                let Ret::Math(stride) = recurse(egraph, trajectory, current, g) else {
                    panic!()
                };
                let r = g.add_node(match enode.op.as_str() {
                    "LoopIn" => GraphTerm::LoopIn {
                        range,
                        stride,
                        marker: "".to_string(),
                    },
                    "LoopOut" => GraphTerm::LoopOut {
                        range,
                        stride,
                        marker: "".to_string(),
                    },
                    _ => panic!(),
                });
                g.add_edge(child_one, r, ());
                Ret::Expr(r)
            }

            "Add" | "Mul" | "Max" | "SMEMLoad" | "SMEMRead" => {
                *current += 1;
                let Ret::Expr(child_one) = recurse(egraph, trajectory, current, g) else {
                    panic!()
                };
                *current += 1;
                // println!("bin: {}", enode.op);
                let Ret::Expr(child_two) = recurse(egraph, trajectory, current, g) else {
                    panic!()
                };
                let r = g.add_node(match enode.op.as_str() {
                    "SMEMLoad" => GraphTerm::SMEMLoad,
                    "SMEMRead" => GraphTerm::SMEMRead,
                    "Add" => GraphTerm::Add,
                    "Mul" => GraphTerm::Mul,
                    "Max" => GraphTerm::Max,
                    _ => panic!(),
                });
                g.add_edge(child_one, r, ());
                g.add_edge(child_two, r, ());
                Ret::Expr(r)
            }
            "Exp" | "Sin" | "Recip" | "Neg" | "Sqrt" => {
                *current += 1;
                let Ret::Expr(child_one) = recurse(egraph, trajectory, current, g) else {
                    panic!()
                };
                let r = g.add_node(match enode.op.as_str() {
                    "Exp2" => GraphTerm::Exp2,
                    "Log2" => GraphTerm::Log2,
                    "Sin" => GraphTerm::Sin,
                    "Recip" => GraphTerm::Recip,
                    "Neg" => GraphTerm::Neg,
                    "Sqrt" => GraphTerm::Sqrt,
                    _ => panic!(),
                });
                g.add_edge(child_one, r, ());
                Ret::Expr(r)
            }
            // ----------- literals & vars -----------
            op if op.starts_with("MNum:") => {
                let num: i64 = op["MNum:".len()..].parse().expect("invalid MNum literal");
                Ret::Math(Expression::from(num as usize))
            }
            op if op.starts_with("MVar:") => {
                let name = op["MVar:".len()..].to_owned();
                Ret::Math(Expression::from(name.chars().next().unwrap()))
            }
            op if op.starts_with("Boxed(\"") => {
                let name = op.replace("Boxed(\"", "").replace("\")", "");
                Ret::Math(Expression::from(name.chars().next().unwrap()))
            }

            // ----------- unary ops -----------
            "MNeg" | "MRecip" => {
                *current += 1;
                let Ret::Math(c0) = recurse(egraph, trajectory, current, g) else {
                    panic!()
                };
                Ret::Math(match enode.op.as_str() {
                    "MNeg" => c0 * -1,
                    "MRecip" => 1 / c0,
                    _ => unreachable!(),
                })
            }

            // ----------- binary ops -----------
            "MAdd" | "MSub" | "MMul" | "MDiv" | "MMod" | "MMin" | "MMax" | "MAnd" | "MOr"
            | "MGte" | "MLt" | "MFloorTo" => {
                *current += 1;
                let Ret::Math(lhs) = recurse(egraph, trajectory, current, g) else {
                    panic!()
                };
                *current += 1;
                let Ret::Math(rhs) = recurse(egraph, trajectory, current, g) else {
                    panic!()
                };
                Ret::Math(match enode.op.as_str() {
                    "MAdd" => lhs + rhs,
                    "MSub" => lhs - rhs,
                    "MMul" => lhs * rhs,
                    "MDiv" => lhs / rhs,
                    "MMod" => lhs % rhs,
                    "MMin" => lhs.min(rhs),
                    "MMax" => lhs.max(rhs),
                    "MAnd" => lhs & rhs,
                    "MOr" => lhs | rhs,
                    "MFloorTo" => lhs / rhs * rhs, // NOT CORRECT, NEED FLOORTO IN EXPRESSIONS
                    _ => unreachable!(),
                })
            }
            "MAccum" => {
                *current += 1;
                Ret::Math(Expression::from(Term::Acc('a')))
            }
            "Loop" => {
                *current += 2; // skip loop label
                recurse(egraph, trajectory, current, g)
            }
            "MNum" | "MVar" => {
                *current += 1;
                recurse(egraph, trajectory, current, g)
            }
            _ => {
                if let Ok(n) = enode.op.parse::<usize>() {
                    Ret::Math(Expression::from(n))
                } else {
                    panic!("unsupported op '{}'", enode.op)
                }
            }
        }
    }

    recurse(egraph, trajectory, &mut 0, &mut g);
    g
}

fn cost<'a>(
    kernels: &StableGraph<Kernel, (usize, usize), Directed>,
    inputs: &[(NodeIndex, Vec<f32>)],
) -> Option<(Cost, Vec<Vec<f32>>)> {
    todo!()
    // // Get buffer info
    // // let (buffer_sizes, buffer_map) = produce_buffer_map(kernels);
    // // Warm up resources (buffer allocation, kernel compiler, etc.)
    // for _ in 0..WARMUP_TRIALS {
    //     run_graph(inputs, &kernels, &FxHashMap::default());
    // }
    // // Test runtime
    // let mut micros = vec![];
    // let mut outputs = vec![];
    // let mut m;
    // for _ in 0..TRIALS {
    //     (outputs, m) = run_graph(inputs, &kernels, &FxHashMap::default());
    //     micros.push(m);
    // }
    // Some((micros.into_iter().sum::<u128>() / TRIALS as u128, outputs))
}
