use std::collections::HashMap;
use std::usize;

use crate::Kernel;
use crate::run::{assign_buffers, compile_kernels, run_graph};
use crate::utils::{display_graph, print_kernels};
use crate::{GPUArch, GraphTerm};
use colored::Colorize;
use egraph_serialize::{ClassId, EGraph, NodeId};
use itertools::Itertools;
use luminal::prelude::NodeIndex;
use luminal::prelude::petgraph::prelude::StableGraph;
use luminal::prelude::petgraph::{Directed, Direction};
use luminal::shape::{Expression, Term};
use metal_rs::objc::rc::autoreleasepool;
use metal_rs::{Buffer, Device, MTLResourceOptions};
use rand::{Rng, rng};
use rustc_hash::{FxHashMap, FxHashSet};

const WARMUP_TRIALS: usize = 0;
const TRIALS: usize = 1;
const MAX_SEARCHED_GRAPHS: usize = 10_000;
const MAX_CYCLES: usize = 1;
const INVALID_IR: &[&str] = &[
    "SwapLoops",
    "TileLoop",
    "UnpadLoop",
    "Unary",
    "Binary",
    "MReplace",
    "MergeLoops",
];

type Cost = u128; // Execution time in microseconds

fn is_expression_enode(enode_label: &str) -> bool {
    matches!(
        enode_label,
        "MNum"
            | "MVar"
            | "MAdd"
            | "MSub"
            | "MMul"
            | "MDiv"
            | "MMod"
            | "MMin"
            | "MMax"
            | "MAnd"
            | "MOr"
            | "MGte"
            | "MLt"
            | "MFloorTo"
            | "MReplace"
            | "MAccum"
    ) || enode_label.starts_with("MNum:")
        || enode_label.starts_with("MVar:")
}

fn shortest_from_enode<'a>(
    egraph: &'a EGraph,
    enode: &'a NodeId,
    seen: &mut FxHashMap<&'a NodeId, usize>,
    junk: &mut FxHashSet<&'a NodeId>,
    cache: &mut FxHashMap<&'a NodeId, Option<Vec<&'a NodeId>>>,
) -> Option<Vec<&'a NodeId>> {
    if let Some(cached) = cache.get(enode) {
        return cached.clone();
    }
    if INVALID_IR.contains(&egraph.nodes[enode].op.as_str()) || junk.contains(enode) {
        cache.insert(enode, None);
        return None;
    }
    if seen.get(&enode).copied().unwrap_or(0) >= MAX_CYCLES {
        cache.insert(enode, None);
        return None;
    }

    *seen.entry(enode).or_insert(0) += 1;

    let out = if egraph.nodes[enode].children.is_empty() {
        // Leaf → path is just this enode
        Some(vec![enode])
    } else {
        // For each child class, take its shortest; if any child has no path → this enode invalid
        let mut acc: Vec<&'a NodeId> = vec![enode];
        let mut ok = true;

        for child in &egraph.nodes[enode].children {
            let child_class = egraph.nid_to_cid(child);
            if let Some(child_path) = extract_shortest(egraph, child_class, seen, junk, cache) {
                acc.extend(child_path);
            } else {
                ok = false;
                break;
            }
        }

        if ok { Some(acc) } else { None }
    };

    *seen.get_mut(&enode).unwrap() -= 1;

    if out.is_none() {
        junk.insert(enode);
    }
    cache.insert(enode, out.clone());
    out
}

pub fn extract_shortest<'a>(
    egraph: &'a EGraph,
    class: &'a ClassId,
    seen: &mut FxHashMap<&'a NodeId, usize>,
    junk: &mut FxHashSet<&'a NodeId>,
    cache: &mut FxHashMap<&'a NodeId, Option<Vec<&'a NodeId>>>,
) -> Option<Vec<&'a NodeId>> {
    // Try all enodes in the class and keep the shortest
    let mut best: Option<Vec<&'a NodeId>> = None;
    for enode in &egraph.classes()[class].nodes {
        if INVALID_IR.contains(&egraph.nodes[enode].op.as_str()) || junk.contains(enode) {
            junk.insert(enode);
            continue;
        }
        if seen.get(&enode).copied().unwrap_or(0) >= MAX_CYCLES {
            continue;
        }

        if let Some(path) = shortest_from_enode(egraph, enode, seen, junk, cache) {
            if best.as_ref().map_or(true, |b| path.len() < b.len()) {
                best = Some(path);
            }
        } else {
            junk.insert(enode);
        }
    }
    best
}

fn extract_trajectories<'a>(
    egraph: &'a EGraph,
    current_class: &'a ClassId,
    seen: &mut FxHashMap<&'a NodeId, usize>,
    junk_cache: &mut FxHashSet<&'a NodeId>,
    trajectory_cache: &mut FxHashMap<&'a NodeId, Vec<Vec<&'a NodeId>>>,
    waiting: usize,
) -> Vec<Vec<&'a NodeId>> {
    let mut trajectories = vec![];
    'enode_loop: for enode in &egraph.classes()[current_class].nodes {
        if INVALID_IR.contains(&egraph.nodes[enode].op.as_str()) {
            junk_cache.insert(enode);
            continue;
        } else if junk_cache.contains(&enode)
            || seen.get(&enode).copied().unwrap_or_default() >= MAX_CYCLES
        {
            continue;
        }
        let mut enode_trajectories = vec![];
        *seen.entry(enode).or_insert(0) += 1;
        for child in &egraph.nodes[enode].children {
            // Ask what's the child's trajectories
            if !trajectory_cache.contains_key(child) {
                let child_trajectories = if is_expression_enode(&egraph.nodes[child].op) {
                    extract_shortest(
                        egraph,
                        egraph.nid_to_cid(child),
                        seen,
                        junk_cache,
                        &mut FxHashMap::default(),
                    )
                    .map(|i| vec![i])
                    .unwrap_or_default()
                } else {
                    extract_trajectories(
                        egraph,
                        egraph.nid_to_cid(child),
                        seen,
                        junk_cache,
                        trajectory_cache,
                        (waiting * enode_trajectories.len().max(1)) + trajectories.len(),
                    )
                };
                if child_trajectories.is_empty() {
                    // bad enode
                    junk_cache.insert(enode);
                    *seen.get_mut(&enode).unwrap() -= 1;
                    continue 'enode_loop;
                }
                trajectory_cache.insert(child, child_trajectories.clone());
            }

            if enode_trajectories.is_empty() {
                // First child
                for mut child_trajectory in trajectory_cache[child].clone() {
                    child_trajectory.insert(0, enode);
                    enode_trajectories.push(child_trajectory);
                }
            } else if !trajectory_cache[child].is_empty() {
                // Cartisian product the current trajectories with the new trajectories
                enode_trajectories = enode_trajectories
                    .into_iter()
                    .cartesian_product(&trajectory_cache[child])
                    .map(|(p, n)| [p, n.clone()].concat())
                    .collect();
            }
        }
        *seen.get_mut(&enode).unwrap() -= 1;

        if egraph.nodes[enode].children.is_empty() {
            // Leaf node → single-element trajectory
            trajectories.push(vec![enode]);
        } else {
            // Add combined trajectories
            trajectories.extend(enode_trajectories);
        }
        if trajectories.len() * waiting > MAX_SEARCHED_GRAPHS {
            break; // Only pick the first valid (non cycling) enode for expressions
        }
    }
    trajectories
}

pub fn search(
    egraph: &EGraph,
    inputs: &[(String, Vec<f32>)],
    arch: GPUArch,
    dyn_vars: &FxHashMap<char, usize>,
) -> Option<StableGraph<GraphTerm, ()>> {

    let trajectories = extract_trajectories(
        egraph,
        &egraph.root_eclasses[0],
        &mut FxHashMap::default(),
        &mut FxHashSet::default(),
        &mut FxHashMap::default(),
        1,
    );

    // Now we have DFS trajectories
    let mut ref_outputs: Vec<Vec<f32>> = vec![];
    let mut best_time = u128::MAX;
    let mut shortest = "".to_string();
    let mut best_graph = None;
    let mut valid_graphs = 0;
    let total_trajectories = trajectories.len().min(MAX_SEARCHED_GRAPHS);
    let mut ui_functions = None;
    if option_env!("DISPLAY").is_some() {
        ui_functions = Some(crate::utils::search_ui());
    };
    'trajectory_loop: for (n, trajectory) in trajectories
        .into_iter()
        .take(MAX_SEARCHED_GRAPHS)
        .enumerate()
    {
        // Build termdag
        let graph = extraction_to_graph(egraph, &trajectory);
        // convert inputs to reference nodes in graph
        let inputs = inputs.into_iter().map(|(l, d)| (graph.node_indices().find(|n| matches!(graph.node_weight(*n).unwrap(), GraphTerm::GMEM { label } if label == l)).unwrap(), d.clone())).collect_vec();
        let root = graph.externals(Direction::Outgoing).next().unwrap();
        let Some((kernels, gmem_mapping)) =
            crate::codegen::codegen(graph.clone(), vec![root], arch.clone(), 0, dyn_vars, false)
        else {
            continue;
        };
        match &arch {
            GPUArch::CUDA => {
                if let Some((_, s, _, _)) = &ui_functions {
                    s(format!(
                        "Graph {valid_graphs} ({:.1}%) ",
                        (n as f32 / total_trajectories as f32) * 100.0
                    ));
                }
            }
            GPUArch::Metal(_) => {
                if let Some((us, outs)) = cost(&kernels, &inputs, &gmem_mapping, dyn_vars) {
                    valid_graphs += 1;
                    if let Some((progress, logs, title, _)) = &ui_functions {
                        progress(((n as f32 / total_trajectories as f32) * 100.0) as u16);
                        logs(print_kernels(&kernels));
                        title(format!("Graph {valid_graphs} {us}µs"));
                    } else if option_env!("DEBUG").is_some() {
                        println!("{}", print_kernels(&kernels));
                        println!("Graph {valid_graphs} {us}µs");
                    }
                    if ref_outputs.is_empty() {
                        ref_outputs = outs;
                    } else {
                        for (a, b) in ref_outputs.iter().zip(&outs) {
                            for (x, y) in a.iter().zip(b) {
                                if (x - y).abs() >= 1e-3 {
                                    if option_env!("DEBUG").is_some() {
                                        println!("REF: {:?} New: {:?}", ref_outputs, outs);
                                        // display_graph(&graph, &[]);
                                        println!(
                                            "{} {x} != {y}",
                                            "Output Mismatch".bold().on_bright_red()
                                        );
                                    }
                                    continue 'trajectory_loop;
                                }
                            }
                        }
                        if option_env!("DEBUG").is_some() {
                            println!("{}", "Outputs Validated".bold().on_bright_green());
                        }
                    }
                    let kernel_string = print_kernels(&kernels);
                    // if kernel_string.len() < shortest.len() || shortest.is_empty() {

                    // }
                    if us < best_time {
                        best_time = us;
                        best_graph = Some(graph);
                        shortest = kernel_string;
                    }
                }
            }
        }
    }
    if let Some((_, _, _, e)) = &ui_functions {
        e();
    }
    println!("SHORTEST ({}ms): {shortest}", best_time / 1000);
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
        Loop(String, Expression),
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
                        label: egraph.nodes[&enode.children[0]]
                            .op
                            .replace("Boxed(\"", "")
                            .replace("\")", ""),
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
                let Ret::Loop(label, range) = recurse(egraph, trajectory, current, g) else {
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
                        marker: label,
                    },
                    "LoopOut" => GraphTerm::LoopOut {
                        range,
                        stride,
                        marker: label,
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
                let label = egraph.nodes[trajectory[*current + 1]]
                    .op
                    .replace("Boxed(\"", "")
                    .replace("\")", "");
                *current += 2; // skip loop label
                let Ret::Math(e) = recurse(egraph, trajectory, current, g) else {
                    panic!()
                };
                Ret::Loop(label, e)
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
    gmem_mapping: &HashMap<NodeIndex, usize>,
    dyn_vars: &FxHashMap<char, usize>,
) -> Option<(Cost, Vec<Vec<f32>>)> {
    autoreleasepool(|| {
        // Get buffer info
        let (int_buffers, int_buffer_map) = assign_buffers(&kernels);
        let compiled_kernels = compile_kernels(&kernels);
        let device = Device::system_default().unwrap();
        // Copy input buffers over
        let mut inputs = inputs
            .into_iter()
            .map(|(n, b)| (gmem_mapping[n], (copy_metal_buffer(b, &device), false)))
            .collect::<FxHashMap<_, _>>();
        // Warm up resources (buffer allocation, kernel compiler, etc.)
        for _ in 0..WARMUP_TRIALS {
            run_graph(
                &mut inputs,
                &kernels,
                dyn_vars,
                &compiled_kernels,
                &int_buffers,
                &int_buffer_map,
            );
        }
        // Test runtime
        let mut micros = vec![];
        let mut outputs = vec![];
        let mut m;
        for _ in 0..TRIALS {
            (outputs, m) = run_graph(
                &mut inputs,
                &kernels,
                dyn_vars,
                &compiled_kernels,
                &int_buffers,
                &int_buffer_map,
            );
            micros.push(m);
        }
        Some((
            micros.into_iter().sum::<u128>() / TRIALS as u128,
            outputs.iter().map(copy_metal_buffer_back).collect_vec(),
        ))
    })
}

pub fn copy_metal_buffer(v: &Vec<f32>, device: &Device) -> Buffer {
    assert!(v.len() > 0);
    let buf = device.new_buffer_with_data(
        v.as_ptr() as *const _,
        (v.len() * std::mem::size_of::<f32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    buf
}
pub fn copy_metal_buffer_back(v: &Buffer) -> Vec<f32> {
    let mut data = vec![0f32; v.length() as usize / size_of::<f32>()];
    let ptr = v.contents() as *mut f32;
    for (i, d) in data.iter_mut().enumerate() {
        *d = unsafe { *ptr.add(i) };
    }
    data
}

pub fn make_test_inputs(
    graph: &StableGraph<GraphTerm, ()>,
    dyn_map: &FxHashMap<char, usize>,
) -> Vec<(String, Vec<f32>)> {
    // Go through each GMEM and work out the size
    let mut inputs = vec![];
    let mut rng = rng();
    for node in graph.externals(Direction::Incoming) {
        if let GraphTerm::GMEM { label } = graph.node_weight(node).unwrap() {
            // Walk down the loopins to find the max size
            let mut size = Expression::from(0);
            let mut curr = graph
                .neighbors_directed(node, Direction::Outgoing)
                .next()
                .unwrap();
            loop {
                if let GraphTerm::LoopIn { range, stride, .. } = graph.node_weight(curr).unwrap() {
                    size = size.max(stride.substitute('z', *range - 1) + 1);
                    curr = graph
                        .neighbors_directed(curr, Direction::Outgoing)
                        .next()
                        .unwrap();
                } else {
                    break;
                }
            }
            inputs.push((
                label.clone(),
                (0..size.exec(&dyn_map).unwrap())
                    .map(|_| rng.random())
                    .collect(),
            ));
        }
    }
    inputs
}
