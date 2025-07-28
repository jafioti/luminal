use std::{
    collections::{BTreeMap, BTreeSet, HashMap},
    fs::File,
    io::Read,
};

use cudarc::{
    driver::{CudaFunction, CudaSlice, LaunchConfig, PushKernelArg},
    nvrtc::CompileOptions,
};
use itertools::Itertools;
use luminal::{
    prelude::{
        NodeIndex,
        petgraph::{
            Direction,
            algo::toposort,
            prelude::StableGraph,
            visit::{EdgeRef, IntoEdgeReferences},
        },
    },
    shape::Expression,
};
use rustc_hash::{FxHashMap, FxHashSet};

use crate::Kernel;

pub fn assign_buffers(
    graph: &StableGraph<Kernel, (usize, usize)>,
) -> (Vec<Expression>, HashMap<NodeIndex, Vec<usize>>) {
    // Count consumers only for producer outputs we manage (exclude "Inputs")
    let mut use_count: HashMap<(NodeIndex, usize), usize> = HashMap::new();
    for e in graph.edge_references() {
        let src = e.source();
        if graph[src].code != "Inputs" {
            let (src_out, _) = *e.weight();
            *use_count.entry((src, src_out)).or_default() += 1;
        }
    }

    let mut master = vec![]; // capacities by global buffer index
    let mut buf_map = HashMap::new(); // node -> output_idx -> buffer_idx
    let mut free_by_cap = HashMap::<Expression, Vec<usize>>::new(); // exact-size reuse

    for node in toposort(graph, None).unwrap() {
        let k = &graph[node];
        if k.code == "Inputs" {
            continue; // user-provided; ignore
        }

        // Allocate exact-size buffers for this node's outputs
        let mut outs = vec![];
        for &cap in &k.outputs {
            let buf_idx = if let Some(idx) = free_by_cap.get_mut(&cap).map(|l| l.pop()).flatten() {
                // reuse
                idx
            } else {
                // allocate new buffer
                master.push(cap);
                master.len() - 1
            };
            outs.push(buf_idx);
        }
        buf_map.insert(node, outs);

        // Free producer buffers whose last consumer just ran (exclude "Inputs")
        for e in graph.edges_directed(node, Direction::Incoming) {
            let src = e.source();
            if graph[src].code == "Inputs" {
                continue;
            }
            let (src_out_idx, _) = *e.weight();
            if let Some(c) = use_count.get_mut(&(src, src_out_idx)) {
                *c -= 1;
                if *c == 0 {
                    let buf_idx = buf_map[&src][src_out_idx];
                    free_by_cap
                        .entry(master[buf_idx])
                        .or_default()
                        .push(buf_idx);
                }
            }
        }
    }

    (master, buf_map)
}

pub fn compile_kernels(
    kernels: &StableGraph<Kernel, (usize, usize)>,
) -> FxHashMap<String, CudaFunction> {
    let ctx = cudarc::driver::CudaContext::new(0).unwrap();
    let mut compiled = FxHashMap::default();
    for kernel in kernels.node_weights() {
        if !compiled.contains_key(&kernel.code)
            && kernel.code != "Inputs"
            && kernel.code != "Outputs"
        {
            let ptx = cudarc::nvrtc::compile_ptx_with_opts(
                &kernel.code,
                CompileOptions {
                    include_paths: vec!["/usr/include".into()],
                    options: vec![
                        "--gpu-architecture=compute_75".into(),
                        "--relocatable-device-code=false".into(),
                        "--std=c++14".into(),
                    ],
                    ..Default::default()
                },
            )
            .unwrap();
            let module = ctx.load_module(ptx).unwrap();
            let k = module.load_function("kernel_name").unwrap();
            compiled.insert(kernel.code.clone(), k);
        }
    }
    compiled
}

pub fn run_graph(
    inputs: &mut HashMap<usize, (CudaSlice<f32>, bool)>,
    kernels: &StableGraph<Kernel, (usize, usize)>,
    dyn_vars: &FxHashMap<char, usize>,
    compiled_kernels: &FxHashMap<String, CudaFunction>,
    intermediate_buffers: &Vec<Expression>,
    intermediate_buffer_map: &HashMap<NodeIndex, Vec<usize>>,
) -> (Vec<Vec<f32>>, u128) {
    let ctx = cudarc::driver::CudaContext::new(0).unwrap();
    let stream = ctx.default_stream();
    // Allocate buffers
    let mut buffers = intermediate_buffers
        .iter()
        .map(|e| unsafe { stream.alloc(e.exec(dyn_vars).unwrap()).unwrap() })
        .collect_vec();
    let mut exec_time = 0;
    let input_node = kernels
        .node_indices()
        .find(|n| kernels[*n].code == "Inputs")
        .unwrap();
    for node in toposort(kernels, None).unwrap() {
        let kernel = &kernels[node];
        if kernel.code == "Inputs" {
            // Inputs should already be in the buffer map
        } else if kernel.code == "Outputs" {
            // Run
            let start = std::time::Instant::now();
            let time_taken_micros = start.elapsed().as_micros();
            let outputs = kernels
                .edges_directed(node, Direction::Incoming)
                .sorted_by_key(|e| e.weight().1)
                .map(|e| {
                    stream
                        .memcpy_dtov(&buffers[intermediate_buffer_map[&e.source()][e.weight().0]])
                        .unwrap()
                })
                .collect();
            println!("Exec Millis: {}", exec_time / 1000);
            return (outputs, time_taken_micros);
        } else if kernel.code.starts_with("Diff") {
            // Load file and diff numbers
            let diff_name = kernel.code.replace("Diff", "");

            let (input, input_index) = kernels
                .edges_directed(node, Direction::Incoming)
                .sorted_by_key(|n| n.weight().1)
                .map(|n| (n.source(), n.weight().0))
                .next()
                .unwrap();
            let buffer = &buffers[intermediate_buffer_map[&input][input_index]];
            let data: Vec<f32> = stream.memcpy_dtov(buffer).unwrap();
            let mut file = File::open(format!("{diff_name}.bin")).unwrap();
            let mut file_buffer = Vec::new();
            file.read_to_end(&mut file_buffer).unwrap();
            assert_eq!(file_buffer.len() % std::mem::size_of::<f32>(), 0);

            let num_floats = file_buffer.len() / std::mem::size_of::<f32>();
            let floats: Vec<f32> = unsafe {
                let ptr = file_buffer.as_ptr() as *const f32;
                Vec::from_raw_parts(ptr as *mut f32, num_floats, num_floats)
            };
            let mut matched = true;
            println!("Diff {} | {}", data.len(), floats.len());
            for (ind, (i, j)) in data.iter().zip(floats).enumerate() {
                if (i - j).abs() > 1e-5 {
                    matched = false;
                    println!("Diff {diff_name} failed: curr: {i} != file: {j}, index {ind}");
                    break;
                }
            }
            std::mem::forget(file_buffer);
            if matched {
                println!("DIFF {diff_name} MATCHED");
            }
            let dest_buffer = &mut buffers[intermediate_buffer_map[&node][0]];
            stream.memcpy_htod(&data, dest_buffer).unwrap();
        } else {
            // println!("Grid {:?} TB: {:?}", kernel.grid, kernel.threadblock);
            // println!("{}", kernel.code);

            let mut builder = stream.launch_builder(&compiled_kernels[&kernel.code]);

            // set inputs
            for (input, input_index) in kernels
                .edges_directed(node, Direction::Incoming)
                .sorted_by_key(|n| n.weight().1)
                .map(|n| (n.source(), n.weight().0))
            {
                if input == input_node {
                    builder.arg(&inputs[&input_index].0);
                } else {
                    builder.arg(&buffers[intermediate_buffer_map[&input][input_index]]);
                }
            }
            // set output
            let mut output_views = (0..kernel.outputs.len())
                .map(|o| buffers[intermediate_buffer_map[&node][o]].as_view_mut())
                .collect_vec();
            for o in &mut output_views {
                builder.arg(o);
            }
            // set dynamic dimensions
            for (_, v) in dyn_vars.iter().sorted_by_key(|(k, _)| **k) {
                builder.arg(v);
            }

            // Set dispatch
            let grid = (
                kernel.grid.0.exec(dyn_vars).unwrap() as u32,
                kernel.grid.1.exec(dyn_vars).unwrap() as u32,
                kernel.grid.2.exec(dyn_vars).unwrap() as u32,
            );
            let tb = (
                kernel.threadblock.0.exec(dyn_vars).unwrap() as u32,
                kernel.threadblock.1.exec(dyn_vars).unwrap() as u32,
                kernel.threadblock.2.exec(dyn_vars).unwrap() as u32,
            );
            assert!(
                tb.0 * tb.1 * tb.2 <= 1024,
                "threadblock is too big: {tb:?} > 1024"
            );
            let now = std::time::Instant::now();
            unsafe {
                builder.launch(LaunchConfig {
                    grid_dim: (grid.0, grid.1, grid.2),
                    block_dim: (tb.0, tb.1, tb.2),
                    shared_mem_bytes: kernel.smem.exec(dyn_vars).unwrap() as u32,
                })
            }
            .unwrap();
            exec_time += now.elapsed().as_micros();
        }
    }
    panic!("No output kernel detected in graph!");
}

// Analyze memory buffers and produce a mapping from node -> Vec<buffer index> and a list of buffers to allocate ahead of time
pub fn produce_buffer_map(
    graph: &StableGraph<Kernel, (u8, u8)>,
) -> (Vec<Expression>, FxHashMap<NodeIndex, Vec<usize>>) {
    // First pass - get clear sets for each node
    #[allow(clippy::type_complexity)]
    let mut first_pass: FxHashMap<
        NodeIndex,
        (
            BTreeMap<NodeIndex, BTreeSet<NodeIndex>>,
            BTreeSet<NodeIndex>,
        ),
    > = FxHashMap::default();
    let toposort = toposort(&graph, None).unwrap();
    // Loop through nodes in graph
    for node in &toposort {
        // Run through parents to build new tenative set and clear set
        let (mut tenative_sets, mut clear_set) = (BTreeMap::default(), BTreeSet::default());
        for parent in graph.neighbors_directed(*node, Direction::Incoming) {
            let parent_children = graph
                .neighbors_directed(parent, Direction::Outgoing)
                .collect::<BTreeSet<_>>();
            tenative_sets.insert(parent, parent_children);
            if let Some((parent_tenative_set, parent_clear_set)) = first_pass.get(&parent) {
                for (node_index, new_tenative_set) in parent_tenative_set.iter().map(|(n, c)| {
                    let mut c = c.clone();
                    c.retain(|n| *n != parent);
                    (*n, c)
                }) {
                    if let Some(set) = tenative_sets.get(&node_index) {
                        *tenative_sets.get_mut(&node_index).unwrap() =
                            btreeset_intersection(new_tenative_set, set);
                    } else {
                        tenative_sets.insert(node_index, new_tenative_set);
                    }
                }
                clear_set.extend(
                    tenative_sets
                        .iter()
                        .filter(|(_, v)| v.is_empty())
                        .map(|(n, _)| *n),
                );
                tenative_sets.retain(|_, v| !v.is_empty());
                clear_set.extend(parent_clear_set);
            }
        }
        first_pass.insert(*node, (tenative_sets, clear_set));
    }

    // Second pass - assign buffers
    let available_buffers = graph
        .node_indices()
        .map(|n| (n, graph[n].outputs.clone()))
        .collect::<FxHashMap<_, _>>();
    // Loop through nodes in graph
    let mut buffers = vec![];
    let mut buffer_map = FxHashMap::default();
    let mut used = FxHashSet::<NodeIndex>::default();
    for node in &toposort {
        buffer_map.insert(*node, vec![]);
        // Assign output buffers
        for required_buffer in &graph[*node].outputs {
            // println!("required :{}", required_buffer);
            // Find an applicable buffer
            if let Some((buffer_index, source_node, _)) = first_pass[node]
                .1
                .iter()
                .filter(|i| !used.contains(i))
                .filter(|i| available_buffers.contains_key(i))
                .flat_map(|i| {
                    available_buffers[i]
                        .iter()
                        .enumerate()
                        .map(|(o, b)| (o, *i, b))
                })
                .find(|(_, _, size)| **size == *required_buffer)
            {
                let buffer = buffer_map.get(&source_node).unwrap()[buffer_index];
                buffer_map.get_mut(node).unwrap().push(buffer);
                // Remove this buffer from first_pass so it can't be used again
                used.insert(source_node);
            } else {
                // Allocate new buffer
                buffer_map.get_mut(node).unwrap().push(buffers.len());
                buffers.push(*required_buffer);
            }
        }
    }

    (buffers, buffer_map)
}

fn btreeset_intersection<T: Ord>(mut a: BTreeSet<T>, b: &BTreeSet<T>) -> BTreeSet<T> {
    a.retain(|i| b.contains(i));
    a
}
