use std::{
    collections::{BTreeMap, BTreeSet, HashMap},
    fs::File,
    io::Read,
};

use cudarc::{
    driver::{CudaSlice, LaunchConfig, PushKernelArg},
    nvrtc::CompileOptions,
};
use itertools::Itertools;
use luminal::{
    prelude::{
        NodeIndex,
        petgraph::{
            Direction, algo::toposort, csr::IndexType, prelude::StableGraph, visit::EdgeRef,
        },
    },
    shape::Expression,
};
use rustc_hash::{FxHashMap, FxHashSet};

use crate::Kernel;

pub fn run_graph(
    buffers: &mut HashMap<(NodeIndex, usize), (CudaSlice<f32>, bool)>,
    kernels: &StableGraph<Kernel, (usize, usize)>,
    dyn_vars: &FxHashMap<char, usize>,
) -> (Vec<Vec<f32>>, u128) {
    let ctx = cudarc::driver::CudaContext::new(0).unwrap();
    let stream = ctx.default_stream();
    // Allocate buffers
    let mut ran = FxHashSet::default();
    let mut mapping: HashMap<usize, usize> = HashMap::default();
    for node in toposort(kernels, None).unwrap() {
        ran.insert(node);
        let kernel = kernels.node_weight(node).unwrap();
        if kernel.code.starts_with("Inputs") {
            // Inputs should already be in the buffer map
        } else if kernel.code == "Outputs" {
            // Run
            let start = std::time::Instant::now();
            let time_taken_micros = start.elapsed().as_micros();
            let outputs = kernels
                .edges_directed(node, Direction::Incoming)
                .sorted_by_key(|e| e.weight().1)
                .map(|e| {
                    let (buffer, _) = &buffers[&(e.source(), e.weight().0)];
                    let data: Vec<f32> = stream.memcpy_dtov(buffer).unwrap();
                    data
                })
                .collect();
            let to_remove = buffers
                .iter()
                .filter(|(_, (_, r))| *r)
                .map(|(k, _)| *k)
                .collect_vec();
            for id in to_remove {
                buffers.remove(&id).unwrap(); // Should we explicitly free this?
            }
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
            let (buffer, to_remove) = buffers.remove(&(input, input_index)).unwrap();
            let data: Vec<f32> = stream.memcpy_dtov(&buffer).unwrap();
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
            for (ind, (i, j)) in data.into_iter().zip(floats).enumerate() {
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
            buffers.insert((node, 0), (buffer, to_remove));
        } else {
            // println!("Grid {:?} TB: {:?}", kernel.grid, kernel.threadblock);
            // println!("{}", kernel.code);

            // compile kernel
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
            let k = module
                .load_function(&format!("kernel{}", node.index()))
                .unwrap();
            let mut builder = stream.launch_builder(&k);

            // set inputs
            for (input, input_index) in kernels
                .edges_directed(node, Direction::Incoming)
                .sorted_by_key(|n| n.weight().1)
                .map(|n| (n.source(), n.weight().0))
            {
                if !buffers.contains_key(&(input, input_index)) {
                    panic!(
                        "Couldn't find buffer, possibly missing input {:?}",
                        (input, input_index)
                    );
                }
            }
            for (i, (input, input_index)) in kernels
                .edges_directed(node, Direction::Incoming)
                .sorted_by_key(|n| n.weight().1)
                .map(|n| (n.source(), n.weight().0))
                .enumerate()
            {
                builder.arg(&buffers[&(input, input_index)].0);
            }
            // set output
            let mut out = kernel
                .outputs
                .iter()
                .map(|s| {
                    stream
                        .alloc_zeros::<f32>(s.exec(dyn_vars).unwrap())
                        .unwrap()
                })
                .collect_vec();
            for o in &mut out {
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
            unsafe {
                builder.launch(LaunchConfig {
                    grid_dim: (grid.0, grid.1, grid.2),
                    block_dim: (tb.0, tb.1, tb.2),
                    shared_mem_bytes: kernel.smem.exec(dyn_vars).unwrap() as u32,
                })
            }
            .unwrap();

            // Insert outputs into buffers
            for (i, buf) in out.into_iter().enumerate() {
                buffers.insert((node, i), (buf, true));
            }

            // Go through inputs and free buffers that aren't going to be used again
            for (in_node, in_ind) in kernels
                .edges_directed(node, Direction::Incoming)
                .map(|e| (e.source(), e.weight().0))
            {
                if kernels
                    .edges_directed(in_node, Direction::Outgoing)
                    .all(|e| e.weight().0 == in_ind && ran.contains(&e.target()))
                {
                    // All consumers have already ran, deallocate
                    if let Some(buf) = buffers.remove(&(in_node, in_ind)) {
                        // Should we explicitly free this?
                    }
                }
            }
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
        .map(|n| (n, graph.node_weight(n).unwrap().outputs.clone()))
        .collect::<FxHashMap<_, _>>();
    // Loop through nodes in graph
    let mut buffers = vec![];
    let mut buffer_map = FxHashMap::default();
    let mut used = FxHashSet::<NodeIndex>::default();
    for node in &toposort {
        buffer_map.insert(*node, vec![]);
        // Assign output buffers
        for required_buffer in &graph.node_weight(*node).unwrap().outputs {
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
