use std::collections::{BTreeMap, BTreeSet, HashMap};

use itertools::Itertools;
use luminal::{
    prelude::{
        NodeIndex,
        petgraph::{Direction, algo::toposort, prelude::StableGraph, visit::EdgeRef},
    },
    shape::Expression,
};
use metal_rs::{Buffer, objc::rc::autoreleasepool};
use rustc_hash::{FxHashMap, FxHashSet};

use crate::Kernel;

// // Take inputs and buffer maps and create buffer storage to feed into run_graph
// pub fn setup_buffers(
//     inputs: &[(NodeIndex, Vec<f32>)],
//     gmem_map: &FxHashMap<NodeIndex, usize>,
//     buffer_map: &FxHashMap<NodeIndex, Vec<usize>>,
//     buffer_sizes: &Vec<Expression>,
//     inputs_kernel: NodeIndex,
//     device: Device,
// ) -> (Vec<Buffer>, FxHashMap<(NodeIndex, u8), Buffer> {
//     let mut buffers = FxHashMap::default();
//     for (input_node, input_data) in inputs {
//         buffers.insert(
//             (inputs_kernel, gmem_map[input_node]),
//             device.new_buffer_with_data(
//                 input_data.as_ptr() as *mut _,
//                 (input_data.len() * std::mem::size_of::<f32>()) as u64,
//                 MTLResourceOptions::StorageModeShared,
//             ),
//         );
//     }
//     for (buffer_sizes)

//     todo!()
// }

pub fn run_graph(
    mut inputs: Vec<(NodeIndex, Box<dyn FnOnce() -> Vec<f32>>)>,
    kernels: &StableGraph<Kernel, (usize, usize)>,
    dyn_vars: &FxHashMap<char, usize>,
) -> (Vec<Vec<f32>>, u128) {
    use metal_rs::{
        CompileOptions, ComputePassDescriptor, ComputePipelineDescriptor, Device,
        MTLResourceOptions, MTLSize,
    };
    autoreleasepool(|| {
        let device = Device::system_default().unwrap();
        let queue = device.new_command_queue();
        // let command_buffer = queue.new_command_buffer();
        // Allocate buffers
        let mut buffers = HashMap::<(NodeIndex, usize), Option<Buffer>>::default();
        let mut ran = FxHashSet::default();
        let mut mapping: HashMap<usize, usize> = HashMap::default();
        for node in toposort(kernels, None).unwrap() {
            ran.insert(node);
            let kernel = kernels.node_weight(node).unwrap();
            if kernel.code.starts_with("Inputs") {
                mapping = serde_json::from_str(&kernel.code.replace("Inputs", "")).unwrap();
                buffers = inputs
                    .iter()
                    .map(|(name, _)| ((node, mapping[&name.index()]), None))
                    .collect();

                // buffers.extend(inputs.into_iter().map(|(name, buf)| {
                //     ((node, mapping[&name.index()]), {
                //         let buf = buf();
                //         device.new_buffer_with_data(
                //             buf.as_ptr() as *mut _,
                //             (buf.len() * std::mem::size_of::<f32>()) as u64,
                //             MTLResourceOptions::StorageModeShared,
                //         )
                //     })
                // }));
                // inputs = vec![];
                println!("INPUTS: {}", buffers.len());
            } else if kernel.code == "Outputs" {
                // Run
                let start = std::time::Instant::now();
                // command_buffer.commit();
                // command_buffer.wait_until_completed();
                let time_taken_micros = start.elapsed().as_micros();
                let outputs = kernels
                    .edges_directed(node, Direction::Incoming)
                    .map(|e| {
                        let buffer = buffers[&(e.source(), e.weight().0)].as_ref().unwrap();
                        let mut curr_data =
                            vec![0.0; buffer.length() as usize / std::mem::size_of::<f32>()];
                        let ptr = buffer.contents() as *mut f32;
                        for (i, d) in curr_data.iter_mut().enumerate() {
                            *d = unsafe { *ptr.add(i) };
                        }
                        curr_data
                    })
                    .collect();
                // Copy outputs back
                return (outputs, time_taken_micros);
            } else {
                // println!("Grid {:?} TB: {:?}", kernel.grid, kernel.threadblock);
                // println!("{}", kernel.code);

                // compile kernel
                let command_buffer = queue.new_command_buffer();
                let encoder = command_buffer
                    .compute_command_encoder_with_descriptor(ComputePassDescriptor::new());
                let options = CompileOptions::new();
                // options.set_fast_math_enabled(true);
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
                    println!("inp");
                    if let Some(b) = &buffers[&(input, input_index)] {
                        encoder.set_buffer(i as u64, Some(b), 0);
                    } else {
                        let entry = inputs
                            .iter_mut()
                            .find(|(n, _)| mapping[&n.index()] == input_index)
                            .unwrap();
                        let mut other =
                            Box::new(|| vec![]) as Box<dyn FnOnce() -> Vec<f32> + 'static>;
                        std::mem::swap(&mut entry.1, &mut other);
                        let buf = other();
                        let buffer = device.new_buffer_with_data(
                            buf.as_ptr() as *mut _,
                            (buf.len() * std::mem::size_of::<f32>()) as u64,
                            MTLResourceOptions::StorageModeShared,
                        );
                        encoder.set_buffer(i as u64, Some(&buffer), 0);
                        let k = (input, input_index);
                        *buffers.get_mut(&k).unwrap() = Some(buffer);
                        // assert!(buffers[&k].is_some());
                    };
                    // println!("Inp {i}: {}", buffers[&(input, input_index)].length());
                }
                // set output
                let n_inputs = kernels.edges_directed(node, Direction::Incoming).count();
                for (i, size) in kernel.outputs.iter().enumerate() {
                    buffers.insert(
                        (node, i),
                        Some(device.new_buffer(
                            (size.exec(&dyn_vars).unwrap() * std::mem::size_of::<f32>()) as u64,
                            MTLResourceOptions::StorageModeShared,
                        )),
                    );
                    encoder.set_buffer(
                        (i + n_inputs) as u64,
                        Some(buffers[&(node, i)].as_ref().unwrap()),
                        0,
                    );
                }
                // set smem
                if !kernel.smem.is_empty() {
                    encoder.set_threadgroup_memory_length(
                        0,
                        (kernel.smem.exec(dyn_vars).unwrap() * std::mem::size_of::<f32>()) as u64,
                    );
                }

                // Set dispatch
                encoder.dispatch_thread_groups(
                    MTLSize::new(
                        kernel.grid.0.exec(dyn_vars).unwrap() as u64,
                        kernel.grid.1.exec(dyn_vars).unwrap() as u64,
                        kernel.grid.2.exec(dyn_vars).unwrap() as u64,
                    ),
                    MTLSize::new(
                        kernel.threadblock.0.exec(dyn_vars).unwrap() as u64,
                        kernel.threadblock.1.exec(dyn_vars).unwrap() as u64,
                        kernel.threadblock.2.exec(dyn_vars).unwrap() as u64,
                    ),
                );
                encoder.end_encoding();
                command_buffer.commit();
                command_buffer.wait_until_completed();
                // for (input, input_index) in kernels
                //     .edges_directed(node, Direction::Incoming)
                //     .sorted_by_key(|n| n.weight().1)
                //     .map(|n| (n.source(), n.weight().0))
                // {
                //     let mut curr_data = vec![
                //         0.0;
                //         buffers[&(input, input_index)].as_ref().unwrap().length()
                //             as usize
                //             / std::mem::size_of::<f32>()
                //     ];
                //     let ptr =
                //         buffers[&(input, input_index)].as_ref().unwrap().contents() as *mut f32;
                //     // if curr_data.is_empty() {
                //     //     panic!(
                //     //         "input empty: {} | {}",
                //     //         buffer_sizes[buffer_map[&input][input_index as usize]],
                //     //         buffers[buffer_map[&input][input_index as usize]].length()
                //     //     );
                //     // }
                //     for (i, d) in curr_data.iter_mut().enumerate() {
                //         *d = unsafe { *ptr.add(i) };
                //     }
                //     println!("{:?}", &curr_data[..10.min(curr_data.len())]);
                // }
                println!("---");
                for i in 0..kernel.outputs.len() {
                    let mut curr_data = vec![
                        0.0;
                        buffers[&(node, i)].as_ref().unwrap().length() as usize
                            / std::mem::size_of::<f32>()
                    ];
                    let ptr = buffers[&(node, i)].as_ref().unwrap().contents() as *mut f32;
                    for (i, d) in curr_data.iter_mut().enumerate() {
                        *d = unsafe { *ptr.add(i) };
                    }
                    println!("{:?}", &curr_data[..10.min(curr_data.len())]);
                    for (i, n) in curr_data.into_iter().enumerate() {
                        if n.is_nan() {
                            panic!("{} | {}", n, i);
                        }
                    }
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
                            println!("Freeing {:?}", buf.as_ref().unwrap().length());
                            buf.unwrap()
                                .set_purgeable_state(metal_rs::MTLPurgeableState::Empty);
                        }
                    }
                }
            }
        }
        panic!("No output kernel detected in graph!");
    })
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
