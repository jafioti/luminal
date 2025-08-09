use std::collections::HashMap;

use itertools::Itertools;
use luminal::prelude::{
    petgraph::{visit::EdgeRef, Direction},
    *,
};
use luminal_2::{
    codegen::{codegen, stitch_meta_graph_together},
    extract::{make_test_inputs, search},
    run::{assign_buffers, compile_kernels, run_graph},
    translate::{translate_graph_meta, InitData},
    utils::build_search_space,
    GPUArch, GraphTerm,
};
use metal_rs::{objc::rc::autoreleasepool, Buffer, Device, MTLResourceOptions};
use rustc_hash::FxHashMap;

fn main() {
    autoreleasepool(|| {
        // let weight = (0..4 * 5).map(|_| rng.random()).collect_vec();
        // Create a new graph
        let mut cx = Graph::new();
        // Randomly initialize a linear layer with an input size of 4 and an output size of 5
        // let model = Linear::new(4, 5, false, &mut cx);
        // model.weight.set(weight.clone());
        // Make an input tensor
        let a = cx.named_tensor("A", (2, 2)).set(vec![1.1, 2.1, 3.1, 4.1]);
        let b = cx.named_tensor("B", (2, 2)).set(vec![0.1, 1.1, 18.1, 1.1]);
        let c = a.matmul(b).retrieve();
        // Execute the graph
        cx.set_dyn_dim('a', 3);
        cx.set_dyn_dim('b', 2);
        cx.execute_debug();
        let (mut new_graph, mut old_to_new_mapping, mut accs) = translate_graph_meta(&cx);
        // luminal_2::utils::display_graph(&new_graph.node_weights().next().unwrap(), &[]);
        // Insert accs into the old_to_new_mapping
        for (meta_node, nodes) in &accs {
            for (node, _) in nodes {
                old_to_new_mapping.insert(*node, (*meta_node, *node));
            }
        }

        // Search each subgraph
        for graph_node in new_graph.node_indices().collect_vec() {
            let graph = new_graph.node_weight_mut(graph_node).unwrap();
            let search_space = build_search_space(graph, 9);
            let inputs = make_test_inputs(graph, &cx.dyn_map);
            let searched_graph = search(
                &search_space,
                &inputs,
                GPUArch::Metal(HashMap::default()),
                &cx.dyn_map,
            )
            .unwrap();
            // screw it just say that the new only output of this graph is the only output (this doesn't work with multiple outputs)
            // adjust meta-edges
            let old_output = graph.externals(Direction::Outgoing).next().unwrap();
            let new_output = searched_graph
                .externals(Direction::Outgoing)
                .next()
                .unwrap();
            let old_inputs = graph
                .node_indices()
                .filter_map(|n| {
                    if let GraphTerm::GMEM { label } = graph.node_weight(n).unwrap() {
                        Some((n, label.clone()))
                    } else {
                        None
                    }
                })
                .collect::<HashMap<_, _>>();
            let new_inputs = searched_graph
                .node_indices()
                .filter_map(|n| {
                    if let GraphTerm::GMEM { label } = searched_graph.node_weight(n).unwrap() {
                        Some((label.clone(), n))
                    } else {
                        None
                    }
                })
                .collect::<HashMap<_, _>>();
            *graph = searched_graph;
            for edge in new_graph
                .edges_directed(graph_node, Direction::Outgoing)
                .map(|e| e.id())
                .collect_vec()
            {
                let (input, _) = new_graph.edge_weight_mut(edge).unwrap();
                *input = new_output;
            }
            // Update old-to-new-mappings
            for (_, (meta, v)) in &mut old_to_new_mapping {
                if *meta != graph_node {
                    continue;
                }
                if *v == old_output {
                    *v = new_output;
                }
                if let Some(gmem_label) = old_inputs.get(v) {
                    *v = new_inputs[gmem_label];
                }
            }
        }
        let outputs = vec![old_to_new_mapping[&c.id]];
        for (_, nodes) in &mut accs {
            for (node, _) in nodes {
                *node = old_to_new_mapping[node].1;
            }
        }
        let (new_graph, accs, outputs) = stitch_meta_graph_together(new_graph, accs, outputs);
        // luminal_2::utils::display_graph(&new_graph, &[]);
        let (kernels, gmem_mapping) = codegen(
            new_graph.clone(),
            outputs,
            GPUArch::Metal(HashMap::default()),
            0,
            &cx.dyn_map,
        )
        .unwrap();
        // print_kernels(&kernels);

        // let w1 = weight.clone();
        let a_data = vec![1.1_f32, 2.1, 3.1, 4.1];
        let b_data = vec![0.1_f32, 1.1, 18.1, 1.1];
        let device = Device::system_default().unwrap();
        let mut inputs = FxHashMap::default();
        inputs.insert(
            gmem_mapping[&old_to_new_mapping[&a.id].1],
            (copy_metal_buffer(&a_data, &device), true),
        );
        inputs.insert(
            gmem_mapping[&old_to_new_mapping[&b.id].1],
            (copy_metal_buffer(&b_data, &device), true),
        );
        for (label, val) in &accs {
            match val {
                InitData::Expr(e) => {
                    let val = e.exec(&cx.dyn_map).unwrap();
                    inputs.insert(gmem_mapping[label], {
                        let v = vec![val as f32];
                        (copy_metal_buffer(&v, &device), true)
                    });
                }
                InitData::Data(d) => {
                    inputs.insert(gmem_mapping[&label], (copy_metal_buffer(d, &device), true));
                }
            }
        }

        let compiled_kernels = compile_kernels(&kernels);
        let (int_buffers, int_buffer_map) = assign_buffers(&kernels);
        let (outputs, _) = run_graph(
            &mut inputs,
            &kernels,
            &cx.dyn_map,
            &compiled_kernels,
            &int_buffers,
            &int_buffer_map,
        );
        println!("{:?}", c.data());
        println!("{:?}", copy_metal_buffer_back(&outputs[0]));
    });
}

pub fn copy_metal_buffer(v: &Vec<f32>, device: &Device) -> Buffer {
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
