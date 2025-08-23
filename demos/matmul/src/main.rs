use std::{collections::HashMap, ffi::c_void, ptr::NonNull};

use itertools::Itertools;
use luminal::prelude::{
    petgraph::{visit::EdgeRef, Direction},
    *,
};
use luminal_2::{
    codegen::{codegen, stitch_meta_graph_together},
    extract::{make_test_inputs, search},
    run::{assign_buffers, compile_kernels, run_graph},
    translate::{translate_graph, InitData},
    utils::build_search_space,
    Buffer, Device, GPUArch, GraphTerm,
};
use objc2_metal::{MTLBuffer, MTLCreateSystemDefaultDevice, MTLDevice, MTLResourceOptions};
use rustc_hash::FxHashMap;

fn main() {
    objc2::rc::autoreleasepool(|_| {
        #[allow(non_snake_case)]
        let (M, K, N) = (512, 512, 512);
        let mut cx = Graph::new();
        let a = cx.named_tensor("A", (M, K));
        let b = cx.named_tensor("B", (K, N));
        let out = a.matmul(b);
        let (mut new_graph, mut mapping, accs) = translate_graph(&cx);
        // Search each subgraph
        for graph_node in new_graph.node_indices().collect_vec() {
            let graph = new_graph.node_weight_mut(graph_node).unwrap();
            let search_space = build_search_space(graph, 3);
            let inputs = make_test_inputs(graph, &cx.dyn_map, &accs);
            let searched_graph = search(
                &search_space,
                &inputs,
                GPUArch::Metal(HashMap::default()),
                &cx.dyn_map,
            )
            .unwrap();
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
            for (_, (meta, v)) in &mut mapping {
                if *meta != graph_node {
                    continue;
                }
                if *v == old_output {
                    *v = new_output;
                }
                if let Some(gmem_label) = old_inputs.get(v) {
                    if let Some(new) = new_inputs.get(gmem_label) {
                        *v = *new;
                    }
                }
            }
        }
        let outputs = vec![mapping[&out.id]];
        let (graph, meta_to_final, outputs) = stitch_meta_graph_together(new_graph, outputs);
        let mut gmem_to_node_mapping = FxHashMap::default();
        for n in graph.node_indices() {
            if let Some(GraphTerm::GMEM { label }) = graph.node_weight(n) {
                gmem_to_node_mapping.insert(label.clone(), n);
            }
        }
        let mut unified_map = FxHashMap::default();
        for (k, v) in mapping {
            unified_map.insert(k, meta_to_final[&v]);
        }
        let (kernels, gmem_mapping) = codegen(
            graph,
            outputs,
            GPUArch::Metal(HashMap::default()),
            0,
            &HashMap::default(),
            false,
        )
        .unwrap();

        let compiled = compile_kernels(&kernels);
        let (int_buffers, int_buffer_map) = assign_buffers(&kernels);

        let device = MTLCreateSystemDefaultDevice().unwrap();
        let mut inputs = FxHashMap::default();
        inputs.insert(
            gmem_mapping[&unified_map[&a.id]],
            (copy_metal_buffer(&vec![1.; M * K], &device), false),
        );
        inputs.insert(
            gmem_mapping[&unified_map[&b.id]],
            (copy_metal_buffer(&vec![1.; K * M], &device), false),
        );
        for (label, val) in &accs {
            if let Some(node) = gmem_to_node_mapping.get(label) {
                match val {
                    InitData::Expr(e) => {
                        let val = e.exec(&cx.dyn_map).unwrap();
                        inputs.insert(gmem_mapping[node], {
                            let v = vec![val as f32];
                            (copy_metal_buffer(&v, &device), true)
                        });
                    }
                    InitData::Data(d) => {
                        inputs.insert(gmem_mapping[node], (copy_metal_buffer(d, &device), true));
                    }
                }
            }
        }

        let (outputs, _) = run_graph(
            &mut inputs,
            &kernels,
            &FxHashMap::default(),
            &compiled,
            &int_buffers,
            &int_buffer_map,
        );
        println!("{:?}", &copy_metal_buffer_back(&outputs[0])[..10]);
    });
    expression_cleanup();
}

pub fn copy_metal_buffer(v: &Vec<f32>, device: &Device) -> Buffer {
    unsafe {
        device
            .newBufferWithBytes_length_options(
                NonNull::new(v.as_ptr() as *mut c_void).unwrap(),
                v.len() * std::mem::size_of::<f32>(),
                MTLResourceOptions::StorageModeShared,
            )
            .unwrap()
    }
}

pub fn copy_metal_buffer_back(v: &Buffer) -> Vec<f32> {
    let mut data = vec![0f32; v.length() as usize / size_of::<f32>()];
    let ptr = v.contents().as_ptr() as *mut f32;
    for (i, d) in data.iter_mut().enumerate() {
        *d = unsafe { *ptr.add(i) };
    }
    data
}
