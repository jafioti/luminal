use std::collections::HashMap;

use itertools::Itertools;
use luminal::prelude::{
    petgraph::{prelude::StableGraph, visit::EdgeRef, Direction},
    *,
};
use luminal_2::{
    codegen::{codegen, stitch_meta_graph_together},
    extract::{make_test_inputs, search},
    run::{assign_buffers, compile_kernels, run_graph},
    translate::{translate_graph_meta, InitData},
    utils::{binary, build_search_space, loop_in, loop_out, print_kernels},
    GPUArch, GraphTerm,
};
use metal_rs::{objc::rc::autoreleasepool, Buffer, Device, MTLResourceOptions};
use rustc_hash::FxHashMap;

fn main() {
    autoreleasepool(|| {
        let (M, K, N) = (512, 512, 512);
        let mut cx = Graph::new();
        let a = cx.named_tensor("A", (M, K));
        let b = cx.named_tensor("B", (K, N));
        let out = a.matmul(b);
        let (graph, gmem_mapping, accs) = translate_graph_meta(&cx);
        let graph = graph.node_weights().next().unwrap().clone();
        let egraph = build_search_space(&graph, 1);
        let inputs = make_test_inputs(
            &graph,
            &FxHashMap::default(),
            &[("acc_0".to_string(), vec![0.0])],
        );
        let out_graph = search(
            &egraph,
            &inputs,
            GPUArch::Metal(HashMap::default()),
            &FxHashMap::default(),
        )
        .unwrap();
        let a_orig = out_graph
            .node_indices()
            .find(|i| {
                *out_graph.node_weight(*i).unwrap()
                    == GraphTerm::GMEM {
                        label: "A Load".to_string(),
                    }
            })
            .unwrap();
        let b_orig = out_graph
            .node_indices()
            .find(|i| {
                *out_graph.node_weight(*i).unwrap()
                    == GraphTerm::GMEM {
                        label: "B Load".to_string(),
                    }
            })
            .unwrap();
        let acc_orig = out_graph.node_indices().find(|i| {
            *out_graph.node_weight(*i).unwrap()
                == GraphTerm::GMEM {
                    label: "acc_0".to_string(),
                }
        }); // there may be no acc here anymore
        let out = out_graph.externals(Direction::Outgoing).next().unwrap();
        luminal_2::utils::display_graph(&out_graph, &[]);
        let (kernels, gmem_mapping) = codegen(
            out_graph,
            vec![out],
            GPUArch::Metal(HashMap::default()),
            0,
            &HashMap::default(),
            false,
        )
        .unwrap();

        let compiled = compile_kernels(&kernels);
        let (int_buffers, int_buffer_map) = assign_buffers(&kernels);

        let device = Device::system_default().unwrap();
        let a_buffer = copy_metal_buffer(&vec![1.; M * K], &device);
        let b_buffer = copy_metal_buffer(&vec![1.; K * M], &device);
        let acc_buffer = copy_metal_buffer(&vec![0.0], &device);

        let mut inputs = FxHashMap::default();
        inputs.insert(gmem_mapping[&a_orig], (a_buffer, false));
        inputs.insert(gmem_mapping[&b_orig], (b_buffer, false));
        if let Some(acc_orig) = acc_orig {
            inputs.insert(gmem_mapping[&acc_orig], (acc_buffer, false));
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
