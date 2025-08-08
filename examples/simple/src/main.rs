use std::{collections::HashMap, sync::Arc};

use cudarc::driver::{CudaSlice, CudaStream};
use itertools::Itertools;
use luminal::prelude::{petgraph::Direction, *};
use luminal_2::{
    codegen::{codegen, stitch_meta_graph_together},
    extract::search,
    run::{assign_buffers, compile_kernels, run_graph},
    translate::{translate_graph_meta, InitData},
    utils::{build_search_space, print_kernels},
    GPUArch,
};
use rand::rng;
use rustc_hash::FxHashMap;

fn main() {
    let mut rng = rng();
    // let weight = (0..4 * 5).map(|_| rng.random()).collect_vec();
    // Create a new graph
    let mut cx = Graph::new();
    // Randomly initialize a linear layer with an input size of 4 and an output size of 5
    // let model = Linear::new(4, 5, false, &mut cx);
    // model.weight.set(weight.clone());
    // Make an input tensor
    let a = cx
        .tensor((1, 'a', 2, 2))
        .set(vec![
            1.1, 2.1, 3.1, 4.1, 5.1, 3.1, 1.2, 2.2, 3.2, 4.2, 5.2, 3.2,
        ])
        .permute((0, 2, 1, 3));
    let b = cx
        .tensor((1, 2, 'b', 2))
        .set(vec![0.1, 1.1, 18.1, 1.1, 3.1, 2.1, 0.2, 1.2]);
    let c = a.concat_along(b, 2).retrieve();
    // let c = cx.triu(5, 1);
    // Feed tensor through model
    // let b = model.forward(a).retrieve();
    // Execute the graph
    cx.set_dyn_dim('a', 3);
    cx.set_dyn_dim('b', 2);
    cx.execute_debug();
    let (new_graph, old_to_new_mapping, accs) = translate_graph_meta(&cx);
    let (new_graph, accs) = stitch_meta_graph_together(new_graph, accs);

    // luminal_2::utils::display_graph(&new_graph, &[]);
    // Print the results
    // for sub_graph_id in new_graph.node_indices() {
    //     let sub_graph = &new_graph[sub_graph_id];
    let (kernels, gmem_mapping) = codegen(
        new_graph.clone(),
        vec![old_to_new_mapping[&c.id].1],
        GPUArch::CUDA,
        0,
        &cx.dyn_map,
    )
    .unwrap();
    print_kernels(&kernels);
    // }

    // let w1 = weight.clone();
    let a_data = vec![
        1.1_f32, 2.1, 3.1, 4.1, 5.1, 3.1, 1.2, 2.2, 3.2, 4.2, 5.2, 3.2,
    ];
    let b_data = vec![
        0.1_f32, 1.1, 18.1, 1.1, 3.1, 2.1, 0.2, 1.2, 18.2, 1.2, 3.2, 2.2,
    ];
    let ctx = cudarc::driver::CudaContext::new(0).unwrap();
    let stream = ctx.default_stream();
    let mut inputs = FxHashMap::default();
    inputs.insert(
        gmem_mapping[&old_to_new_mapping[&a.id].1],
        (copy_cuda_buffer(&a_data, &stream), true),
    );
    inputs.insert(
        gmem_mapping[&old_to_new_mapping[&b.id].1],
        (copy_cuda_buffer(&b_data, &stream), true),
    );
    for (label, val) in &accs {
        match val {
            InitData::Expr(e) => {
                let val = e.exec(&cx.dyn_map).unwrap();
                inputs.insert(gmem_mapping[label], {
                    let v = vec![val as f32];
                    (copy_cuda_buffer(&v, &stream), true)
                });
            }
            InitData::Data(d) => {
                let d = d.clone();
                inputs.insert(gmem_mapping[label], (copy_cuda_buffer(&d, &stream), true));
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
    println!("{:?}", stream.memcpy_dtov(&outputs[0]).unwrap());
}

pub fn copy_cuda_buffer(v: &Vec<f32>, stream: &Arc<CudaStream>) -> CudaSlice<f32> {
    let mut buffer = unsafe { stream.alloc::<f32>(v.len()).unwrap() };
    stream.memcpy_htod(v, &mut buffer).unwrap();
    buffer
}
