use std::{collections::HashMap, sync::Arc};

use itertools::Itertools;
use luminal::prelude::{petgraph::Direction, *};
use luminal_2::{
    codegen::{codegen, stitch_meta_graph_together},
    extract::search,
    run::{assign_buffers, compile_kernels, run_graph},
    translate::{translate_graph_meta, InitData},
    utils::{build_search_space, print_kernels},
    GPUArch, GT2,
};
use metal_rs::{objc::rc::autoreleasepool, Buffer, Device, MTLResourceOptions};
use rand::rng;
use rustc_hash::FxHashMap;

fn main() {
    autoreleasepool(|| {
        let mut rng = rng();
        // let weight = (0..4 * 5).map(|_| rng.random()).collect_vec();
        // Create a new graph
        let mut cx = Graph::new();
        // Randomly initialize a linear layer with an input size of 4 and an output size of 5
        // let model = Linear::new(4, 5, false, &mut cx);
        // model.weight.set(weight.clone());
        // Make an input tensor
        let a = cx.tensor((1, 3, 2, 2)).set(vec![
            1.1, 2.1, 3.1, 4.1, 5.1, 3.1, 1.2, 2.2, 3.2, 4.2, 5.2, 3.2,
        ]);
        let b = cx.tensor((1, 3, 2, 'b')).set(vec![
            0.1_f32, 1.1, 18.1, 1.1, 3.1, 2.1, 0.2, 1.2, 18.2, 1.2, 3.2, 2.2,
        ]);
        let c = a.matmul(b).graph_break().sin().retrieve();
        // Execute the graph
        cx.set_dyn_dim('a', 3);
        cx.set_dyn_dim('b', 2);
        cx.execute_debug();
        let (new_graph, old_to_new_mapping, accs) = translate_graph_meta(&cx);
        for g in new_graph.node_weights() {
            luminal_2::utils::display_graph(g, &[]);
        }
        let (new_graph, accs) = stitch_meta_graph_together(new_graph, accs);
        let (kernels, gmem_mapping) = codegen(
            new_graph.clone(),
            vec![old_to_new_mapping[&c.id].1],
            GPUArch::Metal(HashMap::default()),
            0,
            &cx.dyn_map,
        )
        .unwrap();
        print_kernels(&kernels);

        // let w1 = weight.clone();
        let a_data = vec![
            1.1_f32, 2.1, 3.1, 4.1, 5.1, 3.1, 1.2, 2.2, 3.2, 4.2, 5.2, 3.2,
        ];
        let b_data = vec![
            0.1_f32, 1.1, 18.1, 1.1, 3.1, 2.1, 0.2, 1.2, 18.2, 1.2, 3.2, 2.2,
        ];
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
                    let d = d.clone();
                    inputs.insert(gmem_mapping[label], (copy_metal_buffer(&d, &device), true));
                }
            }
        }

        let compiled_kernels = compile_kernels(&kernels);
        println!("Compiled");
        let (int_buffers, int_buffer_map) = assign_buffers(&kernels);
        println!("Assigned");
        let (outputs, _) = run_graph(
            &mut inputs,
            &kernels,
            &cx.dyn_map,
            &compiled_kernels,
            &int_buffers,
            &int_buffer_map,
        );
        println!("Ran");
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
