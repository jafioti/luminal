use std::collections::HashMap;

use itertools::Itertools;
use luminal::prelude::{petgraph::Direction, *};
use luminal_2::{
    codegen::codegen,
    extract::search,
    run::run_graph,
    translate::{translate_graph, InitData},
    utils::{build_search_space, print_kernels},
    GPUArch,
};
use luminal_metal::{Buffer, Device, MTLResourceOptions};
use rand::rng;

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
    let (new_graph, old_to_new_mapping, accs) = translate_graph(&cx);

    // luminal_2::utils::display_graph(&new_graph, &[]);
    // Print the results
    let kernels = codegen(
        new_graph.clone(),
        vec![old_to_new_mapping[&c.id]],
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
    let mut inputs = vec![
        (
            old_to_new_mapping[&a.id],
            Box::new(move || {
                Device::system_default().unwrap().new_buffer_with_data(
                    a_data.as_ptr() as *const _,
                    (a_data.len() * size_of::<f32>()) as u64,
                    MTLResourceOptions::StorageModeShared,
                )
            }) as Box<dyn FnOnce() -> Buffer + 'static>,
        ),
        (
            old_to_new_mapping[&b.id],
            Box::new(move || {
                Device::system_default().unwrap().new_buffer_with_data(
                    b_data.as_ptr() as *const _,
                    (b_data.len() * size_of::<f32>()) as u64,
                    MTLResourceOptions::StorageModeShared,
                )
            }) as Box<dyn FnOnce() -> Buffer + 'static>,
        ),
        // (old_to_new_mapping[&model.weight.id], Box::new(move || w1)),
    ];
    for (label, val) in &accs {
        match val {
            InitData::Expr(e) => {
                let val = e.exec(&cx.dyn_map).unwrap();
                inputs.push((
                    *label,
                    Box::new(move || {
                        let v = vec![val as f32];
                        Device::system_default().unwrap().new_buffer_with_data(
                            v.as_ptr() as *const _,
                            size_of::<f32>() as u64,
                            MTLResourceOptions::StorageModeShared,
                        )
                    }),
                ));
            }
            InitData::Data(d) => {
                let d = d.clone();
                inputs.push((
                    *label,
                    Box::new(move || {
                        Device::system_default().unwrap().new_buffer_with_data(
                            d.as_ptr() as *const _,
                            (d.len() * size_of::<f32>()) as u64,
                            MTLResourceOptions::StorageModeShared,
                        )
                    }),
                ))
            }
        }
    }

    let outputs = run_graph(inputs, &kernels, &cx.dyn_map);
    println!("{:?}", c.data());
    println!("{:?}", outputs);
    // let root = new_graph
    //     .externals(petgraph::Direction::Outgoing)
    //     .next()
    //     .unwrap();
    // let kernels = codegen(
    //     new_graph,
    //     vec![old_to_new_mapping[&b.id]],
    //     GPUArch::Metal(HashMap::default()),
    //     0,
    // )
    // .unwrap();
    // let mut inputs = vec![
    //     (old_to_new_mapping[&a.id], vec![1_f32, 2., 3., 4.]),
    //     (old_to_new_mapping[&model.weight.id], weight),
    // ];
    // for (label, val) in accs {
    //     inputs.push((label, val));
    // }
    // let egraph = build_search_space(&new_graph, 5, false);
    // let kernels = search(&egraph, &inputs, GPUArch::Metal(HashMap::default())).unwrap();
    // println!("kernels : {}", kernels.node_count() - 2);
    // // let outputs = run_graph(inputs, &kernels);
    // println!("{:?}", outputs);
}
