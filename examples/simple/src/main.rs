use std::collections::HashMap;

use itertools::Itertools;
use luminal::prelude::{petgraph::Direction, *};
use luminal_2::{
    codegen::codegen,
    extract::search,
    run::run_graph,
    translate::translate_graph,
    utils::{build_search_space, print_kernels},
    GPUArch,
};
use luminal_nn::Linear;
use rand::{rng, Rng};
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
    let a = cx.tensor(('a', 'b')).set(vec![1., 2., 3., 4.]);
    // let b = cx.tensor('b').set(vec![0., 1., 0.]);
    let c = a.pad_along(0, 2, 1);
    println!("{:?}", c.shape.padding);
    let c = c.contiguous().retrieve();
    // let c = cx.triu(5, 1);
    // Feed tensor through model
    // let b = model.forward(a).retrieve();

    // Execute the graph
    cx.set_dyn_dim('a', 2);
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
    // luminal_2::utils::display_graph(&kernels, &[]);
    // let w1 = weight.clone();
    let mut inputs = vec![
        (
            old_to_new_mapping[&a.id],
            Box::new(move || vec![1_f32, 2., 3., 4.]) as Box<dyn FnOnce() -> Vec<f32> + 'static>,
        ),
        // (
        //     old_to_new_mapping[&b.id],
        //     Box::new(move || vec![0_f32, 1., 0.]) as Box<dyn FnOnce() -> Vec<f32> + 'static>,
        // ),
        // (old_to_new_mapping[&model.weight.id], Box::new(move || w1)),
    ];
    for (label, val) in &accs {
        let val = val.clone();
        inputs.push((
            label.clone(),
            Box::new(move || val) as Box<dyn FnOnce() -> Vec<f32> + 'static>,
        ));
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
