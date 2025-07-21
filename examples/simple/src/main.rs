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
    // let orig = cx.tensor(3).set([1., 1., 1.]);
    let triu = cx.triu(10, 0).retrieve();
    // let triu = (intermediate.pool_last_dim(3, 1, 1).sum(1) - 1.).retrieve();

    // let triu = cx.arange(3).retrieve();

    // // Randomly initialize a linear layer with an input size of 4 and an output size of 5
    // let model = Linear::new(4, 5, false, &mut cx);
    // model.weight.set(weight.clone());
    // // Make an input tensor
    // let a = cx.tensor(4).set(vec![1., 2., 3., 4.]);
    // // Feed tensor through model
    // let b = model.forward(a).retrieve();

    // Execute the graph
    cx.execute_debug();
    // cx.display();
    let (new_graph, old_to_new_mapping, accs) = translate_graph(&cx);
    // luminal_2::utils::display_graph(&new_graph, &[]);
    // Print the results
    println!("{:?}", accs);
    println!("B: {:?}", triu.data());
    let root = new_graph.externals(Direction::Outgoing).next().unwrap();
    let kernels = codegen(
        new_graph.clone(),
        root,
        GPUArch::Metal(HashMap::default()),
        0,
    )
    .unwrap();
    // luminal_2::utils::display_graph(&kernels, &[]);
    let mut inputs = vec![
        // (
        //     old_to_new_mapping[&orig.id],
        //     Box::new(move || vec![1_f32, 1., 1.]) as Box<dyn FnOnce() -> Vec<f32> + 'static>,
        // ),
        // (
        //     old_to_new_mapping[&model.weight.id],
        //     Box::new(move || weight),
        // ),
    ];
    for (label, val) in accs {
        inputs.push((
            label,
            Box::new(move || val) as Box<dyn FnOnce() -> Vec<f32> + 'static>,
        ));
    }

    let outputs = run_graph(inputs, &kernels, &FxHashMap::default());
    println!("{:?}", outputs);
    // let root = new_graph
    //     .externals(petgraph::Direction::Outgoing)
    //     .next()
    //     .unwrap();
    // let kernels = codegen(new_graph, root, GPUArch::Metal(HashMap::default()), 0).unwrap();
    // // let egraph = build_search_space(&new_graph, 5, false);
    // // let kernels = search(&egraph, &inputs, GPUArch::Metal(HashMap::default())).unwrap();
    // println!("kernels : {}", kernels.node_count() - 2);
    // let outputs = run_graph(inputs, &kernels);
    // println!("{:?}", outputs);
}
