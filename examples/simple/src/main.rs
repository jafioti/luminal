use itertools::Itertools;
use luminal::prelude::*;
//use luminal_2::{
// codegen::codegen, extract::search, run::run_graph, translate::translate_graph,
// utils::build_search_space, GPUArch,
//};
use luminal_nn::Linear;
use rand::{rng, Rng};

fn main() {
    let mut rng = rng();
    let weight = (0..4 * 5).map(|_| rng.random()).collect_vec();
    // Create a new graph
    let mut cx = Graph::new();
    // Randomly initialize a linear layer with an input size of 4 and an output size of 5
    let model = Linear::new(4, 5, false, &mut cx);
    model.weight.set(weight.clone());
    // Make an input tensor
    let a = cx.tensor(4).set(vec![1., 2., 3., 4.]);
    // Feed tensor through model
    let b = model.forward(a).retrieve();

    // Execute the graph
    cx.execute_debug();
    //let (new_graph, accs) = translate_graph(&cx);
    // luminal_2::utils::display_graph(&new_graph, &[]);
    // Print the results
    //println!("{:?}", accs);
    println!("B: {:?}", b.data());
    // let root = new_graph.externals(Direction::Outgoing).next().unwrap();
    // let kernels = codegen(
    //     new_graph.clone(),
    //     root,
    //     GPUArch::Metal(HashMap::default()),
    //     0,
    // )
    // .unwrap();
    // let mut inputs = vec![
    //     ("Tensor Load".to_string(), vec![1., 2., 3., 4.]),
    //     ("Weight Load".to_string(), weight),
    // ];
    // for (label, val) in accs {
    //     inputs.push((label, vec![val]));
    // }

    // let outputs = run_graph(&inputs, &kernels);
    // println!("{:?}", outputs);

    // let egraph = build_search_space(&new_graph, 5, false);
    // let kernels = search(&egraph, &inputs, GPUArch::Metal(HashMap::default())).unwrap();
    // println!("kernels : {}", kernels.node_count() - 2);
    // let outputs = run_graph(&inputs, &kernels);
    // println!("{:?}", outputs);
}
