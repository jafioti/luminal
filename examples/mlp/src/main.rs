use luminal::prelude::*;
use luminal_nn::Linear;

fn main() {
    // Create a new graph
    let mut cx = Graph::new();
    // Randomly initialize 2 linear layers: the first with an input size of 4 and an output
    // size of 5, and a second with an input size of 5 and an (arbitrary) output size of 2
    let layer1 = Linear::new(4, 5, false, &mut cx).init_rand();
    let layer2 = Linear::new(5, 2, false, &mut cx).init_rand();
    // Make an input tensor
    let a = cx.tensor(4).set(vec![1., 2., 3., 4.]);
    // Feed tensor through the first layer
    let b = layer1.forward(a);
    // Feed ouput tensor of the first layer through the second layer
    let c = layer2.forward(b).retrieve();
    // Display the graph to see the ops
    cx.display();
    // Execute the graph
    cx.execute_debug();
    // Print the results
    println!("C: {:?}", c.data());
}

