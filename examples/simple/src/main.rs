use luminal::{nn::linear::Linear, prelude::*};

fn main() {
    // Create a new graph
    let mut cx = Graph::new();
    // Randomly initialize a linear layer with an input size of 4 and an output size of 5
    let model = Linear::<4, 5>::initialize(&mut cx);
    // Make an input tensor
    let a = cx.tensor::<R1<4>>().set(vec![1., 2., 3., 4.]);
    // Feed tensor through model
    let mut b = model.forward(a).retrieve();

    // Execute the graph
    cx.display();
    cx.execute_debug();
    // Print the results
    println!("B: {:?}", b.data());
}
