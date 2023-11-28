use luminal::{nn::linear::Linear, prelude::*};

fn main() {
    let mut cx = Graph::new();
    let model = Linear::<4, 5>::initialize(&mut cx);
    let a = cx.tensor::<R1<4>>().set(vec![1., 2., 3., 4.]);
    let b = model.forward(a).retrieve();

    cx.execute_debug();

    println!("B: {:?}", b.data());
}
