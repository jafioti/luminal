use crate::{graph::Graph, optimizer::GeneralOptimizer, shape::R1, tensor::Tensor};

#[test]
fn main() {
    let mut cx = Graph::new();
    let b = cx.new_tensor::<R1<3>>();
    let c = cx.new_tensor::<R1<3>>();
    let g = cx.new_tensor::<R1<3>>();
    let e = cx.new_tensor::<R1<3>>();

    let a = b * c + g;
    let d = (b * c / e).exp_2().log_2();

    b.set(vec![1.0, 2.0, 3.0]);
    c.set(vec![1.0, 2.0, 3.0]);
    g.set(vec![1.0, 2.0, 3.0]);
    e.set(vec![1.0, 2.0, 3.0]);

    a.mark();
    d.mark();

    cx.execute();

    let unoptimized_a = a.retrieve().unwrap();
    let unoptimized_d = d.retrieve().unwrap();

    // let pre_optimized = cx.debug_graph();

    cx.optimize(GeneralOptimizer::default());

    // display_graph(&pre_optimized.join(&cx.debug_graph()));

    cx.execute();
    assert_close(&unoptimized_a, &a.retrieve().unwrap());
    assert_close(&unoptimized_d, &d.retrieve().unwrap());
}

/// Ensure two tensors are nearly equal
fn assert_close(a: &Tensor, b: &Tensor) {
    assert_eq!(a.shape.shape(), b.shape.shape(), "Shapes don't match");
    for (a, b) in a.data.iter().zip(b.data.iter()) {
        if (a - b).abs() > 0.01 {
            panic!("{a} is not close to {b}");
        }
    }
}
