mod dynamic;

use crate::prelude::*;

// Integration and other tests

#[test]
fn main() {
    let mut cx = Graph::new();
    let b = cx.new_tensor::<R1<3>>("Input");
    let c = cx.new_tensor::<R1<3>>("Input");
    let g = cx.new_tensor::<R1<3>>("Input");
    let e = cx.new_tensor::<R1<3>>("Input");

    let a = b * c + g;
    let d = (b * c / e).exp_2().log_2();

    b.set(vec![1.0, 2.0, 3.0]);
    c.set(vec![1.0, 2.0, 3.0]);
    g.set(vec![1.0, 2.0, 3.0]);
    e.set(vec![1.0, 2.0, 3.0]);

    a.mark();
    d.mark();

    cx.execute();

    let unoptimized_a = a.data();
    let unoptimized_d = d.data();

    cx.optimize(GenericOptimizer::default());

    cx.execute();
    let a = a.data();
    let d = d.data();
    assert_close_data(&unoptimized_a, &a);
    assert_close_data(&unoptimized_d, &d);
}

#[test]
fn test_matmul() {
    let mut cx = Graph::new();
    let b = cx.new_tensor::<R2<3, 1>>("Input");
    let c = cx.new_tensor::<R2<1, 4>>("Input");

    let a = b.matmul(c);

    a.mark();
    b.set(vec![1.0, 2.0, 3.0]);
    c.set(vec![1.0, 2.0, 3.0, 3.0]);

    cx.execute();

    let unoptimized_a = a.data();

    cx.optimize(GenericOptimizer::default());
    cx.execute();

    assert_close_data(&unoptimized_a, &a.data());
}

#[test]
fn test_shapes() {
    let mut cx = Graph::new();
    let a = cx.new_tensor::<R1<4>>("Input");

    let b: GraphTensor<R2<2, 2>> = a.reshape::<R2<2, 2>>().permute::<_, Axes2<1, 0>>();
    b.mark();
    a.set(vec![1., 2., 3., 4.]);

    cx.execute();

    assert_close_data(&b.data(), &[1., 3., 2., 4.]);
}

/// Ensure two arrays are nearly equal
pub fn assert_close_data(a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len(), "Number of elements doesn't match");
    for (a, b) in a.iter().zip(b.iter()) {
        if (a - b).abs() > 1e-3 {
            panic!("{a} is not close to {b}");
        }
    }
}
