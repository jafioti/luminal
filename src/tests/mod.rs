mod dynamic;

use std::fmt::Debug;

use rand::{thread_rng, Rng};

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
    assert_close(&unoptimized_a, &a);
    assert_close(&unoptimized_d, &d);
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

    assert_close(&unoptimized_a, &a.data());
}

#[test]
fn test_shapes() {
    let mut cx = Graph::new();
    let a = cx.new_tensor::<R1<4>>("Input");

    let b: GraphTensor<R2<2, 2>> = a.reshape::<R2<2, 2>>().permute::<_, Axes2<1, 0>>();
    b.mark();
    a.set(vec![1., 2., 3., 4.]);

    cx.execute();

    assert_close(&b.data(), &[1., 3., 2., 4.]);
}

/// Ensure two arrays are nearly equal
pub fn assert_close(a_vec: &[f32], b_vec: &[f32]) {
    assert_eq!(a_vec.len(), b_vec.len(), "Number of elements doesn't match");
    for (a, b) in a_vec.iter().zip(b_vec.iter()) {
        if (a - b).abs() > 1e-3 {
            panic!(
                "{a} is not close to {b}, avg distance: {}",
                a_vec
                    .iter()
                    .zip(b_vec.iter())
                    .map(|(a, b)| (a - b).abs())
                    .sum::<f32>()
                    / a_vec.len() as f32
            );
        }
    }
}

/// Ensure two arrays are nearly equal to a decimal place
pub fn assert_close_precision(a_vec: &[f32], b_vec: &[f32], precision: u8) {
    assert_eq!(a_vec.len(), b_vec.len(), "Number of elements doesn't match");
    for (a, b) in a_vec.iter().zip(b_vec.iter()) {
        if (a - b).abs() > f32::powf(10., -(precision as f32)) {
            panic!(
                "{a} is not close to {b}, avg distance: {}",
                a_vec
                    .iter()
                    .zip(b_vec.iter())
                    .map(|(a, b)| (a - b).abs())
                    .sum::<f32>()
                    / a_vec.len() as f32
            );
        }
    }
}

/// Ensure two arrays are exactly equal
pub fn assert_exact<T: PartialEq + Debug>(a_vec: &[T], b_vec: &[T]) {
    assert_eq!(a_vec.len(), b_vec.len(), "Number of elements doesn't match");
    for (a, b) in a_vec.iter().zip(b_vec.iter()) {
        if a != b {
            panic!("{a:?} is not equal to {b:?}");
        }
    }
}

pub fn random_vec(n: usize) -> Vec<f32> {
    let mut rng = thread_rng();
    (0..n).map(|_| rng.gen_range(-0.5..0.5)).collect()
}

#[macro_export]
macro_rules! test_imports {
    () => {
        #[allow(unused_imports)]
        use dfdx::prelude::{
            Axes as DAxes, Axes2 as DAxes2, Axes3 as DAxes3, Axes4 as DAxes4, Axes5 as DAxes5,
            Axis as DAxis, Const as DConst, *,
        };
        #[allow(unused_imports)]
        use $crate::{
            prelude::{
                Axes as LAxes, Axes2 as LAxes2, Axes3 as LAxes3, Axes4 as LAxes4, Axes5 as LAxes5,
                Axis as LAxis, Const as LConst, *,
            },
            tests::{assert_close, assert_exact, random_vec},
        };
    };
}
