use crate::{nn::linear::Linear, prelude::*};

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

    cx.optimize(GeneralOptimizer::default());

    cx.execute();
    let a = a.retrieve().unwrap();
    let d = d.retrieve().unwrap();
    assert_close(&unoptimized_a, &a);
    assert_close(&unoptimized_d, &d);
}

#[test]
fn test_matmul() {
    let mut cx = Graph::new();
    let b = cx.new_tensor::<R2<3, 1>>();
    let c = cx.new_tensor::<R2<1, 4>>();

    let a = b.matmul(c);

    a.mark();
    b.set(vec![1.0, 2.0, 3.0]);
    c.set(vec![1.0, 2.0, 3.0, 3.0]);

    cx.execute();

    let unoptimized_a = a.retrieve().unwrap();

    cx.optimize(GeneralOptimizer::default());
    cx.execute();

    assert_close(&unoptimized_a, &a.retrieve().unwrap());
}

#[test]
fn test_linear() {
    unsafe { backtrace_on_stack_overflow::enable() };
    let mut cx = Graph::new();
    let batch = cx.new_tensor::<R2<2, 3>>();
    let a = cx.new_tensor::<R1<3>>();

    let model: Linear<3, 4> = Linear::initialize(&mut cx);
    let b = model.forward(a);
    let batch_out = model.forward(batch);

    b.mark();
    a.mark();
    batch_out.mark();
    a.set(vec![1.0, 2.0, 3.0]);
    batch.set(vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);
    cx.execute();

    let unoptimized_b = b.retrieve().unwrap();
    let unoptimized_batch_out = batch_out.retrieve().unwrap();

    cx.optimize(GeneralOptimizer::default());
    cx.execute();

    assert_close(&unoptimized_b, &b.retrieve().unwrap());
    assert_close(&unoptimized_batch_out, &batch_out.retrieve().unwrap());
}

#[test]
fn test_shapes() {
    let mut cx = Graph::new();
    let a = cx.new_tensor::<R1<4>>();

    let b: GraphTensor<R2<2, 2>> = a.reshape::<R2<2, 2>>().permute::<_, _, Axes2<1, 0>>();

    a.set(vec![1., 2., 3., 4.]);

    cx.execute();

    let b = b.retrieve().unwrap();

    assert_close_data(&extract_real_data(&b), &[1., 3., 2., 4.]);
}

/// Ensure two tensors are nearly equal
fn assert_close(a: &Tensor, b: &Tensor) {
    assert_eq!(a.shape.shape(), b.shape.shape(), "Shapes don't match");
    assert_close_data(&a.data, &b.data);
}

/// Ensure two tensor data arrays are nearly equal
fn assert_close_data(a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len(), "Number of elements doesn't match");
    for (a, b) in a.iter().zip(b.iter()) {
        if (a - b).abs() > 0.01 {
            panic!("{a} is not close to {b}");
        }
    }
}

/// Get the real data as layed out by the shape tracker
fn extract_real_data(tensor: &Tensor) -> Vec<f32> {
    let idx_fn = tensor.shape.index_fn();
    let mut real_data = vec![];
    for i in 0..tensor.data.len() {
        real_data.push(tensor.data[(idx_fn)(i)]);
    }
    real_data
}
