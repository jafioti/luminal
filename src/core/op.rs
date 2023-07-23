use std::fmt::Debug;

use crate::{shape::ShapeTracker, tensor::Tensor};

pub trait Operator: Debug {
    fn name(&self) -> &'static str;
    fn process(&self, inp: Vec<&Tensor>) -> Tensor;
}

#[derive(Debug, Clone)]
pub struct Input;
impl Operator for Input {
    fn name(&self) -> &'static str {
        "Input"
    }
    fn process(&self, _: Vec<&Tensor>) -> Tensor {
        panic!("The graph was run without an input set!");
    }
}

// Movement Op (A -> B)

#[derive(Debug, Clone)]
pub struct Permute(pub Vec<usize>);
impl Operator for Permute {
    fn name(&self) -> &'static str {
        "Permute"
    }
    fn process(&self, inp: Vec<&Tensor>) -> Tensor {
        // We don't need to clone here! We should switch to a more view oriented system
        let mut t = inp[0].clone();
        t.shape.permute(&self.0);
        t
    }
}

#[derive(Debug, Clone)]
pub struct Reshape(pub Vec<usize>);
impl Operator for Reshape {
    fn name(&self) -> &'static str {
        "Reshape"
    }
    fn process(&self, inp: Vec<&Tensor>) -> Tensor {
        // We don't need to clone here! We should switch to a more view oriented system
        let mut t = inp[0].clone();
        t.shape.reshape(self.0.clone());
        t
    }
}

#[derive(Debug, Clone)]
pub struct Expand(pub Vec<(usize, usize)>);
impl Operator for Expand {
    fn name(&self) -> &'static str {
        "Expand"
    }
    fn process(&self, inp: Vec<&Tensor>) -> Tensor {
        // We don't need to clone here! We should switch to a more view oriented system
        let mut t = inp[0].clone();
        for (dim, size) in &self.0 {
            t.shape.expand(*dim, *size);
        }
        t
    }
}

// Below are the primitive operators currently supported

// Unary Op (A -> A)

#[derive(Debug, Clone)]
pub struct Log2;
impl Operator for Log2 {
    fn name(&self) -> &'static str {
        "Log2"
    }
    fn process(&self, tensors: Vec<&Tensor>) -> Tensor {
        let mut t = tensors[0].clone();
        for a in t.data.iter_mut() {
            *a = a.log2();
        }

        t
    }
}

#[derive(Debug, Clone)]
pub struct Exp2;
impl Operator for Exp2 {
    fn name(&self) -> &'static str {
        "Exp2"
    }
    fn process(&self, tensors: Vec<&Tensor>) -> Tensor {
        let mut t = tensors[0].clone();
        for a in t.data.iter_mut() {
            *a = a.exp2();
        }

        t
    }
}

#[derive(Debug, Clone)]
pub struct Sin;
impl Operator for Sin {
    fn name(&self) -> &'static str {
        "Sin"
    }
    fn process(&self, tensors: Vec<&Tensor>) -> Tensor {
        let mut t = tensors[0].clone();
        for a in t.data.iter_mut() {
            *a = a.sin();
        }

        t
    }
}

#[derive(Debug, Clone)]
pub struct Sqrt;
impl Operator for Sqrt {
    fn name(&self) -> &'static str {
        "Sqrt"
    }
    fn process(&self, tensors: Vec<&Tensor>) -> Tensor {
        let mut t = tensors[0].clone();
        for a in t.data.iter_mut() {
            *a = a.sqrt();
        }

        t
    }
}

#[derive(Debug, Clone)]
pub struct Recip;
impl Operator for Recip {
    fn name(&self) -> &'static str {
        "Recip"
    }
    fn process(&self, tensors: Vec<&Tensor>) -> Tensor {
        let mut t = tensors[0].clone();
        for a in t.data.iter_mut() {
            *a = a.recip();
        }

        t
    }
}

// Binary Ops (A x A -> A)

#[derive(Debug, Clone)]
pub struct Add;
impl Operator for Add {
    fn name(&self) -> &'static str {
        "Add"
    }
    fn process(&self, tensors: Vec<&Tensor>) -> Tensor {
        let mut t = Tensor {
            data: vec![0.; tensors[0].shape.shape().iter().product()],
            shape: ShapeTracker::new(tensors[0].shape.shape().clone()),
        };
        let r_idx = t.shape.index_fn();
        let a_idx = tensors[0].shape.index_fn();
        let b_idx = tensors[1].shape.index_fn();
        for i in 0..t.data.len() {
            t.data[(r_idx)(i)] = tensors[0].data[(a_idx)(i)] + tensors[1].data[(b_idx)(i)];
        }
        t
    }
}

#[derive(Debug, Clone)]
pub struct Sub;
impl Operator for Sub {
    fn name(&self) -> &'static str {
        "Subtract"
    }
    fn process(&self, tensors: Vec<&Tensor>) -> Tensor {
        let mut t = Tensor {
            data: vec![0.; tensors[0].shape.shape().iter().product()],
            shape: ShapeTracker::new(tensors[0].shape.shape().clone()),
        };
        let r_idx = t.shape.index_fn();
        let a_idx = tensors[0].shape.index_fn();
        let b_idx = tensors[1].shape.index_fn();
        for i in 0..t.data.len() {
            t.data[(r_idx)(i)] = tensors[0].data[(a_idx)(i)] - tensors[1].data[(b_idx)(i)];
        }
        t
    }
}

#[derive(Debug, Clone)]
pub struct Mul;
impl Operator for Mul {
    fn name(&self) -> &'static str {
        "Mul"
    }
    fn process(&self, tensors: Vec<&Tensor>) -> Tensor {
        let mut t = Tensor {
            data: vec![0.; tensors[0].shape.shape().iter().product()],
            shape: ShapeTracker::new(tensors[0].shape.shape().clone()),
        };
        let r_idx = t.shape.index_fn();
        let a_idx = tensors[0].shape.index_fn();
        let b_idx = tensors[1].shape.index_fn();
        for i in 0..t.data.len() {
            t.data[(r_idx)(i)] = tensors[0].data[(a_idx)(i)] * tensors[1].data[(b_idx)(i)];
        }
        t
    }
}

#[derive(Debug, Clone)]
pub struct Div;
impl Operator for Div {
    fn name(&self) -> &'static str {
        "Div"
    }
    fn process(&self, tensors: Vec<&Tensor>) -> Tensor {
        let mut t = tensors[0].clone();
        let t_idx = t.shape.index_fn();
        let o_idx = tensors[0].shape.index_fn();
        for i in 0..tensors[0].data.len() {
            t.data[(t_idx)(i)] /= tensors[1].data[(o_idx)(i)];
        }
        t
    }
}

#[derive(Debug, Clone)]
pub struct Max;
impl Operator for Max {
    fn name(&self) -> &'static str {
        "Max"
    }
    fn process(&self, tensors: Vec<&Tensor>) -> Tensor {
        let mut t = Tensor {
            data: vec![0.; tensors[0].shape.shape().iter().product()],
            shape: ShapeTracker::new(tensors[0].shape.shape().clone()),
        };
        let r_idx = t.shape.index_fn();
        let a_idx = tensors[0].shape.index_fn();
        let b_idx = tensors[1].shape.index_fn();
        for i in 0..t.data.len() {
            t.data[(r_idx)(i)] = tensors[0].data[(a_idx)(i)].max(tensors[1].data[(b_idx)(i)]);
        }
        t
    }
}

#[derive(Debug, Clone)]
pub struct Mod;
impl Operator for Mod {
    fn name(&self) -> &'static str {
        "Mod"
    }
    fn process(&self, tensors: Vec<&Tensor>) -> Tensor {
        let mut t = Tensor {
            data: vec![0.; tensors[0].shape.shape().iter().product()],
            shape: ShapeTracker::new(tensors[0].shape.shape().clone()),
        };
        let r_idx = t.shape.index_fn();
        let a_idx = tensors[0].shape.index_fn();
        let b_idx = tensors[1].shape.index_fn();
        for i in 0..t.data.len() {
            t.data[(r_idx)(i)] = tensors[0].data[(a_idx)(i)] % tensors[1].data[(b_idx)(i)];
        }
        t
    }
}

// Reduce Ops (A -> B (different shape))

#[derive(Debug, Clone)]
pub struct SumReduce(pub usize);
impl Operator for SumReduce {
    fn name(&self) -> &'static str {
        "SumReduce"
    }
    fn process(&self, tensors: Vec<&Tensor>) -> Tensor {
        let mut shape_tracker = tensors[0].shape.clone();
        let a_idx = shape_tracker.index_fn();
        let dim_stride = shape_tracker.views.last().unwrap().strides[self.0]; // This is probably wrong
        let dim_size = shape_tracker.shape()[self.0];

        let mut result = vec![
            0.0;
            shape_tracker
                .shape()
                .iter()
                .enumerate()
                .filter(|(i, _)| *i != self.0)
                .map(|(_, sh)| sh)
                .product()
        ];

        for (i, result) in result.iter_mut().enumerate() {
            let i = (a_idx)(i * dim_size);
            for j in 0..dim_size {
                *result += tensors[0].data[i + dim_stride * j];
            }
        }

        let mut prev_shape = shape_tracker.shape().clone();
        prev_shape.remove(self.0);
        shape_tracker.reshape(prev_shape);

        Tensor {
            data: result,
            shape: shape_tracker,
        }
    }
}

#[derive(Debug, Clone)]
pub struct MaxReduce(pub usize);
impl Operator for MaxReduce {
    fn name(&self) -> &'static str {
        "MaxReduce"
    }
    fn process(&self, tensors: Vec<&Tensor>) -> Tensor {
        let mut shape_tracker = tensors[0].shape.clone();
        let a_idx = shape_tracker.index_fn();
        let dim_stride = shape_tracker.views.last().unwrap().strides[self.0]; // This is probably wrong
        let dim_size = shape_tracker.shape()[self.0];

        let mut result: Vec<f32> = vec![
            0.0;
            shape_tracker
                .shape()
                .iter()
                .enumerate()
                .filter(|(i, _)| *i != self.0)
                .map(|(_, sh)| sh)
                .product()
        ];

        for (i, result) in result.iter_mut().enumerate() {
            let i = (a_idx)(i * dim_size);
            for j in 0..dim_size {
                *result = (*result).max(tensors[0].data[i + dim_stride * j]);
            }
        }

        let mut prev_shape = shape_tracker.shape().clone();
        prev_shape.remove(self.0);
        shape_tracker.reshape(prev_shape);

        Tensor {
            data: result,
            shape: shape_tracker,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{prelude::*, tests::assert_close_data};
    use dfdx::prelude::*;

    // Movement op tests

    #[test]
    fn test_reshape() {
        let mut cx = Graph::new();
        let a = cx.new_tensor::<R2<2, 3>>();
        a.set(vec![1., 2., 3., 1., 2., 3.]);
        let b = a.reshape::<R1<6>>();
        cx.execute();

        let d_dev = Cpu::default();
        let d_a = d_dev.tensor([[1., 2., 3.], [1., 2., 3.]]);
        let d_b: dfdx::tensor::Tensor<Rank1<6>, f32, Cpu> = d_a.reshape();

        assert_close_data(&b.retrieve().unwrap().real_data(), &d_b.as_vec());
    }

    #[test]
    fn test_permute() {
        let mut cx = Graph::new();
        let a = cx.new_tensor::<R2<2, 3>>();
        a.set(vec![1., 2., 3., 1., 2., 3.]);
        let b: GraphTensor<R2<3, 2>> = a.permute();
        cx.execute();

        let d_dev = Cpu::default();
        let d_a = d_dev.tensor([[1., 2., 3.], [1., 2., 3.]]);
        let d_b: dfdx::tensor::Tensor<Rank2<3, 2>, f32, Cpu> = d_a.permute();

        assert_close_data(&b.retrieve().unwrap().real_data(), &d_b.as_vec());
    }

    #[test]
    fn test_expand() {
        let mut cx = Graph::new();
        let a = cx.new_tensor::<R1<3>>();
        a.set(vec![1., 2., 3.]);
        let b: GraphTensor<R2<3, 2>> = a.expand();
        cx.execute();

        let d_dev = Cpu::default();
        let d_a = d_dev.tensor([1., 2., 3.]);
        let d_b: dfdx::tensor::Tensor<Rank2<3, 2>, f32, Cpu> = d_a.broadcast();

        let r = b.retrieve().unwrap().real_data();
        assert_close_data(&r, &d_b.as_vec());
    }

    // Unary op tests

    #[test]
    fn test_log2() {
        // We can't use dfdx because it doesn't implement this op
        let mut cx = Graph::new();
        let a = cx.new_tensor::<R1<3>>();
        a.set(vec![1., 2., 3.]);
        let b = a.log_2();
        cx.execute();

        assert_close_data(
            &b.retrieve().unwrap().real_data(),
            &vec![1., 2., 3.]
                .into_iter()
                .map(|i: f32| i.log2())
                .collect::<Vec<_>>(),
        );
    }

    #[test]
    fn test_exp2() {
        // We can't use dfdx because it doesn't implement this op
        let mut cx = Graph::new();
        let a = cx.new_tensor::<R1<3>>();
        a.set(vec![1., 2., 3.]);
        let b = a.exp_2();
        cx.execute();

        assert_close_data(
            &b.retrieve().unwrap().real_data(),
            &vec![1., 2., 3.]
                .into_iter()
                .map(|i: f32| i.exp2())
                .collect::<Vec<_>>(),
        );
    }

    #[test]
    fn test_recip() {
        let mut cx = Graph::new();
        let a = cx.new_tensor::<R1<3>>();
        a.set(vec![1., 2., 3.]);
        let b = a.recip();
        cx.execute();

        let d_dev = Cpu::default();
        let d_a = d_dev.tensor([1., 2., 3.]);
        let d_b = d_a.recip();

        assert_close_data(&b.retrieve().unwrap().real_data(), &d_b.as_vec());
    }

    #[test]
    fn test_sin() {
        let mut cx = Graph::new();
        let a = cx.new_tensor::<R1<3>>();
        a.set(vec![1., 2., 3.]);
        let b = a.sin();
        cx.execute();

        let d_dev = Cpu::default();
        let d_a = d_dev.tensor([1., 2., 3.]);
        let d_b = d_a.sin();

        assert_close_data(&b.retrieve().unwrap().real_data(), &d_b.as_vec());
    }

    #[test]
    fn test_sqrt() {
        let mut cx = Graph::new();
        let a = cx.new_tensor::<R1<3>>();
        a.set(vec![1., 2., 3.]);
        let b = a.sqrt();
        cx.execute();

        let d_dev = Cpu::default();
        let d_a = d_dev.tensor([1., 2., 3.]);
        let d_b = d_a.sqrt();

        assert_close_data(&b.retrieve().unwrap().real_data(), &d_b.as_vec());
    }

    // Binary op tests

    #[test]
    fn test_add() {
        let mut cx = Graph::new();
        let a = cx.new_tensor::<R1<3>>();
        a.set(vec![1., 2., 3.]);
        let b = cx.new_tensor::<R1<3>>();
        b.set(vec![1., 2., 3.]);
        let c = a + b;
        cx.execute();

        let d_dev = Cpu::default();
        let d_a = d_dev.tensor([1., 2., 3.]);
        let d_b = d_dev.tensor([1., 2., 3.]);
        let d_c = d_a + d_b;

        assert_close_data(&c.retrieve().unwrap().real_data(), &d_c.as_vec());
    }

    #[test]
    fn test_sub() {
        let mut cx = Graph::new();
        let a = cx.new_tensor::<R1<3>>();
        a.set(vec![1., 2., 3.]);
        let b = cx.new_tensor::<R1<3>>();
        b.set(vec![1., 2., 3.]);
        let c = a - b;
        cx.execute();

        let d_dev = Cpu::default();
        let d_a = d_dev.tensor([1., 2., 3.]);
        let d_b = d_dev.tensor([1., 2., 3.]);
        let d_c = d_a - d_b;

        assert_close_data(&c.retrieve().unwrap().real_data(), &d_c.as_vec());
    }

    #[test]
    fn test_mul() {
        let mut cx = Graph::new();
        let a = cx.new_tensor::<R1<3>>();
        a.set(vec![1., 2., 3.]);
        let b = cx.new_tensor::<R1<3>>();
        b.set(vec![1., 2., 3.]);
        let c = a * b;
        cx.execute();

        let d_dev = Cpu::default();
        let d_a = d_dev.tensor([1., 2., 3.]);
        let d_b = d_dev.tensor([1., 2., 3.]);
        let d_c = d_a * d_b;

        assert_close_data(&c.retrieve().unwrap().real_data(), &d_c.as_vec());
    }

    #[test]
    fn test_div() {
        let mut cx = Graph::new();
        let a = cx.new_tensor::<R1<3>>();
        a.set(vec![1., 2., 3.]);
        let b = cx.new_tensor::<R1<3>>();
        b.set(vec![1., 2., 3.]);
        let c = a / b;
        cx.execute();

        let d_dev = Cpu::default();
        let d_a = d_dev.tensor([1., 2., 3.]);
        let d_b = d_dev.tensor([1., 2., 3.]);
        let d_c = d_a / d_b;

        assert_close_data(&c.retrieve().unwrap().real_data(), &d_c.as_vec());
    }

    #[test]
    fn test_max() {
        let mut cx = Graph::new();
        let a = cx.new_tensor::<R1<3>>();
        a.set(vec![1., 2., 3.]);
        let b = cx.new_tensor::<R1<3>>();
        b.set(vec![1., 2., 3.]);
        let c = a.max(b);
        cx.execute();

        let d_dev = Cpu::default();
        let d_a = d_dev.tensor([1., 2., 3.]);
        let d_b = d_dev.tensor([1., 2., 3.]);
        let d_c = d_a.maximum(d_b);

        assert_close_data(&c.retrieve().unwrap().real_data(), &d_c.as_vec());
    }

    // Reduction op tests

    #[test]
    fn test_sum_reduce() {
        let mut cx = Graph::new();
        let a = cx.new_tensor::<R2<2, 3>>();
        a.set(vec![1., 2., 3., 1., 2., 3.]);
        let b = a.sum_reduce::<_, crate::prelude::Axis<1>>();
        cx.execute();

        let d_dev = Cpu::default();
        let d_a = d_dev.tensor([[1., 2., 3.], [1., 2., 3.]]);
        let d_b = d_a.sum::<_, dfdx::shapes::Axis<1>>();

        assert_close_data(&b.retrieve().unwrap().real_data(), &d_b.as_vec());
    }

    #[test]
    fn test_max_reduce() {
        let mut cx = Graph::new();
        let a = cx.new_tensor::<R2<2, 3>>();
        a.set(vec![1., 2., 3., 1., 2., 3.]);
        let b = a.max_reduce::<_, crate::prelude::Axis<1>>();
        cx.execute();

        let d_dev = Cpu::default();
        let d_a = d_dev.tensor([[1., 2., 3.], [1., 2., 3.]]);
        let d_b = d_a.max::<_, dfdx::shapes::Axis<1>>();

        assert_close_data(&b.retrieve().unwrap().real_data(), &d_b.as_vec());
    }

    // Other tests (matmul, batch matmul, etc.)
    #[test]
    fn test_matrix_vector() {
        let mut cx = Graph::new();
        let a = cx.new_tensor::<R1<3>>();
        a.set(vec![1., 2., 3.]);
        let b = cx.new_tensor::<R2<3, 2>>();
        b.set(vec![1., 2., 3., 1., 2., 3.]);
        let c = a.matmul(b);
        cx.execute();

        let d_dev = Cpu::default();
        let d_a = d_dev.tensor([1., 2., 3.]);
        let d_b = d_dev.tensor([[1., 2.], [3., 1.], [2., 3.]]);
        let d_c = d_a.matmul(d_b);

        let r = c.retrieve().unwrap();
        assert_close_data(&r.real_data(), &d_c.as_vec());
    }

    #[test]
    fn test_matmul() {
        let mut cx = Graph::new();
        let a = cx.new_tensor::<R2<2, 3>>();
        a.set(vec![1., 2., 3., 1., 2., 3.]);
        let b = cx.new_tensor::<R2<3, 3>>();
        b.set(vec![1., 2., 3., 1., 2., 3., 1., 2., 3.]);
        let c = a.matmul(b);
        cx.execute();

        let d_dev = Cpu::default();
        let d_a = d_dev.tensor([[1., 2., 3.], [1., 2., 3.]]);
        let d_b = d_dev.tensor([[1., 2., 3.], [1., 2., 3.], [1., 2., 3.]]);
        let d_c = d_a.matmul(d_b);

        let r = c.retrieve().unwrap();
        assert_close_data(&r.real_data(), &d_c.as_vec());
    }

    #[test]
    fn test_batch_matmul() {
        let mut cx = Graph::new();
        let a = cx.new_tensor::<R3<2, 3, 1>>();
        a.set(vec![1., 2., 3., 1., 2., 3.]);
        let b = cx.new_tensor::<R2<1, 4>>();
        b.set(vec![1., 2., 3., 1.]);
        let c = a.matmul(b);

        cx.execute();

        let d_dev = Cpu::default();
        let d_a = d_dev.tensor([[[1.], [2.], [3.]], [[1.], [2.], [3.]]]);
        let d_b = d_dev.tensor([[1., 2., 3., 1.]]);
        let d_c = d_a.matmul(d_b);

        let r = c.retrieve().unwrap();
        assert_close_data(&r.real_data(), &d_c.as_vec());
    }
}
