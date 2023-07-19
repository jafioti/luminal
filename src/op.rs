use std::fmt::Debug;

use crate::{shape::default_strides, tensor::Tensor};

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
        panic!("You Fool.")
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
        let mut t = tensors[0].clone();
        let t_idx = t.shape.index_fn();
        let o_idx = tensors[0].shape.index_fn();
        for i in 0..tensors[0].data.len() {
            t.data[(t_idx)(i)] += tensors[1].data[(o_idx)(i)];
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
        let mut t = tensors[0].clone();
        let t_idx = t.shape.index_fn();
        let o_idx = tensors[0].shape.index_fn();
        for i in 0..tensors[0].data.len() {
            t.data[(t_idx)(i)] -= tensors[1].data[(o_idx)(i)];
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
        let mut t = tensors[0].clone();
        let t_idx = t.shape.index_fn();
        let o_idx = tensors[0].shape.index_fn();
        for i in 0..tensors[0].data.len() {
            t.data[(t_idx)(i)] *= tensors[1].data[(o_idx)(i)];
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
        let mut t = tensors[0].clone();
        let t_idx = t.shape.index_fn();
        let o_idx = tensors[0].shape.index_fn();
        for i in 0..tensors[0].data.len() {
            t.data[(t_idx)(i)] = t.data[(t_idx)(i)].max(tensors[1].data[(o_idx)(i)]);
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
        let mut t = tensors[0].clone();
        let t_idx = t.shape.index_fn();
        let o_idx = tensors[0].shape.index_fn();
        for i in 0..tensors[0].shape.shape().iter().product() {
            t.data[(t_idx)(i)] %= tensors[1].data[(o_idx)(i)];
        }

        t
    }
}

// Reduce Ops (A -> B (different shape))

#[derive(Debug, Clone)]
pub struct ReduceSum(pub usize);
impl Operator for ReduceSum {
    fn name(&self) -> &'static str {
        "ReduceSum"
    }
    fn process(&self, tensors: Vec<&Tensor>) -> Tensor {
        let mut shape_tracker = tensors[0].shape.clone();
        let dim_shape = shape_tracker.shape()[self.0];
        let dim_stride = default_strides(shape_tracker.shape())[self.0];
        let a_idx = shape_tracker.index_fn();

        let mut result = vec![0.0; tensors[0].data.len() / dim_shape];

        for i in 0..tensors[0].data.len() {
            let j = i / dim_stride % dim_shape; // map the index to 'shape' and 'stride'
            result[(a_idx)(i - j * dim_stride)] += tensors[0].data[(a_idx)(i)]; // perform sum at the 'j'th position
        }

        // Not sure if this works
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
pub struct ReduceMax(pub usize);
impl Operator for ReduceMax {
    fn name(&self) -> &'static str {
        "ReduceMax"
    }
    fn process(&self, tensors: Vec<&Tensor>) -> Tensor {
        let mut shape_tracker = tensors[0].shape.clone();
        let dim_shape = shape_tracker.shape()[self.0];
        let dim_stride = default_strides(shape_tracker.shape())[self.0];
        let a_idx = shape_tracker.index_fn();

        let mut result: Vec<f32> = vec![0.0; tensors[0].data.len() / dim_shape];

        for i in 0..tensors[0].data.len() {
            let j = i / dim_stride % dim_shape; // map the index to 'shape' and 'stride'
            result[(a_idx)(i - j * dim_stride)] =
                result[(a_idx)(i - j * dim_stride)].max(tensors[0].data[(a_idx)(i)]);
        }

        // Not sure if this works
        let mut prev_shape = shape_tracker.shape().clone();
        prev_shape.remove(self.0);
        shape_tracker.reshape(prev_shape);

        Tensor {
            data: result,
            shape: shape_tracker,
        }
    }
}
