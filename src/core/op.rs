#![allow(clippy::needless_range_loop)]

use std::fmt::Debug;

use crate::{
    prelude::{
        simple_tracker::{Dim, ShapeTracker},
        TraitObjEq,
    },
    tensor::Tensor,
};

pub enum InputTensor<'a> {
    Owned(Tensor),
    Borrowed(&'a Tensor),
}

impl<'a> InputTensor<'a> {
    pub fn borrowed(&'a self) -> &'a Tensor {
        match self {
            InputTensor::Owned(t) => t,
            InputTensor::Borrowed(t) => t,
        }
    }
}

pub trait Operator: Debug + TraitObjEq {
    fn process(
        &self,
        inp: Vec<(
            InputTensor,
            crate::core::shape::simple_tracker::ShapeTracker,
        )>,
    ) -> Tensor;
}

/// An opaque function running on CPU that takes in tensor references and outputs a new tensor
#[allow(clippy::type_complexity)]
pub struct Function(
    pub String,
    pub  Box<
        dyn Fn(
            Vec<(
                InputTensor,
                crate::core::shape::simple_tracker::ShapeTracker,
            )>,
        ) -> Tensor,
    >,
);

impl PartialEq for Function {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl Operator for Function {
    fn process(&self, inp: Vec<(InputTensor, ShapeTracker)>) -> Tensor {
        (self.1)(inp)
    }
}

impl Debug for Function {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// An operand that does nothing. It's purpose is to force a shape resolution between multiple tensors
#[derive(Debug, Clone, Default, PartialEq)]
pub struct NoOp;

impl Operator for NoOp {
    fn process(&self, mut inp: Vec<(InputTensor, ShapeTracker)>) -> Tensor {
        match inp.pop().unwrap().0 {
            InputTensor::Owned(t) => t,
            InputTensor::Borrowed(t) => Tensor {
                data: t.data.clone(),
            },
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Print(pub String);
impl Operator for Print {
    fn process(&self, inp: Vec<(InputTensor, ShapeTracker)>) -> Tensor {
        println!("{}", self.0);
        for (i, (tensor, tracker)) in inp.iter().enumerate() {
            println!("{} Data: {:?}", i + 1, tensor.borrowed().data);
            println!("{} Shape: {:?}", i + 1, tracker);
        }
        Tensor {
            data: Box::<Vec<f32>>::default(),
        }
    }
}

/// Ensure a tensor is contiguously layed out in memory. May involve copying
#[derive(Debug, Clone, PartialEq)]
pub struct Contiguous;
impl Operator for Contiguous {
    fn process(&self, inp: Vec<(InputTensor, ShapeTracker)>) -> Tensor {
        // Copy data over to new tensor
        let src = inp[0]
            .0
            .borrowed()
            .data
            .as_any()
            .downcast_ref::<Vec<f32>>()
            .unwrap();
        let mut res = vec![0.; inp[0].1.n_elements()];
        for i in 0..res.len() {
            if let Some(n) = inp[0].1.index(i) {
                res[i] = src[n];
            }
        }
        Tensor {
            data: Box::new(res),
        }
    }
}

// Below are the primitive operators currently supported

// Unary Op (A -> A)

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Log2;
impl Operator for Log2 {
    fn process(&self, mut inp: Vec<(InputTensor, ShapeTracker)>) -> Tensor {
        let mut t = match inp.pop().unwrap().0 {
            InputTensor::Borrowed(t) => t.clone(),
            InputTensor::Owned(t) => t,
        };
        for a in t
            .data
            .as_any_mut()
            .downcast_mut::<Vec<f32>>()
            .unwrap()
            .iter_mut()
        {
            *a = a.log2();
        }

        t
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Exp2;
impl Operator for Exp2 {
    fn process(&self, mut inp: Vec<(InputTensor, ShapeTracker)>) -> Tensor {
        let mut t = match inp.pop().unwrap().0 {
            InputTensor::Borrowed(t) => t.clone(),
            InputTensor::Owned(t) => t,
        };
        for a in t
            .data
            .as_any_mut()
            .downcast_mut::<Vec<f32>>()
            .unwrap()
            .iter_mut()
        {
            *a = a.exp2();
        }

        t
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Sin;
impl Operator for Sin {
    fn process(&self, mut inp: Vec<(InputTensor, ShapeTracker)>) -> Tensor {
        let mut t = match inp.pop().unwrap().0 {
            InputTensor::Borrowed(t) => t.clone(),
            InputTensor::Owned(t) => t,
        };
        for a in t
            .data
            .as_any_mut()
            .downcast_mut::<Vec<f32>>()
            .unwrap()
            .iter_mut()
        {
            *a = a.sin();
        }
        t
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Sqrt;
impl Operator for Sqrt {
    fn process(&self, mut inp: Vec<(InputTensor, ShapeTracker)>) -> Tensor {
        let mut t = match inp.pop().unwrap().0 {
            InputTensor::Borrowed(t) => t.clone(),
            InputTensor::Owned(t) => t,
        };
        for a in t
            .data
            .as_any_mut()
            .downcast_mut::<Vec<f32>>()
            .unwrap()
            .iter_mut()
        {
            *a = a.sqrt();
        }
        t
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Recip;
impl Operator for Recip {
    fn process(&self, mut inp: Vec<(InputTensor, ShapeTracker)>) -> Tensor {
        let mut t = match inp.pop().unwrap().0 {
            InputTensor::Borrowed(t) => t.clone(),
            InputTensor::Owned(t) => t,
        };
        for a in t
            .data
            .as_any_mut()
            .downcast_mut::<Vec<f32>>()
            .unwrap()
            .iter_mut()
        {
            *a = a.recip();
        }
        t
    }
}

// Binary Ops (A x A -> A)

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Add;
impl Operator for Add {
    fn process(&self, inp: Vec<(InputTensor, ShapeTracker)>) -> Tensor {
        let (a_data, b_data) = (
            inp[0]
                .0
                .borrowed()
                .data
                .as_any()
                .downcast_ref::<Vec<f32>>()
                .unwrap(),
            inp[1]
                .0
                .borrowed()
                .data
                .as_any()
                .downcast_ref::<Vec<f32>>()
                .unwrap(),
        );
        let mut data = vec![0.; inp[0].1.n_elements()];
        for i in 0..data.len() {
            data[i] = inp[0].1.index(i).map(|i| a_data[i]).unwrap_or_default()
                + inp[1].1.index(i).map(|i| b_data[i]).unwrap_or_default();
        }
        Tensor {
            data: Box::new(data),
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Mul;
impl Operator for Mul {
    fn process(&self, inp: Vec<(InputTensor, ShapeTracker)>) -> Tensor {
        let (a_data, b_data) = (
            inp[0]
                .0
                .borrowed()
                .data
                .as_any()
                .downcast_ref::<Vec<f32>>()
                .unwrap(),
            inp[1]
                .0
                .borrowed()
                .data
                .as_any()
                .downcast_ref::<Vec<f32>>()
                .unwrap(),
        );
        println!("Mulling with {:?} and {:?}", inp[0].1, inp[1].1);
        let mut data = vec![0.; inp[0].1.n_elements()];
        for i in 0..data.len() {
            data[i] = inp[0].1.index(i).map(|i| a_data[i]).unwrap_or_default()
                * inp[1].1.index(i).map(|i| b_data[i]).unwrap_or_default();
        }
        Tensor {
            data: Box::new(data),
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Mod;
impl Operator for Mod {
    fn process(&self, inp: Vec<(InputTensor, ShapeTracker)>) -> Tensor {
        let (a_data, b_data) = (
            inp[0]
                .0
                .borrowed()
                .data
                .as_any()
                .downcast_ref::<Vec<f32>>()
                .unwrap(),
            inp[1]
                .0
                .borrowed()
                .data
                .as_any()
                .downcast_ref::<Vec<f32>>()
                .unwrap(),
        );
        let mut data = vec![0.; inp[0].1.n_elements()];
        for i in 0..data.len() {
            data[i] = inp[0].1.index(i).map(|i| a_data[i]).unwrap_or_default()
                % inp[1].1.index(i).map(|i| b_data[i]).unwrap_or_default();
        }
        Tensor {
            data: Box::new(data),
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct LessThan;
impl Operator for LessThan {
    fn process(&self, inp: Vec<(InputTensor, ShapeTracker)>) -> Tensor {
        let (a_data, b_data) = (
            inp[0]
                .0
                .borrowed()
                .data
                .as_any()
                .downcast_ref::<Vec<f32>>()
                .unwrap(),
            inp[1]
                .0
                .borrowed()
                .data
                .as_any()
                .downcast_ref::<Vec<f32>>()
                .unwrap(),
        );
        let mut data = vec![0.; inp[0].1.n_elements()];
        for i in 0..data.len() {
            let a = inp[0].1.index(i).map(|i| a_data[i]).unwrap_or_default();
            let b = inp[1].1.index(i).map(|i| b_data[i]).unwrap_or_default();
            data[i] = if a < b { 1. } else { 0. };
        }
        Tensor {
            data: Box::new(data),
        }
    }
}

// Reduce Ops (A -> B (different shape))

#[derive(Debug, Clone, Default, PartialEq)]
pub struct SumReduce(pub usize);
impl Operator for SumReduce {
    fn process(&self, inp: Vec<(InputTensor, ShapeTracker)>) -> Tensor {
        let front_size: usize = inp[0]
            .1
            .shape()
            .iter()
            .take(self.0)
            .filter_map(|i| if let Dim::Known(n) = i { Some(n) } else { None })
            .product();
        let back_size: usize = inp[0]
            .1
            .shape()
            .iter()
            .skip(self.0 + 1)
            .filter_map(|i| if let Dim::Known(n) = i { Some(n) } else { None })
            .product();
        let dim_size = match inp[0].1.shape()[self.0] {
            Dim::Known(n) => n,
            Dim::Unknown(_) => panic!("Can't reduce over an unknown dimension"),
        };
        let mut result: Vec<f32> = vec![0.0; front_size * back_size];
        let a_data = inp[0]
            .0
            .borrowed()
            .data
            .as_any()
            .downcast_ref::<Vec<f32>>()
            .unwrap();

        for i in 0..front_size {
            for j in 0..back_size {
                for k in 0..dim_size {
                    let original_index = i * dim_size * back_size + k * back_size + j;
                    let new_index = i * back_size + j;
                    if let Some(n) = inp[0].1.index(original_index) {
                        result[new_index] += a_data[n];
                    }
                }
            }
        }
        Tensor {
            data: Box::new(result),
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct MaxReduce(pub usize);
impl Operator for MaxReduce {
    fn process(&self, inp: Vec<(InputTensor, ShapeTracker)>) -> Tensor {
        let front_size: usize = inp[0]
            .1
            .shape()
            .iter()
            .take(self.0)
            .filter_map(|i| if let Dim::Known(n) = i { Some(n) } else { None })
            .product();
        let back_size: usize = inp[0]
            .1
            .shape()
            .iter()
            .skip(self.0 + 1)
            .filter_map(|i| if let Dim::Known(n) = i { Some(n) } else { None })
            .product();
        let dim_size = match inp[0].1.shape()[self.0] {
            Dim::Known(n) => n,
            Dim::Unknown(_) => panic!("Can't reduce over an unknown dimension"),
        };
        let mut result: Vec<f32> = vec![-f32::INFINITY; front_size * back_size];
        let a_data = inp[0]
            .0
            .borrowed()
            .data
            .as_any()
            .downcast_ref::<Vec<f32>>()
            .unwrap();

        for i in 0..front_size {
            for j in 0..back_size {
                for k in 0..dim_size {
                    let original_index = i * dim_size * back_size + k * back_size + j;
                    let new_index = i * back_size + j;
                    if let Some(n) = inp[0].1.index(original_index) {
                        result[new_index] = result[new_index].max(a_data[n]);
                    }
                }
            }
        }
        Tensor {
            data: Box::new(result),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{prelude::*, tests::assert_close_data};
    use dfdx::prelude::*;
    use itertools::Itertools;

    // Movement op tests

    #[test]
    fn test_reshape() {
        let mut cx = Graph::new();
        let a = cx.new_tensor::<R2<2, 3>>("Input");
        a.set(vec![1., 2., 3., 1., 2., 3.]);
        let b = a.reshape::<R1<6>>();
        b.mark();
        cx.execute();

        let d_dev = Cpu::default();
        let d_a = d_dev.tensor([[1., 2., 3.], [1., 2., 3.]]);
        let d_b: dfdx::tensor::Tensor<Rank1<6>, f32, Cpu> = d_a.reshape();

        assert_close_data(&b.data(), &d_b.as_vec());
    }

    #[test]
    fn test_permute() {
        let mut cx = Graph::new();
        let a = cx.new_tensor::<R2<2, 3>>("Input");
        a.set(vec![1., 2., 3., 1., 2., 3.]);
        let b: GraphTensor<R2<3, 2>> = a.permute();
        b.mark();
        cx.execute();

        let d_dev = Cpu::default();
        let d_a = d_dev.tensor([[1., 2., 3.], [1., 2., 3.]]);
        let d_b: dfdx::tensor::Tensor<Rank2<3, 2>, f32, Cpu> = d_a.permute();

        assert_close_data(&b.data(), &d_b.as_vec());
    }

    #[test]
    fn test_expand() {
        let mut cx = Graph::new();
        let a = cx.new_tensor::<R1<3>>("Input");
        a.set(vec![1., 2., 3.]);
        let b: GraphTensor<R2<3, 2>> = a.expand();
        b.mark();
        cx.execute();

        let d_dev = Cpu::default();
        let d_a = d_dev.tensor([1., 2., 3.]);
        let d_b: dfdx::tensor::Tensor<Rank2<3, 2>, f32, Cpu> = d_a.broadcast();

        assert_close_data(&b.data(), &d_b.as_vec());
    }

    #[test]
    fn test_slice() {
        let mut cx = Graph::new();
        let a = cx.new_tensor::<R2<2, 3>>("Input");
        a.set(vec![1., 2., 3., 1., 2., 3.]);
        let b = a.slice((1.., ..));
        b.mark();
        cx.execute();

        let d_dev = Cpu::default();
        let d_a = d_dev.tensor([[1., 2., 3.], [1., 2., 3.]]);
        let d_b = d_a.slice((1.., ..));

        assert_close_data(&b.dyn_data(&cx.dyn_map), &d_b.as_vec());
    }

    // Unary op tests

    #[test]
    fn test_log2() {
        // We can't use dfdx because it doesn't implement this op
        let mut cx = Graph::new();
        let a = cx.new_tensor::<R1<3>>("Input");
        a.set(vec![1., 2., 3.]);
        let b = a.log_2();
        b.mark();
        cx.execute();

        assert_close_data(
            &b.data(),
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
        let a = cx.new_tensor::<R1<3>>("Input");
        a.set(vec![1., 2., 3.]);
        let b = a.exp_2();
        b.mark();
        cx.execute();

        assert_close_data(
            &b.data(),
            &vec![1., 2., 3.]
                .into_iter()
                .map(|i: f32| i.exp2())
                .collect::<Vec<_>>(),
        );
    }

    #[test]
    fn test_recip() {
        let mut cx = Graph::new();
        let a = cx.new_tensor::<R1<3>>("Input");
        a.set(vec![1., 2., 3.]);
        let b = a.recip();
        b.mark();
        cx.execute();

        let d_dev = Cpu::default();
        let d_a = d_dev.tensor([1., 2., 3.]);
        let d_b = d_a.recip();

        assert_close_data(&b.data(), &d_b.as_vec());
    }

    #[test]
    fn test_sin() {
        let mut cx = Graph::new();
        let a = cx.new_tensor::<R1<3>>("Input");
        a.set(vec![1., 2., 3.]);
        let b = a.sin();
        b.mark();
        cx.execute();

        let d_dev = Cpu::default();
        let d_a = d_dev.tensor([1., 2., 3.]);
        let d_b = d_a.sin();

        assert_close_data(&b.data(), &d_b.as_vec());
    }

    #[test]
    fn test_sqrt() {
        let mut cx = Graph::new();
        let a = cx.new_tensor::<R1<3>>("Input");
        a.set(vec![1., 2., 3.]);
        let b = a.sqrt();
        b.mark();
        cx.execute();

        let d_dev = Cpu::default();
        let d_a = d_dev.tensor([1., 2., 3.]);
        let d_b = d_a.sqrt();

        assert_close_data(&b.data(), &d_b.as_vec());
    }

    // Binary op tests

    #[test]
    fn test_add() {
        let mut cx = Graph::new();
        let a = cx.new_tensor::<R1<3>>("Input");
        a.set(vec![1., 2., 3.]);
        let b = cx.new_tensor::<R1<3>>("Input");
        b.set(vec![1., 2., 3.]);
        let c = a + b;
        c.mark();
        cx.execute();

        let d_dev = Cpu::default();
        let d_a = d_dev.tensor([1., 2., 3.]);
        let d_b = d_dev.tensor([1., 2., 3.]);
        let d_c = d_a + d_b;

        assert_close_data(&c.data(), &d_c.as_vec());
    }

    #[test]
    fn test_sub() {
        let mut cx = Graph::new();
        let a = cx.new_tensor::<R1<3>>("Input");
        a.set(vec![1., 2., 3.]);
        let b = cx.new_tensor::<R1<3>>("Input");
        b.set(vec![1., 2., 3.]);
        let c = a - b;
        c.mark();
        cx.execute();

        let d_dev = Cpu::default();
        let d_a = d_dev.tensor([1., 2., 3.]);
        let d_b = d_dev.tensor([1., 2., 3.]);
        let d_c = d_a - d_b;

        assert_close_data(&c.data(), &d_c.as_vec());
    }

    #[test]
    fn test_mul() {
        let mut cx = Graph::new();
        let a = cx.new_tensor::<R1<3>>("Input");
        a.set(vec![1., 2., 3.]);
        let b = cx.new_tensor::<R1<3>>("Input");
        b.set(vec![1., 2., 3.]);
        let c = a * b;
        c.mark();
        cx.execute();

        let d_dev = Cpu::default();
        let d_a = d_dev.tensor([1., 2., 3.]);
        let d_b = d_dev.tensor([1., 2., 3.]);
        let d_c = d_a * d_b;

        assert_close_data(&c.data(), &d_c.as_vec());
    }

    #[test]
    fn test_permute_mul() {
        let mut cx = Graph::new();
        let a = cx.new_tensor::<R2<3, 2>>("Input");
        a.set(vec![1., 2., 3., 2., 3., 1.]);
        let b = cx.new_tensor::<R2<3, 2>>("Input");
        b.set(vec![1., 2., 3., -1., 3., 0.]);
        let c = a.expand::<R3<3, 2, 3>, crate::prelude::Axis<2>>()
            * b.expand::<R3<3, 2, 3>, crate::prelude::Axis<2>>();
        c.mark();
        cx.execute();

        let d_dev = Cpu::default();
        let d_a = d_dev.tensor([[1., 2.], [3., 2.], [3., 1.]]);
        let d_b = d_dev.tensor([[1., 2.], [3., -1.], [3., 0.]]);
        let d_c = d_a.broadcast::<Rank3<3, 2, 3>, dfdx::prelude::Axis<2>>()
            * d_b.broadcast::<Rank3<3, 2, 3>, dfdx::prelude::Axis<2>>();

        assert_close_data(&c.data(), &d_c.as_vec());
    }

    #[test]
    fn test_div() {
        let mut cx = Graph::new();
        let a = cx.new_tensor::<R1<3>>("Input");
        a.set(vec![1., 2., 3.]);
        let b = cx.new_tensor::<R1<3>>("Input");
        b.set(vec![1., 2., 3.]);
        let c = a / b;
        c.mark();
        cx.execute();

        let d_dev = Cpu::default();
        let d_a = d_dev.tensor([1., 2., 3.]);
        let d_b = d_dev.tensor([1., 2., 3.]);
        let d_c = d_a / d_b;

        assert_close_data(&c.data(), &d_c.as_vec());
    }

    #[test]
    fn test_max() {
        let mut cx = Graph::new();
        let a = cx.new_tensor::<R1<3>>("Input");
        a.set(vec![1., 0., 3.]);
        let b = cx.new_tensor::<R1<3>>("Input");
        b.set(vec![1., 2., -2.]);
        let c = a.max(b);
        c.mark();

        cx.execute();

        let d_dev = Cpu::default();
        let d_a = d_dev.tensor([1., 0., 3.]);
        let d_b = d_dev.tensor([1., 2., -2.]);
        let d_c = d_a.maximum(d_b);

        assert_close_data(&c.data(), &d_c.as_vec());
    }

    #[test]
    fn test_mod() {
        let mut cx = Graph::new();
        let a = cx.new_tensor::<R1<3>>("Input");
        a.set(vec![1., 2., 3.]);
        let b = cx.new_tensor::<R1<3>>("Input");
        b.set(vec![1., 2., 3.]);
        let c = a % b;
        c.mark();
        cx.execute();

        // No dfdx equivalent

        assert_close_data(
            &c.data(),
            &[1., 2., 3.]
                .into_iter()
                .zip([1., 2., 3.].into_iter())
                .map(|(a, b)| a % b)
                .collect_vec(),
        );
    }

    // Reduction op tests

    #[test]
    fn test_sum_reduce() {
        let mut cx = Graph::new();
        let a = cx.new_tensor::<R3<2, 2, 3>>("Input");
        a.set(vec![1., 2., 3., 1., 2., 3., 1., 2., 3., 1., 2., 3.]);
        let b = a.sum_reduce::<_, crate::prelude::Axis<1>>();
        let c = a.sum_reduce::<_, crate::prelude::Axis<0>>();
        let d = a.sum_reduce::<_, crate::prelude::Axis<2>>();
        b.mark();
        c.mark();
        d.mark();
        cx.execute();

        let d_dev = Cpu::default();
        let d_a = d_dev.tensor([[[1., 2., 3.], [1., 2., 3.]], [[1., 2., 3.], [1., 2., 3.]]]);
        let d_b = d_a.clone().sum::<_, dfdx::shapes::Axis<1>>();
        let d_c = d_a.clone().sum::<_, dfdx::shapes::Axis<0>>();
        let d_d = d_a.sum::<_, dfdx::shapes::Axis<2>>();

        assert_close_data(&b.data(), &d_b.as_vec());
        assert_close_data(&c.data(), &d_c.as_vec());
        assert_close_data(&d.data(), &d_d.as_vec());
    }

    #[test]
    fn test_sum_reduce2() {
        let mut cx = Graph::new();
        let a = cx.new_tensor::<R4<1, 2, 2, 3>>("Input");
        a.set(vec![
            34.4, -96.0, 144.0, 43.0, 560.0, 180.0, 39.6, -120.0, 180.0, 49.5, 700.0, 225.0,
        ]);
        let b = a.sum_reduce::<_, crate::prelude::Axis<3>>();
        b.mark();
        cx.execute();

        let d_dev = Cpu::default();
        let d_a = d_dev.tensor_from_vec(
            vec![
                34.4, -96.0, 144.0, 43.0, 560.0, 180.0, 39.6, -120.0, 180.0, 49.5, 700.0, 225.0,
            ],
            (
                dfdx::shapes::Const::<1>,
                dfdx::shapes::Const::<2>,
                dfdx::shapes::Const::<2>,
                dfdx::shapes::Const::<3>,
            ),
        );
        let d_b = d_a.sum::<_, dfdx::shapes::Axis<3>>();

        assert_close_data(&b.data(), &d_b.as_vec());
    }

    #[test]
    fn test_max_reduce() {
        let mut cx = Graph::new();
        let a = cx.new_tensor::<R3<2, 2, 3>>("Input");
        a.set(vec![1., 2., 3., 1., 2., 3., 1., 2., 3., 1., 2., 3.]);
        let b = a.max_reduce::<_, crate::prelude::Axis<1>>();
        let c = a.max_reduce::<_, crate::prelude::Axis<0>>();
        let d = a.max_reduce::<_, crate::prelude::Axis<2>>();
        b.mark();
        c.mark();
        d.mark();

        cx.execute();

        let d_dev = Cpu::default();
        let d_a = d_dev.tensor([[[1., 2., 3.], [1., 2., 3.]], [[1., 2., 3.], [1., 2., 3.]]]);
        let d_b = d_a.clone().max::<_, dfdx::shapes::Axis<1>>();
        let d_c = d_a.clone().max::<_, dfdx::shapes::Axis<0>>();
        let d_d = d_a.max::<_, dfdx::shapes::Axis<2>>();

        assert_close_data(&b.data(), &d_b.as_vec());
        assert_close_data(&c.data(), &d_c.as_vec());
        assert_close_data(&d.data(), &d_d.as_vec());
    }
}
