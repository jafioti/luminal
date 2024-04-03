#![allow(clippy::needless_range_loop)]

use std::{any::Any, fmt::Debug, path::PathBuf};

use crate::{
    prelude::{tracker::ShapeTracker, TraitObjEq},
    tensor::Tensor,
};

use super::shape::symbolic::BigExpression;
use colored::Colorize;
use itertools::Itertools;
use rustc_hash::FxHashMap;

/// Either an owned or borrowed tensor that gets consumed by ops
pub enum InputTensor<'a> {
    /// An owned tensor
    Owned(Tensor),
    /// A borrowed tensor
    Borrowed(&'a Tensor),
}

impl<'a> InputTensor<'a> {
    /// Borrow the tensor
    pub fn borrowed(&'a self) -> &'a Tensor {
        match self {
            InputTensor::Owned(t) => t,
            InputTensor::Borrowed(t) => t,
        }
    }

    /// Unwrap or clone the tensor, depending on if it's owned or not
    pub fn cloned(self) -> Tensor {
        match self {
            InputTensor::Owned(t) => t,
            InputTensor::Borrowed(t) => t.clone(),
        }
    }
}

pub trait Operator: Debug + TraitObjEq {
    /// Process the input tensors and produce output tensors
    fn process(&mut self, inp: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor>;
    /// Implement custom functionality
    #[allow(unused)]
    fn custom(&mut self, key: &str, input: Box<dyn Any>) -> Option<Box<dyn Any>> {
        None
    }
}

/// An opaque function running on CPU that takes in Vec<f32> tensors and outputs Vec<f32> tensors
#[allow(clippy::type_complexity)]
pub struct Function(
    pub String,
    pub Box<dyn Fn(Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor>>,
);

impl PartialEq for Function {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl Operator for Function {
    fn process(&mut self, inp: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        (self.1)(inp)
    }
}

impl Debug for Function {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// An op to print the value of a tensor
#[derive(Clone, Default, PartialEq)]
pub struct Print(pub String);

impl Debug for Print {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Print-{}", self.0)
    }
}

impl Operator for Print {
    fn process(&mut self, inp: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        for (i, (tensor, tracker)) in inp.iter().enumerate() {
            println!("{}", self.0);
            let d = tensor.borrowed().downcast_ref::<Vec<f32>>().unwrap();
            println!("{} Data: {:?}", i + 1, &d[..d.len().min(10)]);
            println!("{} Shape: {:?}", i + 1, tracker);
        }
        vec![]
    }
}

/// An op to diff a tensor with a binary file
#[derive(Clone, Default, PartialEq)]
pub struct Diff(pub PathBuf, pub f32);

impl Debug for Diff {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Diff-{}", self.0.as_os_str().to_str().unwrap())
    }
}

impl Operator for Diff {
    fn process(&mut self, mut inp: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        // Get tensor data and file data
        let (tensor, shape) = inp.pop().unwrap();
        let d = tensor.borrowed().downcast_ref::<Vec<f32>>().unwrap();
        let mut data = vec![0.; d.len()];
        let (ind, val) = (shape.index_expression(), shape.valid_expression());
        let mut stack = vec![];
        #[allow(unused_mut)]
        for (i, mut r) in data.iter_mut().enumerate() {
            if val.exec_single_var_stack(i, &mut stack) != 0 {
                *r = d[ind.exec_single_var_stack(i, &mut stack)];
            }
        }
        let bin_data = std::fs::read(&self.0)
            .unwrap()
            .chunks(4)
            .map(|i| f32::from_ne_bytes([i[0], i[1], i[2], i[3]]))
            .collect::<Vec<_>>();
        if data.len() != bin_data.len() {
            println!(
                "{}",
                format!(
                    "{} | Length mismatch! Data: {}, File: {}",
                    self.0.as_os_str().to_str().unwrap(),
                    data.len(),
                    bin_data.len()
                )
                .bold()
                .red()
            );
            println!("Data Shape: {shape:?}");
            return vec![];
        }
        let data_nan = data.iter().any(|i| i.is_nan());
        let file_nan = bin_data.iter().any(|i| i.is_nan());
        if data_nan {
            println!(
                "{}",
                format!("{} | Data contains nan!", self.0.to_str().unwrap())
                    .bold()
                    .red()
            );
        }
        if file_nan {
            println!(
                "{}",
                format!("{} | File contains nan!", self.0.to_str().unwrap())
                    .bold()
                    .red()
            );
        }
        if data_nan || file_nan {
            return vec![];
        }
        let mut matched = true;
        for (i, (a, b)) in data.iter().zip(bin_data.iter()).enumerate() {
            if (a - b).abs() > self.1 {
                println!(
                    "{}",
                    format!("{} | Mismatch!", self.0.to_str().unwrap())
                        .bold()
                        .red()
                );
                if let Some((i, _)) = data.iter().enumerate().find(|(_, i)| i.is_nan()) {
                    println!("Index {} is nan!", i.to_string().bold());
                }
                println!("{a} is not equal to {b}, index {i}");
                let avg_dist = data
                    .iter()
                    .zip(bin_data.iter())
                    .map(|(a, b)| (a - b).abs())
                    .sum::<f32>()
                    / data.len() as f32;
                let max_dist = data
                    .iter()
                    .zip(bin_data.iter())
                    .map(|(a, b)| (a - b).abs())
                    .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .unwrap();
                let sum_dist = data
                    .iter()
                    .zip(bin_data.iter())
                    .map(|(a, b)| (a - b) * (a - b))
                    .sum::<f32>();
                println!(
                    "Avg dist: {}, Max dist: {} Sum dist: {}",
                    avg_dist.to_string().bold().red(),
                    max_dist.to_string().bold().red(),
                    sum_dist.to_string().bold().red(),
                );
                println!("Data Shape: {shape:?}");
                println!("{}: {:?}", "This".bold(), &data[..10]);
                println!("{}: {:?}", "File".bold(), &bin_data[..10]);
                println!(
                    "Largest Mismatches: {:?}",
                    data.iter()
                        .zip(bin_data.iter())
                        .filter(|(a, b)| (**a - **b).abs() > 0.01)
                        .sorted_by(|(a, b), (c, d)| (**c - **d)
                            .abs()
                            .partial_cmp(&(**a - **b).abs())
                            .unwrap_or(std::cmp::Ordering::Equal))
                        .take(10)
                        .collect::<Vec<_>>()
                );
                println!(
                    "A avg: {} B avg: {}",
                    data.iter().sum::<f32>() / data.len() as f32,
                    bin_data.iter().sum::<f32>() / bin_data.len() as f32
                );
                println!(
                    "A max: {} B max: {}",
                    data.iter()
                        .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                        .unwrap(),
                    bin_data
                        .iter()
                        .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                        .unwrap()
                );
                println!(
                    "A min: {} B min: {}",
                    data.iter()
                        .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                        .unwrap(),
                    bin_data
                        .iter()
                        .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                        .unwrap()
                );
                matched = false;
                break;
            }
        }
        if matched {
            println!(
                "{}",
                format!("{} matched", self.0.to_str().unwrap())
                    .bold()
                    .bright_green()
            );
        }
        vec![]
    }
}

/// A constant value placed on the graph at runtime. Can either be an expression evaluated at runtime, or a constant float
#[derive(Debug, Clone, PartialEq)]
pub enum ConstantValue {
    Expression(BigExpression),
    Float(f32),
}

/// Produces a single number constant from an expression or a float
#[derive(Clone, PartialEq)]
pub struct Constant(pub ConstantValue, pub *const FxHashMap<char, usize>);
impl Debug for Constant {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Constant(",)?;
        match &self.0 {
            ConstantValue::Expression(e) => e.fmt(f)?,
            ConstantValue::Float(fl) => fl.fmt(f)?,
        }
        write!(f, ")")
    }
}

impl Operator for Constant {
    fn process(&mut self, _: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        vec![Tensor::new(vec![match &self.0 {
            ConstantValue::Expression(e) => {
                e.exec(unsafe { self.1.as_ref().unwrap() }).unwrap() as f32
            }
            ConstantValue::Float(f) => *f,
        }])]
    }
}

/// Ensure a tensor is contiguously layed out in memory. May involve copying
#[derive(Debug, Clone, PartialEq)]
pub struct Contiguous;
impl Operator for Contiguous {
    fn process(&mut self, inp: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        // Copy data over to new tensor
        let src = get_vec_from_tensor(&inp[0].0);
        let mut res = vec![0.; inp[0].1.n_elements().to_usize().unwrap()];
        let ind = inp[0].1.index_expression();
        let val = inp[0].1.valid_expression();
        let mut stack = vec![];
        for i in 0..res.len() {
            if val.exec_single_var_stack(i, &mut stack) != 0 {
                res[i] = src[ind.exec_single_var_stack(i, &mut stack)];
            }
        }
        vec![Tensor::new(res)]
    }
}

// Below are all the primitive operators

// Unary Op (A -> A)

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Log2;
impl Operator for Log2 {
    fn process(&mut self, mut inp: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        if !inp[0].1.is_reshaped() {
            let mut tensor = inp.pop().unwrap().0.cloned();
            for a in get_vec_from_tensor_owned(&mut tensor).iter_mut() {
                *a = a.log2();
            }
            vec![tensor]
        } else {
            let mut data = vec![0.; inp[0].1.n_elements().to_usize().unwrap()];
            let inp_data = get_vec_from_tensor(&inp[0].0);
            let (ind, val) = (inp[0].1.index_expression(), inp[0].1.valid_expression());
            let mut stack = vec![];
            for i in 0..data.len() {
                if val.exec_single_var_stack(i, &mut stack) != 0 {
                    data[i] = inp_data[ind.exec_single_var_stack(i, &mut stack)].log2();
                }
            }
            vec![Tensor::new(data)]
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Exp2;
impl Operator for Exp2 {
    fn process(&mut self, mut inp: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        if !inp[0].1.is_reshaped() {
            let mut tensor = inp.pop().unwrap().0.cloned();
            for a in get_vec_from_tensor_owned(&mut tensor).iter_mut() {
                *a = a.exp2();
            }
            vec![tensor]
        } else {
            let mut data = vec![0.; inp[0].1.n_elements().to_usize().unwrap()];
            let inp_data = get_vec_from_tensor(&inp[0].0);
            let (ind, val) = (inp[0].1.index_expression(), inp[0].1.valid_expression());
            let mut stack = vec![];
            for i in 0..data.len() {
                if val.exec_single_var_stack(i, &mut stack) != 0 {
                    data[i] = inp_data[ind.exec_single_var_stack(i, &mut stack)].exp2();
                }
            }
            vec![Tensor::new(data)]
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Sin;
impl Operator for Sin {
    fn process(&mut self, mut inp: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        if !inp[0].1.is_reshaped() {
            let mut tensor = inp.pop().unwrap().0.cloned();
            for a in get_vec_from_tensor_owned(&mut tensor).iter_mut() {
                *a = a.sin();
            }
            vec![tensor]
        } else {
            let mut data = vec![0.; inp[0].1.n_elements().to_usize().unwrap()];
            let inp_data = get_vec_from_tensor(&inp[0].0);
            let (ind, val) = (inp[0].1.index_expression(), inp[0].1.valid_expression());
            let mut stack = vec![];
            for i in 0..data.len() {
                if val.exec_single_var_stack(i, &mut stack) != 0 {
                    data[i] = inp_data[ind.exec_single_var_stack(i, &mut stack)].sin();
                }
            }
            vec![Tensor::new(data)]
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Recip;
impl Operator for Recip {
    fn process(&mut self, mut inp: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        if !inp[0].1.is_reshaped() {
            let mut tensor = inp.pop().unwrap().0.cloned();
            for a in get_vec_from_tensor_owned(&mut tensor).iter_mut() {
                *a = a.recip();
            }
            vec![tensor]
        } else {
            let mut data = vec![0.; inp[0].1.n_elements().to_usize().unwrap()];
            let inp_data = get_vec_from_tensor(&inp[0].0);
            let (ind, val) = (inp[0].1.index_expression(), inp[0].1.valid_expression());
            let mut stack = vec![];
            for i in 0..data.len() {
                if val.exec_single_var_stack(i, &mut stack) != 0 {
                    data[i] = inp_data[ind.exec_single_var_stack(i, &mut stack)].recip();
                }
            }
            vec![Tensor::new(data)]
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Sqrt;
impl Operator for Sqrt {
    fn process(&mut self, mut inp: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        let mut tensor = inp.pop().unwrap().0.cloned();
        for a in get_vec_from_tensor_owned(&mut tensor).iter_mut() {
            *a = a.sqrt();
        }
        vec![tensor]
    }
}

// Binary Ops (A x A -> A)

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Add;
impl Operator for Add {
    fn process(&mut self, inp: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        let (a_data, b_data) = (
            get_vec_from_tensor(&inp[0].0),
            get_vec_from_tensor(&inp[1].0),
        );
        let (a_ind, a_val, b_ind, b_val) = (
            inp[0].1.index_expression(),
            inp[0].1.valid_expression(),
            inp[1].1.index_expression(),
            inp[1].1.valid_expression(),
        );
        let mut stack = vec![];
        let mut data = vec![0.; inp[0].1.n_elements().to_usize().unwrap()];
        for i in 0..data.len() {
            let lhs = if a_val.exec_single_var_stack(i, &mut stack) != 0 {
                a_data[a_ind.exec_single_var_stack(i, &mut stack)]
            } else {
                0.0
            };
            let rhs = if b_val.exec_single_var_stack(i, &mut stack) != 0 {
                b_data[b_ind.exec_single_var_stack(i, &mut stack)]
            } else {
                0.0
            };
            data[i] = lhs + rhs;
        }
        vec![Tensor::new(data)]
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Mul;
impl Operator for Mul {
    fn process(&mut self, inp: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        let (a_data, b_data) = (
            get_vec_from_tensor(&inp[0].0),
            get_vec_from_tensor(&inp[1].0),
        );
        let mut data = vec![0.; inp[0].1.n_elements().to_usize().unwrap()];
        let (a_ind, a_val, b_ind, b_val) = (
            inp[0].1.index_expression(),
            inp[0].1.valid_expression(),
            inp[1].1.index_expression(),
            inp[1].1.valid_expression(),
        );
        let mut stack = vec![];
        for i in 0..data.len() {
            data[i] = if a_val.exec_single_var_stack(i, &mut stack) != 0 {
                a_data[a_ind.exec_single_var_stack(i, &mut stack)]
            } else {
                0.0
            } * if b_val.exec_single_var_stack(i, &mut stack) != 0 {
                b_data[b_ind.exec_single_var_stack(i, &mut stack)]
            } else {
                0.0
            };
        }
        vec![Tensor::new(data)]
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Mod;
impl Operator for Mod {
    fn process(&mut self, inp: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        let (a_data, b_data) = (
            get_vec_from_tensor(&inp[0].0),
            get_vec_from_tensor(&inp[1].0),
        );
        let mut data = vec![0.; inp[0].1.n_elements().to_usize().unwrap()];
        let (a_ind, a_val, b_ind, b_val) = (
            inp[0].1.index_expression(),
            inp[0].1.valid_expression(),
            inp[1].1.index_expression(),
            inp[1].1.valid_expression(),
        );
        let mut stack = vec![];
        for i in 0..data.len() {
            data[i] = if a_val.exec_single_var_stack(i, &mut stack) != 0 {
                a_data[a_ind.exec_single_var_stack(i, &mut stack)]
            } else {
                0.0
            } % if b_val.exec_single_var_stack(i, &mut stack) != 0 {
                b_data[b_ind.exec_single_var_stack(i, &mut stack)]
            } else {
                0.0
            };
        }
        vec![Tensor::new(data)]
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct LessThan;
impl Operator for LessThan {
    fn process(&mut self, inp: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        let (a_data, b_data) = (
            get_vec_from_tensor(&inp[0].0),
            get_vec_from_tensor(&inp[1].0),
        );
        let mut data = vec![0.; inp[0].1.n_elements().to_usize().unwrap()];
        let (a_ind, a_val, b_ind, b_val) = (
            inp[0].1.index_expression(),
            inp[0].1.valid_expression(),
            inp[1].1.index_expression(),
            inp[1].1.valid_expression(),
        );
        let mut stack = vec![];
        for i in 0..data.len() {
            let a = if a_val.exec_single_var_stack(i, &mut stack) != 0 {
                a_data[a_ind.exec_single_var_stack(i, &mut stack)]
            } else {
                0.0
            };
            let b = if b_val.exec_single_var_stack(i, &mut stack) != 0 {
                b_data[b_ind.exec_single_var_stack(i, &mut stack)]
            } else {
                0.0
            };
            data[i] = (a < b) as i32 as f32;
        }
        vec![Tensor::new(data)]
    }
}

// Reduce Ops (A -> B (different shape))

#[derive(Debug, Clone, PartialEq)]
pub struct SumReduce(pub usize);
impl Operator for SumReduce {
    fn process(&mut self, inp: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        let sh = inp[0]
            .1
            .shape()
            .iter()
            .map(|e| e.to_usize().unwrap())
            .collect::<Vec<_>>();
        let front_size = sh.iter().take(self.0).product::<usize>().max(1);
        let back_size = sh.iter().skip(self.0 + 1).product::<usize>().max(1);
        let dim_size = sh[self.0];
        let mut result: Vec<f32> = vec![0.0; front_size * back_size];
        let input = get_vec_from_tensor(&inp[0].0);
        let (ind, val) = (inp[0].1.index_expression(), inp[0].1.valid_expression());
        let mut stack = vec![];

        for i in 0..front_size {
            for j in 0..back_size {
                for k in 0..dim_size {
                    let original_index = i * dim_size * back_size + k * back_size + j;
                    if val.exec_single_var_stack(original_index, &mut stack) != 0 {
                        result[i * back_size + j] +=
                            input[ind.exec_single_var_stack(original_index, &mut stack)];
                    }
                }
            }
        }
        vec![Tensor::new(result)]
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct MaxReduce(pub usize);
impl Operator for MaxReduce {
    fn process(&mut self, inp: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        let sh = inp[0]
            .1
            .shape()
            .iter()
            .map(|e| e.to_usize().unwrap())
            .collect::<Vec<_>>();
        let front_size = sh.iter().take(self.0).product::<usize>().max(1);
        let back_size = sh.iter().skip(self.0 + 1).product::<usize>().max(1);
        let dim_size = sh[self.0];
        let mut result: Vec<f32> = vec![-f32::INFINITY; front_size * back_size];
        let a_data = get_vec_from_tensor(&inp[0].0);
        let (ind, val) = (inp[0].1.index_expression(), inp[0].1.valid_expression());
        let mut stack = vec![];

        for i in 0..front_size {
            for j in 0..back_size {
                for k in 0..dim_size {
                    let original_index = i * dim_size * back_size + k * back_size + j;
                    let new_index = i * back_size + j;
                    if val.exec_single_var_stack(original_index, &mut stack) != 0 {
                        result[new_index] = result[new_index]
                            .max(a_data[ind.exec_single_var_stack(original_index, &mut stack)]);
                    }
                }
            }
        }
        vec![Tensor::new(result)]
    }
}

pub fn get_vec_from_tensor<'a>(tensor: &'a InputTensor<'a>) -> &'a Vec<f32> {
    tensor.borrowed().downcast_ref::<Vec<f32>>().unwrap()
}

pub fn get_vec_from_tensor_owned(tensor: &mut Tensor) -> &mut Vec<f32> {
    tensor.downcast_mut::<Vec<f32>>().unwrap()
}

#[cfg(test)]
mod tests {
    use crate::{
        prelude::{symbolic::Expression, *},
        tests::assert_close,
    };
    use dfdx::prelude::*;
    use itertools::Itertools;

    // Movement op tests

    #[test]
    fn test_reshape() {
        let mut cx = Graph::new();
        let a = cx.tensor::<R2<2, 3>>().set([[1., 2., 3.], [1., 2., 3.]]);
        let b = a.reshape::<R1<6>>().retrieve();
        cx.execute();

        let d_dev = Cpu::default();
        let d_a = d_dev.tensor([[1., 2., 3.], [1., 2., 3.]]);
        let d_b: dfdx::tensor::Tensor<Rank1<6>, f32, Cpu> = d_a.reshape();

        assert_close(&b.data(), &d_b.as_vec());
    }

    #[test]
    fn test_permute() {
        let mut cx = Graph::new();
        let a = cx.tensor::<R2<2, 3>>().set([[1., 2., 3.], [1., 2., 3.]]);
        let b: GraphTensor<R2<3, 2>> = a.permute().retrieve();
        cx.execute();

        let d_dev = Cpu::default();
        let d_a = d_dev.tensor([[1., 2., 3.], [1., 2., 3.]]);
        let d_b: dfdx::tensor::Tensor<Rank2<3, 2>, f32, Cpu> = d_a.permute();

        assert_close(&b.data(), &d_b.as_vec());
    }

    #[test]
    fn test_expand() {
        let mut cx = Graph::new();
        let a = cx.tensor::<R1<3>>().set([1., 2., 3.]);
        let b: GraphTensor<R2<3, 2>> = a.expand().retrieve();
        cx.execute();

        let d_dev = Cpu::default();
        let d_a = d_dev.tensor([1., 2., 3.]);
        let d_b: dfdx::tensor::Tensor<Rank2<3, 2>, f32, Cpu> = d_a.broadcast();

        assert_close(&b.data(), &d_b.as_vec());
    }

    #[test]
    fn test_slice() {
        let mut cx = Graph::new();
        let a = cx.tensor::<R2<2, 3>>().set([[1., 2., 3.], [1., 2., 3.]]);
        let b = a.slice((Expression::from(1).., ..)).retrieve();
        cx.execute();

        let d_dev = Cpu::default();
        let d_a = d_dev.tensor([[1., 2., 3.], [1., 2., 3.]]);
        let d_b = d_a.slice((1.., ..));

        assert_close(&b.data(), &d_b.as_vec());
    }

    // Unary op tests

    #[test]
    fn test_log2() {
        // We can't use dfdx because it doesn't implement this op
        let mut cx = Graph::new();
        let a = cx.tensor::<R1<3>>().set([1., 2., 3.]);
        let b = a.log2().retrieve();
        cx.execute();

        assert_close(
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
        let a = cx.tensor::<R1<3>>().set([1., 2., 3.]);
        let b = a.exp2().retrieve();
        cx.execute();

        assert_close(
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
        let a = cx.tensor::<R1<3>>().set([1., 2., 3.]);
        let b = a.recip().retrieve();
        cx.execute();

        let d_dev = Cpu::default();
        let d_a = d_dev.tensor([1., 2., 3.]);
        let d_b = d_a.recip();

        assert_close(&b.data(), &d_b.as_vec());
    }

    #[test]
    fn test_sin() {
        let mut cx = Graph::new();
        let a = cx.tensor::<R1<3>>().set([1., 2., 3.]);
        let b = a.sin().retrieve();
        cx.execute();

        let d_dev = Cpu::default();
        let d_a = d_dev.tensor([1., 2., 3.]);
        let d_b = d_a.sin();

        assert_close(&b.data(), &d_b.as_vec());
    }

    #[test]
    fn test_sqrt() {
        let mut cx = Graph::new();
        let a = cx.tensor::<R1<3>>().set([1., 2., 3.]);
        let b = a.sqrt().retrieve();
        cx.execute();

        let d_dev = Cpu::default();
        let d_a = d_dev.tensor([1., 2., 3.]);
        let d_b = d_a.sqrt();

        assert_close(&b.data(), &d_b.as_vec());
    }

    // Binary op tests

    #[test]
    fn test_add() {
        let mut cx = Graph::new();
        let a = cx.tensor::<R1<3>>().set([1., 2., 3.]);
        let b = cx.tensor::<R1<3>>().set([1., 2., 3.]);
        let c = (a + b).retrieve();
        cx.execute();

        let d_dev = Cpu::default();
        let d_a = d_dev.tensor([1., 2., 3.]);
        let d_b = d_dev.tensor([1., 2., 3.]);
        let d_c = d_a + d_b;

        assert_close(&c.data(), &d_c.as_vec());
    }

    #[test]
    fn test_sub() {
        let mut cx = Graph::new();
        let a = cx.tensor::<R1<3>>().set([1., 2., 3.]);
        let b = cx.tensor::<R1<3>>().set([1., 2., 3.]);
        let c = (a - b).retrieve();
        cx.execute();

        let d_dev = Cpu::default();
        let d_a = d_dev.tensor([1., 2., 3.]);
        let d_b = d_dev.tensor([1., 2., 3.]);
        let d_c = d_a - d_b;

        assert_close(&c.data(), &d_c.as_vec());
    }

    #[test]
    fn test_mul() {
        let mut cx = Graph::new();
        let a = cx.tensor::<R1<3>>().set([1., 2., 3.]);
        let b = cx.tensor::<R1<3>>().set([1., 2., 3.]);
        let c = (a * b).retrieve();
        cx.execute();

        let d_dev = Cpu::default();
        let d_a = d_dev.tensor([1., 2., 3.]);
        let d_b = d_dev.tensor([1., 2., 3.]);
        let d_c = d_a * d_b;

        assert_close(&c.data(), &d_c.as_vec());
    }

    #[test]
    fn test_permute_mul() {
        let mut cx = Graph::new();
        let a = cx.tensor::<R2<3, 2>>().set([[1., 2.], [3., 2.], [3., 1.]]);
        let b = cx.tensor::<R2<3, 2>>().set([[1., 2.], [3., -1.], [3., 0.]]);
        let c = a.expand::<R3<3, 2, 3>, crate::prelude::Axis<2>>()
            * b.expand::<R3<3, 2, 3>, crate::prelude::Axis<2>>();
        c.retrieve();
        cx.execute();

        let d_dev = Cpu::default();
        let d_a = d_dev.tensor([[1., 2.], [3., 2.], [3., 1.]]);
        let d_b = d_dev.tensor([[1., 2.], [3., -1.], [3., 0.]]);
        let d_c = d_a.broadcast::<Rank3<3, 2, 3>, dfdx::prelude::Axis<2>>()
            * d_b.broadcast::<Rank3<3, 2, 3>, dfdx::prelude::Axis<2>>();

        assert_close(&c.data(), &d_c.as_vec());
    }

    #[test]
    fn test_div() {
        let mut cx = Graph::new();
        let a = cx.tensor::<R1<3>>().set([1., 2., 3.]);
        let b = cx.tensor::<R1<3>>().set([1., 2., 3.]);
        let c = (a / b).retrieve();
        cx.execute();

        let d_dev = Cpu::default();
        let d_a = d_dev.tensor([1., 2., 3.]);
        let d_b = d_dev.tensor([1., 2., 3.]);
        let d_c = d_a / d_b;

        assert_close(&c.data(), &d_c.as_vec());
    }

    #[test]
    fn test_max() {
        let mut cx = Graph::new();
        let a = cx.tensor::<R1<3>>().set([1., 0., 3.]);
        let b = cx.tensor::<R1<3>>().set([1., 2., -2.]);
        let c = a.max(b).retrieve();
        cx.execute();

        let d_dev = Cpu::default();
        let d_a = d_dev.tensor([1., 0., 3.]);
        let d_b = d_dev.tensor([1., 2., -2.]);
        let d_c = d_a.maximum(d_b);

        assert_close(&c.data(), &d_c.as_vec());
    }

    #[test]
    fn test_mod() {
        let mut cx = Graph::new();
        let a = cx.tensor::<R1<3>>().set([1., 2., 3.]);
        let b = cx.tensor::<R1<3>>().set([1., 2., 3.]);
        let c = (a % b).retrieve();
        cx.execute();

        // No dfdx equivalent

        assert_close(
            &c.data(),
            &[1., 2., 3.]
                .into_iter()
                .zip([1., 2., 3.])
                .map(|(a, b)| a % b)
                .collect_vec(),
        );
    }

    // Reduction op tests

    #[test]
    fn test_sum_reduce() {
        let mut cx = Graph::new();
        let a = cx
            .tensor::<R3<2, 2, 3>>()
            .set([[[1., 2., 3.], [1., 2., 3.]], [[1., 2., 3.], [1., 2., 3.]]]);
        let b = a.sum_reduce::<_, crate::prelude::Axis<1>>().retrieve();
        let c = a.sum_reduce::<_, crate::prelude::Axis<0>>().retrieve();
        let d = a.sum_reduce::<_, crate::prelude::Axis<2>>().retrieve();
        cx.execute();

        let d_dev = Cpu::default();
        let d_a = d_dev.tensor([[[1., 2., 3.], [1., 2., 3.]], [[1., 2., 3.], [1., 2., 3.]]]);
        let d_b = d_a.clone().sum::<_, dfdx::shapes::Axis<1>>();
        let d_c = d_a.clone().sum::<_, dfdx::shapes::Axis<0>>();
        let d_d = d_a.sum::<_, dfdx::shapes::Axis<2>>();

        assert_close(&b.data(), &d_b.as_vec());
        assert_close(&c.data(), &d_c.as_vec());
        assert_close(&d.data(), &d_d.as_vec());
    }

    #[test]
    fn test_sum_reduce2() {
        let mut cx = Graph::new();
        let a = cx.tensor::<R4<1, 2, 2, 3>>().set([[
            [[34.4, -96.0, 144.0], [43.0, 560.0, 180.0]],
            [[39.6, -120.0, 180.0], [49.5, 700.0, 225.0]],
        ]]);
        let b = a.sum_reduce::<_, crate::prelude::Axis<3>>().retrieve();
        cx.execute();

        let d_dev = Cpu::default();
        let d_a = d_dev.tensor([[
            [[34.4, -96.0, 144.0], [43.0, 560.0, 180.0]],
            [[39.6, -120.0, 180.0], [49.5, 700.0, 225.0]],
        ]]);
        let d_b = d_a.sum::<_, dfdx::shapes::Axis<3>>();

        assert_close(&b.data(), &d_b.as_vec());
    }

    #[test]
    fn test_max_reduce() {
        let mut cx = Graph::new();
        let a = cx
            .tensor::<R3<2, 2, 3>>()
            .set([[[1., 2., 3.], [1., 2., 3.]], [[1., 2., 3.], [1., 2., 3.]]]);
        let b = a.max_reduce::<_, crate::prelude::Axis<1>>().retrieve();
        let c = a.max_reduce::<_, crate::prelude::Axis<0>>().retrieve();
        let d = a.max_reduce::<_, crate::prelude::Axis<2>>().retrieve();
        cx.execute();

        let d_dev = Cpu::default();
        let d_a = d_dev.tensor([[[1., 2., 3.], [1., 2., 3.]], [[1., 2., 3.], [1., 2., 3.]]]);
        let d_b = d_a.clone().max::<_, dfdx::shapes::Axis<1>>();
        let d_c = d_a.clone().max::<_, dfdx::shapes::Axis<0>>();
        let d_d = d_a.max::<_, dfdx::shapes::Axis<2>>();

        assert_close(&b.data(), &d_b.as_vec());
        assert_close(&c.data(), &d_c.as_vec());
        assert_close(&d.data(), &d_d.as_vec());
    }
}
