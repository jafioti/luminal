#![allow(clippy::needless_range_loop)]

use std::{any::Any, fmt::Debug};

use petgraph::stable_graph::NodeIndex;

use crate::{
    prelude::{to_shapes_strides, RealDim, TensorView},
    shape::ShapeTracker,
    tensor::Tensor,
};

pub trait Operator: Debug {
    fn name(&self) -> &'static str;
    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;
    fn process(
        &self,
        inp: Vec<(&Tensor, TensorView)>,
        nid: NodeIndex,
    ) -> (Option<Tensor>, TensorView);
}

/// An opaque function running on CPU that takes in tensor references and outputs a new tensor
#[allow(clippy::type_complexity)]
pub struct Function(
    pub Box<dyn Fn(Vec<(&Tensor, TensorView)>, NodeIndex) -> (Option<Tensor>, TensorView)>,
);
impl Operator for Function {
    fn name(&self) -> &'static str {
        "Function"
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
    fn process(
        &self,
        input: Vec<(&Tensor, TensorView)>,
        nid: NodeIndex,
    ) -> (Option<Tensor>, TensorView) {
        (self.0)(input, nid)
    }
}

impl Debug for Function {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Function")
    }
}

#[derive(Debug, Clone)]
pub struct Print(pub String);
impl Operator for Print {
    fn name(&self) -> &'static str {
        "Print"
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
    fn process(
        &self,
        input: Vec<(&Tensor, TensorView)>,
        nid: NodeIndex,
    ) -> (Option<Tensor>, TensorView) {
        println!("{}", self.0);
        for (i, (tensor, view)) in input.iter().enumerate() {
            println!("{} Data: {:?}", i + 1, tensor.real_data(view));
            println!("{} Shape: {:?}", i + 1, view.shape.shape());
            println!(
                "{} Idx: {}",
                i + 1,
                view.shape.index_fn_node().to_string_no_range()
            );
        }
        (
            None,
            TensorView {
                tensor_id: nid,
                shape: ShapeTracker::new(vec![]),
            },
        )
    }
}

// Movement Op (A -> B)

#[derive(Debug, Clone)]
pub struct Permute(pub Vec<usize>);
impl Operator for Permute {
    fn name(&self) -> &'static str {
        "Permute"
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
    fn process(
        &self,
        inp: Vec<(&Tensor, TensorView)>,
        _: NodeIndex,
    ) -> (Option<Tensor>, TensorView) {
        let mut view = inp[0].1.clone();
        view.shape.permute(&self.0);
        (None, view)
    }
}

#[derive(Debug, Clone)]
pub struct Reshape(pub Vec<RealDim>);
impl Operator for Reshape {
    fn name(&self) -> &'static str {
        "Reshape"
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
    fn process(
        &self,
        inp: Vec<(&Tensor, TensorView)>,
        _: NodeIndex,
    ) -> (Option<Tensor>, TensorView) {
        let mut view = inp[0].1.clone();
        // Figure out the proper shapes
        let inp_elements: usize = view.shape.shape().iter().product();
        let known_elements: usize = self
            .0
            .iter()
            .filter_map(|i| {
                if let RealDim::Const(n) = i {
                    Some(n)
                } else {
                    None
                }
            })
            .product();
        let mut encountered_dyn = false;
        let real_shape = self
            .0
            .iter()
            .map(|i| match i {
                RealDim::Const(n) => *n,
                RealDim::Dyn => {
                    if encountered_dyn {
                        panic!("Encountered two dyn dims in a reshape!");
                    }
                    encountered_dyn = true;
                    inp_elements / known_elements
                }
            })
            .collect();
        view.shape.reshape(real_shape);
        (None, view)
    }
}

#[derive(Debug, Clone)]
pub struct Expand(pub usize, pub RealDim);
impl Operator for Expand {
    fn name(&self) -> &'static str {
        "Expand"
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
    fn process(
        &self,
        inp: Vec<(&Tensor, TensorView)>,
        _: NodeIndex,
    ) -> (Option<Tensor>, TensorView) {
        let mut view = inp[0].1.clone();
        view.shape.expand(self.0, self.1);
        (None, view)
    }
}

#[derive(Debug, Clone)]
pub struct Slice(pub Vec<(usize, usize)>);
impl Operator for Slice {
    fn name(&self) -> &'static str {
        "Slice"
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
    fn process(
        &self,
        inp: Vec<(&Tensor, TensorView)>,
        _: NodeIndex,
    ) -> (Option<Tensor>, TensorView) {
        let mut view = inp[0].1.clone();
        view.shape.slice(&self.0);
        (None, view)
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
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
    fn process(
        &self,
        inp: Vec<(&Tensor, TensorView)>,
        nid: NodeIndex,
    ) -> (Option<Tensor>, TensorView) {
        let (mut t, mut view) = (inp[0].0.clone(), inp[0].1.clone());
        for a in t
            .data
            .as_any_mut()
            .downcast_mut::<Vec<f32>>()
            .unwrap()
            .iter_mut()
        {
            *a = a.log2();
        }

        view.tensor_id = nid;
        (Some(t), view)
    }
}

#[derive(Debug, Clone)]
pub struct Exp2;
impl Operator for Exp2 {
    fn name(&self) -> &'static str {
        "Exp2"
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
    fn process(
        &self,
        inp: Vec<(&Tensor, TensorView)>,
        nid: NodeIndex,
    ) -> (Option<Tensor>, TensorView) {
        let (mut t, mut view) = (inp[0].0.clone(), inp[0].1.clone());
        for a in t
            .data
            .as_any_mut()
            .downcast_mut::<Vec<f32>>()
            .unwrap()
            .iter_mut()
        {
            *a = a.exp2();
        }

        view.tensor_id = nid;
        (Some(t), view)
    }
}

#[derive(Debug, Clone)]
pub struct Sin;
impl Operator for Sin {
    fn name(&self) -> &'static str {
        "Sin"
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
    fn process(
        &self,
        inp: Vec<(&Tensor, TensorView)>,
        nid: NodeIndex,
    ) -> (Option<Tensor>, TensorView) {
        let (mut t, mut view) = (inp[0].0.clone(), inp[0].1.clone());
        for a in t
            .data
            .as_any_mut()
            .downcast_mut::<Vec<f32>>()
            .unwrap()
            .iter_mut()
        {
            *a = a.sin();
        }

        view.tensor_id = nid;
        (Some(t), view)
    }
}

#[derive(Debug, Clone)]
pub struct Sqrt;
impl Operator for Sqrt {
    fn name(&self) -> &'static str {
        "Sqrt"
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
    fn process(
        &self,
        inp: Vec<(&Tensor, TensorView)>,
        nid: NodeIndex,
    ) -> (Option<Tensor>, TensorView) {
        let (mut t, mut view) = (inp[0].0.clone(), inp[0].1.clone());
        for a in t
            .data
            .as_any_mut()
            .downcast_mut::<Vec<f32>>()
            .unwrap()
            .iter_mut()
        {
            *a = a.sqrt();
        }

        view.tensor_id = nid;
        (Some(t), view)
    }
}

#[derive(Debug, Clone)]
pub struct Recip;
impl Operator for Recip {
    fn name(&self) -> &'static str {
        "Recip"
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
    fn process(
        &self,
        inp: Vec<(&Tensor, TensorView)>,
        nid: NodeIndex,
    ) -> (Option<Tensor>, TensorView) {
        let (mut t, mut view) = (inp[0].0.clone(), inp[0].1.clone());
        for a in t
            .data
            .as_any_mut()
            .downcast_mut::<Vec<f32>>()
            .unwrap()
            .iter_mut()
        {
            *a = a.recip();
        }

        view.tensor_id = nid;
        (Some(t), view)
    }
}

// Binary Ops (A x A -> A)

#[derive(Debug, Clone)]
pub struct Add;
impl Operator for Add {
    fn name(&self) -> &'static str {
        "Add"
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
    fn process(
        &self,
        inp: Vec<(&Tensor, TensorView)>,
        nid: NodeIndex,
    ) -> (Option<Tensor>, TensorView) {
        let res_shape = inp[0].1.shape.get_real_shape([&inp[1].1.shape]).unwrap();
        let (mut left_shape, mut right_shape) = (inp[0].1.shape.clone(), inp[1].1.shape.clone());
        left_shape.views.last_mut().unwrap().shape = res_shape.clone();
        left_shape.views.last_mut().unwrap().shape_strides = to_shapes_strides(
            &left_shape.views.last().unwrap().shape,
            &left_shape.views.last().unwrap().strides,
        );
        right_shape.views.last_mut().unwrap().shape = res_shape.clone();
        right_shape.views.last_mut().unwrap().shape_strides = to_shapes_strides(
            &right_shape.views.last().unwrap().shape,
            &right_shape.views.last().unwrap().strides,
        );
        let ((a_idx, a_valid), (b_idx, b_valid)) =
            (left_shape.index_node(), right_shape.index_node());
        let a_data = inp[0].0.data.as_any().downcast_ref::<Vec<f32>>().unwrap();
        let b_data = inp[1].0.data.as_any().downcast_ref::<Vec<f32>>().unwrap();
        let mut data = vec![0.; res_shape.iter().product()];
        for i in 0..data.len() as i32 {
            data[i as usize] = if a_valid.solve(i) != 0 {
                a_data[a_idx.solve(i) as usize]
            } else {
                0.
            } + if b_valid.solve(i) != 0 {
                b_data[b_idx.solve(i) as usize]
            } else {
                0.
            };
        }
        (
            Some(Tensor {
                data: Box::new(data),
            }),
            TensorView {
                tensor_id: nid,
                shape: ShapeTracker::new(res_shape),
            },
        )
    }
}

#[derive(Debug, Clone)]
pub struct Sub;
impl Operator for Sub {
    fn name(&self) -> &'static str {
        "Sub"
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
    fn process(
        &self,
        inp: Vec<(&Tensor, TensorView)>,
        nid: NodeIndex,
    ) -> (Option<Tensor>, TensorView) {
        let res_shape = inp[0].1.shape.get_real_shape([&inp[1].1.shape]).unwrap();
        let (mut left_shape, mut right_shape) = (inp[0].1.shape.clone(), inp[1].1.shape.clone());
        left_shape.views.last_mut().unwrap().shape = res_shape.clone();
        left_shape.views.last_mut().unwrap().shape_strides = to_shapes_strides(
            &left_shape.views.last().unwrap().shape,
            &left_shape.views.last().unwrap().strides,
        );
        right_shape.views.last_mut().unwrap().shape = res_shape.clone();
        right_shape.views.last_mut().unwrap().shape_strides = to_shapes_strides(
            &right_shape.views.last().unwrap().shape,
            &right_shape.views.last().unwrap().strides,
        );
        let ((a_idx, a_valid), (b_idx, b_valid)) =
            (left_shape.index_node(), right_shape.index_node());
        let a_data = inp[0].0.data.as_any().downcast_ref::<Vec<f32>>().unwrap();
        let b_data = inp[1].0.data.as_any().downcast_ref::<Vec<f32>>().unwrap();
        let mut data = vec![0.; res_shape.iter().product()];
        for i in 0..data.len() as i32 {
            data[i as usize] = if a_valid.solve(i) != 0 {
                a_data[a_idx.solve(i) as usize]
            } else {
                0.
            } - if b_valid.solve(i) != 0 {
                b_data[b_idx.solve(i) as usize]
            } else {
                0.
            };
        }
        (
            Some(Tensor {
                data: Box::new(data),
            }),
            TensorView {
                tensor_id: nid,
                shape: ShapeTracker::new(res_shape),
            },
        )
    }
}

#[derive(Debug, Clone)]
pub struct Mul;
impl Operator for Mul {
    fn name(&self) -> &'static str {
        "Mul"
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
    fn process(
        &self,
        inp: Vec<(&Tensor, TensorView)>,
        nid: NodeIndex,
    ) -> (Option<Tensor>, TensorView) {
        let res_shape = inp[0].1.shape.get_real_shape([&inp[1].1.shape]).unwrap();
        let (mut left_shape, mut right_shape) = (inp[0].1.shape.clone(), inp[1].1.shape.clone());
        left_shape.views.last_mut().unwrap().shape = res_shape.clone();
        left_shape.views.last_mut().unwrap().shape_strides = to_shapes_strides(
            &left_shape.views.last().unwrap().shape,
            &left_shape.views.last().unwrap().strides,
        );
        right_shape.views.last_mut().unwrap().shape = res_shape.clone();
        right_shape.views.last_mut().unwrap().shape_strides = to_shapes_strides(
            &right_shape.views.last().unwrap().shape,
            &right_shape.views.last().unwrap().strides,
        );
        let ((a_idx, a_valid), (b_idx, b_valid)) =
            (left_shape.index_node(), right_shape.index_node());
        println!(
            "Left Shape: {:?} Right Shape: {:?}",
            left_shape, right_shape
        );
        let a_data = inp[0].0.data.as_any().downcast_ref::<Vec<f32>>().unwrap();
        let b_data = inp[1].0.data.as_any().downcast_ref::<Vec<f32>>().unwrap();
        println!("A Data: {} B Data: {}", a_data.len(), b_data.len());
        let mut data = vec![0.; res_shape.iter().product()];
        for i in 0..data.len() as i32 {
            data[i as usize] = if a_valid.solve(i) != 0 {
                a_data[a_idx.solve(i) as usize]
            } else {
                0.
            } * if b_valid.solve(i) != 0 {
                b_data[b_idx.solve(i) as usize]
            } else {
                0.
            };
        }
        (
            Some(Tensor {
                data: Box::new(data),
            }),
            TensorView {
                tensor_id: nid,
                shape: ShapeTracker::new(res_shape),
            },
        )
    }
}

#[derive(Debug, Clone)]
pub struct Div;
impl Operator for Div {
    fn name(&self) -> &'static str {
        "Div"
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
    fn process(
        &self,
        inp: Vec<(&Tensor, TensorView)>,
        nid: NodeIndex,
    ) -> (Option<Tensor>, TensorView) {
        let res_shape = inp[0].1.shape.get_real_shape([&inp[1].1.shape]).unwrap();
        let (mut left_shape, mut right_shape) = (inp[0].1.shape.clone(), inp[1].1.shape.clone());
        left_shape.views.last_mut().unwrap().shape = res_shape.clone();
        left_shape.views.last_mut().unwrap().shape_strides = to_shapes_strides(
            &left_shape.views.last().unwrap().shape,
            &left_shape.views.last().unwrap().strides,
        );
        right_shape.views.last_mut().unwrap().shape = res_shape.clone();
        right_shape.views.last_mut().unwrap().shape_strides = to_shapes_strides(
            &right_shape.views.last().unwrap().shape,
            &right_shape.views.last().unwrap().strides,
        );
        let ((a_idx, a_valid), (b_idx, b_valid)) =
            (left_shape.index_node(), right_shape.index_node());
        let a_data = inp[0].0.data.as_any().downcast_ref::<Vec<f32>>().unwrap();
        let b_data = inp[1].0.data.as_any().downcast_ref::<Vec<f32>>().unwrap();
        let mut data = vec![0.; res_shape.iter().product()];
        for i in 0..data.len() as i32 {
            data[i as usize] = if a_valid.solve(i) != 0 {
                a_data[a_idx.solve(i) as usize]
            } else {
                0.
            } / if b_valid.solve(i) != 0 {
                b_data[b_idx.solve(i) as usize]
            } else {
                0.
            };
        }
        (
            Some(Tensor {
                data: Box::new(data),
            }),
            TensorView {
                tensor_id: nid,
                shape: ShapeTracker::new(res_shape),
            },
        )
    }
}

#[derive(Debug, Clone)]
pub struct Max;
impl Operator for Max {
    fn name(&self) -> &'static str {
        "Max"
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
    fn process(
        &self,
        inp: Vec<(&Tensor, TensorView)>,
        nid: NodeIndex,
    ) -> (Option<Tensor>, TensorView) {
        let res_shape = inp[0].1.shape.get_real_shape([&inp[1].1.shape]).unwrap();
        let (mut left_shape, mut right_shape) = (inp[0].1.shape.clone(), inp[1].1.shape.clone());
        left_shape.views.last_mut().unwrap().shape = res_shape.clone();
        left_shape.views.last_mut().unwrap().shape_strides = to_shapes_strides(
            &left_shape.views.last().unwrap().shape,
            &left_shape.views.last().unwrap().strides,
        );
        right_shape.views.last_mut().unwrap().shape = res_shape.clone();
        right_shape.views.last_mut().unwrap().shape_strides = to_shapes_strides(
            &right_shape.views.last().unwrap().shape,
            &right_shape.views.last().unwrap().strides,
        );
        let ((a_idx, a_valid), (b_idx, b_valid)) =
            (left_shape.index_node(), right_shape.index_node());
        let a_data = inp[0].0.data.as_any().downcast_ref::<Vec<f32>>().unwrap();
        let b_data = inp[1].0.data.as_any().downcast_ref::<Vec<f32>>().unwrap();
        let mut data = vec![0.; res_shape.iter().product()];
        for i in 0..data.len() as i32 {
            data[i as usize] = if a_valid.solve(i) != 0 {
                a_data[a_idx.solve(i) as usize]
            } else {
                0.
            }
            .max(if b_valid.solve(i) != 0 {
                b_data[b_idx.solve(i) as usize]
            } else {
                0.
            });
        }
        (
            Some(Tensor {
                data: Box::new(data),
            }),
            TensorView {
                tensor_id: nid,
                shape: ShapeTracker::new(res_shape),
            },
        )
    }
}

#[derive(Debug, Clone)]
pub struct Mod;
impl Operator for Mod {
    fn name(&self) -> &'static str {
        "Mod"
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
    fn process(
        &self,
        inp: Vec<(&Tensor, TensorView)>,
        nid: NodeIndex,
    ) -> (Option<Tensor>, TensorView) {
        let res_shape = inp[0].1.shape.get_real_shape([&inp[1].1.shape]).unwrap();
        let (mut left_shape, mut right_shape) = (inp[0].1.shape.clone(), inp[1].1.shape.clone());
        left_shape.views.last_mut().unwrap().shape = res_shape.clone();
        left_shape.views.last_mut().unwrap().shape_strides = to_shapes_strides(
            &left_shape.views.last().unwrap().shape,
            &left_shape.views.last().unwrap().strides,
        );
        right_shape.views.last_mut().unwrap().shape = res_shape.clone();
        right_shape.views.last_mut().unwrap().shape_strides = to_shapes_strides(
            &right_shape.views.last().unwrap().shape,
            &right_shape.views.last().unwrap().strides,
        );
        let ((a_idx, a_valid), (b_idx, b_valid)) =
            (left_shape.index_node(), right_shape.index_node());
        let a_data = inp[0].0.data.as_any().downcast_ref::<Vec<f32>>().unwrap();
        let b_data = inp[1].0.data.as_any().downcast_ref::<Vec<f32>>().unwrap();
        let mut data = vec![0.; res_shape.iter().product()];
        for i in 0..data.len() as i32 {
            data[i as usize] = if a_valid.solve(i) != 0 {
                a_data[a_idx.solve(i) as usize]
            } else {
                0.
            } % if b_valid.solve(i) != 0 {
                b_data[b_idx.solve(i) as usize]
            } else {
                0.
            };
        }
        (
            Some(Tensor {
                data: Box::new(data),
            }),
            TensorView {
                tensor_id: nid,
                shape: ShapeTracker::new(res_shape),
            },
        )
    }
}

// Reduce Ops (A -> B (different shape))

#[derive(Debug, Clone)]
pub struct SumReduce(pub usize);
impl Operator for SumReduce {
    fn name(&self) -> &'static str {
        "SumReduce"
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
    fn process(
        &self,
        inp: Vec<(&Tensor, TensorView)>,
        nid: NodeIndex,
    ) -> (Option<Tensor>, TensorView) {
        let front_size: usize = inp[0].1.shape.shape().iter().take(self.0).product();
        let back_size: usize = inp[0].1.shape.shape().iter().skip(self.0 + 1).product();
        let dim_size = inp[0].1.shape.shape()[self.0];
        let mut result: Vec<f32> = vec![0.0; front_size * back_size];
        let a_data = inp[0].0.data.as_any().downcast_ref::<Vec<f32>>().unwrap();
        let (a_idx, a_valid) = inp[0].1.shape.index_node();

        for i in 0..front_size {
            for j in 0..back_size {
                for k in 0..dim_size {
                    let original_index = i * dim_size * back_size + k * back_size + j;
                    let new_index = i * back_size + j;
                    if a_valid.solve(original_index as i32) != 0 {
                        result[new_index] += a_data[a_idx.solve(original_index as i32) as usize];
                    }
                }
            }
        }
        let mut shape = inp[0].1.shape.shape().clone();
        shape.remove(self.0);
        (
            Some(Tensor {
                data: Box::new(result),
            }),
            TensorView {
                tensor_id: nid,
                shape: ShapeTracker::new(shape),
            },
        )
    }
}

#[derive(Debug, Clone)]
pub struct MaxReduce(pub usize);
impl Operator for MaxReduce {
    fn name(&self) -> &'static str {
        "MaxReduce"
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
    fn process(
        &self,
        inp: Vec<(&Tensor, TensorView)>,
        nid: NodeIndex,
    ) -> (Option<Tensor>, TensorView) {
        let front_size: usize = inp[0].1.shape.shape().iter().take(self.0).product();
        let back_size: usize = inp[0].1.shape.shape().iter().skip(self.0 + 1).product();
        let dim_size = inp[0].1.shape.shape()[self.0];
        let mut result: Vec<f32> = vec![-f32::INFINITY; front_size * back_size];
        let a_data = inp[0].0.data.as_any().downcast_ref::<Vec<f32>>().unwrap();
        let (a_idx, a_valid) = inp[0].1.shape.index_node();

        for i in 0..front_size {
            for j in 0..back_size {
                for k in 0..dim_size {
                    let original_index = i * dim_size * back_size + k * back_size + j;
                    let new_index = i * back_size + j;
                    if a_valid.solve(original_index as i32) != 0 {
                        result[new_index] = result[new_index]
                            .max(a_data[a_idx.solve(original_index as i32) as usize]);
                    }
                }
            }
        }
        let mut shape = inp[0].1.shape.shape().clone();
        shape.remove(self.0);
        (
            Some(Tensor {
                data: Box::new(result),
            }),
            TensorView {
                tensor_id: nid,
                shape: ShapeTracker::new(shape),
            },
        )
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
        let a = cx.new_tensor::<R2<2, 3>>();
        a.set(vec![1., 2., 3., 1., 2., 3.]);
        let b = a.reshape::<R1<6>>();
        cx.execute();

        let d_dev = Cpu::default();
        let d_a = d_dev.tensor([[1., 2., 3.], [1., 2., 3.]]);
        let d_b: dfdx::tensor::Tensor<Rank1<6>, f32, Cpu> = d_a.reshape();

        assert_close_data(
            &b.retrieve().unwrap().real_data(b.view().unwrap()).unwrap(),
            &d_b.as_vec(),
        );
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

        assert_close_data(
            &b.retrieve().unwrap().real_data(b.view().unwrap()).unwrap(),
            &d_b.as_vec(),
        );
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

        assert_close_data(
            &b.retrieve().unwrap().real_data(b.view().unwrap()).unwrap(),
            &d_b.as_vec(),
        );
    }

    #[test]
    fn test_slice() {
        let mut cx = Graph::new();
        let a = cx.new_tensor::<R2<2, 3>>();
        a.set(vec![1., 2., 3., 1., 2., 3.]);
        let b = a.slice((1.., ..));
        cx.execute();

        let d_dev = Cpu::default();
        let d_a = d_dev.tensor([[1., 2., 3.], [1., 2., 3.]]);
        let d_b = d_a.slice((1.., ..));

        assert_close_data(
            &b.retrieve().unwrap().real_data(b.view().unwrap()).unwrap(),
            &d_b.as_vec(),
        );
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
            &b.retrieve().unwrap().real_data(b.view().unwrap()).unwrap(),
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
            &b.retrieve().unwrap().real_data(b.view().unwrap()).unwrap(),
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

        assert_close_data(
            &b.retrieve().unwrap().real_data(b.view().unwrap()).unwrap(),
            &d_b.as_vec(),
        );
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

        assert_close_data(
            &b.retrieve().unwrap().real_data(b.view().unwrap()).unwrap(),
            &d_b.as_vec(),
        );
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

        assert_close_data(
            &b.retrieve().unwrap().real_data(b.view().unwrap()).unwrap(),
            &d_b.as_vec(),
        );
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

        assert_close_data(
            &c.retrieve().unwrap().real_data(c.view().unwrap()).unwrap(),
            &d_c.as_vec(),
        );
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

        assert_close_data(
            &c.retrieve().unwrap().real_data(c.view().unwrap()).unwrap(),
            &d_c.as_vec(),
        );
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

        assert_close_data(
            &c.retrieve().unwrap().real_data(c.view().unwrap()).unwrap(),
            &d_c.as_vec(),
        );
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

        assert_close_data(
            &c.retrieve().unwrap().real_data(c.view().unwrap()).unwrap(),
            &d_c.as_vec(),
        );
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

        assert_close_data(
            &c.retrieve().unwrap().real_data(c.view().unwrap()).unwrap(),
            &d_c.as_vec(),
        );
    }

    #[test]
    fn test_mod() {
        let mut cx = Graph::new();
        let a = cx.new_tensor::<R1<3>>();
        a.set(vec![1., 2., 3.]);
        let b = cx.new_tensor::<R1<3>>();
        b.set(vec![1., 2., 3.]);
        let c = a % b;
        cx.execute();

        // No dfdx equivalent

        assert_close_data(
            &c.retrieve().unwrap().real_data(c.view().unwrap()).unwrap(),
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
        let a = cx.new_tensor::<R3<2, 2, 3>>();
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

        assert_close_data(
            &b.retrieve().unwrap().real_data(b.view().unwrap()).unwrap(),
            &d_b.as_vec(),
        );
        assert_close_data(
            &c.retrieve().unwrap().real_data(c.view().unwrap()).unwrap(),
            &d_c.as_vec(),
        );
        assert_close_data(
            &d.retrieve().unwrap().real_data(d.view().unwrap()).unwrap(),
            &d_d.as_vec(),
        );
    }

    #[test]
    fn test_sum_reduce2() {
        let mut cx = Graph::new();
        let a = cx.new_tensor::<R4<1, 2, 2, 3>>();
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

        assert_close_data(
            &b.retrieve().unwrap().real_data(b.view().unwrap()).unwrap(),
            &d_b.as_vec(),
        );
    }

    #[test]
    fn test_max_reduce() {
        let mut cx = Graph::new();
        let a = cx.new_tensor::<R3<2, 2, 3>>();
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

        assert_close_data(
            &b.retrieve().unwrap().real_data(b.view().unwrap()).unwrap(),
            &d_b.as_vec(),
        );
        assert_close_data(
            &c.retrieve().unwrap().real_data(c.view().unwrap()).unwrap(),
            &d_c.as_vec(),
        );
        assert_close_data(
            &d.retrieve().unwrap().real_data(d.view().unwrap()).unwrap(),
            &d_d.as_vec(),
        );
    }
}
