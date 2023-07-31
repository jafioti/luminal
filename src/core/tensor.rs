use std::{any::Any, fmt::Debug};

use dyn_clone::{clone_trait_object, DynClone};
use petgraph::stable_graph::NodeIndex;

use crate::shape::ShapeTracker;

/// An entirely dynamic tensor with data
#[derive(Debug, Clone)]
pub struct Tensor {
    pub data: Box<dyn Data>,
}

#[derive(Debug, Clone)]
pub struct TensorView {
    pub tensor_id: NodeIndex,
    pub shape: ShapeTracker,
}

/// Some sort of data, for instance a Vec<f32> on CPU or CudaSlice<f32> on GPU
pub trait Data: Any + Debug + DynClone {
    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;
}

clone_trait_object!(Data);

impl Data for Vec<f32> {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

impl Tensor {
    /// Get the real data as layed out by the shape tracker
    pub fn real_data(&self, view: &TensorView) -> Option<Vec<f32>> {
        let Some(self_data) = self.data.as_any().downcast_ref::<Vec<f32>>() else {
            return None;
        };
        let mut data = vec![0.; view.shape.shape().iter().product()];
        let idx = view.shape.index_fn();
        for (i, r) in data.iter_mut().enumerate() {
            *r = self_data[(idx)(i)];
        }

        Some(data)
    }
}
