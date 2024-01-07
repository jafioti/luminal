use std::{any::Any, fmt::Debug};

use dyn_clone::{clone_trait_object, DynClone};

/// An entirely dynamic tensor with data
#[derive(Debug, Clone)]
pub struct Tensor {
    pub data: Box<dyn Data>,
}

impl Tensor {
    pub fn new<T: Data>(data: T) -> Self {
        Self {
            data: Box::new(data),
        }
    }
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
