use std::{any::Any, fmt::Debug};

use dyn_clone::{clone_trait_object, DynClone};

/// A tensor with data. The data can be anything that implements the Data trait
#[derive(Debug, Clone)]
pub struct Tensor {
    data: Box<dyn Data>,
}

impl Tensor {
    pub fn new<T: Data>(data: T) -> Self {
        Self {
            data: Box::new(data),
        }
    }
    pub fn downcast_ref<T: Data>(&self) -> Option<&T> {
        self.data.as_any().downcast_ref()
    }
    pub fn downcast_mut<T: Data>(&mut self) -> Option<&mut T> {
        self.data.as_any_mut().downcast_mut()
    }
    pub fn is<T: Data>(&self) -> bool {
        self.data.as_any().is::<T>()
    }
}

/// Some sort of data, for instance a Vec<f32> on CPU, CudaSlice<f32> on Nvidia GPUs, or metal::Buffer for Apple GPUs
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
