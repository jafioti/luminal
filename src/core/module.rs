use std::collections::HashMap;

use crate::prelude::Graph;

/// A module that can initialize it's variables on the graph
pub trait InitModule {
    fn initialize(cx: &mut Graph) -> Self;
}

/// A module with a forward pass
pub trait Module<I> {
    type Output;
    fn forward(&self, input: I) -> Self::Output;
}

/// An already-initilized module that can be loaded from a state dict
pub trait LoadModule {
    fn load(&mut self, state_dict: &mut StateDict);
}

pub struct StateDict {
    /// Map from a tensor key to the tensor data and shape
    pub data: HashMap<String, (Vec<f32>, Vec<usize>)>,
}
