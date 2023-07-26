use crate::{graph::Graph, shape::*, tensor::Tensor};
use std::marker::PhantomData;

use petgraph::graph::NodeIndex;

#[derive(Clone, Copy)]
pub struct GraphTensor<S: Shape> {
    pub id: NodeIndex,
    pub graph_ref: *mut Graph,
    pub(crate) _phantom: PhantomData<S>,
}

impl<S: Shape> GraphTensor<S> {
    pub(crate) fn from_id(id: NodeIndex, graph_ref: *mut Graph) -> Self {
        Self {
            id,
            graph_ref,
            _phantom: Default::default(),
        }
    }

    /// Get the shape tracker for this tensor
    pub fn shape_tracker(&self) -> &Vec<RealDim> {
        let graph = unsafe { self.graph_ref.as_mut().unwrap() };
        &graph.graph.node_weight(self.id).unwrap().1
    }

    /// Mark this tensor to be retrieved later
    pub fn mark(&self) {
        unsafe {
            self.graph_ref.as_mut().unwrap().no_delete.insert(self.id);
            self.graph_ref.as_mut().unwrap().to_retrieve.insert(self.id);
        }
    }

    /// Get the value of the tensor (if the graph was executed)
    pub fn retrieve(self) -> Option<Tensor> {
        unsafe { self.graph_ref.as_mut().unwrap().get_tensor(self.id) }
    }

    /// Set the value of the tensor, with dynamic dimensions.
    pub fn set_dyn(&self, data: Vec<f32>, shape: Vec<usize>) {
        unsafe {
            self.graph_ref
                .as_mut()
                .unwrap()
                .set_tensor(*self, data, shape)
        }
    }
}

impl<S: ConstShape> GraphTensor<S> {
    /// Set the value of the tensor matching the constant shape
    pub fn set(&self, data: Vec<f32>) {
        unsafe {
            self.graph_ref.as_mut().unwrap().set_tensor(
                *self,
                data,
                <S as ConstShape>::realized_shape(),
            )
        }
    }
}
