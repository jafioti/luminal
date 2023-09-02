use crate::{
    graph::Graph,
    op::{self, Function},
    prelude::{Data, TensorView},
    shape::*,
    tensor::Tensor,
};
use std::marker::PhantomData;

use petgraph::graph::NodeIndex;

#[derive(Clone, Copy)]
pub struct GraphTensor<S: Shape> {
    pub id: NodeIndex,
    pub graph_ref: *mut Graph,
    pub(crate) _phantom: PhantomData<S>,
    pub shape: crate::core::shape::simple_tracker::ShapeTracker,
}

impl<S: Shape> GraphTensor<S> {
    pub fn from_id(
        id: NodeIndex,
        shape: crate::core::shape::simple_tracker::ShapeTracker,
        graph_ref: *mut Graph,
    ) -> Self {
        Self {
            id,
            graph_ref,
            shape,
            _phantom: Default::default(),
        }
    }

    /// Mark this tensor to be retrieved later
    pub fn mark(&self) {
        unsafe {
            self.graph_ref.as_mut().unwrap().no_delete.insert(self.id);
            self.graph_ref.as_mut().unwrap().to_retrieve.insert(self.id);
        }
    }

    /// Remove this tensor's data from the graph. All other views pointing to the same tensor become invalidated.
    pub fn drop(&self) {
        let graph = unsafe { self.graph_ref.as_mut().unwrap() };
        graph
            .tensors
            .remove(&graph.views.remove(&self.id).unwrap().tensor_id);
    }

    /// Mark this tensor to not be deleted, but not retrieved either
    pub fn mark_no_delete(&self) {
        unsafe {
            self.graph_ref.as_mut().unwrap().no_delete.insert(self.id);
        }
    }

    /// Get the value of the tensor (if the graph was executed)
    pub fn retrieve(&self) -> Option<Tensor> {
        unsafe { self.graph_ref.as_mut().unwrap().get_tensor(self.id) }
    }

    /// Get the contiguous data of the tensor
    pub fn data(&self) -> Vec<f32> {
        self.retrieve()
            .unwrap()
            .real_data(self.view().unwrap())
            .unwrap()
    }

    /// Set the value of the tensor, with dynamic dimensions.
    pub fn set_dyn<T: Data + Clone>(&self, data: T, shape: Vec<usize>) {
        let graph = unsafe { self.graph_ref.as_mut().unwrap() };
        let node = graph
            .graph
            .node_weight_mut(self.id)
            .unwrap()
            .as_any_mut()
            .downcast_mut::<Function>()
            .unwrap();
        // We shouldn't do cloning here!
        node.1 = Box::new(move |_| Tensor {
            data: Box::new(data.clone()),
        });
    }

    /// Set the name of a tensor
    pub fn set_name(&self, name: &str) {
        let graph = unsafe { self.graph_ref.as_mut().unwrap() };
        let node = graph
            .graph
            .node_weight_mut(self.id)
            .unwrap()
            .as_any_mut()
            .downcast_mut::<Function>()
            .unwrap();
        node.0 = name.to_string();
    }

    pub fn debug(&self, message: &str) {
        let graph = unsafe { self.graph_ref.as_mut().unwrap() };
        graph
            .add_op(op::Print(message.to_string()))
            .input(self.id, self.shape)
            .finish();
    }

    pub fn view(&self) -> Option<&TensorView> {
        unsafe { self.graph_ref.as_mut().unwrap().get_view(self.id) }
    }
}

impl<S: ConstShape> GraphTensor<S> {
    /// Set the value of the tensor matching the constant shape
    pub fn set<T: Data + Clone>(&self, data: T) {
        let graph = unsafe { self.graph_ref.as_mut().unwrap() };
        let node = graph
            .graph
            .node_weight_mut(self.id)
            .unwrap()
            .as_any_mut()
            .downcast_mut::<Function>()
            .unwrap();
        // We shouldn't do cloning here!
        node.1 = Box::new(move |_| Tensor {
            data: Box::new(data.clone()),
        });
    }
}
