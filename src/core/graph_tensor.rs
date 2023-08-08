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
}

impl<S: Shape> GraphTensor<S> {
    pub fn from_id(id: NodeIndex, graph_ref: *mut Graph) -> Self {
        Self {
            id,
            graph_ref,
            _phantom: Default::default(),
        }
    }

    /// Get the shape tracker for this tensor
    pub fn shape(&self) -> &Vec<RealDim> {
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
    pub fn set_dyn<T: Data + Clone>(&self, data: T, shape: Vec<usize>) {
        let graph = unsafe { self.graph_ref.as_mut().unwrap() };
        let node = graph
            .graph
            .node_weight_mut(self.id)
            .unwrap()
            .0
            .as_any_mut()
            .downcast_mut::<Function>()
            .unwrap();
        // We shouldn't do cloning here!
        node.1 = Box::new(move |_, i| {
            (
                Some(Tensor {
                    data: Box::new(data.clone()),
                }),
                TensorView {
                    tensor_id: i,
                    shape: ShapeTracker::new(shape.clone()),
                },
            )
        });
    }

    /// Set the name of a tensor
    pub fn set_name(&self, name: &str) {
        let graph = unsafe { self.graph_ref.as_mut().unwrap() };
        let node = graph
            .graph
            .node_weight_mut(self.id)
            .unwrap()
            .0
            .as_any_mut()
            .downcast_mut::<Function>()
            .unwrap();
        node.0 = name.to_string();
    }

    pub fn debug(&self, message: &str) {
        let graph = unsafe { self.graph_ref.as_mut().unwrap() };
        graph
            .add_op(op::Print(message.to_string()), vec![])
            .input(self.id)
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
            .0
            .as_any_mut()
            .downcast_mut::<Function>()
            .unwrap();
        // We shouldn't do cloning here!
        node.1 = Box::new(move |_, i| {
            (
                Some(Tensor {
                    data: Box::new(data.clone()),
                }),
                TensorView {
                    tensor_id: i,
                    shape: ShapeTracker::new(<S as ConstShape>::realized_shape()),
                },
            )
        });
    }
}
