use crate::{
    graph::Graph,
    op::{self, Function},
    prelude::{remap_id, Data},
    shape::*,
    tensor::Tensor,
};
use std::{any::TypeId, collections::HashMap, marker::PhantomData};

use petgraph::graph::NodeIndex;

#[derive(Clone, Copy)]
pub struct GraphTensor<S: Shape> {
    pub id: NodeIndex,
    pub graph_ref: *mut Graph,
    pub(crate) _phantom: PhantomData<S>,
    pub shape: crate::core::shape::tracker::ShapeTracker,
}

impl<S: Shape> GraphTensor<S> {
    pub fn from_id(
        id: NodeIndex,
        shape: crate::core::shape::tracker::ShapeTracker,
        graph_ref: *mut Graph,
    ) -> Self {
        Self {
            id,
            graph_ref,
            shape,
            _phantom: Default::default(),
        }
    }

    pub fn id(&self) -> NodeIndex {
        remap_id(self.id, &self.graph().id_remap)
    }

    /// Mark this tensor to be retrieved later
    pub fn mark(&self) {
        self.graph().no_delete.insert(self.id());
        self.graph().to_retrieve.insert(self.id());
    }

    /// Remove this tensor's data from the graph.
    pub fn drop(&self) {
        self.graph().tensors.remove(&(self.id(), 0));
    }

    /// Mark this tensor to not be deleted, but not retrieved either
    pub fn mark_no_delete(&self) {
        self.graph().no_delete.insert(self.id());
    }

    /// Get the value of the tensor (if the graph was executed)
    pub fn retrieve(&self) -> Option<&Tensor> {
        self.graph().get_tensor_ref(self.id, 0)
    }

    #[allow(clippy::mut_from_ref)]
    pub fn graph(&self) -> &mut Graph {
        unsafe { self.graph_ref.as_mut().unwrap() }
    }

    /// Set the value of the tensor, with dynamic dimensions.
    pub fn set_dyn<T: Data + Clone>(&self, data: T, shape: Vec<usize>) {
        // Report dyn dim values to graph dyn map
        for (d, s) in S::realized_shape().iter().zip(shape.iter()) {
            if let Dim::Unknown(c) = d {
                self.graph().dyn_map.insert(*c, *s);
            }
        }
        let node = self
            .graph()
            .graph
            .node_weight_mut(self.id())
            .unwrap()
            .as_any_mut()
            .downcast_mut::<Function>()
            .unwrap();
        // We shouldn't do cloning here!
        node.1 = Box::new(move |_| {
            vec![Tensor {
                data: Box::new(data.clone()),
            }]
        });
    }

    /// Set the name of a tensor
    pub fn set_name(&self, name: &str) {
        let node = self
            .graph()
            .graph
            .node_weight_mut(self.id)
            .unwrap()
            .as_any_mut()
            .downcast_mut::<Function>()
            .unwrap();
        node.0 = name.to_string();
    }

    /// Set type of this tensor
    pub fn set_type(&self, type_id: TypeId) {
        let node = self
            .graph()
            .graph
            .node_weight_mut(self.id)
            .unwrap()
            .as_any_mut()
            .downcast_mut::<Function>()
            .unwrap();
        node.2 = type_id;
    }

    pub fn debug(&self, message: &str) {
        let id = self
            .graph()
            .add_op(op::Print(message.to_string()))
            .input(self.id, 0, self.shape)
            .finish();
        self.graph().no_delete.insert(id);
    }

    pub fn dyn_data(&self, dyn_dim_map: &HashMap<char, usize>) -> Vec<f32> {
        let st = self.shape.resolve_global_dyn_dims(dyn_dim_map);
        let tensor = self.retrieve().unwrap();
        let orig_data = tensor.data.as_any().downcast_ref::<Vec<f32>>().unwrap();
        let mut data = vec![0.; st.n_elements()];
        let ind = st.indexer();
        #[allow(unused_mut)]
        for (i, mut r) in data.iter_mut().enumerate() {
            if let Some(n) = ind.index(i) {
                *r = orig_data[n];
            }
        }
        data
    }
}

impl<S: ConstShape> GraphTensor<S> {
    /// Set the value of the tensor matching the constant shape
    pub fn set<T: Data + Clone>(&self, data: T) {
        let node = self
            .graph()
            .graph
            .node_weight_mut(self.id())
            .unwrap()
            .as_any_mut()
            .downcast_mut::<Function>()
            .unwrap();
        // We shouldn't do cloning here!
        node.1 = Box::new(move |_| {
            vec![Tensor {
                data: Box::new(data.clone()),
            }]
        });
    }

    /// Get the contiguous data of the tensor
    pub fn data(self) -> Vec<f32> {
        let tensor = self.retrieve().unwrap();
        let orig_data = tensor.data.as_any().downcast_ref::<Vec<f32>>().unwrap();
        let mut data = vec![0.; self.shape.n_elements()];
        let ind = self.shape.indexer();
        #[allow(unused_mut)]
        for (i, mut r) in data.iter_mut().enumerate() {
            if let Some(n) = ind.index(i) {
                *r = orig_data[n];
            }
        }
        data
    }
}
