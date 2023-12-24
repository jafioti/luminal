use crate::{
    graph::Graph,
    op::{self, Function},
    prelude::{remap_id, Data},
    shape::*,
    tensor::Tensor,
};
use std::fmt::Debug;
use std::marker::PhantomData;

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

    /// Get remapped graph id of this node
    pub fn id(&self) -> NodeIndex {
        remap_id(self.id, &self.graph().id_remap)
    }

    /// Mark this tensor to not be deleted
    pub fn keep(self) -> Self {
        self.graph().no_delete.insert(self.id());
        self
    }

    /// Mark this tensor to be retrieved later
    pub fn retrieve(self) -> Self {
        self.keep();
        self.graph().to_retrieve.insert(self.id());
        self
    }

    /// Remove this tensor's data from the graph.
    pub fn drop(&self) {
        self.graph().tensors.remove(&(self.id(), 0));
    }

    #[allow(clippy::mut_from_ref)]
    pub fn graph(&self) -> &mut Graph {
        unsafe { self.graph_ref.as_mut().unwrap() }
    }

    /// Set the value of the tensor, with dynamic dimensions.
    pub fn set_dyn<T: Data + Clone>(self, data: T, shape: Vec<usize>) -> Self {
        // Report dyn dim values to graph dyn map
        for (d, s) in S::realized_shape().iter().zip(shape.iter()) {
            if let Some(c) = d.to_symbols().pop() {
                self.graph().dyn_map.insert(c, *s);
            }
        }
        let node = self
            .graph()
            .graph
            .node_weight_mut(self.id)
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
        self
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

    pub fn print(&self, message: &str) {
        let id = self
            .graph()
            .add_op(op::Print(message.to_string()))
            .input(self.id, 0, self.shape)
            .finish();
        self.graph().no_delete.insert(id);
    }

    pub fn no_shape(self) -> GraphTensor<()> {
        GraphTensor::from_id(self.id, self.shape, self.graph_ref)
    }

    /// Get the contiguous data of the tensor
    pub fn data(&self) -> Vec<f32> {
        let st = self.shape.resolve_global_dyn_dims(&self.graph().dyn_map);
        let tensor = self.graph().get_tensor_ref(self.id, 0).unwrap();
        let orig_data = tensor.data.as_any().downcast_ref::<Vec<f32>>().unwrap();
        let mut data = vec![0.; st.n_elements().to_usize().unwrap()];
        let ind = st.index_expression();
        let val = st.valid_expression();
        #[allow(unused_mut)]
        for (i, mut r) in data.iter_mut().enumerate() {
            if val.exec_single_var(i) != 0 {
                *r = orig_data[ind.exec_single_var(i)];
            }
        }
        data
    }
}

impl<S: ConstShape> GraphTensor<S> {
    /// Set the value of the tensor matching the constant shape
    pub fn set<T: Data + Clone>(self, data: T) -> Self {
        let node = self
            .graph()
            .graph
            .node_weight_mut(self.id)
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
        self
    }
}

pub fn pretty_print_tensor_recursive(
    f: &mut std::fmt::Formatter<'_>,
    data: &[f32],
    shape: &[usize],
    level: usize,
) -> std::fmt::Result {
    if shape.is_empty() {
        // Base case: no dimensions left
        return Ok(());
    }

    let indent = "  ".repeat(level);

    if shape.len() == 1 {
        // If this is the innermost dimension, print the raw data in a single line
        write!(f, "{}[", indent)?;
        if data.len() > 10 {
            for (i, value) in data.iter().take(5).enumerate() {
                write!(f, "{:.4}", value)?;
                if i < data.len() - 1 {
                    write!(f, ", ")?;
                }
            }
            write!(f, "..., ")?;
            for (i, value) in data.iter().skip(data.len() - 5).enumerate() {
                write!(f, "{:.4}", value)?;
                if i < data.len() - 1 {
                    write!(f, ", ")?;
                }
            }
        } else {
            for (i, value) in data.iter().enumerate() {
                write!(f, "{:.4}", value)?;
                if i < data.len() - 1 {
                    write!(f, ", ")?;
                }
            }
        }
        write!(f, "]")?; // No newline after the innermost array
    } else {
        // For higher dimensions, handle the nesting
        writeln!(f, "{indent}[")?;
        let stride = shape[1..].iter().product();
        if data.len() / stride > 10 {
            for (i, chunk) in data.chunks(stride).take(5).enumerate() {
                pretty_print_tensor_recursive(f, chunk, &shape[1..], level + 1)?;
                if i < shape[0] - 1 {
                    writeln!(f, ",")?; // Place the comma right after the bracket and then a newline
                }
            }
            writeln!(f, "{indent}  ..., ")?;
            for (i, chunk) in data
                .chunks(stride)
                .skip(data.len() / stride - 5)
                .enumerate()
            {
                pretty_print_tensor_recursive(f, chunk, &shape[1..], level + 1)?;
                if i < shape[0] - 1 {
                    writeln!(f, ",")?; // Place the comma right after the bracket and then a newline
                }
            }
        } else {
            for (i, chunk) in data.chunks(stride).enumerate() {
                pretty_print_tensor_recursive(f, chunk, &shape[1..], level + 1)?;
                if i < shape[0] - 1 {
                    writeln!(f, ",")?; // Place the comma right after the bracket and then a newline
                }
            }
        }
        writeln!(f)?; // Add a newline before closing the current dimension bracket
        write!(f, "{indent}]")?; // Close the current dimension bracket
    }

    // Only add a newline after the top-level closing bracket
    if level == 0 {
        writeln!(f)?;
    }

    Ok(())
}

impl<S: Shape> Debug for GraphTensor<S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Get the data
        let data = self.data();

        // Get the shape
        let shape = self
            .shape
            .shape()
            .iter()
            .map(|expr| expr.exec(&self.graph().dyn_map).unwrap())
            .collect::<Vec<_>>();

        // Print the shape
        writeln!(f, "Tensor with Shape: {:?}", shape)?;

        // Print the data by going dimension by dimension, recursively
        pretty_print_tensor_recursive(f, &data, &shape, 0)
    }
}

pub trait MarkTensors {
    /// Mark all tensors in this collection to be kept
    fn keep(&self);
    /// Mark all tensors in this collection to be retrieved
    fn retrieve(&self);
}

impl<S: Shape> MarkTensors for GraphTensor<S> {
    fn keep(&self) {
        GraphTensor::keep(*self);
    }

    fn retrieve(&self) {
        GraphTensor::retrieve(*self);
    }
}

impl<S: MarkTensors> MarkTensors for Vec<S> {
    fn keep(&self) {
        for t in self {
            t.keep();
        }
    }

    fn retrieve(&self) {
        for t in self {
            t.retrieve();
        }
    }
}
impl<S: MarkTensors> MarkTensors for &[S] {
    fn keep(&self) {
        for t in *self {
            t.keep();
        }
    }

    fn retrieve(&self) {
        for t in *self {
            t.retrieve();
        }
    }
}

macro_rules! tuple_impls {
    ([$($name:ident),+] , [$($idx:tt),+]) => {
        impl<
        $($name:
            MarkTensors, )+
        > MarkTensors for ($($name,)+) {
            fn keep(&self) {
                $(self.$idx.keep();)+
            }
            fn retrieve(&self) {
                $(self.$idx.retrieve();)+
            }
        }
    };
}

tuple_impls!([M1], [0]);
tuple_impls!([M1, M2], [0, 1]);
tuple_impls!([M1, M2, M3], [0, 1, 2]);
tuple_impls!([M1, M2, M3, M4], [0, 1, 2, 3]);
tuple_impls!([M1, M2, M3, M4, M5], [0, 1, 2, 3, 4]);
tuple_impls!([M1, M2, M3, M4, M5, M6], [0, 1, 2, 3, 4, 5]);
tuple_impls!([M1, M2, M3, M4, M5, M6, M7], [0, 1, 2, 3, 4, 5, 6]);
tuple_impls!([M1, M2, M3, M4, M5, M6, M7, M8], [0, 1, 2, 3, 4, 5, 6, 7]);
tuple_impls!(
    [M1, M2, M3, M4, M5, M6, M7, M8, M9],
    [0, 1, 2, 3, 4, 5, 6, 7, 8]
);
tuple_impls!(
    [M1, M2, M3, M4, M5, M6, M7, M8, M9, M10],
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
);
