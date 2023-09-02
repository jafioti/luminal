use crate::{
    op::{Function, NoOp},
    prelude::{simple_tracker::Dim, *},
};

impl<S: Shape> GraphTensor<S> {
    /// Cumulative summation along a dimension
    pub fn cumsum<const DIM: usize>(self) -> GraphTensor<S> {
        let graph = unsafe { self.graph_ref.as_mut().unwrap() };
        let id = graph
            .add_op(Function(
                "CumSum".to_string(),
                Box::new(move |inp| {
                    let front_size: usize = inp[0]
                        .1
                        .shape()
                        .iter()
                        .take(DIM)
                        .filter_map(|i| match i {
                            Dim::Known(n) => Some(n),
                            Dim::Unknown => None,
                        })
                        .product();
                    let back_size: usize = inp[0]
                        .1
                        .shape()
                        .iter()
                        .skip(DIM + 1)
                        .filter_map(|i| match i {
                            Dim::Known(n) => Some(n),
                            Dim::Unknown => None,
                        })
                        .product();
                    let dim_size = match inp[0].1.shape()[DIM] {
                        Dim::Known(n) => n,
                        Dim::Unknown => panic!(),
                    };
                    let mut scratchpad: Vec<f32> = vec![0.0; front_size * back_size];
                    let a_data = inp[0]
                        .0
                        .borrowed()
                        .data
                        .as_any()
                        .downcast_ref::<Vec<f32>>()
                        .unwrap();
                    let mut result = vec![0.0; front_size * back_size * dim_size];

                    for i in 0..front_size {
                        for j in 0..back_size {
                            for k in 0..dim_size {
                                let original_index = i * dim_size * back_size + k * back_size + j;
                                let new_index = i * back_size + j;
                                if let Some(n) = inp[0].1.index(original_index) {
                                    scratchpad[new_index] += a_data[n];
                                }
                                result[original_index] = scratchpad[new_index];
                            }
                        }
                    }
                    Tensor {
                        data: Box::new(result),
                    }
                }),
            ))
            .input(self.id, self.shape)
            .finish();
        GraphTensor::from_id(id, self.shape, graph)
    }

    pub fn match_shape<Dst: Shape>(self, rhs: GraphTensor<Dst>) -> GraphTensor<Dst> {
        let graph = unsafe { self.graph_ref.as_mut().unwrap() };
        GraphTensor::from_id(
            graph
                .add_op(NoOp)
                .input(self.id, self.shape)
                .input(rhs.id, rhs.shape)
                .finish(),
            self.shape,
            graph,
        )
    }

    /// Create an arange of the same shape, along a certian dimension
    pub fn arange<const DIM: usize>(self) -> GraphTensor<S> {
        let graph = unsafe { self.graph_ref.as_mut().unwrap() };
        graph.constant(1.).match_shape(self).cumsum::<DIM>() - 1.0
    }
}

impl Graph {
    pub fn constant<T>(&mut self, i: T) -> GraphTensor<R0>
    where
        Vec<T>: Data + Clone,
    {
        let t = self.new_tensor("Const");
        t.set(vec![i]);
        t
    }
}
