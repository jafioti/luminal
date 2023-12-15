use crate::{
    op::{Constant, Function},
    prelude::*,
};

impl<S: Shape> GraphTensor<S> {
    /// Cumulative summation along a dimension
    pub fn cumsum<const DIM: usize>(self) -> GraphTensor<S> {
        let id = self
            .graph()
            .add_op(Function(
                "CumSum".to_string(),
                Box::new(move |inp| {
                    let front_size: usize = inp[0]
                        .1
                        .shape()
                        .iter()
                        .take(DIM)
                        .filter_map(|i| i.to_usize())
                        .product();
                    let back_size: usize = inp[0]
                        .1
                        .shape()
                        .iter()
                        .skip(DIM + 1)
                        .filter_map(|i| i.to_usize())
                        .product();
                    let dim_size = inp[0].1.shape()[DIM].to_usize().unwrap();
                    let mut scratchpad: Vec<f32> = vec![0.0; front_size * back_size];
                    let a_data = inp[0]
                        .0
                        .borrowed()
                        .data
                        .as_any()
                        .downcast_ref::<Vec<f32>>()
                        .unwrap();
                    let mut result = vec![0.0; front_size * back_size * dim_size];
                    let ind = inp[0].1.index_expression();
                    let val = inp[0].1.valid_expression();

                    for i in 0..front_size {
                        for j in 0..back_size {
                            for k in 0..dim_size {
                                let original_index = i * dim_size * back_size + k * back_size + j;
                                let new_index = i * back_size + j;
                                if val.exec_single_var(original_index) != 0 {
                                    scratchpad[new_index] +=
                                        a_data[ind.exec_single_var(original_index)];
                                }
                                result[original_index] = scratchpad[new_index];
                            }
                        }
                    }
                    vec![Tensor {
                        data: Box::new(result),
                    }]
                }),
            ))
            .input(self.id, 0, self.shape)
            .finish();
        GraphTensor::from_id(id, self.shape, self.graph_ref)
    }

    /// Create an arange of the same shape, along a certian dimension
    pub fn arange<const DIM: usize>(self) -> GraphTensor<S> {
        self.graph().constant(1.).expand::<S, _>().cumsum::<DIM>() - 1.0
    }
}

impl Graph {
    pub fn constant(&mut self, i: f32) -> GraphTensor<R0> {
        GraphTensor::from_id(
            self.add_op(Constant(i)).finish(),
            ShapeTracker::new(&[]),
            self,
        )
    }
}
