use crate::prelude::*;

impl<S: Shape> GraphTensor<S> {
    // /// Cumulative summation along a dimension
    // pub fn cumsum<const DIM: usize>(self) -> GraphTensor<S> {
    //     let graph = unsafe { self.graph_ref.as_mut().unwrap() };
    //     let id = graph
    //         .add_op(
    //             Function(
    //                 "CumSum".to_string(),
    //                 Box::new(move |inp, i| {
    //                     let front_size: usize = inp[0].1.shape.shape().iter().take(DIM).product();
    //                     let back_size: usize =
    //                         inp[0].1.shape.shape().iter().skip(DIM + 1).product();
    //                     let dim_size = inp[0].1.shape.shape()[DIM];
    //                     let mut scratchpad: Vec<f32> = vec![0.0; front_size * back_size];
    //                     let a_data = inp[0].0.data.as_any().downcast_ref::<Vec<f32>>().unwrap();
    //                     let mut result = vec![0.0; front_size * back_size * dim_size];
    //                     let (a_idx, a_valid) = inp[0].1.shape.index_node();

    //                     for i in 0..front_size {
    //                         for j in 0..back_size {
    //                             for k in 0..dim_size {
    //                                 let original_index =
    //                                     i * dim_size * back_size + k * back_size + j;
    //                                 let new_index = i * back_size + j;
    //                                 if a_valid.solve(original_index as i32) != 0 {
    //                                     scratchpad[new_index] +=
    //                                         a_data[a_idx.solve(original_index as i32) as usize];
    //                                 }
    //                                 result[original_index] = scratchpad[new_index];
    //                             }
    //                         }
    //                     }
    //                     (
    //                         Some(Tensor {
    //                             data: Box::new(result),
    //                         }),
    //                         TensorView {
    //                             tensor_id: i,
    //                             shape: ShapeTracker::new(inp[0].1.shape.shape().clone()),
    //                         },
    //                     )
    //                 }),
    //             ),
    //             S::realized_shape(),
    //         )
    //         .input(self.id)
    //         .finish();
    //     GraphTensor::from_id(id, graph)
    // }
}

impl Graph {
    // pub fn arange<N: Dim>(&mut self) -> GraphTensor<(N,)> {
    //     self.constant(1.).expand::<(N,), _>().cumsum::<0>() - 1.0
    // }

    pub fn constant<T>(&mut self, i: T) -> GraphTensor<R0>
    where
        Vec<T>: Data + Clone,
    {
        let t = self.new_tensor("Const");
        t.set(vec![i]);
        t
    }
}
