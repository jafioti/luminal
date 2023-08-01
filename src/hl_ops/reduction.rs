use itertools::Itertools;

use crate::{op, prelude::*};

impl<S: Shape> GraphTensor<S> {
    pub fn sum_reduce<Dst: Shape, Ax: Axes>(self) -> GraphTensor<Dst>
    where
        S: HasAxes<Ax> + ReduceShapeTo<Dst, Ax>,
    {
        let graph = unsafe { self.graph_ref.as_mut().unwrap() };
        let mut shape = self.shape_tracker().clone();

        let mut new_id = self.id;
        for dim in Ax::as_array().into_iter().collect_vec().into_iter().rev() {
            shape.remove(dim as usize);
            new_id = graph
                .add_op(op::SumReduce(dim as usize), shape.clone())
                .input(new_id)
                .finish();
        }
        GraphTensor::from_id(new_id, self.graph_ref)
    }

    pub fn max_reduce<Dst: Shape, Ax: Axes>(self) -> GraphTensor<Dst>
    where
        S: HasAxes<Ax> + ReduceShapeTo<Dst, Ax>,
    {
        let graph = unsafe { self.graph_ref.as_mut().unwrap() };
        let mut shape = self.shape_tracker().clone();

        let mut new_id = self.id;
        for dim in Ax::as_array().into_iter().collect_vec().into_iter().rev() {
            shape.remove(dim as usize);
            new_id = graph
                .add_op(op::MaxReduce(dim as usize), shape.clone())
                .input(new_id)
                .finish();
        }
        GraphTensor::from_id(new_id, self.graph_ref)
    }

    pub fn mean_reduce<Dst: Shape, Ax: Axes>(self) -> GraphTensor<Dst>
    where
        S: HasAxes<Ax> + ReduceShapeTo<Dst, Ax>,
    {
        let graph = unsafe { self.graph_ref.as_mut().unwrap() };
        let mut shape = <S as Shape>::realized_shape();

        let mut node_id = self.id;
        for dim in Ax::as_array().into_iter().collect_vec().into_iter().rev() {
            // Reduce shape
            shape.remove(dim as usize);
            // Sum reduce
            node_id = graph
                .add_op(op::SumReduce(dim as usize), shape.clone())
                .input(node_id)
                .finish();
            // Create div tensor
            let size_t = graph
                .add_op(
                    op::Function(Box::new(move |inp, i| {
                        let s = inp[0].1.shape.shape()[dim as usize];
                        (
                            Some(Tensor {
                                data: Box::new(vec![s as f32]),
                            }),
                            TensorView {
                                tensor_id: i,
                                shape: ShapeTracker::new(vec![]),
                            },
                        )
                    })),
                    vec![],
                )
                .input(self.id)
                .finish();
            let size_t: GraphTensor<R0> = GraphTensor::from_id(size_t, graph);
            let mut size_t = size_t.id;
            let mut size_t_shape = vec![];
            // Expand div tensor
            for (dim, size) in shape.iter().enumerate() {
                size_t_shape.insert(dim, *size);
                size_t = graph
                    .add_op(op::Expand(dim, *size), size_t_shape.clone())
                    .input(size_t)
                    .finish();
            }
            // Divide by div tensor
            node_id = graph
                .add_op(op::Div, shape.clone())
                .input(node_id)
                .input(size_t)
                .finish();
        }
        GraphTensor::from_id(node_id, self.graph_ref)
    }
}

#[cfg(test)]
mod tests {
    use crate::{prelude::*, tests::assert_close_data};
    use dfdx::prelude::*;

    #[test]
    fn test_mean_reduce() {
        let mut cx = Graph::new();
        let a = cx.new_tensor::<R2<2, 3>>();
        a.set(vec![1., 2., 3., 1., 2., 3.]);
        let b = a.mean_reduce::<_, crate::prelude::Axis<1>>();
        b.mark();

        cx.execute();

        let d_dev = Cpu::default();
        let d_a = d_dev.tensor([[1., 2., 3.], [1., 2., 3.]]);
        let d_b = d_a.mean::<_, dfdx::shapes::Axis<1>>();

        assert_close_data(
            &b.retrieve().unwrap().real_data(b.view().unwrap()).unwrap(),
            &d_b.as_vec(),
        );
    }
}
