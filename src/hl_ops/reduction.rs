use itertools::Itertools;

use crate::{op, prelude::*};

impl<S: Shape> GraphTensor<S> {
    pub fn sum_reduce<Dst: Shape, Ax: Axes>(self) -> GraphTensor<Dst>
    where
        S: HasAxes<Ax> + ReduceShapeTo<Dst, Ax>,
    {
        let graph = unsafe { self.graph_ref.as_mut().unwrap() };
        let mut shape = self.shape;

        let mut new_id = self.id;
        for dim in Ax::as_array().into_iter().collect_vec().into_iter().rev() {
            new_id = graph
                .add_op(op::SumReduce(dim as usize))
                .input(new_id, shape)
                .finish();
            // Reduce shape
            shape.remove_dim(dim as usize);
        }
        GraphTensor::from_id(new_id, shape, self.graph_ref)
    }

    pub fn max_reduce<Dst: Shape, Ax: Axes>(self) -> GraphTensor<Dst>
    where
        S: HasAxes<Ax> + ReduceShapeTo<Dst, Ax>,
    {
        let graph = unsafe { self.graph_ref.as_mut().unwrap() };
        let mut shape = self.shape;

        let mut new_id = self.id;
        for dim in Ax::as_array().into_iter().collect_vec().into_iter().rev() {
            new_id = graph
                .add_op(op::MaxReduce(dim as usize))
                .input(new_id, shape)
                .finish();
            // Reduce shape
            shape.remove_dim(dim as usize);
        }
        GraphTensor::from_id(new_id, shape, self.graph_ref)
    }

    pub fn mean_reduce<Dst: Shape, Ax: Axes>(self) -> GraphTensor<Dst>
    where
        S: HasAxes<Ax> + ReduceShapeTo<Dst, Ax>,
    {
        let graph = unsafe { self.graph_ref.as_mut().unwrap() };
        let mut shape = self.shape;
        let mut node_id = self.id;
        for dim in Ax::as_array().into_iter().collect_vec().into_iter().rev() {
            // Create div tensor
            // Create ones tensor and expand up to full tensor shape
            let mut ones = graph.constant(1.0).id;
            let mut ones_shape =
                crate::core::shape::simple_tracker::ShapeTracker::new(&shape.shape());
            ones = graph
                .add_op(op::NoOp)
                .input(ones, ones_shape)
                .input(node_id, shape)
                .finish();
            // Sum reduce on current dimension
            let div_tensor = graph
                .add_op(op::SumReduce(dim as usize))
                .input(ones, ones_shape)
                .finish();
            ones_shape.remove_dim(dim as usize);
            // Sum reduce
            node_id = graph
                .add_op(op::SumReduce(dim as usize))
                .input(node_id, shape)
                .finish();
            shape.remove_dim(dim as usize);

            // Divide by div tensor
            let mul_tensor = graph.add_op(op::Recip).input(div_tensor, shape).finish();
            node_id = graph
                .add_op(op::Mul)
                .input(node_id, shape)
                .input(mul_tensor, shape)
                .finish();
        }
        GraphTensor::from_id(node_id, shape, self.graph_ref)
    }
}

#[cfg(test)]
mod tests {
    use crate::{prelude::*, tests::assert_close_data};
    use dfdx::prelude::*;

    #[test]
    fn test_mean_reduce() {
        let mut cx = Graph::new();
        let a = cx.new_tensor::<R2<2, 3>>("Input");
        a.set(vec![1., 2., 3., 1., 2., 3.]);
        let b = a.mean_reduce::<_, crate::prelude::Axis<1>>();
        b.mark();

        cx.execute();

        let d_dev = Cpu::default();
        let d_a = d_dev.tensor([[1., 2., 3.], [1., 2., 3.]]);
        let d_b = d_a.mean::<_, dfdx::shapes::Axis<1>>();

        assert_close_data(&b.data(), &d_b.as_vec());
    }
}
