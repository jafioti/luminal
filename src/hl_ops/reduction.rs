use itertools::Itertools;

use crate::{op, prelude::*};

impl<S: Shape> GraphTensor<S> {
    pub fn sum_reduce<Dst: Shape, Ax: Axes>(self) -> GraphTensor<Dst>
    where
        S: HasAxes<Ax> + ReduceShapeTo<Dst, Ax>,
    {
        let graph = unsafe { self.graph_ref.as_mut().unwrap() };
        let mut shape = self.shape().clone();

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
        let mut shape = self.shape().clone();

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
            // Create div tensor
            // Create ones tensor and expand up to full tensor shape
            let mut ones = graph.constant(1.0).id;
            let mut ones_shape = vec![];
            for (dim, size) in shape.iter().enumerate() {
                ones_shape.insert(dim, *size);
                ones = graph
                    .add_op(op::Expand(dim, *size), ones_shape.clone())
                    .input(ones)
                    .finish();
            }
            ones = graph
                .add_op(op::NoOp, ones_shape.clone())
                .input(ones)
                .input(node_id)
                .finish();
            ones_shape.remove(dim as usize);
            // Sum reduce on current dimension
            let div_tensor = graph
                .add_op(op::SumReduce(dim as usize), ones_shape)
                .input(ones)
                .finish();
            // Reduce shape
            shape.remove(dim as usize);
            // Sum reduce
            node_id = graph
                .add_op(op::SumReduce(dim as usize), shape.clone())
                .input(node_id)
                .finish();

            // Divide by div tensor
            let mul_tensor = graph
                .add_op(op::Recip, shape.clone())
                .input(div_tensor)
                .finish();
            node_id = graph
                .add_op(op::Mul, shape.clone())
                .input(node_id)
                .input(mul_tensor)
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
