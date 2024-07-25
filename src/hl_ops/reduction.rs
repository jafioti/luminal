use crate::{
    op::{self},
    prelude::*,
};

impl GraphTensor {
    /// Reduce a dimension of the tensor by summing all elements along that axis.
    pub fn sum_reduce(self, axes: impl ToAxes) -> GraphTensor {
        let mut shape = self.shape;

        let mut new_id = self.id;
        for dim in axes.to_axes().into_iter().rev() {
            new_id = self
                .graph()
                .add_op(op::SumReduce(dim))
                .input(new_id, 0, shape)
                .finish();
            // Reduce shape
            shape.remove_dim(dim);
        }
        GraphTensor::from_id(new_id, shape, self.graph_ref)
    }

    /// Reduce a dimension of the tensor by taking the maximum of all elements along that axis.
    pub fn max_reduce(self, axes: impl ToAxes) -> GraphTensor {
        let mut shape = self.shape;

        let mut new_id = self.id;
        for dim in axes.to_axes().into_iter().rev() {
            new_id = self
                .graph()
                .add_op(op::MaxReduce(dim))
                .input(new_id, 0, shape)
                .finish();
            // Reduce shape
            shape.remove_dim(dim);
        }
        GraphTensor::from_id(new_id, shape, self.graph_ref)
    }

    /// Reduce a dimension of the tensor by taking the mean of all elements along that axis.
    pub fn mean_reduce(self, axes: impl ToAxes) -> GraphTensor {
        let mut shape = self.shape;
        let mut node_id = self.id;
        for dim in axes.to_axes().into_iter().rev() {
            // Sum reduce
            node_id = self
                .graph()
                .add_op(op::SumReduce(dim))
                .input(node_id, 0, shape)
                .finish();

            // Divide by size of dimension
            let div_tensor = self.graph().constant_expr(shape.remove_dim(dim)).id;
            let mul_tensor = self
                .graph()
                .add_op(op::Recip)
                .input(div_tensor, 0, ShapeTracker::new(()))
                .finish();
            node_id = self
                .graph()
                .add_op(op::Mul)
                .input(node_id, 0, shape)
                .input(mul_tensor, 0, ShapeTracker::fake(shape))
                .finish();
        }
        GraphTensor::from_id(node_id, shape, self.graph_ref)
    }

    /// Reduce a dimension of the tensor by multiplying all elements along that axis.
    pub fn prod_reduce(self, axes: impl ToAxes) -> GraphTensor {
        self.ln().sum_reduce(axes).exp()
    }
}

#[cfg(test)]
mod tests {
    crate::test_imports!();

    #[test]
    fn test_sum_reduce() {
        let mut cx = Graph::new();
        let a_data = random_vec(6);
        let a = cx.tensor((2, 3));
        a.set(a_data.clone());
        let b = a.sum_reduce(1);
        b.retrieve();

        cx.execute();

        let d_dev = Cpu::default();
        let d_a = d_dev.tensor_from_vec(a_data, (DConst::<2>, DConst::<3>));
        let d_b = d_a.sum::<_, DAxis<1>>();

        assert_close(&b.data(), &d_b.as_vec());
    }

    #[test]
    fn test_max_reduce() {
        let mut cx = Graph::new();
        let a_data = random_vec(6);
        let a = cx.tensor((2, 3));
        a.set(a_data.clone());
        let b = a.max_reduce(1);
        b.retrieve();

        cx.execute();

        let d_dev = Cpu::default();
        let d_a = d_dev.tensor_from_vec(a_data, (DConst::<2>, DConst::<3>));
        let d_b = d_a.max::<_, DAxis<1>>();

        assert_close(&b.data(), &d_b.as_vec());
    }

    #[test]
    fn test_mean_reduce() {
        let mut cx = Graph::new();
        let a_data = random_vec(6);
        let a = cx.tensor((2, 3));
        a.set(a_data.clone());
        let b = a.mean_reduce(1);
        b.retrieve();

        cx.execute();

        let d_dev = Cpu::default();
        let d_a = d_dev.tensor_from_vec(a_data, (DConst::<2>, DConst::<3>));
        let d_b = d_a.mean::<_, DAxis<1>>();

        assert_close(&b.data(), &d_b.as_vec());
    }
}
