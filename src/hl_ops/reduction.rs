use crate::{
    op::{self},
    prelude::*,
};

impl GraphTensor {
    /// Reduce a dimension of the tensor by summing all elements along that axis.
    pub fn sum_reduce(self, axes: impl ToAxes) -> GraphTensor {
        let (mut shape, mut id) = (self.shape, self.id);
        // Sum reduce each dimension
        for dim in axes.to_axes().into_iter().rev() {
            id = self
                .graph()
                .add_op(op::SumReduce(dim))
                .input(id, 0, shape)
                .finish();
            shape.remove_dim(dim);
        }
        GraphTensor::from_id(id, shape, self.graph_ref)
    }

    /// Reduce a dimension of the tensor by taking the maximum of all elements along that axis.
    pub fn max_reduce(self, axes: impl ToAxes) -> GraphTensor {
        let (mut shape, mut id) = (self.shape, self.id);
        // Max reduce each dimension
        for dim in axes.to_axes().into_iter().rev() {
            id = self
                .graph()
                .add_op(op::MaxReduce(dim))
                .input(id, 0, shape)
                .finish();
            shape.remove_dim(dim);
        }
        GraphTensor::from_id(id, shape, self.graph_ref)
    }

    /// Reduce a dimension of the tensor by taking the mean of all elements along that axis.
    pub fn mean_reduce(self, axes: impl ToAxes) -> GraphTensor {
        let reduced_elements = axes
            .to_axes()
            .into_iter()
            .map(|i| self.dims()[i])
            .product::<Expression>();
        (self / reduced_elements).sum_reduce(axes)
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
        let a = cx.tensor((2, 3)).set(a_data.clone());
        let b = a.sum_reduce(1).retrieve();

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
        let a = cx.tensor((2, 3)).set(a_data.clone());
        let b = a.max_reduce(1).retrieve();

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
        let a = cx.tensor((2, 3)).set(a_data.clone());
        let b = a.mean_reduce(1).retrieve();

        cx.execute();

        let d_dev = Cpu::default();
        let d_a = d_dev.tensor_from_vec(a_data, (DConst::<2>, DConst::<3>));
        let d_b = d_a.mean::<_, DAxis<1>>();

        assert_close(&b.data(), &d_b.as_vec());
    }
}
