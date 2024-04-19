use crate::{op, prelude::*};

impl<S: Shape> GraphTensor<S> {
    pub fn permute<Dst: Shape, Ax: Axes>(mut self) -> GraphTensor<Dst>
    where
        S: PermuteShapeTo<Dst, Ax>,
    {
        self.shape
            .permute(&Ax::as_array().into_iter().collect::<Vec<_>>());
        GraphTensor::from_id(self.id, self.shape, self.graph_ref)
    }

    pub fn expand<Dst: Shape, Ax: Axes>(mut self) -> GraphTensor<Dst>
    where
        S: BroadcastShapeTo<Dst, Ax>,
    {
        let new_dims = Dst::realized_shape();
        if !new_dims.is_empty() {
            for (i, dim) in Ax::as_array().into_iter().map(|i| (i, new_dims[i])) {
                self.shape.expand(i, dim);
            }
        }

        GraphTensor::from_id(self.id, self.shape, self.graph_ref)
    }

    pub fn expand_to<Dst: Shape>(mut self, shape: ShapeTracker) -> GraphTensor<Dst> {
        for (i, s) in shape.indexes.iter().map(|i| shape.dims[*i]).enumerate() {
            if self.shape.len() <= i || self.shape.dims[self.shape.indexes[i]] != s {
                self.shape.expand(i, s);
            }
        }

        GraphTensor::from_id(self.id, self.shape, self.graph_ref)
    }

    pub fn reshape<N: Shape>(mut self) -> GraphTensor<N> {
        // Insert contiguous call
        self = self.contiguous();
        GraphTensor::from_id(
            self.id,
            ShapeTracker::new(&N::realized_shape()),
            self.graph_ref,
        )
    }

    /// Dynamically reshape with annotations for the shape tracker
    pub fn dyn_reshape<N: Shape>(mut self, shape: Vec<Expression>) -> GraphTensor<N> {
        if !self.shape.indexes.iter().enumerate().all(|(a, b)| a == *b) {
            // Insert contiguous call
            self = self.contiguous();
        }

        GraphTensor::from_id(self.id, ShapeTracker::new(&shape), self.graph_ref)
    }

    pub fn realize<Dst: Shape<Concrete = <<S as HasShape>::Shape as Shape>::Concrete>>(
        self,
    ) -> GraphTensor<Dst>
    where
        S: RealizeShapeTo<Dst>,
    {
        GraphTensor::from_id(self.id, self.shape, self.graph_ref)
    }

    pub fn sync_shape(self) -> Self {
        GraphTensor::from_id(
            self.id,
            ShapeTracker::new(&S::realized_shape()),
            self.graph_ref,
        )
    }

    pub fn contiguous(self) -> GraphTensor<S> {
        if !self.shape.is_reshaped() {
            return self;
        }
        let new_id = self
            .graph()
            .add_op(op::Contiguous)
            .input(self.id, 0, self.shape)
            .finish();
        GraphTensor::from_id(new_id, self.shape.contiguous(), self.graph_ref)
    }

    /// Take a slice of the original tensor. Any dimension with bounds becomes a dynamic dimension
    pub fn slice<Slice: SliceOfShape<S>>(
        mut self,
        slice: Slice,
    ) -> GraphTensor<Slice::OutputShape> {
        let ranges = slice.to_range_vec();
        // This exists because currently padding and slicing on the same dimension (even on opposite sides) is unsupported
        if ranges.iter().zip(self.shape.indexes).any(|(range, ind)| {
            (range.0 != 0.into() || range.1 != i32::MAX.into())
                && (self.shape.padding[self.shape.indexes[ind]].0 != 0.into()
                    || self.shape.padding[self.shape.indexes[ind]].1 != 0.into())
        }) {
            self = self.contiguous();
        }
        self.shape.slice(&ranges);
        GraphTensor::from_id(self.id, self.shape, self.graph_ref)
    }

    /// Cut out 'size' elements every 'spacing' elements in the last dimension. 'size' must be smaller than the last dimension
    pub fn excise<Dst: Shape>(mut self, spacing: usize, size: usize) -> GraphTensor<Dst> {
        let n_dims = self.shape.len();
        // Pad out to a multiple of spacing + size
        let total_size = (self.shape.dims[self.shape.indexes[n_dims - 1]] + ((spacing + size) - 1))
            / (spacing + size)
            * (spacing + size);
        let padding = total_size - self.shape.dims[self.shape.indexes[n_dims - 1]];
        self.shape.padding[self.shape.indexes[n_dims - 1]].1 = padding;

        self = self.contiguous();
        // Expand a new dimension to do the slicing on
        let n_rows = total_size / (spacing + size);
        self.shape.expand(n_dims, (spacing + size).into());
        // self = self.contiguous();
        self.shape.dims[self.shape.indexes[n_dims - 1]] = n_rows;
        self.shape.fake[self.shape.indexes[n_dims]] = false;

        // Slice
        self.shape.slices[self.shape.indexes[n_dims]].1 = spacing.into();

        self = self.contiguous();

        self.shape.remove_dim(n_dims);
        GraphTensor::from_id(self.id, self.shape, self.graph_ref)
    }

    /// Pool elements along the last dimension, pools are exposed as a new dimension
    pub fn pool_last_dim<Dst: Shape>(
        mut self,
        kernel: Expression,
        stride: Expression,
        dilation: usize,
    ) -> GraphTensor<Dst> {
        let n_dims = self.shape.len();
        let full_kernel = kernel + dilation;
        let dim_size = self.shape.dims[self.shape.indexes[n_dims - 1]];
        let number_of_windows = ((dim_size - full_kernel) / stride) + 1;
        // Expand new dimension
        self.shape.expand(n_dims - 1, number_of_windows);

        let orig_width = BigExpression::from(dim_size);

        self = self.contiguous();
        if n_dims > 1 {
            // View as single dimension of matrix with wider width
            let mat_size = (orig_width.clone() + stride) * number_of_windows;
            let actual_size = orig_width.clone() * self.shape.dims[self.shape.indexes[n_dims - 1]];
            // Reshape into single dimension to pad
            self.shape.remove_dim(n_dims);
            self.shape.dims[self.shape.indexes[n_dims - 1]] = actual_size.clone().into();
            self.shape.padding[self.shape.indexes[n_dims - 1]].1 = (mat_size - actual_size).into();
            self = self.contiguous();
            // Reshape back (mats should be full now)
            self.shape.add_dim(n_dims, (orig_width + stride).into());
        } else {
            self.shape.dims[self.shape.indexes[n_dims]] = (orig_width + stride).into();
        }
        self.shape.dims[self.shape.indexes[n_dims - 1]] = number_of_windows;
        // Slice down to kernel size
        self.shape.slices[self.shape.indexes[n_dims]].1 = full_kernel;
        self.shape.slices[self.shape.indexes[n_dims - 1]].1 = number_of_windows;
        self = self.contiguous();

        if dilation > 0 {
            // Remove dilations
            self = self.contiguous();
            self.excise(1, dilation)
        } else {
            GraphTensor::from_id(self.id, self.shape, self.graph_ref)
        }
    }

    pub fn pad<Dst: Shape, Start: Into<Expression> + Copy, End: Into<Expression> + Copy>(
        mut self,
        ranges: &[(Start, End)],
    ) -> GraphTensor<Dst> {
        let ranges = ranges
            .iter()
            .map(|i| (i.0.into(), i.1.into()))
            .collect::<Vec<_>>();
        // This exists because currently padding and slicing on the same dimension (even on opposite sides) is unsupported
        if ranges.iter().zip(self.shape.indexes).any(|(range, ind)| {
            (range.0 != 0.into() || range.1 != 0.into())
                && (self.shape.slices[self.shape.indexes[ind]].0 != 0.into()
                    || self.shape.slices[self.shape.indexes[ind]].1 != i32::MAX.into())
        }) {
            self = self.contiguous();
        }
        self.shape.pad(&ranges);
        GraphTensor::from_id(self.id, self.shape, self.graph_ref)
    }

    pub fn concat_along<Dst: Shape, Ax: Axes<Array = [usize; 1]>, Rhs: Shape>(
        self,
        rhs: GraphTensor<Rhs>,
    ) -> GraphTensor<Dst> {
        let dim = Ax::as_array()[0];
        // Create padding
        let mut a_padding = vec![(Expression::default(), Expression::default()); self.shape.len()];
        a_padding[dim].1 = rhs.shape.shape()[dim].clone().into();
        let mut b_padding = vec![(Expression::default(), Expression::default()); rhs.shape.len()];
        b_padding[dim].0 = self.shape.shape()[dim].clone().into();
        // Pad and add
        (self.pad(&a_padding) + rhs.pad(&b_padding)).sync_shape()
    }
}

#[cfg(test)]
mod tests {
    use dfdx::{
        shapes::Rank2,
        tensor::{Cpu, TensorFrom, TensorFromVec},
        tensor_ops::{RealizeTo, TryConcatAlong},
    };

    crate::test_imports!();

    #[test]
    fn test_concat_1d() {
        let mut cx = Graph::new();
        let a = cx.tensor::<R1<4>>();
        a.set(vec![1.4325, 2.492428, 3.127365, 3.54865]);
        let b = cx.tensor::<R1<3>>();
        b.set(vec![2.30434, 2.2343113, 1.4393]);
        let c = a.concat_along::<R1<7>, LAxis<0>, _>(b);
        c.retrieve();
        cx.execute();

        let d_dev = Cpu::default();
        let d_a = d_dev.tensor([1.4325, 2.492428, 3.127365, 3.54865]);
        let d_b = d_dev.tensor([2.30434, 2.2343113, 1.4393]);
        let d_c = (d_a.realize::<(usize,)>(), d_b.realize::<(usize,)>()).concat_along(DAxis::<0>);

        assert_close(&c.data(), &d_c.as_vec());
    }

    #[test]
    fn test_concat_self() {
        let mut cx = Graph::new();
        let a = cx
            .tensor::<(LConst<4>,)>()
            .set(vec![1.4325, 2.492428, 3.127365, 3.54865]);
        let b = a.concat_along::<(LConst<8>,), LAxis<0>, _>(a).retrieve();
        cx.execute();

        let d_dev = Cpu::default();
        let d_a = d_dev.tensor([1.4325, 2.492428, 3.127365, 3.54865]);
        let d_b =
            (d_a.clone().realize::<(usize,)>(), d_a.realize::<(usize,)>()).concat_along(DAxis::<0>);

        assert_close(&b.data(), &d_b.as_vec());
    }

    #[test]
    fn test_concat_2d() {
        let mut cx = Graph::new();
        let a = cx.tensor::<R2<3, 2>>();
        a.set(vec![1.4325, 2.492428, 3.127365, 33.2834, 4.18734, 23.854]);
        let b = cx.tensor::<R2<3, 2>>();
        b.set(vec![2.30434, 2.2343113, 1.4393, 482.4312, 8.1234, 54.2054]);
        let c = a.concat_along::<R2<3, 4>, LAxis<1>, _>(b);
        let d = a.concat_along::<R2<6, 2>, LAxis<0>, _>(b);
        c.retrieve();
        d.retrieve();
        cx.execute();

        let d_dev = Cpu::default();
        let d_a = d_dev.tensor_from_vec(
            vec![1.4325, 2.492428, 3.127365, 33.2834, 4.18734, 23.854],
            (dfdx::shapes::Const::<3>, dfdx::shapes::Const::<2>),
        );
        let d_b = d_dev.tensor_from_vec(
            vec![2.30434, 2.2343113, 1.4393, 482.4312, 8.1234, 54.2054],
            (dfdx::shapes::Const::<3>, dfdx::shapes::Const::<2>),
        );
        let d_c = (
            d_a.clone().realize::<(dfdx::shapes::Const<3>, usize)>(),
            d_b.clone().realize::<(dfdx::shapes::Const<3>, usize)>(),
        )
            .concat_along(dfdx::shapes::Axis::<1>);
        let d_d = (
            d_a.realize::<(usize, dfdx::shapes::Const<2>)>(),
            d_b.realize::<(usize, dfdx::shapes::Const<2>)>(),
        )
            .concat_along(dfdx::shapes::Axis::<0>);

        assert_close(&c.data(), &d_c.as_vec());
        assert_close(&d.data(), &d_d.as_vec());
    }

    #[test]
    fn test_pad_2d() {
        let mut cx = Graph::new();
        let a = cx
            .tensor::<R2<3, 2>>()
            .set(vec![1.4325, 2.492428, 3.127365, 33.2834, 4.18734, 23.854]);
        let b = a
            .pad::<R2<3, 4>, usize, usize>(&[(0, 0), (0, 2)])
            .retrieve();
        cx.execute();

        let d_dev = Cpu::default();
        let d_a = d_dev.tensor_from_vec(
            vec![1.4325, 2.492428, 3.127365, 33.2834, 4.18734, 23.854],
            (dfdx::shapes::Const::<3>, dfdx::shapes::Const::<2>),
        );
        let d_b = d_dev.tensor_from_vec(
            vec![0., 0., 0., 0., 0., 0.],
            (dfdx::shapes::Const::<3>, dfdx::shapes::Const::<2>),
        );
        let d_b = (
            d_a.realize::<(dfdx::shapes::Const<3>, usize)>(),
            d_b.realize::<(dfdx::shapes::Const<3>, usize)>(),
        )
            .concat_along(dfdx::shapes::Axis::<1>)
            .realize::<Rank2<3, 4>>();

        assert_close(&b.data(), &d_b.as_vec());
    }

    #[test]
    fn test_slice_2d() {
        let mut cx = Graph::new();
        let a = cx.tensor::<R2<3, 2>>();
        a.set(vec![1.4325, 2.492428, 3.127365, 33.2834, 4.18734, 23.854]);
        let b = a.slice((.., ..Expression::from(1))).realize::<R2<3, 1>>();
        b.retrieve();
        cx.execute();

        let d_dev = Cpu::default();
        let d_a = d_dev.tensor_from_vec(
            vec![1.4325, 2.492428, 3.127365, 33.2834, 4.18734, 23.854],
            (dfdx::shapes::Const::<3>, dfdx::shapes::Const::<2>),
        );
        let d_b = d_a.slice((.., ..1)).realize::<Rank2<3, 1>>();

        assert_close(&b.data(), &d_b.as_vec());
    }

    #[test]
    fn test_cumsum() {
        let mut cx = Graph::new();
        let a = cx.constant(1.).expand::<R1<3>, _>();
        let b = a.cumsum_last_dim().retrieve();
        let c = a
            .expand::<R2<3, 3>, LAxis<1>>()
            .permute::<_, LAxes2<1, 0>>()
            .cumsum_last_dim()
            .permute::<_, LAxes2<1, 0>>()
            .retrieve();
        cx.execute();

        assert_exact(&b.data(), &[1., 2., 3.]);
        assert_exact(&c.data(), &[1., 1., 1., 2., 2., 2., 3., 3., 3.]);
    }

    #[test]
    fn test_pool_1d() {
        let mut cx = Graph::new();

        let inp1 = cx.tensor::<R1<5>>().set(vec![1., 2., 3., 4., 5.]);
        // Stride 1
        let out1 = inp1
            .pool_last_dim::<R2<3, 3>>(3.into(), 1.into(), 0)
            .retrieve();
        // Stride 2
        let out2 = inp1
            .pool_last_dim::<R2<2, 3>>(3.into(), 2.into(), 0)
            .retrieve();
        // Stride 3
        let out3 = inp1
            .pool_last_dim::<R2<1, 3>>(3.into(), 3.into(), 0)
            .retrieve();

        cx.execute();

        assert_exact(&out1.data(), &[1., 2., 3., 2., 3., 4., 3., 4., 5.]);
        assert_exact(&out2.data(), &[1., 2., 3., 3., 4., 5.]);
        assert_exact(&out3.data(), &[1., 2., 3.]);
    }

    #[test]
    fn test_pool_1d_dims() {
        let mut cx = Graph::new();

        let inp1 = cx.tensor::<R2<4, 4>>().set(vec![
            1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16.,
        ]);
        // Stride 1
        let out1 = inp1
            .pool_last_dim::<R3<4, 2, 3>>(3.into(), 1.into(), 0)
            .retrieve();

        cx.execute();

        assert_exact(
            &out1.data(),
            &[
                1., 2., 3., 2., 3., 4., 5., 6., 7., 6., 7., 8., 9., 10., 11., 10., 11., 12., 13.,
                14., 15., 14., 15., 16.,
            ],
        );
    }

    #[test]
    fn test_pool_2d() {
        let mut cx = Graph::new();

        let inp1 = cx.tensor::<R2<4, 4>>().set(vec![
            1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16.,
        ]);
        // 3x3 kernel
        let out1 = inp1
            // Pool first dim first by moving it to end
            .permute::<_, LAxes2<1, 0>>()
            .pool_last_dim::<R3<4, 2, 3>>(3.into(), 1.into(), 0)
            // Now move other dim to end
            .permute::<_, LAxes3<1, 2, 0>>()
            .pool_last_dim::<R4<2, 3, 2, 3>>(3.into(), 1.into(), 0)
            // Now swap middle two dims
            .permute::<_, LAxes4<0, 2, 1, 3>>()
            // Now merge both pooled dimensions
            .reshape::<R3<4, 3, 3>>()
            .retrieve();

        cx.execute();

        assert_exact(
            &out1.data(),
            &[
                1.00, 2.00, 3.00, 5.00, 6.00, 7.00, 9.00, 10.00, 11.00, 2.00, 3.00, 4.00, 6.00,
                7.00, 8.00, 10.00, 11.00, 12.00, 5.00, 6.00, 7.00, 9.00, 10.00, 11.00, 13.00,
                14.00, 15.00, 6.00, 7.00, 8.00, 10.00, 11.00, 12.00, 14.00, 15.00, 16.00,
            ],
        );
    }

    #[test]
    fn test_pool_1d_dilation() {
        let mut cx = Graph::new();

        let inp1 = cx.tensor::<R1<5>>().set(vec![1., 2., 3., 4., 5.]);
        // Stride 1
        let out1 = inp1
            .pool_last_dim::<R2<3, 2>>(2.into(), 1.into(), 1)
            .retrieve();
        // Stride 2
        let out2 = inp1
            .pool_last_dim::<R2<2, 2>>(2.into(), 2.into(), 1)
            .retrieve();
        // Stride 3
        let out3 = inp1
            .pool_last_dim::<R2<1, 2>>(2.into(), 3.into(), 1)
            .retrieve();

        cx.execute();

        assert_exact(&out1.data(), &[1., 3., 2., 4., 3., 5.]);
        assert_exact(&out2.data(), &[1., 3., 3., 5.]);
        assert_exact(&out3.data(), &[1., 3.]);
    }

    #[test]
    fn test_rotate_half() {
        let mut cx = Graph::new();
        let a = cx.tensor::<R2<3, 2>>();
        a.set(vec![1.4325, 2.492428, 3.127365, 33.2834, 4.18734, 23.854]);
        let x1 = a.slice((.., ..Expression::from(1))).contiguous();
        let x2 = a.slice((.., Expression::from(1)..)).contiguous();
        let c = (-x2).concat_along::<R2<3, 2>, LAxis<1>, _>(x1);
        c.retrieve();
        cx.execute();

        let d_dev = Cpu::default();
        let d_a = d_dev.tensor_from_vec(
            vec![1.4325, 2.492428, 3.127365, 33.2834, 4.18734, 23.854],
            (dfdx::shapes::Const::<3>, dfdx::shapes::Const::<2>),
        );
        let d_x1 = d_a.clone().slice((.., ..1));
        let d_x2 = d_a.slice((.., 1..));
        let d_c = (-d_x2, d_x1)
            .concat_along(dfdx::shapes::Axis::<1>)
            .realize::<Rank2<3, 2>>();

        assert_close(&c.data(), &d_c.as_vec());
    }
}
