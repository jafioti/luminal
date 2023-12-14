use crate::{
    op,
    prelude::{symbolic::Expression, *},
};

impl<S: Shape> GraphTensor<S> {
    pub fn permute<Dst: Shape, Ax: Axes>(mut self) -> GraphTensor<Dst>
    where
        S: PermuteShapeTo<Dst, Ax>,
    {
        self.shape.permute(
            &Ax::as_array()
                .into_iter()
                .map(|i| i as usize)
                .collect::<Vec<_>>(),
        );
        GraphTensor::from_id(self.id, self.shape, self.graph_ref)
    }

    pub fn expand<Dst: Shape, Ax: Axes>(mut self) -> GraphTensor<Dst>
    where
        S: BroadcastShapeTo<Dst, Ax>,
    {
        let new_dims = Dst::realized_shape();
        if !new_dims.is_empty() {
            for (i, dim) in Ax::as_array()
                .into_iter()
                .map(|i| (i as usize, new_dims[i as usize]))
            {
                self.shape.expand(i, dim);
            }
        }

        GraphTensor::from_id(self.id, self.shape, self.graph_ref)
    }

    pub fn reshape<N: Shape>(self) -> GraphTensor<N> {
        let id = if !self.shape.is_contiguous() {
            // Insert contiguous call
            self.graph()
                .add_op(op::Contiguous)
                .input(self.id, 0, self.shape)
                .finish()
        } else {
            // Already contiguous
            self.id
        };

        GraphTensor::from_id(id, ShapeTracker::new(&N::realized_shape()), self.graph_ref)
    }

    /// Dynamically reshape with annotations for the shape tracker
    pub fn dyn_reshape<N: Shape>(self, shape: Vec<Expression>) -> GraphTensor<N> {
        let id = if !self.shape.is_contiguous() {
            // Insert contiguous call
            self.graph()
                .add_op(op::Contiguous)
                .input(self.id, 0, self.shape)
                .finish()
        } else {
            // Already contiguous
            self.id
        };

        GraphTensor::from_id(id, ShapeTracker::new(&shape), self.graph_ref)
    }

    pub fn realize<Dst: Shape<Concrete = <<S as HasShape>::Shape as Shape>::Concrete>>(
        self,
    ) -> GraphTensor<Dst>
    where
        S: RealizeShapeTo<Dst>,
    {
        GraphTensor::from_id(self.id, self.shape, self.graph_ref)
    }

    pub fn contiguous(self) -> GraphTensor<S> {
        if self.shape.is_contiguous() && !self.shape.is_sliced() && !self.shape.is_padded() {
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
        self.shape.slice(&slice.to_range_vec());
        GraphTensor::from_id(self.id, self.shape, self.graph_ref)
    }

    /*
    def _pool2d(self, py, px, sy, sx):
        if py > sy or px > sx:
            raise NotImplementedError(
                "pool2d doesn't support kernel_size > stride"
            )

        xup = self.slice(
            (
                (0, self.shape[0]),
                (0, self.shape[1]),
                (0, (self.shape[2]+(sy-py))//sy*sy),
                (0, (self.shape[3]+(sx-px))//sx*sx)
            )
        )

        return xup.reshape(
            shape=(
                xup.shape[0],
                xup.shape[1],
                xup.shape[2]//sy,
                sy,
                xup.shape[3]//sx,
                sx
            )
        )[:, :, :, :py, :, :px]

        // dims: (N, C, H, W)
        // after transpose: (N, C, W, H)
        // indexes: [0, 1, 3, 2]

        // (N, C, H, W)
        // 2x2 pool
        // (N, C, H / 2, W / 2, 2, 2)

        // (N, C, H, W)
        // 2 pool
        // (N, C, H, W / 2, 2)
         */

    // TODO: Pooling goes here
    // For now we assume kernel size = stride
    // TODO: We need to fix this
    pub fn pool<Dst: Shape>(self, kernel_size: &[usize], stride: &[usize]) -> GraphTensor<Dst> {
        let current_shape = self.shape.shape();
        let current_dims = current_shape.len();

        let mut new_shape = current_shape.clone();

        let kernel_sizes: Vec<Expression> =
            kernel_size.iter().map(|&k| Expression::from(k)).collect();

        for i in 0..kernel_sizes.len() {
            let dim = current_dims - kernel_sizes.len() + i;
            new_shape[dim] = new_shape[dim] / kernel_sizes[i];
        }

        for i in 0..kernel_sizes.len() {
            let insert_index = current_dims - 1 + (i * 2);
            new_shape.insert(insert_index, kernel_sizes[i]);
        }

        return self.dyn_reshape(new_shape);
    }

    pub fn pad<Dst: Shape, Start: Into<Expression> + Copy, End: Into<Expression> + Copy>(
        mut self,
        ranges: &[(Start, End)],
    ) -> GraphTensor<Dst> {
        self.shape.pad(
            &ranges
                .iter()
                .map(|i| (i.0.into(), i.1.into()))
                .collect::<Vec<_>>(),
        );
        GraphTensor::from_id(self.id, self.shape, self.graph_ref)
    }

    pub fn concat_along<Dst: Shape, Ax: Axes<Array = [isize; 1]>, Rhs: Shape>(
        self,
        rhs: GraphTensor<Rhs>,
    ) -> GraphTensor<Dst> {
        let dim = Ax::as_array()[0] as usize;
        let mut a_padding = self.shape.padding;
        a_padding[dim].1 = rhs.shape.shape()[dim];
        let mut b_padding = rhs.shape.padding;
        b_padding[dim].0 = self.shape.shape()[dim];
        let lhs = self.pad(&a_padding);
        lhs + rhs.pad(&b_padding)
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
        let a = cx.tensor::<R2<3, 2>>();
        a.set(vec![1.4325, 2.492428, 3.127365, 33.2834, 4.18734, 23.854]);
        let b = a.pad::<R2<3, 4>, usize, usize>(&[(0, 0), (0, 2)]);
        b.retrieve();
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
        let b: GraphTensor<(LConst<3>, LConst<1>)> = a.slice((.., ..1)).realize::<R2<3, 1>>();
        b.retrieve();
        cx.execute();

        let d_dev = Cpu::default();
        let d_a = d_dev.tensor_from_vec(
            vec![1.4325, 2.492428, 3.127365, 33.2834, 4.18734, 23.854],
            (dfdx::shapes::Const::<3>, dfdx::shapes::Const::<2>),
        );
        let d_b = d_a.slice((.., ..1)).realize::<Rank2<3, 1>>();

        println!("{:?}", &b.data());

        assert_close(&b.data(), &d_b.as_vec());
    }

    #[test]
    fn test_pool() {
        let mut cx = Graph::new();
        let a = cx.tensor::<R2<4, 4>>();
        a.set(vec![
            12.0, 20.0, 30.0, 0.0, // row 1
            8.0, 12.0, 2.0, 0.0, // row 2
            34.0, 70.0, 37.0, 4.0, // row 3
            112.0, 100.0, 25.0, 12.0, // row 4
        ]);
        a.retrieve();

        let b = a
            .pool::<R4<2, 2, 2, 2>>(&[2, 2], &[2, 2])
            .max_reduce::<_, LAxes2<2, 3>>();
        b.retrieve();
        cx.execute();

        assert_close(&b.data(), &[30.0, 12.0, 70.0, 112.0]);
    }

    #[test]
    fn test_rotate_half() {
        let mut cx = Graph::new();
        let a = cx.tensor::<R2<3, 2>>();
        a.set(vec![1.4325, 2.492428, 3.127365, 33.2834, 4.18734, 23.854]);
        let x1 = a.slice((.., ..1)).contiguous();
        let x2 = a.slice((.., 1..)).contiguous();
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
