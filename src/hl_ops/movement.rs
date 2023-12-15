use itertools::Itertools;

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

    // For now we assume kernel size <= stride and stride shape is multiple of stride
    pub fn pool<Dst: Shape>(self, kernel_size: &[usize], stride: &[usize]) -> GraphTensor<Dst> {
        let current_shape = self.shape.shape();
        let current_dims = current_shape.len();

        let mut new_shape = current_shape.clone();

        let kernel_sizes: Vec<Expression> =
            kernel_size.iter().map(|&k| Expression::from(k)).collect();

        let strides: Vec<Expression> = stride.iter().map(|&k| Expression::from(k)).collect();

        for i in 0..kernel_sizes.len() {
            let dim = current_dims - kernel_sizes.len() + i;
            new_shape[dim] = (new_shape[dim] - kernel_sizes[i]) / strides[i] + 1;
        }

        for i in 0..kernel_sizes.len() {
            let insert_index = current_dims - 1 + (i * 2);
            new_shape.insert(insert_index, strides[i]);
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

    use crate::prelude::symbolic::Expression;

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
        let b = a.slice((.., ..1)).realize::<R2<3, 1>>();
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
    fn test_pool() {
        // Max Pool Example
        let mut cx = Graph::new();

        // Case 1
        let inp1 = cx.tensor::<R3<5, 2, 4>>();
        inp1.set(vec![
            84., 66., 48., 35., 87., 22., 31., 37., 33., 8., 22., 67., 54., 5., 99., 54., 19., 94.,
            85., 77., 35., 22., 11., 10., 50., 67., 53., 17., 53., 98., 96., 30., 0., 82., 21.,
            30., 35., 79., 70., 22.,
        ]);

        let out1 = inp1
            .pool::<R5<1, 2, 2, 2, 2>>(&[2, 2], &[2, 2])
            .max_reduce::<_, LAxes2<2, 4>>();
        out1.retrieve();

        // Case 2
        let inp2 = cx.tensor::<R3<1, 4, 4>>();
        inp2.set(vec![
            12., 20., 30., 0., 8., 12., 2., 0., 34., 70., 37., 4., 112., 100., 25., 12.,
        ]);

        let out2 = inp2
            .pool::<R5<1, 2, 2, 2, 2>>(&[2, 2], &[2, 2])
            .max_reduce::<_, LAxes2<2, 4>>();
        out2.retrieve();

        // Case 3
        let inp3 = cx.tensor::<R3<1, 8, 6>>();
        inp3.set(vec![
            44., 82., 83., 76., 30., 48., 0., 72., 48., 94., 57., 40., 62., 93., 77., 34., 97.,
            31., 10., 23., 64., 77., 75., 65., 75., 1., 11., 7., 27., 25., 56., 42., 74., 97., 0.,
            43., 61., 20., 40., 59., 83., 80., 53., 59., 43., 47., 84., 9.,
        ]);
        let out3 = inp3
            .pool::<R5<1, 2, 4, 3, 2>>(&[2, 2], &[4, 2])
            .max_reduce::<_, LAxes2<2, 4>>();
        out3.retrieve();

        // Case 4
        let inp4 = cx.tensor::<R3<1, 8, 8>>();
        inp4.set(vec![
            43., 49., 18., 34., 93., 64., 10., 50., 23., 83., 82., 62., 53., 25., 0., 11., 57.,
            17., 76., 96., 64., 96., 43., 63., 61., 41., 52., 68., 1., 28., 51., 25., 35., 95., 2.,
            55., 75., 14., 48., 52., 59., 52., 50., 10., 36., 36., 43., 80., 0., 84., 95., 97.,
            55., 26., 34., 43., 66., 78., 30., 89., 37., 63., 55., 89.,
        ]);
        let out4 = inp4
            .pool::<R5<1, 2, 4, 4, 2>>(&[2, 2], &[4, 2])
            .max_reduce::<_, LAxes2<2, 4>>();
        out4.retrieve();

        // Run all the computations
        cx.execute();

        // println!("out3: {:?}", &out3.data());
        // println!("out3 shape: {:?}", out3.shape.shape());
        println!("out4: {:?}", &out4.data());
        println!("out4 shape: {:?}", out4.shape.shape());

        // Run all the assertions
        assert_close(
            &out1.data(),
            &[87., 48., 54., 99., 94., 85., 98., 96., 82., 70.],
        );
        assert_close(&out2.data(), &[20., 30., 112., 37.]);
        // assert_close(&out3.data(), &[82., 94., 57., 75., 97., 43.]);
        assert_close(&out4.data(), &[83., 93., 95., 75.])
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
