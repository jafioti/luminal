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

    // For now we assume kernel size <= stride
    pub fn pool_2d_old<Dst: Shape>(
        mut self,
        kernel_size: (usize, usize),
        stride: (usize, usize),
    ) -> GraphTensor<Dst> {
        let current_shape = self.shape.shape();
        let current_dims = current_shape.len();

        // let kernel_sizes: Vec<Expression> =
        //     kernel_size.iter().map(|&k| Expression::from(k)).collect();

        // let sliced = self.slice((.., .., .., ..));
        // self.shape.slice

        let (kx, ky) = kernel_size;
        let (sx, sy) = stride;

        let mut slice_vector: Vec<(Expression, Expression)> = Vec::new();
        for i in 0..current_dims {
            if i == current_dims - 2 {
                let end_shape = current_shape[i] + ((sx - kx) / (kx * kx));
                slice_vector.push((Expression::from(0), end_shape));
            } else if i == current_dims - 1 {
                let end_shape = current_shape[i] + ((sy - ky) / (ky * ky));
                slice_vector.push((Expression::from(0), end_shape));
            } else {
                slice_vector.push((Expression::from(0), current_shape[i]));
            }
        }

        let slices_array = slice_vector.as_slice();
        self.shape.slice(slices_array);
        let mut new_shape = self.shape.shape().clone();

        fn new_dim(dim: usize, k: usize, s: usize) -> usize {
            (dim - k) / s + 1
        }

        // Let's see if we can extract the current dim as a number
        let dimx: Expression = current_shape[current_dims - 2];
        let _dimx = dimx.to_usize().unwrap();
        // let new_dimx = _dimx / sx;
        let new_dimx = new_dim(_dimx, kx, sx);

        println!(
            "
        dimx: {_dimx}
        sx: {sx}
        new dimx: {new_dimx}
                "
        );

        let dimy: Expression = current_shape[current_dims - 1];
        let _dimy = dimy.to_usize().unwrap();
        // let new_dimy = _dimy / sy;
        let new_dimy = new_dim(_dimy, ky, sy);
        println!(
            "
        dimy: {_dimy}
        sy: {sy}
        new dimy: {new_dimy}
                "
        );

        // new_shape[current_dims - 2] = current_shape[current_dims - 2] / sx;
        // new_shape[current_dims - 1] = current_shape[current_dims - 1] / sy;
        new_shape[current_dims - 2] = Expression::from(new_dimx);
        new_shape[current_dims - 1] = Expression::from(new_dimy);

        println!("New Shape: {:?}", new_shape);

        new_shape.insert(current_dims - 1, Expression::from(sx));
        new_shape.insert(current_dims - 1 + 2, Expression::from(sy));

        println!("New Shape: {:?}", new_shape);

        // for i in 0..kernel_sizes.len() {
        //     let dim = current_dims - kernel_sizes.len() + i;
        //     new_shape[dim] = new_shape[dim] / stride[i];
        // }

        // for i in 0..kernel_sizes.len() {
        //     let insert_index = current_dims - 1 + (i * 2);
        //     new_shape.insert(insert_index, Expression::from(stride[i]));
        // }

        let mut slice_vector: Vec<(Expression, Expression)> = Vec::new();
        let new_dims = new_shape.len();
        for i in 0..new_dims {
            if i == new_dims - 3 {
                slice_vector.push((Expression::from(0), Expression::from(kx)));
            } else if i == new_dims - 1 {
                slice_vector.push((Expression::from(0), Expression::from(ky)));
            } else {
                slice_vector.push((Expression::from(0), new_shape[i].clone()));
            }
        }

        let mut reshaped = self.dyn_reshape(new_shape);
        let slices_array = slice_vector.as_slice();
        reshaped.shape.slice(slices_array);

        reshaped
    }

    pub fn pool_2d<Dst: Shape>(
        mut self,
        kernel_size: (usize, usize),
        stride: (usize, usize),
    ) -> GraphTensor<Dst> {
        let (kx, ky) = kernel_size;
        let (sx, sy) = stride;

        if sx < kx || sy < ky {
            panic!("Currently do not support case where kernel_size > stride")
        }

        // Get the tensor dims
        let current_shape: Vec<usize> = self
            .shape
            .shape()
            .iter()
            .map(|&dim: &Expression| dim.to_usize().unwrap())
            .collect_vec();

        let mut new_shape: Vec<usize> = Vec::new();
        for i in 0..current_shape.len() {
            if i < current_shape.len() - 2 {
                new_shape.push(current_shape[i]);
            } else if i == current_shape.len() - 2 {
                new_shape.push((current_shape[i] - kx) / sx + 1);
            } else if i == current_shape.len() - 1 {
                new_shape.push((current_shape[i] - ky) / sy + 1);
            }
        }

        let h = current_shape[current_shape.len() - 2];
        let w = current_shape[current_shape.len() - 1];

        let n_x = h / sx;
        let max_x = (n_x - 1) * sx + kx;
        let px = h - max_x;

        let n_y = w / sy;
        let max_y = (n_y - 1) * sy + ky;
        let py = w - max_y;

        // let max_x = (h / sx) * (sx - 1) + kx;
        // let max_y = (w / sy) * (sy - 1) + ky;

        println!("current_shape: {:?}", current_shape);
        println!("(kx, ky) = {:?}", kernel_size);
        println!("(sx, sy) = {:?}", stride);
        println!("new_shape: {:?}", new_shape);
        println!("(h, w), {:?}", (h, w));
        println!("(max_x, max_y): {:?}", (max_x, max_y));
        println!("(px, py): {:?}", (px, py));

        // Now, we slice the it

        // self.shape.slice(&[
        //     (Expression::from(0), Expression::from(new_shape[0])),
        //     (Expression::from(0), Expression::from(max_x)),
        //     (Expression::from(0), Expression::from(max_y)),
        // ]);

        // self.shape.slice(&[
        //     self.shape.slices[0],
        //     (Expression::from(0), Expression::from(max_x)),
        //     (Expression::from(0), Expression::from(max_y)),
        // ]);

        // Instead of slicing, we pad
        /*
        let a = cx.tensor::<R2<3, 2>>();
        a.set(vec![1.4325, 2.492428, 3.127365, 33.2834, 4.18734, 23.854]);
        let b = a.pad::<R2<3, 4>, usize, usize>(&[(0, 0), (0, 2)]);
        */

        self.shape.pad(&[
            self.shape.padding[0],
            (Expression::from(0), Expression::from(px)),
            (Expression::from(0), Expression::from(py)),
        ]);

        println!("padded_shape: {:?}", self.shape);

        // println!("sliced_shape: {:?}", self.shape);

        let mut new_shape_expr = self.shape.shape();

        for i in 0..new_shape_expr.len() {
            new_shape_expr[i] = Expression::from(new_shape[i]);
        }

        new_shape_expr.insert(new_shape.len() - 1, Expression::from(sx));
        new_shape_expr.insert(new_shape.len() - 1 + 2, Expression::from(sy));

        let reshaped: GraphTensor<Dst> = self.dyn_reshape(new_shape_expr);

        // Update the dims directly
        // for i in self.shape.dims {
        //     self.shape.dims[i] = new_shape[i];
        // }

        println!("reshape: {:?}", reshaped.shape);

        reshaped
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
    fn test_dummy_slice() {
        let mut cx = Graph::new();

        let mut a = cx.tensor::<R3<1, 4, 4>>();
        a.set(vec![
            12., 20., 30., 0., 8., 12., 2., 0., 34., 70., 37., 4., 112., 100., 25., 12.,
        ]);

        // I basically want to slice the top part
        a.shape.slice(&[
            (Expression::from(0), Expression::from(1)),
            (Expression::from(0), Expression::from(2)),
            (Expression::from(0), Expression::from(4)),
        ]);

        let b: GraphTensor<_> = a.realize::<R3<1, 2, 4>>();

        b.retrieve();

        cx.execute();

        println!("{:?}", &b.data());
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
            // .pool::<R5<5, 1, 2, 2, 2>>(&[2, 2], &[2, 2])
            .pool_2d::<R5<1, 2, 2, 2, 2>>((2, 2), (2, 2))
            .max_reduce::<_, LAxes2<2, 4>>();
        out1.retrieve();

        // Case 2
        let inp2 = cx.tensor::<R3<1, 4, 4>>();
        inp2.set(vec![
            12., 20., 30., 0., 8., 12., 2., 0., 34., 70., 37., 4., 112., 100., 25., 12.,
        ]);

        let out2 = inp2
            // .pool::<R5<1, 2, 2, 2, 2>>(&[2, 2], &[2, 2])
            .pool_2d::<R5<1, 2, 2, 2, 2>>((2, 2), (2, 2))
            .max_reduce::<_, LAxes2<2, 4>>();
        out2.retrieve();

        // Run all the computations
        cx.execute();

        // Run all the assertions
        assert_close(
            &out1.data(),
            &[87., 48., 54., 99., 94., 85., 98., 96., 82., 70.],
        );
        assert_close(&out2.data(), &[20., 30., 112., 37.]);
    }

    #[test]
    fn test_pool_non_square() {
        let mut cx = Graph::new();
        // Case 3
        let inp3 = cx.tensor::<R3<1, 4, 4>>();
        inp3.set(vec![
            12., 20., 30., 0., 8., 12., 2., 0., 34., 70., 37., 4., 112., 100., 25., 12.,
        ]);

        let out3 = inp3
            // .pool::<R5<1, 2, 2, 2, 2>>(&[2, 2], &[2, 3])
            .pool_2d::<R5<1, 2, 1, 1, 1>>((2, 2), (2, 3))
            .max_reduce::<_, LAxes2<2, 4>>();
        out3.retrieve();

        // Run all the computations
        cx.execute();

        println!(
            "
Expected Shape: {:?}
Actual Shape: {:?}",
            &[1, 2, 1],
            out3.shape.dims
        );
        println!(
            "
Expected Tensor: {:?}
Actual Tensor: {:?}",
            &[20., 112.],
            &out3.data()
        );

        println!(
            "{:?}",
            [12., 20., 30., 0., 8., 12., 2., 0., 34., 70., 37., 4., 112., 100., 25., 12.,]
        );

        assert_close(&out3.data(), &[20., 112.]);
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
