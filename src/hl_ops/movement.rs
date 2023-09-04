use crate::{op, prelude::*};

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
        for (i, dim) in Ax::as_array()
            .into_iter()
            .map(|i| (i as usize, new_dims[i as usize]))
        {
            self.shape.expand(i, dim);
        }

        GraphTensor::from_id(self.id, self.shape, self.graph_ref)
    }

    pub fn reshape<N: Shape>(self) -> GraphTensor<N> {
        let id = if !self.shape.is_contiguous() {
            // Insert contiguous call
            self.graph()
                .add_op(op::Contiguous)
                .input(self.id, self.shape)
                .finish()
        } else {
            // Already contiguous
            self.id
        };

        GraphTensor::from_id(id, ShapeTracker::new(&N::realized_shape()), self.graph_ref)
    }

    /// Dynamically reshape with annotations for the shape tracker
    pub fn dyn_reshape<N: Shape>(self, shape: Vec<Dim>) -> GraphTensor<N> {
        let id = if !self.shape.is_contiguous() {
            // Insert contiguous call
            self.graph()
                .add_op(op::Contiguous)
                .input(self.id, self.shape)
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
        GraphTensor::from_id(
            self.id,
            self.shape.realize(&Dst::realized_shape()),
            self.graph_ref,
        )
    }

    pub fn contiguous(self) -> GraphTensor<S> {
        let new_id = self
            .graph()
            .add_op(op::Contiguous)
            .input(self.id, self.shape)
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

    pub fn pad<Dst: Shape>(mut self, ranges: Vec<(usize, usize)>) -> GraphTensor<Dst> {
        self.shape.pad(&ranges);
        GraphTensor::from_id(self.id, self.shape, self.graph_ref)
    }
}

pub trait TryConcatAlong<Ax>: Sized {
    type Output;
    fn concat_along(self, ax: Ax) -> Self::Output;
}

impl<A, B, Ax> TryConcatAlong<Ax> for (GraphTensor<A>, GraphTensor<B>)
where
    Ax: Axes<Array = [isize; 1]>,
    A: Shape + HasAxes<Ax>,
    B: Shape<Concrete = A::Concrete> + HasAxes<Ax>,
    (A, B): TryConcatAlong<Ax>,
    <(A, B) as TryConcatAlong<Ax>>::Output: Shape,
{
    type Output = GraphTensor<<(A, B) as TryConcatAlong<Ax>>::Output>;
    fn concat_along(self, _: Ax) -> Self::Output {
        let dim = Ax::as_array()[0] as usize;
        let fin = self
            .0
            .graph()
            .add_op(op::Function(
                "Concat".to_string(),
                Box::new(move |mut inps| {
                    let mut pad_shape = vec![(0, 0); A::NUM_DIMS];
                    pad_shape[dim].1 = inps[1].1.shape()[dim]
                        .to_usize()
                        .expect("Tried to concat on a dim with an unknown size");
                    inps[0].1.pad(&pad_shape);
                    let mut pad_shape = vec![(0, 0); A::NUM_DIMS];
                    pad_shape[dim].1 = inps[0].1.shape()[dim]
                        .to_usize()
                        .expect("Tried to concat on a dim with an unknown size");
                    inps[1].1.pad(&pad_shape);
                    let (a_data, b_data) = (
                        inps[0]
                            .0
                            .borrowed()
                            .data
                            .as_any()
                            .downcast_ref::<Vec<f32>>()
                            .unwrap(),
                        inps[1]
                            .0
                            .borrowed()
                            .data
                            .as_any()
                            .downcast_ref::<Vec<f32>>()
                            .unwrap(),
                    );
                    let mut data = vec![0.; inps[0].1.n_elements()];
                    #[allow(clippy::needless_range_loop)]
                    for i in 0..data.len() {
                        data[i] = inps[0].1.index(i).map(|i| a_data[i]).unwrap_or_default()
                            + inps[1].1.index(i).map(|i| b_data[i]).unwrap_or_default();
                    }
                    Tensor {
                        data: Box::new(data),
                    }
                }),
            ))
            .input(self.0.id, self.0.shape)
            .input(self.1.id, self.1.shape)
            .finish();
        GraphTensor::from_id(
            fin,
            ShapeTracker::new(&<(A, B) as TryConcatAlong<Ax>>::Output::realized_shape()),
            self.0.graph_ref,
        )
    }
}

macro_rules! impl_concat {
    ($Ax:expr, $NumDims:expr, [$($Head:tt),*], [$($Tail:tt),*]) => {
        impl<A: Dimension, B: Dimension, $($Head: Dimension, )* $($Tail: Dimension, )*> TryConcatAlong<Axis<$Ax>>
            for (
                ($($Head, )* A, $($Tail, )*),
                ($($Head, )* B, $($Tail, )*),
            )
        where
            A: std::ops::Add<B>,
            <A as std::ops::Add<B>>::Output: Dimension,
            {
                type Output = (
                    $($Head, )*
                    <A as std::ops::Add<B>>::Output,
                    $($Tail, )*
                );
                fn concat_along(self, _: Axis<$Ax>) -> Self::Output {
                    let (lhs, rhs) = self;
                    let lhs_dims = lhs.concrete();
                    let rhs_dims = rhs.concrete();
                    for i in 0..$NumDims {
                        if i != $Ax {
                            assert_eq!(lhs_dims[i], rhs_dims[i]);
                        }
                    }
                    let mut out_dims = lhs_dims;
                    out_dims[$Ax] += rhs_dims[$Ax];
                    Self::Output::from_concrete(&out_dims).unwrap()
                }
            }
    };
}

impl_concat!(0, 1, [], []);
impl_concat!(0, 2, [], [D1]);
impl_concat!(0, 3, [], [D1, D2]);
impl_concat!(0, 4, [], [D1, D2, D3]);
impl_concat!(0, 5, [], [D1, D2, D3, D4]);
impl_concat!(0, 6, [], [D1, D2, D3, D4, D5]);

impl_concat!(1, 2, [D0], []);
impl_concat!(1, 3, [D0], [D2]);
impl_concat!(1, 4, [D0], [D2, D3]);
impl_concat!(1, 5, [D0], [D2, D3, D4]);
impl_concat!(1, 6, [D0], [D2, D3, D4, D5]);

impl_concat!(2, 3, [D0, D1], []);
impl_concat!(2, 4, [D0, D1], [D3]);
impl_concat!(2, 5, [D0, D1], [D3, D4]);
impl_concat!(2, 6, [D0, D1], [D3, D4, D5]);

impl_concat!(3, 4, [D0, D1, D2], []);
impl_concat!(3, 5, [D0, D1, D2], [D4]);
impl_concat!(3, 6, [D0, D1, D2], [D4, D5]);

impl_concat!(4, 5, [D0, D1, D2, D3], []);
impl_concat!(4, 6, [D0, D1, D2, D3], [D5]);

impl_concat!(5, 6, [D0, D1, D2, D3, D4], []);

#[cfg(test)]
mod tests {
    use dfdx::{
        tensor::{Cpu, TensorFrom, TensorFromVec},
        tensor_ops::{RealizeTo, TryConcatAlong as DfdxTryConcatAlong},
    };

    use crate::{prelude::*, tests::assert_close_data};

    use super::TryConcatAlong;

    #[test]
    fn test_concat_1d() {
        let mut cx = Graph::new();
        let a = cx.new_tensor::<R1<4>>("Input");
        a.set(vec![1.4325, 2.492428, 3.127365, 3.54865]);
        let b = cx.new_tensor::<R1<3>>("Input");
        b.set(vec![2.30434, 2.2343113, 1.4393]);
        let c = (a.realize::<(Dyn<'-'>,)>(), b.realize::<(Dyn<'-'>,)>()).concat_along(Axis::<0>);
        c.mark();
        cx.execute();

        let d_dev = Cpu::default();
        let d_a = d_dev.tensor([1.4325, 2.492428, 3.127365, 3.54865]);
        let d_b = d_dev.tensor([2.30434, 2.2343113, 1.4393]);
        let d_c = (d_a.realize::<(usize,)>(), d_b.realize::<(usize,)>())
            .concat_along(dfdx::shapes::Axis::<0>);

        assert_close_data(&c.data(), &d_c.as_vec());
    }

    #[test]
    fn test_concat_2d() {
        let mut cx = Graph::new();
        let a = cx.new_tensor::<R2<3, 2>>("Input");
        a.set(vec![1.4325, 2.492428, 3.127365, 33.2834, 4.18734, 23.854]);
        let b = cx.new_tensor::<R2<3, 2>>("Input");
        b.set(vec![2.30434, 2.2343113, 1.4393, 482.4312, 8.1234, 54.2054]);
        let c = (
            a.realize::<(Const<3>, Dyn<'-'>)>(),
            b.realize::<(Const<3>, Dyn<'-'>)>(),
        )
            .concat_along(Axis::<1>);
        let d = (
            a.realize::<(Dyn<'-'>, Const<2>)>(),
            b.realize::<(Dyn<'-'>, Const<2>)>(),
        )
            .concat_along(Axis::<0>);
        c.mark();
        d.mark();
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

        assert_close_data(&c.data(), &d_c.as_vec());
        assert_close_data(&d.data(), &d_d.as_vec());
    }
}
