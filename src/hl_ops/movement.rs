use itertools::Itertools;

use crate::{op, prelude::*};

impl<S: Shape> GraphTensor<S> {
    pub fn permute<Dst: Shape, Ax: Axes>(self) -> GraphTensor<Dst>
    where
        S: PermuteShapeTo<Dst, Ax>,
    {
        let graph = unsafe { self.graph_ref.as_mut().unwrap() };
        let new_id = graph
            .add_op(
                op::Permute(Ax::as_array().into_iter().map(|i| i as usize).collect_vec()),
                <Dst as Shape>::realized_shape(),
            )
            .input(self.id)
            .finish();
        GraphTensor::from_id(new_id, self.graph_ref)
    }

    pub fn expand<Dst: Shape, Ax: Axes>(self) -> GraphTensor<Dst>
    where
        S: BroadcastShapeTo<Dst, Ax>,
    {
        let graph = unsafe { self.graph_ref.as_mut().unwrap() };
        let new_shape = <Dst as Shape>::realized_shape();
        let mut shape = self.shape().clone();

        let mut new_id = self.id;
        for (dim, size) in Ax::as_array()
            .into_iter()
            .map(|i| (i as usize, new_shape[i as usize]))
        {
            shape.insert(dim, size);
            new_id = graph
                .add_op(op::Expand(dim, size), shape.clone())
                .input(new_id)
                .finish();
        }
        GraphTensor::from_id(new_id, self.graph_ref)
    }

    pub fn reshape<N: ConstShape>(self) -> GraphTensor<N> {
        // <S as AssertSameNumel<N>>::assert_same_numel();
        let graph = unsafe { self.graph_ref.as_mut().unwrap() };
        let new_id = graph
            .add_op(
                op::Reshape(
                    <N as ConstShape>::realized_shape()
                        .into_iter()
                        .map(ReshapeDim::Const)
                        .collect(),
                ),
                <N as Shape>::realized_shape(),
            )
            .input(self.id)
            .finish();
        GraphTensor::from_id(new_id, self.graph_ref)
    }

    /// Dynamically reshape. Panics if destination shape doesn't match given shape.
    pub fn dyn_reshape<N: Shape>(self, shape: Vec<ReshapeDim>) -> GraphTensor<N> {
        for (a, b) in N::realized_shape().iter().zip(shape.iter()) {
            match a {
                RealDim::Const(n) => assert_eq!(ReshapeDim::Const(*n), *b),
                RealDim::Dyn => {}
            }
        }
        let graph = unsafe { self.graph_ref.as_mut().unwrap() };
        let new_id = graph
            .add_op(
                op::Reshape(shape.clone()),
                shape
                    .iter()
                    .map(|i| match i {
                        ReshapeDim::Const(n) => RealDim::Const(*n),
                        ReshapeDim::PrevDim(_) => RealDim::Dyn,
                    })
                    .collect(),
            )
            .input(self.id)
            .finish();
        GraphTensor::from_id(new_id, self.graph_ref)
    }

    pub fn realize<Dst: Shape<Concrete = <<S as HasShape>::Shape as Shape>::Concrete>>(
        self,
    ) -> GraphTensor<Dst>
    where
        S: RealizeShapeTo<Dst>,
    {
        let GraphTensor { id, graph_ref, .. } = self;
        GraphTensor::from_id(id, graph_ref)
    }

    pub fn contiguous(self) -> GraphTensor<S> {
        let graph = unsafe { self.graph_ref.as_mut().unwrap() };
        let new_id = graph
            .add_op(op::Contiguous, S::realized_shape())
            .input(self.id)
            .finish();
        GraphTensor::from_id(new_id, self.graph_ref)
    }

    /// Take a slice of the original tensor. Any dimension with bounds becomes a dynamic dimension
    pub fn slice<Slice: SliceOfShape<S>>(self, slice: Slice) -> GraphTensor<Slice::OutputShape> {
        let slice = slice.to_range_vec();
        let graph = unsafe { self.graph_ref.as_mut().unwrap() };
        let new_id = graph
            .add_op(op::Slice(slice), Slice::OutputShape::realized_shape())
            .input(self.id)
            .finish();
        GraphTensor::from_id(new_id, self.graph_ref)
    }

    pub fn pad<Dst: Shape>(self, ranges: Vec<(i32, i32)>) -> GraphTensor<Dst> {
        let graph = unsafe { self.graph_ref.as_mut().unwrap() };
        let new_id = graph
            .add_op(
                op::Function(
                    "Pad".to_string(),
                    Box::new(move |inps, _| {
                        let (id, mut st) = (inps[0].1.tensor_id, inps[0].1.shape.clone());
                        st.pad(&ranges);
                        (
                            None,
                            TensorView {
                                tensor_id: id,
                                shape: st,
                            },
                        )
                    }),
                ),
                Dst::realized_shape(),
            )
            .input(self.id)
            .finish();
        GraphTensor::from_id(new_id, self.graph_ref)
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
        let (orig_left, orig_right) = self;
        let graph = unsafe { orig_left.graph_ref.as_mut().unwrap() };
        let dim = Ax::as_array()[0] as usize;
        let pad_a = graph
            .add_op(
                op::Function(
                    "ConcatPad".to_string(),
                    Box::new(move |inps, _| {
                        let mut pad_shape = vec![(0, 0); A::NUM_DIMS];
                        pad_shape[dim] = (0, inps[1].1.shape.shape()[dim] as i32);
                        let (id, mut st) = (inps[0].1.tensor_id, inps[0].1.shape.clone());
                        st.pad(&pad_shape);
                        (
                            None,
                            TensorView {
                                tensor_id: id,
                                shape: st,
                            },
                        )
                    }),
                ),
                <(A, B) as TryConcatAlong<Ax>>::Output::realized_shape(),
            )
            .input(orig_left.id)
            .input(orig_right.id)
            .finish();
        let left = GraphTensor::from_id(pad_a, orig_left.graph_ref);
        let pad_b = graph
            .add_op(
                op::Function(
                    "ConcatPad".to_string(),
                    Box::new(move |inps, _| {
                        let mut pad_shape = vec![(0, 0); A::NUM_DIMS];
                        pad_shape[dim] = (inps[1].1.shape.shape()[dim] as i32, 0);
                        let (id, mut st) = (inps[0].1.tensor_id, inps[0].1.shape.clone());
                        st.pad(&pad_shape);
                        (
                            None,
                            TensorView {
                                tensor_id: id,
                                shape: st,
                            },
                        )
                    }),
                ),
                <(A, B) as TryConcatAlong<Ax>>::Output::realized_shape(),
            )
            .input(orig_right.id)
            .input(orig_left.id)
            .finish();
        let right = GraphTensor::from_id(pad_b, orig_right.graph_ref);
        left + right
    }
}

macro_rules! impl_concat {
    ($Ax:expr, $NumDims:expr, [$($Head:tt),*], [$($Tail:tt),*]) => {
        impl<A: Dim, B: Dim, $($Head: Dim, )* $($Tail: Dim, )*> TryConcatAlong<Axis<$Ax>>
            for (
                ($($Head, )* A, $($Tail, )*),
                ($($Head, )* B, $($Tail, )*),
            )
        where
            A: std::ops::Add<B>,
            <A as std::ops::Add<B>>::Output: Dim,
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
        let c = (a.realize::<(usize,)>(), b.realize::<(usize,)>()).concat_along(Axis::<0>);
        c.mark();
        cx.execute();

        let d_dev = Cpu::default();
        let d_a = d_dev.tensor([1.4325, 2.492428, 3.127365, 3.54865]);
        let d_b = d_dev.tensor([2.30434, 2.2343113, 1.4393]);
        let d_c = (d_a.realize::<(usize,)>(), d_b.realize::<(usize,)>())
            .concat_along(dfdx::shapes::Axis::<0>);

        assert_close_data(
            &c.retrieve().unwrap().real_data(c.view().unwrap()).unwrap(),
            &d_c.as_vec(),
        );
    }

    #[test]
    fn test_concat_2d() {
        let mut cx = Graph::new();
        let a = cx.new_tensor::<R2<3, 2>>("Input");
        a.set(vec![1.4325, 2.492428, 3.127365, 33.2834, 4.18734, 23.854]);
        let b = cx.new_tensor::<R2<3, 2>>("Input");
        b.set(vec![2.30434, 2.2343113, 1.4393, 482.4312, 8.1234, 54.2054]);
        let c = (
            a.realize::<(Const<3>, usize)>(),
            b.realize::<(Const<3>, usize)>(),
        )
            .concat_along(Axis::<1>);
        let d = (
            a.realize::<(usize, Const<2>)>(),
            b.realize::<(usize, Const<2>)>(),
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

        assert_close_data(
            &c.retrieve().unwrap().real_data(c.view().unwrap()).unwrap(),
            &d_c.as_vec(),
        );
        assert_close_data(
            &d.retrieve().unwrap().real_data(c.view().unwrap()).unwrap(),
            &d_d.as_vec(),
        );
    }
}
