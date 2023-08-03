use super::*;
use std::ops::{Bound, Range, RangeBounds, RangeFrom, RangeFull, RangeTo, RangeToInclusive};

fn get_start_bound(bound: Bound<&usize>) -> usize {
    match bound {
        Bound::Included(x) => *x,
        Bound::Excluded(x) => x + 1,
        Bound::Unbounded => 0,
    }
}

fn get_end_bound(bound: Bound<&usize>, size: usize) -> usize {
    match bound {
        Bound::Excluded(x) => *x,
        Bound::Included(x) => x + 1,
        Bound::Unbounded => size,
    }
}

fn real_dim_to_size(r: RealDim) -> usize {
    if let RealDim::Const(n) = r {
        n
    } else {
        usize::MAX
    }
}

pub trait RangeToDim<D: Dim> {
    type Dim: Dim;
}

impl<D: Dim> RangeToDim<D> for RangeFrom<usize> {
    type Dim = usize;
}
impl<D: Dim> RangeToDim<D> for RangeTo<usize> {
    type Dim = usize;
}
impl<D: Dim> RangeToDim<D> for RangeToInclusive<usize> {
    type Dim = usize;
}
impl<D: Dim> RangeToDim<D> for Range<usize> {
    type Dim = usize;
}
impl<D: Dim> RangeToDim<D> for RangeFull {
    type Dim = D;
}

pub trait SliceOfShape<S: Shape> {
    type OutputShape: Shape;
    fn to_range_vec(&self) -> Vec<(usize, usize)>;
}

impl SliceOfShape<R0> for () {
    type OutputShape = R0;
    fn to_range_vec(&self) -> Vec<(usize, usize)> {
        vec![]
    }
}

impl<A: Dim, R: RangeBounds<usize> + RangeToDim<A>> SliceOfShape<(A,)> for (R,) {
    type OutputShape = (R::Dim,);
    fn to_range_vec(&self) -> Vec<(usize, usize)> {
        vec![(
            get_start_bound(self.0.start_bound()),
            get_end_bound(self.0.end_bound(), real_dim_to_size(A::const_size())),
        )]
    }
}

impl<
        A: Dim,
        B: Dim,
        R1: RangeBounds<usize> + RangeToDim<A>,
        R2: RangeBounds<usize> + RangeToDim<B>,
    > SliceOfShape<(A, B)> for (R1, R2)
{
    type OutputShape = (R1::Dim, R2::Dim);
    fn to_range_vec(&self) -> Vec<(usize, usize)> {
        vec![
            (
                get_start_bound(self.0.start_bound()),
                get_end_bound(self.0.end_bound(), real_dim_to_size(A::const_size())),
            ),
            (
                get_start_bound(self.1.start_bound()),
                get_end_bound(self.1.end_bound(), real_dim_to_size(B::const_size())),
            ),
        ]
    }
}

impl<
        A: Dim,
        B: Dim,
        C: Dim,
        R1: RangeBounds<usize> + RangeToDim<A>,
        R2: RangeBounds<usize> + RangeToDim<B>,
        R3: RangeBounds<usize> + RangeToDim<C>,
    > SliceOfShape<(A, B, C)> for (R1, R2, R3)
{
    type OutputShape = (R1::Dim, R2::Dim, R3::Dim);
    fn to_range_vec(&self) -> Vec<(usize, usize)> {
        vec![
            (
                get_start_bound(self.0.start_bound()),
                get_end_bound(self.0.end_bound(), real_dim_to_size(A::const_size())),
            ),
            (
                get_start_bound(self.1.start_bound()),
                get_end_bound(self.1.end_bound(), real_dim_to_size(B::const_size())),
            ),
            (
                get_start_bound(self.2.start_bound()),
                get_end_bound(self.2.end_bound(), real_dim_to_size(C::const_size())),
            ),
        ]
    }
}

impl<
        A: Dim,
        B: Dim,
        C: Dim,
        D: Dim,
        R1: RangeBounds<usize> + RangeToDim<A>,
        R2: RangeBounds<usize> + RangeToDim<B>,
        R3: RangeBounds<usize> + RangeToDim<C>,
        R4: RangeBounds<usize> + RangeToDim<C>,
    > SliceOfShape<(A, B, C, D)> for (R1, R2, R3, R4)
{
    type OutputShape = (R1::Dim, R2::Dim, R3::Dim, R4::Dim);
    fn to_range_vec(&self) -> Vec<(usize, usize)> {
        vec![
            (
                get_start_bound(self.0.start_bound()),
                get_end_bound(self.0.end_bound(), real_dim_to_size(A::const_size())),
            ),
            (
                get_start_bound(self.1.start_bound()),
                get_end_bound(self.1.end_bound(), real_dim_to_size(B::const_size())),
            ),
            (
                get_start_bound(self.2.start_bound()),
                get_end_bound(self.2.end_bound(), real_dim_to_size(C::const_size())),
            ),
            (
                get_start_bound(self.3.start_bound()),
                get_end_bound(self.3.end_bound(), real_dim_to_size(D::const_size())),
            ),
        ]
    }
}

impl<
        A: Dim,
        B: Dim,
        C: Dim,
        D: Dim,
        E: Dim,
        R1: RangeBounds<usize> + RangeToDim<A>,
        R2: RangeBounds<usize> + RangeToDim<B>,
        R3: RangeBounds<usize> + RangeToDim<C>,
        R4: RangeBounds<usize> + RangeToDim<C>,
        R5: RangeBounds<usize> + RangeToDim<C>,
    > SliceOfShape<(A, B, C, D, E)> for (R1, R2, R3, R4, R5)
{
    type OutputShape = (R1::Dim, R2::Dim, R3::Dim, R4::Dim, R5::Dim);
    fn to_range_vec(&self) -> Vec<(usize, usize)> {
        vec![
            (
                get_start_bound(self.0.start_bound()),
                get_end_bound(self.0.end_bound(), real_dim_to_size(A::const_size())),
            ),
            (
                get_start_bound(self.1.start_bound()),
                get_end_bound(self.1.end_bound(), real_dim_to_size(B::const_size())),
            ),
            (
                get_start_bound(self.2.start_bound()),
                get_end_bound(self.2.end_bound(), real_dim_to_size(C::const_size())),
            ),
            (
                get_start_bound(self.3.start_bound()),
                get_end_bound(self.3.end_bound(), real_dim_to_size(D::const_size())),
            ),
            (
                get_start_bound(self.4.start_bound()),
                get_end_bound(self.4.end_bound(), real_dim_to_size(E::const_size())),
            ),
        ]
    }
}
