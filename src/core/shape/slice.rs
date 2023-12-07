use super::*;
use std::ops::{Bound, Range, RangeBounds, RangeFrom, RangeFull, RangeTo, RangeToInclusive};

fn get_start_bound<D: ExprInterface + Copy>(bound: Bound<D>) -> Expression {
    match bound {
        Bound::Included(x) => x.expr(),
        Bound::Excluded(x) => x.expr() + 1.expr(),
        Bound::Unbounded => 0.expr(),
    }
}

fn get_end_bound<D: ExprInterface + Copy, S: ExprInterface>(
    bound: Bound<D>,
    size: S,
) -> Expression {
    match bound {
        Bound::Excluded(x) => x.expr(),
        Bound::Included(x) => x.expr() + 1.expr(),
        Bound::Unbounded => size.expr(),
    }
}

fn dim_to_size(r: Expression) -> usize {
    r.to_usize().unwrap_or(usize::MAX)
}

pub trait RangeToDim<D: Dimension> {
    type Dimension: Dimension;
}

impl<D: Dimension> RangeToDim<D> for RangeFrom<usize> {
    type Dimension = Dyn<'-'>;
}
impl<D: Dimension> RangeToDim<D> for RangeTo<usize> {
    type Dimension = Dyn<'-'>;
}
impl<D: Dimension> RangeToDim<D> for RangeToInclusive<usize> {
    type Dimension = Dyn<'-'>;
}
impl<D: Dimension> RangeToDim<D> for Range<usize> {
    type Dimension = Dyn<'-'>;
}
impl<D: Dimension> RangeToDim<D> for RangeFrom<Expression> {
    type Dimension = Dyn<'-'>;
}
impl<D: Dimension> RangeToDim<D> for RangeTo<Expression> {
    type Dimension = Dyn<'-'>;
}
impl<D: Dimension> RangeToDim<D> for RangeToInclusive<Expression> {
    type Dimension = Dyn<'-'>;
}
impl<D: Dimension> RangeToDim<D> for Range<Expression> {
    type Dimension = Dyn<'-'>;
}
impl<D: Dimension> RangeToDim<D> for RangeFull {
    type Dimension = D;
}

pub trait SliceOfShape<S: Shape> {
    type OutputShape: Shape;
    fn to_range_vec(&self) -> Vec<(Expression, Expression)>;
}

impl SliceOfShape<R0> for () {
    type OutputShape = R0;
    fn to_range_vec(&self) -> Vec<(Expression, Expression)> {
        vec![]
    }
}

impl<A: Dimension, R: RangeBounds<usize> + RangeToDim<A>> SliceOfShape<(A,)> for (R,) {
    type OutputShape = (R::Dimension,);
    fn to_range_vec(&self) -> Vec<(Expression, Expression)> {
        vec![(
            get_start_bound(self.0.start_bound()),
            get_end_bound(self.0.end_bound(), dim_to_size(A::const_size())),
        )]
    }
}

impl<
        A: Dimension,
        B: Dimension,
        R1: RangeBounds<usize> + RangeToDim<A>,
        R2: RangeBounds<usize> + RangeToDim<B>,
    > SliceOfShape<(A, B)> for (R1, R2)
{
    type OutputShape = (R1::Dimension, R2::Dimension);
    fn to_range_vec(&self) -> Vec<(Expression, Expression)> {
        vec![
            (
                get_start_bound(self.0.start_bound()),
                get_end_bound(self.0.end_bound(), dim_to_size(A::const_size())),
            ),
            (
                get_start_bound(self.1.start_bound()),
                get_end_bound(self.1.end_bound(), dim_to_size(B::const_size())),
            ),
        ]
    }
}

impl<
        A: Dimension,
        B: Dimension,
        C: Dimension,
        R1: RangeBounds<usize> + RangeToDim<A>,
        R2: RangeBounds<usize> + RangeToDim<B>,
        R3: RangeBounds<usize> + RangeToDim<C>,
    > SliceOfShape<(A, B, C)> for (R1, R2, R3)
{
    type OutputShape = (R1::Dimension, R2::Dimension, R3::Dimension);
    fn to_range_vec(&self) -> Vec<(Expression, Expression)> {
        vec![
            (
                get_start_bound(self.0.start_bound()),
                get_end_bound(self.0.end_bound(), dim_to_size(A::const_size())),
            ),
            (
                get_start_bound(self.1.start_bound()),
                get_end_bound(self.1.end_bound(), dim_to_size(B::const_size())),
            ),
            (
                get_start_bound(self.2.start_bound()),
                get_end_bound(self.2.end_bound(), dim_to_size(C::const_size())),
            ),
        ]
    }
}

impl<
        A: Dimension,
        B: Dimension,
        C: Dimension,
        D: Dimension,
        R1: RangeBounds<usize> + RangeToDim<A>,
        R2: RangeBounds<usize> + RangeToDim<B>,
        R3: RangeBounds<usize> + RangeToDim<C>,
        R4: RangeBounds<usize> + RangeToDim<C>,
    > SliceOfShape<(A, B, C, D)> for (R1, R2, R3, R4)
{
    type OutputShape = (R1::Dimension, R2::Dimension, R3::Dimension, R4::Dimension);
    fn to_range_vec(&self) -> Vec<(Expression, Expression)> {
        vec![
            (
                get_start_bound(self.0.start_bound()),
                get_end_bound(self.0.end_bound(), dim_to_size(A::const_size())),
            ),
            (
                get_start_bound(self.1.start_bound()),
                get_end_bound(self.1.end_bound(), dim_to_size(B::const_size())),
            ),
            (
                get_start_bound(self.2.start_bound()),
                get_end_bound(self.2.end_bound(), dim_to_size(C::const_size())),
            ),
            (
                get_start_bound(self.3.start_bound()),
                get_end_bound(self.3.end_bound(), dim_to_size(D::const_size())),
            ),
        ]
    }
}

impl<
        A: Dimension,
        B: Dimension,
        C: Dimension,
        D: Dimension,
        E: Dimension,
        R1: RangeBounds<usize> + RangeToDim<A>,
        R2: RangeBounds<usize> + RangeToDim<B>,
        R3: RangeBounds<usize> + RangeToDim<C>,
        R4: RangeBounds<usize> + RangeToDim<C>,
        R5: RangeBounds<usize> + RangeToDim<C>,
    > SliceOfShape<(A, B, C, D, E)> for (R1, R2, R3, R4, R5)
{
    type OutputShape = (
        R1::Dimension,
        R2::Dimension,
        R3::Dimension,
        R4::Dimension,
        R5::Dimension,
    );
    fn to_range_vec(&self) -> Vec<(Expression, Expression)> {
        vec![
            (
                get_start_bound(self.0.start_bound()),
                get_end_bound(self.0.end_bound(), dim_to_size(A::const_size())),
            ),
            (
                get_start_bound(self.1.start_bound()),
                get_end_bound(self.1.end_bound(), dim_to_size(B::const_size())),
            ),
            (
                get_start_bound(self.2.start_bound()),
                get_end_bound(self.2.end_bound(), dim_to_size(C::const_size())),
            ),
            (
                get_start_bound(self.3.start_bound()),
                get_end_bound(self.3.end_bound(), dim_to_size(D::const_size())),
            ),
            (
                get_start_bound(self.4.start_bound()),
                get_end_bound(self.4.end_bound(), dim_to_size(E::const_size())),
            ),
        ]
    }
}
