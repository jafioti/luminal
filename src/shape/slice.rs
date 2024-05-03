use crate::prelude::*;
use std::ops::{Bound, Range, RangeBounds, RangeFrom, RangeFull, RangeTo, RangeToInclusive};

fn get_start_bound<D: Into<Expression> + Copy>(bound: Bound<D>) -> Expression {
    match bound {
        Bound::Included(x) => x.into(),
        Bound::Excluded(x) => x.into() + Expression::from(1),
        Bound::Unbounded => 0.into(),
    }
}

fn get_end_bound<D: Into<Expression> + Copy, S: Into<Expression>>(
    bound: Bound<D>,
    size: S,
) -> Expression {
    match bound {
        Bound::Excluded(x) => x.into(),
        Bound::Included(x) => x.into() + Expression::from(1),
        Bound::Unbounded => size.into(),
    }
}

pub trait RangeToDim<D: Dimension> {
    type Dimension: Dimension;
    fn bounds(&self, size: impl Into<Expression>) -> (Expression, Expression);
}

impl<D: Dimension> RangeToDim<D> for RangeFrom<usize> {
    type Dimension = Dyn<'-'>;
    fn bounds(&self, size: impl Into<Expression>) -> (Expression, Expression) {
        (
            get_start_bound(self.start_bound()),
            get_end_bound(self.end_bound(), size),
        )
    }
}
impl<D: Dimension> RangeToDim<D> for RangeTo<usize> {
    type Dimension = Dyn<'-'>;
    fn bounds(&self, size: impl Into<Expression>) -> (Expression, Expression) {
        (
            get_start_bound(self.start_bound()),
            get_end_bound(self.end_bound(), size),
        )
    }
}
impl<D: Dimension> RangeToDim<D> for RangeToInclusive<usize> {
    type Dimension = Dyn<'-'>;
    fn bounds(&self, size: impl Into<Expression>) -> (Expression, Expression) {
        (
            get_start_bound(self.start_bound()),
            get_end_bound(self.end_bound(), size),
        )
    }
}
impl<D: Dimension> RangeToDim<D> for Range<usize> {
    type Dimension = Dyn<'-'>;
    fn bounds(&self, size: impl Into<Expression>) -> (Expression, Expression) {
        (
            get_start_bound(self.start_bound()),
            get_end_bound(self.end_bound(), size),
        )
    }
}
impl<D: Dimension> RangeToDim<D> for RangeFrom<Expression> {
    type Dimension = Dyn<'-'>;
    fn bounds(&self, size: impl Into<Expression>) -> (Expression, Expression) {
        (
            get_start_bound(self.start_bound()),
            get_end_bound(self.end_bound(), size),
        )
    }
}
impl<D: Dimension> RangeToDim<D> for RangeTo<Expression> {
    type Dimension = Dyn<'-'>;
    fn bounds(&self, size: impl Into<Expression>) -> (Expression, Expression) {
        (
            get_start_bound(self.start_bound()),
            get_end_bound(self.end_bound(), size),
        )
    }
}
impl<D: Dimension> RangeToDim<D> for RangeToInclusive<Expression> {
    type Dimension = Dyn<'-'>;
    fn bounds(&self, size: impl Into<Expression>) -> (Expression, Expression) {
        (
            get_start_bound(self.start_bound()),
            get_end_bound(self.end_bound(), size),
        )
    }
}
impl<D: Dimension> RangeToDim<D> for Range<Expression> {
    type Dimension = Dyn<'-'>;
    fn bounds(&self, size: impl Into<Expression>) -> (Expression, Expression) {
        (
            get_start_bound(self.start_bound()),
            get_end_bound(self.end_bound(), size),
        )
    }
}
impl<D: Dimension> RangeToDim<D> for RangeFull {
    type Dimension = D;
    fn bounds(&self, size: impl Into<Expression>) -> (Expression, Expression) {
        (0.into(), size.into())
    }
}
impl<D: Dimension, R: RangeToDim<D>> RangeToDim<D> for (R,) {
    type Dimension = R::Dimension;
    fn bounds(&self, size: impl Into<Expression>) -> (Expression, Expression) {
        self.0.bounds(size)
    }
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

impl<A: Dimension, R: RangeToDim<A>> SliceOfShape<(A,)> for R {
    type OutputShape = (R::Dimension,);
    fn to_range_vec(&self) -> Vec<(Expression, Expression)> {
        vec![self.bounds(A::const_size())]
    }
}

impl<A: Dimension, B: Dimension, R1: RangeToDim<A>, R2: RangeToDim<B>> SliceOfShape<(A, B)>
    for (R1, R2)
{
    type OutputShape = (R1::Dimension, R2::Dimension);
    fn to_range_vec(&self) -> Vec<(Expression, Expression)> {
        vec![
            self.0.bounds(A::const_size()),
            self.1.bounds(B::const_size()),
        ]
    }
}

impl<
        A: Dimension,
        B: Dimension,
        C: Dimension,
        R1: RangeToDim<A>,
        R2: RangeToDim<B>,
        R3: RangeToDim<C>,
    > SliceOfShape<(A, B, C)> for (R1, R2, R3)
{
    type OutputShape = (R1::Dimension, R2::Dimension, R3::Dimension);
    fn to_range_vec(&self) -> Vec<(Expression, Expression)> {
        vec![
            self.0.bounds(A::const_size()),
            self.1.bounds(B::const_size()),
            self.2.bounds(C::const_size()),
        ]
    }
}

impl<
        A: Dimension,
        B: Dimension,
        C: Dimension,
        D: Dimension,
        R1: RangeToDim<A>,
        R2: RangeToDim<B>,
        R3: RangeToDim<C>,
        R4: RangeToDim<C>,
    > SliceOfShape<(A, B, C, D)> for (R1, R2, R3, R4)
{
    type OutputShape = (R1::Dimension, R2::Dimension, R3::Dimension, R4::Dimension);
    fn to_range_vec(&self) -> Vec<(Expression, Expression)> {
        vec![
            self.0.bounds(A::const_size()),
            self.1.bounds(B::const_size()),
            self.2.bounds(C::const_size()),
            self.3.bounds(D::const_size()),
        ]
    }
}

impl<
        A: Dimension,
        B: Dimension,
        C: Dimension,
        D: Dimension,
        E: Dimension,
        R1: RangeToDim<A>,
        R2: RangeToDim<B>,
        R3: RangeToDim<C>,
        R4: RangeToDim<C>,
        R5: RangeToDim<C>,
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
            self.0.bounds(A::const_size()),
            self.1.bounds(B::const_size()),
            self.2.bounds(C::const_size()),
            self.3.bounds(D::const_size()),
            self.4.bounds(E::const_size()),
        ]
    }
}
