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

pub trait SliceRange<D: Dimension> {
    type Dimension: Dimension;
    fn bounds(&self, size: impl Into<Expression>) -> (Expression, Expression);
}

impl<D: Dimension> SliceRange<D> for RangeFrom<usize> {
    type Dimension = Dyn<'-'>;
    fn bounds(&self, size: impl Into<Expression>) -> (Expression, Expression) {
        (
            get_start_bound(self.start_bound()),
            get_end_bound(self.end_bound(), size),
        )
    }
}
impl<D: Dimension> SliceRange<D> for RangeTo<usize> {
    type Dimension = Dyn<'-'>;
    fn bounds(&self, size: impl Into<Expression>) -> (Expression, Expression) {
        (
            get_start_bound(self.start_bound()),
            get_end_bound(self.end_bound(), size),
        )
    }
}
impl<D: Dimension> SliceRange<D> for RangeToInclusive<usize> {
    type Dimension = Dyn<'-'>;
    fn bounds(&self, size: impl Into<Expression>) -> (Expression, Expression) {
        (
            get_start_bound(self.start_bound()),
            get_end_bound(self.end_bound(), size),
        )
    }
}
impl<D: Dimension> SliceRange<D> for Range<usize> {
    type Dimension = Dyn<'-'>;
    fn bounds(&self, size: impl Into<Expression>) -> (Expression, Expression) {
        (
            get_start_bound(self.start_bound()),
            get_end_bound(self.end_bound(), size),
        )
    }
}
impl<D: Dimension> SliceRange<D> for RangeFrom<Expression> {
    type Dimension = Dyn<'-'>;
    fn bounds(&self, size: impl Into<Expression>) -> (Expression, Expression) {
        (
            get_start_bound(self.start_bound()),
            get_end_bound(self.end_bound(), size),
        )
    }
}
impl<D: Dimension> SliceRange<D> for RangeTo<Expression> {
    type Dimension = Dyn<'-'>;
    fn bounds(&self, size: impl Into<Expression>) -> (Expression, Expression) {
        (
            get_start_bound(self.start_bound()),
            get_end_bound(self.end_bound(), size),
        )
    }
}
impl<D: Dimension> SliceRange<D> for RangeToInclusive<Expression> {
    type Dimension = Dyn<'-'>;
    fn bounds(&self, size: impl Into<Expression>) -> (Expression, Expression) {
        (
            get_start_bound(self.start_bound()),
            get_end_bound(self.end_bound(), size),
        )
    }
}
impl<D: Dimension> SliceRange<D> for Range<Expression> {
    type Dimension = Dyn<'-'>;
    fn bounds(&self, size: impl Into<Expression>) -> (Expression, Expression) {
        (
            get_start_bound(self.start_bound()),
            get_end_bound(self.end_bound(), size),
        )
    }
}
impl<D: Dimension> SliceRange<D> for RangeFull {
    type Dimension = D;
    fn bounds(&self, size: impl Into<Expression>) -> (Expression, Expression) {
        (0.into(), size.into())
    }
}
impl<D: Dimension, R: SliceRange<D>> SliceRange<D> for (R,) {
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

impl<A: Dimension, R: SliceRange<A>> SliceOfShape<(A,)> for R {
    type OutputShape = (R::Dimension,);
    fn to_range_vec(&self) -> Vec<(Expression, Expression)> {
        vec![self.bounds(A::size())]
    }
}

impl<A: Dimension, B: Dimension, R1: SliceRange<A>, R2: SliceRange<B>> SliceOfShape<(A, B)>
    for (R1, R2)
{
    type OutputShape = (R1::Dimension, R2::Dimension);
    fn to_range_vec(&self) -> Vec<(Expression, Expression)> {
        vec![self.0.bounds(A::size()), self.1.bounds(B::size())]
    }
}

impl<
        A: Dimension,
        B: Dimension,
        C: Dimension,
        R1: SliceRange<A>,
        R2: SliceRange<B>,
        R3: SliceRange<C>,
    > SliceOfShape<(A, B, C)> for (R1, R2, R3)
{
    type OutputShape = (R1::Dimension, R2::Dimension, R3::Dimension);
    fn to_range_vec(&self) -> Vec<(Expression, Expression)> {
        vec![
            self.0.bounds(A::size()),
            self.1.bounds(B::size()),
            self.2.bounds(C::size()),
        ]
    }
}

impl<
        A: Dimension,
        B: Dimension,
        C: Dimension,
        D: Dimension,
        R1: SliceRange<A>,
        R2: SliceRange<B>,
        R3: SliceRange<C>,
        R4: SliceRange<C>,
    > SliceOfShape<(A, B, C, D)> for (R1, R2, R3, R4)
{
    type OutputShape = (R1::Dimension, R2::Dimension, R3::Dimension, R4::Dimension);
    fn to_range_vec(&self) -> Vec<(Expression, Expression)> {
        vec![
            self.0.bounds(A::size()),
            self.1.bounds(B::size()),
            self.2.bounds(C::size()),
            self.3.bounds(D::size()),
        ]
    }
}

impl<
        A: Dimension,
        B: Dimension,
        C: Dimension,
        D: Dimension,
        E: Dimension,
        R1: SliceRange<A>,
        R2: SliceRange<B>,
        R3: SliceRange<C>,
        R4: SliceRange<C>,
        R5: SliceRange<C>,
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
            self.0.bounds(A::size()),
            self.1.bounds(B::size()),
            self.2.bounds(C::size()),
            self.3.bounds(D::size()),
            self.4.bounds(E::size()),
        ]
    }
}
