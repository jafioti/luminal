mod permute;
mod symbolic;
mod tracker;

pub use permute::*;
pub use tracker::*;

use std::fmt::Debug;

pub type R1<const D: usize> = (Const<D>,);
pub type R2<const A: usize, const B: usize> = (Const<A>, Const<B>);
pub type R3<const A: usize, const B: usize, const C: usize> = (Const<A>, Const<B>, Const<C>);

pub trait Shape: Debug + Copy {
    type AddRight<const N: usize>: Shape;
    type AddLeft<const N: usize>: Shape;
    fn realized_shape() -> Vec<usize>;

    const NUM_DIMS: usize;
    const NUM_ELEMENTS: usize;
}

#[derive(Debug, Clone, Copy)]
pub struct Const<const U: usize>;

impl<const A: usize> Shape for Const<A> {
    type AddRight<const N: usize> = R2<A, N>;
    type AddLeft<const N: usize> = R2<N, A>;
    fn realized_shape() -> Vec<usize> {
        vec![A]
    }

    const NUM_DIMS: usize = 1;
    const NUM_ELEMENTS: usize = A;
}

impl<const A: usize> Shape for (Const<A>,) {
    type AddRight<const N: usize> = R2<A, N>;
    type AddLeft<const N: usize> = R2<N, A>;
    fn realized_shape() -> Vec<usize> {
        vec![A]
    }

    const NUM_DIMS: usize = 1;
    const NUM_ELEMENTS: usize = A;
}

impl<const A: usize, const B: usize> Shape for (Const<A>, Const<B>) {
    type AddRight<const N: usize> = Const<A>;
    type AddLeft<const N: usize> = Const<A>;
    fn realized_shape() -> Vec<usize> {
        vec![A, B]
    }

    const NUM_DIMS: usize = 2;
    const NUM_ELEMENTS: usize = A * B;
}

pub trait AssertSameNumel<Dst: Shape>: Shape {
    const TYPE_CHECK: ();
    fn assert_same_numel() {
        #[allow(clippy::let_unit_value)]
        let _ = <Self as AssertSameNumel<Dst>>::TYPE_CHECK;
    }
}

impl<Src: Shape, Dst: Shape> AssertSameNumel<Dst> for Src {
    const TYPE_CHECK: () = assert!(Src::NUM_ELEMENTS == Dst::NUM_ELEMENTS);
}
