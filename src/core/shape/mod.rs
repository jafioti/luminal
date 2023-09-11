mod axes;
mod broadcast;
mod permute;
mod realize;
pub mod simple_tracker;
mod slice;

pub use realize::*;
pub use slice::*;

#[cfg(test)]
mod test;

pub use axes::*;
pub use broadcast::*;
pub use permute::*;
pub use simple_tracker::*;

// This currently is a lot more complicated than it needs to be, because it's based on dfdx and is ready to do dynamic dimensions.
// TODO: Actually use dynamic dimensions
// TODO: Simplify this code

/// Represents a single dimension of a multi dimensional [Shape]
pub trait Dimension:
    'static + Copy + Clone + std::fmt::Debug + Send + Sync + Eq + PartialEq
{
    // fn size(&self) -> usize;
    fn const_size() -> Dim;
    // fn from_size(size: usize) -> Option<Self>;
}

/// Represents a single dimension where all
/// instances are guaranteed to be the same size at compile time.
pub trait ConstDim: Default + Dimension {
    const SIZE: usize;
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Dyn<const C: char>;

impl<const C: char> Dimension for Dyn<C> {
    fn const_size() -> Dim {
        Dim::Unknown(C)
    }
}

/// Represents a [Dim] with size known at compile time
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Const<const M: usize>;
impl<const M: usize> Dimension for Const<M> {
    fn const_size() -> Dim {
        Dim::Known(M)
    }
}

impl<const M: usize> ConstDim for Const<M> {
    const SIZE: usize = M;
}

impl<const N: usize, const C: char> core::ops::Add<Const<N>> for Dyn<C> {
    type Output = Dyn<C>;
    fn add(self, _: Const<N>) -> Self::Output {
        todo!();
    }
}
impl<const N: usize, const C: char> core::ops::Add<Dyn<C>> for Const<N> {
    type Output = Dyn<C>;
    fn add(self, _: Dyn<C>) -> Self::Output {
        todo!();
    }
}

impl<const N: usize, const C: char> core::ops::Mul<Const<N>> for Dyn<C> {
    type Output = Dyn<C>;
    fn mul(self, _: Const<N>) -> Self::Output {
        todo!();
    }
}
impl<const N: usize, const C: char> core::ops::Mul<Dyn<C>> for Const<N> {
    type Output = Dyn<C>;
    fn mul(self, _: Dyn<C>) -> Self::Output {
        todo!();
    }
}

impl<const N: usize, const C: char> core::ops::Div<Const<N>> for Dyn<C> {
    type Output = Dyn<C>;
    fn div(self, _: Const<N>) -> Self::Output {
        todo!();
    }
}
impl<const N: usize, const C: char> core::ops::Div<Dyn<C>> for Const<N> {
    type Output = Dyn<C>;
    fn div(self, _: Dyn<C>) -> Self::Output {
        todo!();
    }
}

impl<const A: char, const C: char> core::ops::Add<Dyn<A>> for Dyn<C> {
    type Output = Dyn<'-'>;
    fn add(self, _: Dyn<A>) -> Self::Output {
        todo!();
    }
}

impl<const A: char, const C: char> core::ops::Mul<Dyn<A>> for Dyn<C> {
    type Output = Dyn<'-'>;
    fn mul(self, _: Dyn<A>) -> Self::Output {
        todo!();
    }
}

impl<const A: char, const C: char> core::ops::Div<Dyn<A>> for Dyn<C> {
    type Output = Dyn<'-'>;
    fn div(self, _: Dyn<A>) -> Self::Output {
        todo!();
    }
}

/// Represents either `[T; N]` or `Vec<T>`
pub trait Array<T>: IntoIterator<Item = T> {
    type Dim: Dimension;
    fn dim(&self) -> Self::Dim;
}
impl<T, const N: usize> Array<T> for [T; N] {
    type Dim = Const<N>;
    fn dim(&self) -> Self::Dim {
        Const
    }
}
impl<T> Array<T> for std::vec::Vec<T> {
    type Dim = Dyn<'-'>;
    fn dim(&self) -> Self::Dim {
        Dyn::<'-'>
    }
}

/// A collection of dimensions ([Dim]) that change how a multi-dimensional
/// array is interacted with.
pub trait Shape:
    'static
    + std::fmt::Debug
    + Clone
    + Copy
    + Send
    + Sync
    + Eq
    + PartialEq
    + HasAxes<Self::AllAxes>
    + HasAxes<Self::LastAxis>
    + ReduceShapeTo<(), Self::AllAxes>
    + ReduceShape<Self::LastAxis>
{
    /// The number of dimensions the shape has
    const NUM_DIMS: usize;

    /// Is `[usize; Self::NUM_DIMS]`, but that is not usable yet.
    type Concrete: std::fmt::Debug
        + Clone
        + Copy
        + Default
        + Eq
        + PartialEq
        + std::ops::Index<usize, Output = usize>
        + std::ops::IndexMut<usize>
        + Send
        + Sync
        + IntoIterator<Item = usize>
        + Into<std::vec::Vec<usize>>
        + AsRef<[usize]>;

    /// All the axes of this shape
    type AllAxes: Axes;

    /// The last axis of this shape
    type LastAxis: Axes;

    fn realized_shape() -> Vec<Dim>;
    fn to_tracker() -> crate::core::shape::simple_tracker::ShapeTracker;
}

/// Represents a [Shape] that has all [ConstDim]s
pub trait ConstShape: Default + Shape {
    const NUMEL: usize;
    fn realized_shape() -> Vec<usize>;
}

/// Represents something that has a [Shape].
pub trait HasShape {
    type WithShape<New: Shape>: HasShape<Shape = New>;
    type Shape: Shape;
    fn shape(&self) -> &Self::Shape;
}

impl<S: Shape> HasShape for S {
    type WithShape<New: Shape> = New;
    type Shape = Self;
    fn shape(&self) -> &Self::Shape {
        self
    }
}

/// Compile time known shape with 0 dimensions
pub type R0 = ();
/// Compile time known shape with 1 dimensions
pub type R1<const M: usize> = (Const<M>,);
/// Compile time known shape with 2 dimensions
pub type R2<const M: usize, const N: usize> = (Const<M>, Const<N>);
/// Compile time known shape with 3 dimensions
pub type R3<const M: usize, const N: usize, const O: usize> = (Const<M>, Const<N>, Const<O>);
/// Compile time known shape with 4 dimensions
pub type R4<const M: usize, const N: usize, const O: usize, const P: usize> =
    (Const<M>, Const<N>, Const<O>, Const<P>);
/// Compile time known shape with 5 dimensions
pub type R5<const M: usize, const N: usize, const O: usize, const P: usize, const Q: usize> =
    (Const<M>, Const<N>, Const<O>, Const<P>, Const<Q>);
#[rustfmt::skip]
/// Compile time known shape with 6 dimensions
pub type R6<const M: usize, const N: usize, const O: usize, const P: usize, const Q: usize, const R: usize> =
    (Const<M>, Const<N>, Const<O>, Const<P>, Const<Q>, Const<R>);

macro_rules! shape {
    (($($D:tt $Idx:tt),*), rank=$Num:expr, all=$All:tt) => {
        impl<$($D: Dimension, )*> Shape for ($($D, )*) {
            const NUM_DIMS: usize = $Num;
            type Concrete = [usize; $Num];
            type AllAxes = $All<$($Idx,)*>;
            type LastAxis = Axis<{$Num - 1}>;

            fn realized_shape() -> Vec<Dim> {
                vec![$($D::const_size(), )*]
            }
            fn to_tracker() -> ShapeTracker {
                ShapeTracker::new(&Self::realized_shape())
            }
        }
        impl<$($D: ConstDim, )*> ConstShape for ($($D, )*) {
            const NUMEL: usize = $($D::SIZE * )* 1;

            fn realized_shape() -> Vec<usize> {
                vec![$($D::SIZE , )*]
            }
         }

        impl Shape for [usize; $Num] {
            const NUM_DIMS: usize = $Num;
            type Concrete = Self;
            type AllAxes = $All<$($Idx,)*>;
            type LastAxis = Axis<{$Num - 1}>;

            fn realized_shape() -> Vec<Dim> {
                vec![Dim::Unknown('-'); $Num]
            }

            fn to_tracker() -> ShapeTracker {
                let st = ShapeTracker::new(&Self::realized_shape());
                st
            }
        }
    };
}

impl Shape for () {
    const NUM_DIMS: usize = 0;
    type Concrete = [usize; 0];
    type AllAxes = Axis<0>;
    type LastAxis = Axis<0>;
    fn realized_shape() -> Vec<Dim> {
        vec![]
    }
    fn to_tracker() -> ShapeTracker {
        ShapeTracker::new(&[])
    }
}
impl ConstShape for () {
    const NUMEL: usize = 1;

    fn realized_shape() -> Vec<usize> {
        vec![]
    }
}

shape!((D1 0), rank=1, all=Axis);
shape!((D1 0, D2 1), rank=2, all=Axes2);
shape!((D1 0, D2 1, D3 2), rank=3, all=Axes3);
shape!((D1 0, D2 1, D3 2, D4 3), rank=4, all=Axes4);
shape!((D1 0, D2 1, D3 2, D4 3, D5 4), rank=5, all=Axes5);
shape!((D1 0, D2 1, D3 2, D4 3, D5 4, D6 5), rank=6, all=Axes6);

/// Marker for shapes that have the same number of elements as `Dst`
pub trait AssertSameNumel<Dst: ConstShape>: ConstShape {
    const TYPE_CHECK: ();
    fn assert_same_numel() {
        #[allow(clippy::let_unit_value)]
        let _ = <Self as AssertSameNumel<Dst>>::TYPE_CHECK;
    }
}

impl<Src: ConstShape, Dst: ConstShape> AssertSameNumel<Dst> for Src {
    const TYPE_CHECK: () = assert!(Src::NUMEL == Dst::NUMEL);
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReshapeDim {
    /// A known size for the dim
    Const(usize),
    /// A reference to the size of a dim of the previous shape
    PrevDim(usize),
}
