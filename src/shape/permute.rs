use crate::shape::*;

/// Represents indices into the dimensions of shapes
pub trait Axes: 'static + Default + Copy + Clone {
    type Array: IntoIterator<Item = isize>;
    fn as_array() -> Self::Array;
}

/// A singular axis, e.g. `Axis<0>` or `Axis<1>`
#[derive(Clone, Copy, Debug, Default)]
pub struct Axis<const I: isize>;
impl<const I: isize> Axes for Axis<I> {
    type Array = [isize; 1];
    #[inline(always)]
    fn as_array() -> Self::Array {
        [I]
    }
}

/// A set of 2 axes, e.g. `Axes2<0, 1>`, or `Axes2<1, 3>`.
#[derive(Clone, Copy, Debug, Default)]
pub struct Axes2<const I: isize, const J: isize>;
impl<const I: isize, const J: isize> Axes for Axes2<I, J> {
    type Array = [isize; 2];
    #[inline(always)]
    fn as_array() -> Self::Array {
        [I, J]
    }
}

/// A set of 3 axes, e.g. `Axes3<1, 3, 4>`
#[derive(Clone, Copy, Debug, Default)]
pub struct Axes3<const I: isize, const J: isize, const K: isize>;
impl<const I: isize, const J: isize, const K: isize> Axes for Axes3<I, J, K> {
    type Array = [isize; 3];
    #[inline(always)]
    fn as_array() -> Self::Array {
        [I, J, K]
    }
}

pub trait PermuteShapeTo<Dst, Ax> {}

#[rustfmt::skip]
macro_rules! d { (0) => { D1 }; (1) => { D2 }; (2) => { D3 }; (3) => { D4 }; (4) => { D5 }; (5) => { D6 }; }

macro_rules! impl_permute {
    ($Ax0:tt, $Ax1:tt) => {
        impl<const D1: usize, const D2: usize>
            PermuteShapeTo<(Const<d!($Ax0)>, Const<d!($Ax1)>), Axes2<$Ax0, $Ax1>>
            for (Const<D1>, Const<D2>)
        {
        }
    };
    ($Ax0:tt, $Ax1:tt, $Ax2:tt) => {
        impl<const D1: usize, const D2: usize, const D3: usize>
            PermuteShapeTo<
                (Const<d!($Ax0)>, Const<d!($Ax1)>, Const<d!($Ax2)>),
                Axes3<$Ax0, $Ax1, $Ax2>,
            > for (Const<D1>, Const<D2>, Const<D3>)
        {
        }
    };
}

/// Expand out all the possible permutations for 2-4d
macro_rules! permutations {
    ([$Ax0:tt, $Ax1:tt]) => {
        impl_permute!($Ax1, $Ax0);
    };
    ([$Ax0:tt, $Ax1:tt, $Ax2:tt]) => {
        impl_permute!($Ax0, $Ax2, $Ax1);
        impl_permute!($Ax1, $Ax0, $Ax2);
        impl_permute!($Ax1, $Ax2, $Ax0);
        impl_permute!($Ax2, $Ax0, $Ax1);
        impl_permute!($Ax2, $Ax1, $Ax0);
    };

    ([$Ax0:tt, $Ax1:tt, $Ax2:tt, $Ax3:tt]) => {
        permutations!($Ax0, [$Ax1, $Ax2, $Ax3]);
        permutations!($Ax1, [$Ax0, $Ax2, $Ax3]);
        permutations!($Ax2, [$Ax0, $Ax1, $Ax3]);
        permutations!($Ax3, [$Ax0, $Ax1, $Ax2]);
    };
    ($Ax0:tt, [$Ax1:tt, $Ax2:tt, $Ax3:tt]) => {
        permutations!($Ax0, $Ax1, [$Ax2, $Ax3]);
        permutations!($Ax0, $Ax2, [$Ax1, $Ax3]);
        permutations!($Ax0, $Ax3, [$Ax1, $Ax2]);
    };
}

permutations!([0, 1]);
permutations!([0, 1, 2]);
