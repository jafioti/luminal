use super::*;
use std::ops::{Bound, RangeBounds};

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

pub trait SliceDim<R: RangeBounds<usize>>: Dim {
    type Sliced: Dim;

    fn slice(&self, range: &R) -> Option<Self::Sliced> {
        let size = self.size();

        let start_bound = get_start_bound(range.start_bound());
        let end_bound = get_end_bound(range.end_bound(), size);

        (end_bound <= size && start_bound <= end_bound)
            .then_some(end_bound - start_bound)
            .and_then(Self::Sliced::from_size)
    }
}

macro_rules! slice_dim_to_usize {
    ($range:ty) => {
        impl<D: Dim> SliceDim<$range> for D {
            type Sliced = usize;
        }
    };
}

slice_dim_to_usize!(std::ops::Range<usize>);
slice_dim_to_usize!(std::ops::RangeTo<usize>);
slice_dim_to_usize!(std::ops::RangeFrom<usize>);
slice_dim_to_usize!(std::ops::RangeInclusive<usize>);
slice_dim_to_usize!(std::ops::RangeToInclusive<usize>);

impl<D: Dim> SliceDim<std::ops::RangeFull> for D {
    type Sliced = D;

    fn slice(&self, _: &std::ops::RangeFull) -> Option<D> {
        Some(*self)
    }
}

pub trait SliceShape<R>: Shape {
    type Sliced: Shape<Concrete = Self::Concrete>;

    fn slice(&self, range: &R) -> Option<Self::Sliced>;
    fn first_idx_in_slice(&self, range: &R) -> usize;
}

impl SliceShape<()> for () {
    type Sliced = Self;

    fn slice(&self, _range: &()) -> Option<Self> {
        Some(())
    }

    fn first_idx_in_slice(&self, _range: &()) -> usize {
        0
    }
}

macro_rules! length {
    () => {0};
    ($x:tt $($xs:tt)*) => {1 + length!($($xs)*)};
}

macro_rules! slice_shape {
    ([$($dim:ident)*] [$($range:ident)*] [$($idx:tt)*]) => {
        impl<$($dim: Dim),*, $($range: RangeBounds<usize>),*> SliceShape<($($range,)*)> for ($($dim,)*)
        where
            $($dim: SliceDim<$range>),*
        {
            type Sliced = ($($dim::Sliced,)*);

            fn slice(&self, range: &($($range,)*)) -> Option<Self::Sliced> {
                Some(($(self.$idx.slice(&range.$idx)?,)*))
            }

            fn first_idx_in_slice(&self, range: &($($range,)*)) -> usize {
                let strides = self.strides();
                $(get_start_bound(range.$idx.start_bound()) * strides[$idx] + )* 0
            }
        }

        impl<$($range: RangeBounds<usize>),*> SliceShape<($($range,)*)> for [usize; {length!($($range)*)}]
        where
            $(usize: SliceDim<$range>),*
        {
            type Sliced = ($(<usize as SliceDim<$range>>::Sliced,)*);

            fn slice(&self, range: &($($range,)*)) -> Option<Self::Sliced> {
                Some(($(self[$idx].slice(&range.$idx)?,)*))
            }

            fn first_idx_in_slice(&self, range: &($($range,)*)) -> usize {
                let strides = self.strides();
                $(get_start_bound(range.$idx.start_bound()) * strides[$idx] + )* 0
            }
        }
    }
}

slice_shape!([D1][R1][0]);
slice_shape!([D1 D2] [R1 R2] [0 1]);
slice_shape!([D1 D2 D3] [R1 R2 R3] [0 1 2]);
slice_shape!([D1 D2 D3 D4] [R1 R2 R3 R4] [0 1 2 3]);
slice_shape!([D1 D2 D3 D4 D5] [R1 R2 R3 R4 R5] [0 1 2 3 4]);
slice_shape!([D1 D2 D3 D4 D5 D6] [R1 R2 R3 R4 R5 R6] [0 1 2 3 4 5]);
