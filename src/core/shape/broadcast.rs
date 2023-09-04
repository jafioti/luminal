use super::*;

/// Marker for shapes that can be reduced to [Shape] `S` along [Axes] `Ax`.
pub trait ReduceShapeTo<S, Ax>: HasAxes<Ax> + Sized {}

/// Marker for shapes that can be broadcasted to [Shape] `S` along [Axes] `Ax`.
pub trait BroadcastShapeTo<S, Ax>: Sized {}

/// Marker for shapes that can have their [Axes] `Ax` reduced. See Self::Reduced
/// for the resulting type.
pub trait ReduceShape<Ax>: Sized + HasAxes<Ax> + ReduceShapeTo<Self::Reduced, Ax> {
    type Reduced: Shape + BroadcastShapeTo<Self, Ax>;
}

impl ReduceShapeTo<(), Axis<0>> for () {}
impl ReduceShape<Axis<0>> for () {
    type Reduced = ();
}
impl<Src: Shape, Dst: Shape + ReduceShapeTo<Src, Ax>, Ax> BroadcastShapeTo<Dst, Ax> for Src {}

macro_rules! broadcast_to_array {
    ($SrcNum:tt, (), $DstNum:tt, ($($DstDims:tt),*), $Axes:ty) => {
        impl ReduceShapeTo<(), $Axes> for [usize; $DstNum] {}
        impl ReduceShape<$Axes> for [usize; $DstNum] {
            type Reduced = ();
        }
    };
    ($SrcNum:tt, ($($SrcDims:tt),*), $DstNum:tt, ($($DstDims:tt),*), $Axes:ty) => {
        impl ReduceShapeTo<[usize; $SrcNum], $Axes> for [usize; $DstNum] {}
        impl ReduceShape<$Axes> for [usize; $DstNum] {
            type Reduced = [usize; $SrcNum];
        }
    };
}

macro_rules! broadcast_to {
    ($SrcNum:tt, ($($SrcDims:tt),*), $DstNum:tt, ($($DstDims:tt),*), ()<>) => {
    };
    ($SrcNum:tt, ($($SrcDims:tt),*), $DstNum:tt, ($($DstDims:tt),*), $Axes:ty) => {
        impl<$($DstDims: Dimension, )*> ReduceShapeTo<($($SrcDims, )*), $Axes> for ($($DstDims, )*) {}
        impl<$($DstDims: Dimension, )*> ReduceShape<$Axes> for ($($DstDims, )*) {
            type Reduced = ($($SrcDims, )*);
        }
        broadcast_to_array!($SrcNum, ($($SrcDims),*), $DstNum, ($($DstDims),*), $Axes);
    };
}

macro_rules! length {
    () => {0};
    ($x:tt $($xs:tt)*) => {1 + length!($($xs)*)};
}

pub(crate) use length;

// Defines all reduce/broadcast rules recursively
macro_rules! broadcast_to_all {
    ([$($s1:ident)*] [$($s2:ident)*] [$($ax:tt)*] [] [$axis:tt $($axes:tt)*]) => {
        broadcast_to!({length!($($s1)*)}, ($($s1),*), {length!($($s2)*)}, ($($s2),*), $axis<$({$ax}),*>);
    };
    (
        [$($s1:ident)*]
        [$($s2:ident)*]
        [$($ax:tt)*]
        [$sh:ident $($shs:ident)*]
        [$axis:tt $($axes:tt)*]
    ) => {
        broadcast_to!({length!($($s1)*)}, ($($s1),*), {length!($($s2)*)}, ($($s2),*), $axis<$({$ax}),*>);

        // Add a broadcasted dimension to the end of s2
        broadcast_to_all!([$($s1)*] [$($s2)* $sh] [$($ax)* {length!($($s2)*)}] [$($shs)*] [$($axes)*]);

        // Add a dimension to both s1 and s2
        broadcast_to_all!([$($s1)* $sh] [$($s2)* $sh] [$($ax)*] [$($shs)*] [$axis $($axes)*]);
    }
}

broadcast_to_all!([] [] [] [A B C D E F] [() Axis Axes2 Axes3 Axes4 Axes5 Axes6]);

/// Internal implementation for broadcasting strides
pub trait BroadcastStridesTo<S: Shape, Ax>: Shape + BroadcastShapeTo<S, Ax> {
    // fn check(&self, dst: &S);
    fn broadcast_strides(&self, strides: Self::Concrete) -> S::Concrete;
}

impl<Src: Shape, Dst: Shape, Ax: Axes> BroadcastStridesTo<Dst, Ax> for Src
where
    Self: BroadcastShapeTo<Dst, Ax>,
{
    #[inline(always)]
    fn broadcast_strides(&self, strides: Self::Concrete) -> Dst::Concrete {
        let mut new_strides: Dst::Concrete = Default::default();
        let mut j = 0;
        for i in 0..Dst::NUM_DIMS {
            if !Ax::as_array().into_iter().any(|x| x == i as isize) {
                new_strides[i] = strides[j];
                j += 1;
            }
        }
        new_strides
    }
}

/// Internal implementation for reducing a shape
pub trait ReduceStridesTo<S: Shape, Ax>: Shape + ReduceShapeTo<S, Ax> {}

impl<Src: Shape, Dst: Shape, Ax: Axes> ReduceStridesTo<Dst, Ax> for Src where
    Self: ReduceShapeTo<Dst, Ax>
{
}
