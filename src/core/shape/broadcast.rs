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
        impl<$($DstDims: Dim, )*> ReduceShapeTo<($($SrcDims, )*), $Axes> for ($($DstDims, )*) {}
        impl<$($DstDims: Dim, )*> ReduceShape<$Axes> for ($($DstDims, )*) {
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
    fn check(&self, dst: &S);
    fn broadcast_strides(&self, strides: Self::Concrete) -> S::Concrete;
}

impl<Src: Shape, Dst: Shape, Ax: Axes> BroadcastStridesTo<Dst, Ax> for Src
where
    Self: BroadcastShapeTo<Dst, Ax>,
{
    #[inline(always)]
    fn check(&self, dst: &Dst) {
        let src_dims = self.concrete();
        let dst_dims = dst.concrete();
        let mut j = 0;
        for i in 0..Dst::NUM_DIMS {
            if !Ax::as_array().into_iter().any(|x| x == i as isize) {
                assert_eq!(dst_dims[i], src_dims[j]);
                j += 1;
            }
        }
    }

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
pub trait ReduceStridesTo<S: Shape, Ax>: Shape + ReduceShapeTo<S, Ax> {
    fn reduced(&self) -> S;
}

impl<Src: Shape, Dst: Shape, Ax: Axes> ReduceStridesTo<Dst, Ax> for Src
where
    Self: ReduceShapeTo<Dst, Ax>,
{
    #[inline(always)]
    fn reduced(&self) -> Dst {
        let src_dims = self.concrete();
        let mut dst_dims: Dst::Concrete = Default::default();
        let mut i_dst = 0;
        for i_src in 0..Src::NUM_DIMS {
            if !Ax::as_array().into_iter().any(|x| x == i_src as isize) {
                dst_dims[i_dst] = src_dims[i_src];
                i_dst += 1;
            }
        }
        Dst::from_concrete(&dst_dims).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_check() {
        BroadcastStridesTo::<(usize, usize), Axis<1>>::check(&(1,), &(1, 2));
    }

    #[test]
    #[should_panic]
    fn test_check_failures() {
        BroadcastStridesTo::<(usize, usize), Axis<1>>::check(&(1,), &(2, 2));
    }

    #[test]
    fn test_no_conflict_reductions() {
        let src = (1, Const::<2>, 3, Const::<4>);

        let dst: (usize, Const<2>) = src.reduced();
        assert_eq!(dst, (1, Const::<2>));

        let dst: (Const<2>, usize) = src.reduced();
        assert_eq!(dst, (Const::<2>, 3));

        let dst: (usize, usize) = src.reduced();
        assert_eq!(dst, (1, 3));
    }

    #[test]
    fn test_conflicting_reductions() {
        let src = (1, 2, Const::<3>);

        let dst = ReduceStridesTo::<_, Axis<1>>::reduced(&src);
        assert_eq!(dst, (1, Const::<3>));

        let dst = ReduceStridesTo::<_, Axis<0>>::reduced(&src);
        assert_eq!(dst, (2, Const::<3>));
    }

    #[test]
    fn test_broadcast_strides() {
        let src = (1,);
        let dst_strides =
            BroadcastStridesTo::<(usize, usize, usize), Axes2<0, 2>>::broadcast_strides(
                &src,
                src.strides(),
            );
        assert_eq!(dst_strides, [0, 1, 0]);
    }
}
