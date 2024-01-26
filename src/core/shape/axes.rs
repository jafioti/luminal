use super::*;

/// Represents indices into the dimensions of shapes
pub trait Axes: 'static + Default + Copy + Clone {
    type Array: IntoIterator<Item = usize>;
    fn as_array() -> Self::Array;
}

/// A singular axis, e.g. `Axis<0>` or `Axis<1>`
#[derive(Clone, Copy, Debug, Default)]
pub struct Axis<const I: usize>;
impl<const I: usize> Axes for Axis<I> {
    type Array = [usize; 1];
    #[inline(always)]
    fn as_array() -> Self::Array {
        [I]
    }
}

/// A set of 2 axes, e.g. `Axes2<0, 1>`, or `Axes2<1, 3>`.
#[derive(Clone, Copy, Debug, Default)]
pub struct Axes2<const I: usize, const J: usize>;
impl<const I: usize, const J: usize> Axes for Axes2<I, J> {
    type Array = [usize; 2];
    #[inline(always)]
    fn as_array() -> Self::Array {
        [I, J]
    }
}

/// A set of 3 axes, e.g. `Axes3<1, 3, 4>`
#[derive(Clone, Copy, Debug, Default)]
pub struct Axes3<const I: usize, const J: usize, const K: usize>;
impl<const I: usize, const J: usize, const K: usize> Axes for Axes3<I, J, K> {
    type Array = [usize; 3];
    #[inline(always)]
    fn as_array() -> Self::Array {
        [I, J, K]
    }
}

/// A set of 4 axes
#[derive(Clone, Copy, Debug, Default)]
pub struct Axes4<const I: usize, const J: usize, const K: usize, const L: usize>;
impl<const I: usize, const J: usize, const K: usize, const L: usize> Axes for Axes4<I, J, K, L> {
    type Array = [usize; 4];
    #[inline(always)]
    fn as_array() -> Self::Array {
        [I, J, K, L]
    }
}

/// A set of 5 axes
#[derive(Clone, Copy, Debug, Default)]
pub struct Axes5<const I: usize, const J: usize, const K: usize, const L: usize, const M: usize>;
impl<const I: usize, const J: usize, const K: usize, const L: usize, const M: usize> Axes
    for Axes5<I, J, K, L, M>
{
    type Array = [usize; 5];
    #[inline(always)]
    fn as_array() -> Self::Array {
        [I, J, K, L, M]
    }
}

/// A set of 6 axes
#[rustfmt::skip]
#[derive(Clone, Copy, Debug, Default)]
pub struct Axes6<const I: usize, const J: usize, const K: usize, const L: usize, const M: usize, const N: usize>;
#[rustfmt::skip]
impl<const I: usize, const J: usize, const K: usize, const L: usize, const M: usize, const N: usize> Axes
    for Axes6<I, J, K, L, M, N>
{
    type Array = [usize; 6];
    #[inline(always)]
    fn as_array() -> Self::Array {
        [I, J, K, L, M, N]
    }
}

/// Represents something that has the axes `Ax`
pub trait HasAxes<Ax> {}

macro_rules! impl_has_axis {
    (($($Vars:tt),*), $Num:tt, $Axis:tt) => {
        impl<$($Vars: Dimension, )*> HasAxes<Axis<$Axis>> for ($($Vars, )*) {
        }

        impl HasAxes<Axis<$Axis>> for [usize; $Num] {
        }
    };
}

impl HasAxes<Axis<0>> for () {}

impl_has_axis!((D1), 1, 0);
impl_has_axis!((D1, D2), 2, 0);
impl_has_axis!((D1, D2), 2, 1);
impl_has_axis!((D1, D2, D3), 3, 0);
impl_has_axis!((D1, D2, D3), 3, 1);
impl_has_axis!((D1, D2, D3), 3, 2);
impl_has_axis!((D1, D2, D3, D4), 4, 0);
impl_has_axis!((D1, D2, D3, D4), 4, 1);
impl_has_axis!((D1, D2, D3, D4), 4, 2);
impl_has_axis!((D1, D2, D3, D4), 4, 3);
impl_has_axis!((D1, D2, D3, D4, D5), 5, 0);
impl_has_axis!((D1, D2, D3, D4, D5), 5, 1);
impl_has_axis!((D1, D2, D3, D4, D5), 5, 2);
impl_has_axis!((D1, D2, D3, D4, D5), 5, 3);
impl_has_axis!((D1, D2, D3, D4, D5), 5, 4);
impl_has_axis!((D1, D2, D3, D4, D5, D6), 6, 0);
impl_has_axis!((D1, D2, D3, D4, D5, D6), 6, 1);
impl_has_axis!((D1, D2, D3, D4, D5, D6), 6, 2);
impl_has_axis!((D1, D2, D3, D4, D5, D6), 6, 3);
impl_has_axis!((D1, D2, D3, D4, D5, D6), 6, 4);
impl_has_axis!((D1, D2, D3, D4, D5, D6), 6, 5);

impl<const I: usize, const J: usize, S> HasAxes<Axes2<I, J>> for S where
    Self: HasAxes<Axis<I>> + HasAxes<Axis<J>>
{
}

impl<const I: usize, const J: usize, const K: usize, S> HasAxes<Axes3<I, J, K>> for S where
    Self: HasAxes<Axis<I>> + HasAxes<Axis<J>> + HasAxes<Axis<K>>
{
}

impl<const I: usize, const J: usize, const K: usize, const L: usize, S> HasAxes<Axes4<I, J, K, L>>
    for S
where
    Self: HasAxes<Axis<I>> + HasAxes<Axis<J>> + HasAxes<Axis<K>> + HasAxes<Axis<L>>,
{
}

impl<const I: usize, const J: usize, const K: usize, const L: usize, const M: usize, S>
    HasAxes<Axes5<I, J, K, L, M>> for S
where
    Self: HasAxes<Axis<I>>
        + HasAxes<Axis<J>>
        + HasAxes<Axis<K>>
        + HasAxes<Axis<L>>
        + HasAxes<Axis<M>>,
{
}

impl<
        const I: usize,
        const J: usize,
        const K: usize,
        const L: usize,
        const M: usize,
        const N: usize,
        S,
    > HasAxes<Axes6<I, J, K, L, M, N>> for S
where
    Self: HasAxes<Axis<I>>
        + HasAxes<Axis<J>>
        + HasAxes<Axis<K>>
        + HasAxes<Axis<L>>
        + HasAxes<Axis<M>>
        + HasAxes<Axis<N>>,
{
}
