use crate::{
    prelude::{metal::prim::MetalAdd, *},
    select_const, select_ty,
};
use petgraph::stable_graph::NodeIndex;

use super::{
    binary::MetalSub,
    prim::{MetalConstant, MetalLessThan, MetalMul},
};

pub fn less_than<T: MetalFloat>(
    s1: SelectEdge,
    s2: SelectEdge,
    ptrs: &mut Vec<NodeIndex>,
) -> SelectEdge {
    s2.edge(s1.edge(select_ty!(MetalLessThan<T>).ptr(ptrs)))
}

pub fn mul<T: MetalFloat>(s1: SelectEdge, s2: SelectEdge, ptrs: &mut Vec<NodeIndex>) -> SelectEdge {
    s2.edge(s1.edge(select_ty!(MetalMul<T>).ptr(ptrs)))
}
pub fn add<T: MetalFloat>(s1: SelectEdge, s2: SelectEdge, ptrs: &mut Vec<NodeIndex>) -> SelectEdge {
    s2.edge(s1.edge(select_ty!(MetalAdd<T>).ptr(ptrs)))
}
pub fn sub<T: MetalFloat>(s1: SelectEdge, s2: SelectEdge, ptrs: &mut Vec<NodeIndex>) -> SelectEdge {
    s2.edge(s1.edge(select_ty!(MetalSub<T>).ptr(ptrs)))
}
pub fn less_than_equal<T: MetalFloat>(
    s1: SelectEdge,
    s2: SelectEdge,
    mut ptrs: &mut Vec<NodeIndex>,
) -> SelectEdge {
    sub::<T>(
        select_const!(1.0, T).ptr(&mut ptrs).into(),
        less_than::<T>(s2, s1, &mut ptrs),
        ptrs,
    )
}
pub fn max<T: MetalFloat>(s1: SelectEdge, s2: SelectEdge, ptrs: &mut Vec<NodeIndex>) -> SelectEdge {
    let a = mul::<T>(
        less_than::<T>(s1.clone(), s2.clone(), ptrs),
        s2.clone(),
        ptrs,
    );
    let b = mul::<T>(less_than_equal::<T>(s2, s1.clone(), ptrs), s1, ptrs);
    add::<T>(a, b, ptrs)
}
pub fn relu<T: MetalFloat>(s1: SelectEdge, mut ptrs: &mut Vec<NodeIndex>) -> SelectEdge {
    max::<T>(s1, select_const!(0.0, T).ptr(&mut ptrs).into(), &mut ptrs)
}
pub fn abs<T: MetalFloat>(s1: SelectEdge, mut ptrs: &mut Vec<NodeIndex>) -> SelectEdge {
    add::<T>(
        relu::<T>(s1.clone(), &mut ptrs),
        relu::<T>(
            mul::<T>(s1, select_const!(-1.0, T).ptr(&mut ptrs).into(), &mut ptrs),
            &mut ptrs,
        ),
        &mut ptrs,
    )
}
