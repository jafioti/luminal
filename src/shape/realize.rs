use crate::prelude::*;

pub trait RealizeShapeTo<Dst: Shape>: Shape {}

impl<Src: Shape<Concrete = Dst::Concrete>, Dst: Shape> RealizeShapeTo<Dst> for Src {}
