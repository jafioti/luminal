mod prim;

pub type MetalFp16Optimizer = (prim::PrimitiveOptimizer, prim::FakeReductionOptimizer);

#[cfg(test)]
mod tests;
