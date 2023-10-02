mod matmul;
mod prim;

pub type MetalFp16Optimizer = (
    prim::PrimitiveOptimizer,
    prim::FakeReductionOptimizer,
    matmul::MetalMatMulOptimizer,
);

#[cfg(test)]
mod tests;
