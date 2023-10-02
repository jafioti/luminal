mod matmul;
mod prim;

pub type MetalFp32Optimizer = (prim::PrimitiveOptimizer, matmul::MetalMatMulOptimizer);

#[cfg(test)]
mod tests;
