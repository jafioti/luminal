mod matmul;
mod mean_reduce;
mod prim;
mod rms_norm;

pub type MetalFp16Optimizer = (
    prim::PrimitiveOptimizer,
    prim::FakeReductionOptimizer,
    matmul::MetalMatMulOptimizer,
    mean_reduce::MeanReduceOptimizer,
    rms_norm::RMSNormOptimizer,
);

#[cfg(test)]
mod tests;
