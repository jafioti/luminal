mod matmul;
mod mean_reduce;
mod other;
mod prim;
mod rms_norm;

pub type MetalFp16Optimizer = (
    prim::PrimitiveOptimizer,
    prim::FakeReductionOptimizer,
    other::MetalCosOptimizer,
    other::MetalExpOptimizer,
    matmul::MetalMatMulOptimizer,
    mean_reduce::MeanReduceOptimizer,
    rms_norm::RMSNormOptimizer,
    prim::CopyOptimizer,
);

#[cfg(test)]
mod tests;
