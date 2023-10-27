mod common_buffer;
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
    common_buffer::CommonBufferOptimizer,
);

#[cfg(test)]
mod tests;
