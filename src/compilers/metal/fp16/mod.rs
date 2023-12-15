use half::f16;

mod matmul;
mod mean_reduce;
mod other;
mod rms_norm;

pub type MetalFp16Compiler = (
    super::prim::PrimitiveCompiler<f16>,
    other::FakeSumReduceCompiler,
    // other::MetalCosCompiler, // For some reason doesn't produce the same outputs, need to test
    other::MetalExpCompiler,
    other::MetalGatherCompiler,
    matmul::MetalMatMulCompiler,
    mean_reduce::MeanReduceCompiler,
    rms_norm::RMSNormCompiler,
    super::other::CopyCompiler<f16>,
    super::common_buffer::CommonBufferCompiler,
);

#[cfg(test)]
mod tests;
