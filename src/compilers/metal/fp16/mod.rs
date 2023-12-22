use half::f16;

mod arange;
mod matmul;
mod mean_reduce;
mod other;
mod rms_norm;

pub type MetalFp16Compiler = (
    super::prim::PrimitiveCompiler<f16>,
    super::binary::MetalBinaryCompilers<f16>,
    // other::MetalCosCompiler, // For some reason doesn't produce the same outputs, need to test
    other::MetalExpCompiler,
    other::MetalGatherCompiler,
    arange::ARangeCompiler,
    matmul::MetalMatMulCompiler,
    mean_reduce::MeanReduceCompiler,
    rms_norm::RMSNormCompiler,
    super::other::CopyCompiler<f16>,
    super::common_buffer::CommonBufferCompiler,
);

#[cfg(test)]
mod tests;
