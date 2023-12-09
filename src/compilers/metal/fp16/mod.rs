mod common_buffer;
mod matmul;
mod mean_reduce;
mod other;
pub mod prim;
mod rms_norm;

pub type MetalFp16Compiler = (
    prim::PrimitiveCompiler,
    prim::FakeReductionCompiler,
    // other::MetalCosCompiler, // For some reason doesn't produce the same outputs, need to test
    other::MetalExpCompiler,
    other::MetalGatherCompiler,
    matmul::MetalMatMulCompiler,
    mean_reduce::MeanReduceCompiler,
    rms_norm::RMSNormCompiler,
    prim::CopyCompiler,
    common_buffer::CommonBufferCompiler,
);

#[cfg(test)]
mod tests;
