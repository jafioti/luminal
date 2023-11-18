mod common_buffer;
mod matmul;
mod mean_reduce;
mod other;
pub mod prim;
mod rms_norm;

pub type MetalFp16Compiler = (
    prim::PrimitiveCompiler,
    prim::FakeReductionCompiler,
    // other::MetalCosCompiler, // Need faster graph selector to reenable this
    other::MetalExpCompiler,
    matmul::MetalMatMulCompiler,
    mean_reduce::MeanReduceCompiler,
    rms_norm::RMSNormCompiler,
    prim::CopyCompiler,
    // common_buffer::CommonBufferCompiler,
);

#[cfg(test)]
mod tests;
