use half::f16;

mod matmul;
mod mean_reduce;
mod other;
mod rms_norm;

pub type MetalFp16Compiler = (
    super::prim::PrimitiveCompiler<f16>,
    (
        super::binary::MetalSubtractionCompiler<f16>,
        super::binary::MetalEqualCompiler<f16>,
        super::other::ARangeCompiler<f16>,
        super::binary::MetalGatherCompiler<f16>,
    ),
    other::MetalExpCompiler,
    matmul::MetalMatMulCompiler,
    mean_reduce::MeanReduceCompiler,
    rms_norm::RMSNormCompiler,
    super::other::CopyCompiler<f16>,
    super::common_buffer::CommonBufferCompiler,
);

#[cfg(test)]
mod tests;
