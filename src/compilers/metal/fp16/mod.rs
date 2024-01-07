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
    (other::MetalExpCompiler, other::MetalSwishCompiler),
    matmul::MetalMatMulCompiler,
    mean_reduce::MeanReduceCompiler,
    rms_norm::RMSNormCompiler,
    super::other::CopyCompiler<f16>,
    super::other::ContiguousElimination<f16>,
    super::command_buffer::CommandBufferCompiler,
    super::storage_buffer::StorageBufferCompiler,
);

#[cfg(test)]
mod tests;
