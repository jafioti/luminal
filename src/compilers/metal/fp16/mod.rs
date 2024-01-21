use half::f16;

mod mean_reduce;
mod std_norm;

pub type MetalFp16Compiler = (
    super::prim::PrimitiveCompiler<f16>,
    (
        super::binary::MetalSubtractionCompiler<f16>,
        super::binary::MetalEqualCompiler<f16>,
        super::other::ARangeCompiler<f16>,
        super::binary::MetalGatherCompiler<f16>,
    ),
    super::other::MetalExpCompiler<f16>,
    super::matmul::MetalMatMulCompiler<f16>,
    mean_reduce::MeanReduceCompiler,
    std_norm::StdNormCompiler,
    super::other::CopyCompiler<f16>,
    super::other::ContiguousElimination<f16>,
    super::elementwise_fusion::ElementwiseFusionCompiler<f16>,
    (
        super::command_buffer::CommandBufferCompiler,
        super::storage_buffer::StorageBufferCompiler,
    ),
);

#[cfg(test)]
mod tests;
