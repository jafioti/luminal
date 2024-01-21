use half::f16;

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
    super::mean_reduce::MeanReduceCompiler<f16>,
    super::std_norm::StdNormCompiler<f16>,
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
