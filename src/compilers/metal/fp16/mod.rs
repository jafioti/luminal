use half::f16;

use crate::prelude::TimedCompiler;

mod matmul;
mod mean_reduce;
mod other;
mod rms_norm;

pub type MetalFp16Compiler = (
    super::prim::PrimitiveCompiler<f16>,
    (
        TimedCompiler<super::binary::MetalSubtractionCompiler<f16>>,
        TimedCompiler<super::binary::MetalEqualCompiler<f16>>,
        TimedCompiler<super::other::ARangeCompiler<f16>>,
        TimedCompiler<super::binary::MetalGatherCompiler<f16>>,
    ),
    other::MetalExpCompiler,
    matmul::MetalMatMulCompiler,
    mean_reduce::MeanReduceCompiler,
    rms_norm::RMSNormCompiler,
    super::other::CopyCompiler<f16>,
    super::command_buffer::CommandBufferCompiler,
    super::storage_buffer::StorageBufferCompiler,
);

#[cfg(test)]
mod tests;
