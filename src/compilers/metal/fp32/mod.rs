mod matmul;

pub type MetalFp32Compiler = (
    super::prim::PrimitiveCompiler<f32>,
    matmul::MetalMatMulCompiler,
    super::other::CopyCompiler<f32>,
    super::common_buffer::CommonBufferCompiler,
);

#[cfg(test)]
mod tests;
