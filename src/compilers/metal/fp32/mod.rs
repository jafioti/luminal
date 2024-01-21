pub type MetalFp32Compiler = (
    super::prim::PrimitiveCompiler<f32>,
    super::matmul::MetalMatMulCompiler<f32>,
    super::other::CopyCompiler<f32>,
    super::command_buffer::CommandBufferCompiler,
);

#[cfg(test)]
mod tests;
