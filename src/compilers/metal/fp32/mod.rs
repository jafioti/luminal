mod matmul;
mod prim;

pub type MetalFp32Compiler = (prim::PrimitiveCompiler, matmul::MetalMatMulCompiler);

#[cfg(test)]
mod tests;
