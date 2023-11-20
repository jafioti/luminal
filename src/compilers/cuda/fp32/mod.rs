mod matmul;
mod prim;

#[cfg(test)]
mod tests;

use cudarc::driver::CudaSlice;

use crate::prelude::*;

// Ops and optimizers specific to CUDA execution

pub type CudaFp32Compiler = (prim::CudaPrimitiveCompiler, matmul::CudaMatMulCompiler);

impl Data for CudaSlice<f32> {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}
