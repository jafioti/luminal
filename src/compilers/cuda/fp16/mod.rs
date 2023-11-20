mod matmul;
mod prim;

#[cfg(test)]
mod tests;

use cudarc::driver::CudaSlice;
use half::f16;

use crate::prelude::*;

// Ops and optimizers specific to CUDA execution

pub type CudaFp16Compiler = (
    prim::CudaPrimitiveCompiler,
    prim::FakeReductionCompiler,
    matmul::CudaMatMulCompiler,
    prim::CopyCompiler,
);

impl Data for CudaSlice<f16> {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}
