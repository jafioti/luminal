mod matmul;
mod prim;

#[cfg(test)]
mod tests;

use cudarc::driver::CudaSlice;

use crate::prelude::*;

// Ops and optimizers specific to CUDA execution

pub type CudaOptimizer = (prim::CudaPrimitiveOptimizer, matmul::CudaMatMulOptimizer);

impl Data for CudaSlice<f32> {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

impl Data for CudaSlice<usize> {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}
