use cudarc::driver::CudaSlice;

use crate::prelude::Data;

mod fp16;
mod fp32;

pub use fp16::CudaFp16Optimizer;
pub use fp32::CudaFp32Optimizer;

impl Data for CudaSlice<usize> {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}
