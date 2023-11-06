use cudarc::driver::CudaSlice;

use crate::prelude::Data;

mod fp16;
mod fp32;

pub use fp16::CudaFp16Optimizer;
pub use fp32::CudaFp32Optimizer;
