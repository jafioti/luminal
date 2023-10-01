mod fp32;
pub use fp32::*;
mod fp16;
pub use fp16::*;
use metal_rs::Buffer;

use crate::prelude::Data;

impl Data for Buffer {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}
