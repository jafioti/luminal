use crate::{op::Operator, prelude::*};

// Ops and optimizers specific to CUDA execution

pub type CudaOptimizer = (CudaPrimitiveOptimizer,);

/// Convert all primitive ops to cuda primitive ops, and insert copy to and from device ops
#[derive(Debug, Default)]
pub struct CudaPrimitiveOptimizer;

impl GraphOptimizer for CudaOptimizer {
    fn optimize(&self, graph: &mut Graph) {
        todo!()
    }
}

/// Copy a tensor to the GPU
#[derive(Debug)]
pub struct CudaCopyToDevice;

impl Operator for CudaCopyToDevice {
    fn name(&self) -> &'static str {
        "CudaCopyToDevice"
    }

    fn process(&self, inp: Vec<&Tensor>) -> Tensor {
        todo!()
    }
}
