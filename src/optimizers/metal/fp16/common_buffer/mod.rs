use crate::prelude::*;

#[derive(Default)]
pub struct CommonBufferOptimizer;

impl GraphOptimizer for CommonBufferOptimizer {
    fn optimize(&self, graph: &mut crate::prelude::Graph) {
        // Look for successive metal kernels implementing MetalKernelForward
    }
}
