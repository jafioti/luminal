use crate::{
    op::{InputTensor, Operator},
    prelude::*,
};
use metal_rs::*;

/// Copy a tensor to the GPU
#[derive(Debug, Clone)]
pub struct MetalCopyToDevice(Device);
impl PartialEq for MetalCopyToDevice {
    fn eq(&self, _: &Self) -> bool {
        false
    }
}

impl Operator for MetalCopyToDevice {
    fn process(&self, mut inp: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        if inp[0].0.borrowed().data.as_any().is::<Buffer>() {
            // Already on device
            return vec![inp.pop().unwrap().0.cloned()];
        }
        let data = inp[0]
            .0
            .borrowed()
            .data
            .as_any()
            .downcast_ref::<Vec<f32>>()
            .unwrap();
        let buffer = self.0.new_buffer_with_data(
            unsafe { std::mem::transmute(data.as_ptr()) },
            (data.len() * std::mem::size_of::<f32>()) as u64,
            MTLResourceOptions::CPUCacheModeDefaultCache,
        );
        vec![Tensor {
            data: Box::new(buffer),
        }]
    }
}

/// Copy a tensor from the GPU
#[derive(Debug, Clone)]
pub struct MetalCopyFromDevice(Device);
impl PartialEq for MetalCopyFromDevice {
    fn eq(&self, _: &Self) -> bool {
        false
    }
}

impl Operator for MetalCopyFromDevice {
    fn process(&self, mut inp: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        if inp[0].0.borrowed().data.as_any().is::<Vec<f32>>()
            || inp[0].0.borrowed().data.as_any().is::<Vec<usize>>()
        {
            // Already off device
            return vec![inp.pop().unwrap().0.cloned()];
        }
        let mut data = vec![0.0; inp[0].1.n_physical_elements()];
        let buffer = inp[0]
            .0
            .borrowed()
            .data
            .as_any()
            .downcast_ref::<Buffer>()
            .unwrap();
        let ptr = buffer.contents() as *mut f32;
        for i in 0..data.len() {
            data[i] = unsafe { *ptr.add(i) };
        }
        vec![Tensor {
            data: Box::new(data),
        }]
    }
}

pub struct PrimitiveOptimizer;

impl GraphOptimizer for PrimitiveOptimizer {
    fn optimize(&self, graph: &mut crate::prelude::Graph) {}
}

fn compile_function(name: &str, code: &str, device: &Device) -> ComputePipelineState {
    let library = device
        .new_library_with_source(code, &CompileOptions::new())
        .unwrap();
    let pipeline_state_descriptor = ComputePipelineDescriptor::new();
    pipeline_state_descriptor
        .set_compute_function(Some(&library.get_function(name, None).unwrap()));

    device
        .new_compute_pipeline_state_with_function(
            pipeline_state_descriptor.compute_function().unwrap(),
        )
        .unwrap()
}
