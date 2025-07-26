pub mod codegen;
pub mod extract;
pub mod run;
pub mod translate;
pub mod utils;

#[cfg(test)]
mod tests;

use luminal::prelude::*;
use luminal_metal::MetalBuffer;
use metal_rs::{
    Buffer, CompileOptions, ComputePassDescriptor, ComputePipelineDescriptor, Device,
    MTLResourceOptions, MTLSize,
};
use std::{collections::HashMap, fmt::Debug, fs::File, io::Write};

#[derive(Clone, PartialEq, Eq)]
pub enum GPUArch {
    CUDA,
    Metal(HashMap<usize, &'static str>),
}

impl GPUArch {
    fn metal_buffer_type(&self, var: usize) -> &'static str {
        match self {
            Self::Metal(m) => m.get(&var).copied().unwrap_or(""),
            _ => "",
        }
    }

    fn add_metal_buffer_type(&mut self, var: usize, buf_type: &'static str) {
        if let Self::Metal(m) = self {
            m.insert(var, buf_type);
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Hash)]
pub struct Kernel {
    pub code: String,
    // launch params
    pub grid: (Expression, Expression, Expression),
    pub threadblock: (Expression, Expression, Expression),
    pub smem: Expression, // sizes of required shared memory buffers
    pub outputs: Vec<Expression>,
}

#[derive(Clone, Debug)]
pub enum GMEMBuffer {
    PrevKernel { kernel: usize, output: usize },
    Input { node: NodeIndex },
}

#[derive(Clone, Debug)]
pub enum GraphTerm {
    GMEM {
        // Signifies global memory
        label: Option<String>,
    },
    LoopIn {
        range: Expression,
        stride: Expression,
        marker: String,
    },
    LoopOut {
        range: Expression,
        stride: Expression,
        marker: String,
    },
    Add,
    Mul,
    Max,
    Exp2,
    Log2,
    Recip,
    Sin,
    Neg,
    Sqrt,
    LessThan,
    Mod,
    SMEM,     // Signifies shared memory
    SMEMLoad, // Takes in an smem pointer and a gmem pointer, copies the gmem element to smem and returns the smem pointer
    SMEMRead, // Takes in an smem pointer and an smemload, returns the smem pointer
    Custom(Kernel),
    Diff(String), // Diff a buffer
}

#[derive(Debug)]
pub struct CompatKernel(Kernel, *mut Graph);
impl Operator for CompatKernel {
    fn process(&mut self, inputs: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        let dyn_vars = &unsafe { self.1.as_ref().unwrap() }.dyn_map;
        let device = Device::system_default().unwrap();
        let queue = device.new_command_queue();
        let command_buffer = queue.new_command_buffer();
        let encoder =
            command_buffer.compute_command_encoder_with_descriptor(ComputePassDescriptor::new());
        let options = CompileOptions::new();
        // options.set_fast_math_enabled(true);
        let lib = device
            .new_library_with_source(&self.0.code, &options)
            .unwrap();
        let pipeline_state_descriptor = ComputePipelineDescriptor::new();
        pipeline_state_descriptor
            .set_compute_function(Some(&lib.get_function("kernel_name", None).unwrap()));
        let pipeline = device
            .new_compute_pipeline_state_with_function(
                pipeline_state_descriptor.compute_function().unwrap(),
            )
            .unwrap();
        encoder.set_compute_pipeline_state(&pipeline);

        // set inputs
        for (i, input) in inputs
            .iter()
            .map(|(b, _)| b.borrowed().downcast_ref::<MetalBuffer>().unwrap())
            .enumerate()
        {
            encoder.set_buffer(i as u64, Some(&input.0), 0);
        }
        // set output
        let mut buffers = vec![];
        for (i, size) in self.0.outputs.iter().enumerate() {
            let buff = vec![0.0_f32; size.exec(dyn_vars).unwrap()];
            buffers.push(device.new_buffer_with_data(
                buff.as_ptr() as *const _,
                (size.exec(dyn_vars).unwrap() * std::mem::size_of::<f32>()) as u64,
                MTLResourceOptions::StorageModeShared,
            ));
            encoder.set_buffer((i + inputs.len()) as u64, Some(buffers.last().unwrap()), 0);
        }
        // set smem
        if !self.0.smem.is_empty() {
            encoder.set_threadgroup_memory_length(
                0,
                (self.0.smem.exec(dyn_vars).unwrap() * std::mem::size_of::<f32>()) as u64,
            );
        }

        // Set dispatch
        encoder.dispatch_thread_groups(
            MTLSize::new(
                self.0.grid.0.exec(dyn_vars).unwrap() as u64,
                self.0.grid.1.exec(dyn_vars).unwrap() as u64,
                self.0.grid.2.exec(dyn_vars).unwrap() as u64,
            ),
            MTLSize::new(
                self.0.threadblock.0.exec(dyn_vars).unwrap() as u64,
                self.0.threadblock.1.exec(dyn_vars).unwrap() as u64,
                self.0.threadblock.2.exec(dyn_vars).unwrap() as u64,
            ),
        );
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
        buffers
            .into_iter()
            .map(|b| Tensor::new(MetalBuffer(b)))
            .collect()
    }
}

pub fn custom_kernel<const O: usize>(
    inputs: &[GraphTensor],
    kernel: Kernel,
    output_shapes: [impl ToShape; O],
    cx: &mut Graph,
) -> [GraphTensor; O] {
    let graph_ref: *mut Graph = cx;
    let mut kernel_op = cx.add_op(CompatKernel(kernel, graph_ref));
    for input in inputs {
        kernel_op = kernel_op.input(input.id, 0, input.shape);
    }
    let kernel_op = kernel_op.finish();
    let mut outputs = [GraphTensor::from_id(kernel_op, ShapeTracker::new(()), cx); O];
    for (i, output_shape) in output_shapes.into_iter().enumerate() {
        outputs[i].shape = ShapeTracker::new(output_shape);
    }
    outputs
}

#[derive(Debug)]
pub struct Diff {
    name: String,
}

impl Operator for Diff {
    fn process(&mut self, inp: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        // Dump
        let buffer = inp[0].0.borrowed().downcast_ref::<MetalBuffer>().unwrap();
        let mut data = vec![0.0; buffer.length() as usize / std::mem::size_of::<f32>()];
        let ptr = buffer.contents() as *mut f32;
        for (i, d) in data.iter_mut().enumerate() {
            *d = unsafe { *ptr.add(i) };
        }
        let mut file = File::create(format!("{}.bin", self.name)).unwrap();
        let bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(
                data.as_ptr() as *const u8,
                data.len() * std::mem::size_of::<f32>(),
            )
        };
        file.write_all(bytes).unwrap();
        vec![Tensor::new(buffer.clone())]
    }
}

pub trait GTDiff {
    fn diff2(self, name: impl ToString) -> Self;
}

impl GTDiff for GraphTensor {
    fn diff2(mut self, name: impl ToString) -> Self {
        if !self.shape.is_contiguous() {
            self = self.contiguous();
        }
        let id = self
            .graph()
            .add_op(Diff {
                name: name.to_string(),
            })
            .input(self.id, 0, self.shape)
            .finish();
        GraphTensor::from_id(id, self.shape, self.graph_ref)
    }
}
