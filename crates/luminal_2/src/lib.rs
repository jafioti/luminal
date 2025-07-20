pub mod codegen;
pub mod extract;
pub mod run;
pub mod translate;
pub mod utils;

#[cfg(test)]
mod tests;

use luminal::prelude::*;
use serde::Serialize;
use std::{collections::HashMap, fmt::Debug};

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

#[derive(Clone, Debug, Serialize)]
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
}

impl Operator for Kernel {
    fn process(&mut self, _: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        unimplemented!("This shouldn't be ran directly. run_graph runs this!");
    }
}

pub fn custom_kernel<const O: usize>(
    inputs: &[GraphTensor],
    kernel: Kernel,
    output_shapes: [impl ToShape; O],
    cx: &mut Graph,
) -> [GraphTensor; O] {
    let mut kernel_op = cx.add_op(kernel);
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
