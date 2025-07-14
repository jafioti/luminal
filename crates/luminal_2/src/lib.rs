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
    code: String,
    // launch params
    grid: (Expression, Expression, Expression),
    threadblock: (Expression, Expression, Expression),
    smem: Expression, // sizes of required shared memory buffers
    outputs: Vec<Expression>,
}

#[derive(Clone)]
pub enum GMEMBuffer {
    PrevKernel { kernel: usize, output: usize },
    Input { label: Option<String> },
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
    Exp,
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
