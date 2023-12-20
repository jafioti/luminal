mod fp16;
mod fp32;
mod prim;

pub use fp16::CudaFp16Compiler;
pub use fp32::CudaFp32Compiler;
use half::f16;
use itertools::Itertools;

use std::{
    collections::{hash_map::DefaultHasher, HashSet},
    fmt::{Debug, Write},
    hash::{Hash, Hasher},
    marker::PhantomData,
    mem::size_of,
    sync::Arc,
};

use cudarc::{
    driver::{CudaDevice, CudaFunction, CudaSlice, DeviceRepr, LaunchAsync, LaunchConfig},
    nvrtc::{compile_ptx_with_opts, CompileOptions},
};

use crate::{op::*, prelude::*};

pub trait CudaFloat {
    fn to_f32(self) -> f32;
    fn from_f32(a: f32) -> Self;
    fn is_f32() -> bool;
    fn type_name() -> &'static str;
}

impl CudaFloat for f32 {
    fn from_f32(a: f32) -> Self {
        a
    }
    fn to_f32(self) -> f32 {
        self
    }
    fn is_f32() -> bool {
        true
    }
    fn type_name() -> &'static str {
        "float"
    }
}

impl CudaFloat for f16 {
    fn from_f32(a: f32) -> Self {
        f16::from_f32(a)
    }
    fn to_f32(self) -> f32 {
        self.to_f32()
    }
    fn is_f32() -> bool {
        false
    }
    fn type_name() -> &'static str {
        "__half"
    }
}
