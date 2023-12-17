use std::{
    any::{Any, TypeId},
    collections::{hash_map::DefaultHasher, HashMap},
    fmt::{Debug, Write},
    hash::{Hash, Hasher},
    sync::Arc,
};

mod fp32;
pub use fp32::*;
pub mod fp16;
pub use fp16::*;
mod common_buffer;
mod other;
mod prim;

use half::f16;
use itertools::Itertools;
use metal_rs::*;

use crate::{
    op::InputTensor,
    prelude::{
        symbolic::{BigExpression, Expression, Term},
        *,
    },
};

impl Data for Buffer {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

pub trait MetalFloat: Copy + 'static {
    fn to_f32(self) -> f32;
    fn from_f32(a: f32) -> Self;
    fn is_f32() -> bool;
    fn type_name() -> &'static str;
}

impl MetalFloat for f32 {
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

impl MetalFloat for f16 {
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
        "half"
    }
}

pub trait MetalKernelForward: Debug {
    fn metal_forward(
        &self,
        inputs: &[(&Buffer, ShapeTracker)],
        dev: &Device,
        command_buffer: &CommandBufferRef,
    ) -> Vec<Buffer>;
}

#[derive(LuminalEq, LuminalPrint, Clone)]
pub struct MetalKernelWrapper(pub Arc<Box<dyn MetalKernelForward>>);

impl Default for MetalKernelWrapper {
    fn default() -> Self {
        Self(Arc::new(Box::new(())))
    }
}

impl MetalKernelForward for () {
    fn metal_forward(
        &self,
        _: &[(&Buffer, ShapeTracker)],
        _: &Device,
        _: &CommandBufferRef,
    ) -> Vec<Buffer> {
        vec![]
    }
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

fn is<T: Any>(type_id: TypeId) -> bool {
    type_id == TypeId::of::<T>()
}

fn hash<T: Hash>(obj: T) -> u64 {
    let mut hasher = DefaultHasher::new();
    obj.hash(&mut hasher);
    hasher.finish()
}

trait DispatchNElements {
    fn dispatch_1d(&self, n: usize);
}

impl DispatchNElements for ComputeCommandEncoderRef {
    fn dispatch_1d(&self, n: usize) {
        self.dispatch_thread_groups(
            MTLSize {
                width: n.div_ceil(256) as u64,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 256,
                height: 1,
                depth: 1,
            },
        );
    }
}

trait SetInt {
    fn set_int(&self, index: usize, value: u32);
}

impl SetInt for ComputeCommandEncoderRef {
    fn set_int(&self, index: usize, value: u32) {
        self.set_bytes(
            index as u64,
            std::mem::size_of::<u32>() as u64,
            &value as *const u32 as *const _,
        );
    }
}

fn input_dyn_dims(
    shapes: &[ShapeTracker],
    dyn_map: &HashMap<char, usize>,
    encoder: &ComputeCommandEncoderRef,
    index: usize,
) {
    for (i, symb) in shapes
        .iter()
        .flat_map(|s| {
            s.dims
                .into_iter()
                .chain(s.padding.into_iter().flat_map(|i| [i.0, i.1]))
                .chain(s.slices.into_iter().flat_map(|i| [i.0, i.1]))
        })
        .flat_map(|e| e.to_symbols())
        .unique()
        .enumerate()
    {
        encoder.set_int(i + index, dyn_map[&symb] as u32);
    }
}

fn render_dyn_dim_inputs(shapes: &[ShapeTracker], offset: usize) -> String {
    shapes
        .iter()
        .flat_map(|st| {
            st.shape()
                .into_iter()
                .chain(
                    st.padding
                        .into_iter()
                        .flat_map(|i| [i.0.into(), i.1.into()]),
                )
                .chain(st.slices.into_iter().flat_map(|i| [i.0.into(), i.1.into()]))
        })
        .flat_map(|d| d.to_symbols())
        .unique()
        .enumerate()
        .fold(String::default(), |mut acc, (i, c)| {
            write!(&mut acc, ", device uint& {c} [[buffer({})]]", i + offset).unwrap();
            acc
        })
}

fn expr_to_metal_string(expr: BigExpression) -> String {
    let mut symbols = vec![];
    for term in expr.terms {
        let new_symbol = match term {
            Term::Num(n) => n.to_string(),
            Term::Var(c) => {
                if c == 'z' {
                    "idx".to_string()
                } else {
                    c.to_string()
                }
            }
            Term::Max => format!(
                "max((uint){}, (uint){})",
                symbols.pop().unwrap(),
                symbols.pop().unwrap()
            ),
            Term::Min => format!(
                "min((uint){}, (uint){})",
                symbols.pop().unwrap(),
                symbols.pop().unwrap()
            ),
            Term::Sub => format!(
                "(uint)max((int){} - (int){}, 0)",
                symbols.pop().unwrap(),
                symbols.pop().unwrap()
            ),
            _ => format!(
                "({}{term:?}{})",
                symbols.pop().unwrap(),
                symbols.pop().unwrap()
            ),
        };
        symbols.push(new_symbol);
    }
    symbols.pop().unwrap()
}

fn get_idx_valid_exps(shape: ShapeTracker) -> (String, String) {
    (
        expr_to_metal_string(shape.index_expression()),
        expr_to_metal_string(shape.valid_expression()),
    )
}

fn get_buffer_from_tensor<'a>(tensor: &'a InputTensor) -> &'a Buffer {
    tensor
        .borrowed()
        .data
        .as_any()
        .downcast_ref::<Buffer>()
        .unwrap()
}
