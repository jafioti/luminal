use std::{
    any::{Any, TypeId},
    collections::{hash_map::DefaultHasher, HashMap},
    fmt::{Debug, Write},
    hash::{Hash, Hasher},
    sync::Arc,
};

#[cfg(test)]
mod tests;

mod binary;
mod command_buffer;
mod elementwise_fusion;
mod matmul;
mod mean_reduce;
mod other;
mod prim;
mod std_norm;
mod storage_buffer;

use half::f16;
use itertools::Itertools;
use metal_rs::*;

use crate::{
    op::InputTensor,
    prelude::{
        symbolic::{BigExpression, Term},
        *,
    },
};

pub type MetalCompiler<T> = (
    prim::PrimitiveCompiler<T>,
    (
        binary::MetalSubtractionCompiler<T>,
        binary::MetalEqualCompiler<T>,
        other::ARangeCompiler<T>,
        binary::MetalGatherCompiler<T>,
    ),
    other::MetalExpCompiler<T>,
    matmul::MetalMatMulCompiler<T>,
    mean_reduce::MeanReduceCompiler<T>,
    std_norm::StdNormCompiler<T>,
    other::CopyCompiler<T>,
    other::ContiguousElimination<T>,
    elementwise_fusion::ElementwiseFusionCompiler<T>,
    (
        command_buffer::CommandBufferCompiler,
        storage_buffer::StorageBufferCompiler,
    ),
);

impl Data for Buffer {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

impl Data for Arc<Box<Buffer>> {
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

pub trait MetalKernel: Debug {
    /// Annotate the buffer sizes of the intermediate buffers
    fn intermediate_buffer_sizes(&self, _: &[ShapeTracker]) -> Vec<BigExpression> {
        vec![]
    }
    /// Annotate the buffer sizes of the output buffers
    fn output_buffer_sizes(&self, input_shapes: &[ShapeTracker]) -> Vec<BigExpression>;
    /// Set up the kernel on the buffer
    fn metal_forward(
        &self,
        inputs: &[(&Buffer, ShapeTracker)],
        command_buffer: &CommandBufferRef,
        intermediate_buffers: &[&Buffer],
        output_buffers: &[&Buffer],
    );
    fn without_command_buffer(
        &self,
        inputs: &[(&Buffer, ShapeTracker)],
        intermediate_buffers: &[&Buffer],
        output_buffers: &[&Buffer],
    ) {
        let dev = Device::system_default().unwrap();
        let queue = dev.new_command_queue();
        let command_buffer = queue.new_command_buffer();
        self.metal_forward(inputs, command_buffer, intermediate_buffers, output_buffers);
    }
    fn without_storage_buffers(
        &self,
        inputs: &[(&Buffer, ShapeTracker)],
        command_buffer: &CommandBufferRef,
        dyn_map: &HashMap<char, usize>,
    ) -> Vec<Buffer> {
        let dev = Device::system_default().unwrap();
        // Allocate storage buffers
        let inp_shapes = inputs.iter().map(|(_, s)| *s).collect::<Vec<_>>();
        let intermediate_buffers = self
            .intermediate_buffer_sizes(&inp_shapes)
            .into_iter()
            .map(|n| {
                dev.new_buffer(
                    n.exec(dyn_map).unwrap() as u64,
                    MTLResourceOptions::StorageModeShared,
                )
            })
            .collect::<Vec<_>>();
        let intermediate_buffers_ref = intermediate_buffers.iter().collect::<Vec<_>>();
        let output_buffers = self
            .output_buffer_sizes(&inp_shapes)
            .into_iter()
            .map(|n| {
                dev.new_buffer(
                    n.exec(dyn_map).unwrap() as u64,
                    MTLResourceOptions::StorageModeShared,
                )
            })
            .collect::<Vec<_>>();
        let output_buffers_ref = output_buffers.iter().collect::<Vec<_>>();
        self.metal_forward(
            inputs,
            command_buffer,
            &intermediate_buffers_ref,
            &output_buffers_ref,
        );
        output_buffers
    }
}

#[derive(LuminalPrint, Clone)]
pub struct MetalKernelWrapper(pub Arc<Box<dyn MetalKernel>>);

// TODO: This is like the worst thing in the world. Please don't do this.
// MetalKernelWrapper doesn't need an actual PartialEq if we can move CSE to beforehand, which requires much more robust compilers
impl PartialEq for MetalKernelWrapper {
    fn eq(&self, other: &Self) -> bool {
        format!("{self:?}") == format!("{other:?}")
    }
}

impl Default for MetalKernelWrapper {
    fn default() -> Self {
        Self(Arc::new(Box::new(())))
    }
}

impl MetalKernel for () {
    fn output_buffer_sizes(&self, _: &[ShapeTracker]) -> Vec<BigExpression> {
        vec![]
    }
    fn metal_forward(
        &self,
        _: &[(&Buffer, ShapeTracker)],
        _: &CommandBufferRef,
        _: &[&Buffer],
        _: &[&Buffer],
    ) {
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
    fn set_i32(&self, index: usize, value: i32);
    fn set_u32(&self, index: usize, value: u32);
    fn set_f32(&self, index: usize, value: f32);
    fn set_i64(&self, index: usize, value: i64);
    fn set_u64(&self, index: usize, value: u64);
    fn set_f64(&self, index: usize, value: f64);
}

impl SetInt for ComputeCommandEncoderRef {
    fn set_i32(&self, index: usize, value: i32) {
        self.set_bytes(
            index as u64,
            std::mem::size_of::<i32>() as u64,
            &value as *const i32 as *const _,
        );
    }
    fn set_u32(&self, index: usize, value: u32) {
        self.set_bytes(
            index as u64,
            std::mem::size_of::<u32>() as u64,
            &value as *const u32 as *const _,
        );
    }
    fn set_f32(&self, index: usize, value: f32) {
        self.set_bytes(
            index as u64,
            std::mem::size_of::<f32>() as u64,
            &value as *const f32 as *const _,
        );
    }
    fn set_i64(&self, index: usize, value: i64) {
        self.set_bytes(
            index as u64,
            std::mem::size_of::<i64>() as u64,
            &value as *const i64 as *const _,
        );
    }
    fn set_u64(&self, index: usize, value: u64) {
        self.set_bytes(
            index as u64,
            std::mem::size_of::<u64>() as u64,
            &value as *const u64 as *const _,
        );
    }
    fn set_f64(&self, index: usize, value: f64) {
        self.set_bytes(
            index as u64,
            std::mem::size_of::<f64>() as u64,
            &value as *const f64 as *const _,
        );
    }
}

fn input_dyn_dims(
    dyn_symbols: &[char],
    dyn_map: &HashMap<char, usize>,
    encoder: &ComputeCommandEncoderRef,
    index: usize,
) {
    for (i, s) in dyn_symbols.iter().enumerate() {
        encoder.set_u32(i + index, dyn_map[s] as u32);
    }
}

fn render_dyn_dim_inputs(shapes: &[ShapeTracker], offset: usize) -> (Vec<char>, String) {
    let symbols: Vec<char> = shapes
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
        .collect();
    (
        symbols.clone(),
        symbols
            .into_iter()
            .enumerate()
            .fold(String::default(), |mut acc, (i, c)| {
                write!(&mut acc, ", device int& {c} [[buffer({})]]", i + offset).unwrap();
                acc
            }),
    )
}

fn expr_to_metal_string(expr: BigExpression) -> String {
    let mut symbols = vec![];
    for term in expr.terms {
        let new_symbol = match term {
            Term::Num(n) => n.to_string(),
            Term::Var(c) => {
                if c == 'z' {
                    "(int)idx".to_string()
                } else {
                    c.to_string()
                }
            }
            Term::Max => format!(
                "max((int){}, (int){})",
                symbols.pop().unwrap(),
                symbols.pop().unwrap()
            ),
            Term::Min => format!(
                "min((int){}, (int){})",
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
        .expect("Tensor does not contain a metal buffer")
}
