use std::{
    any::{Any, TypeId},
    collections::{hash_map::DefaultHasher, HashMap},
    fmt::{Debug, Write},
    hash::{Hash, Hasher},
    marker::PhantomData,
    sync::Arc,
};

mod fp32;
pub use fp32::*;
pub mod fp16;
pub use fp16::*;
use half::f16;
use itertools::Itertools;
use metal_rs::*;
use objc::rc::autoreleasepool;

use crate::{
    op::{InputTensor, Operator},
    prelude::{
        symbolic::{BigExpression, Term},
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

pub trait MetalFloat {
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
    let opts = CompileOptions::new();
    opts.set_fast_math_enabled(false);
    let library = device.new_library_with_source(code, &opts).unwrap();
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
                .chain(st.padding.into_iter().flat_map(|i| [i.0, i.1]))
                .chain(st.slices.into_iter().flat_map(|i| [i.0, i.1]))
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

/// Copy a tensor to the GPU
#[derive(LuminalEq, LuminalPrint, Clone)]
pub struct MetalCopyToDevice<T>(Device, PhantomData<T>);

impl<T> MetalCopyToDevice<T> {
    fn new(dev: Device) -> Self {
        Self(dev, Default::default())
    }
}

impl<T: MetalFloat + 'static> Operator for MetalCopyToDevice<T> {
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
            .unwrap()
            .iter()
            .copied()
            .map(MetalFloat::from_f32)
            .collect::<Vec<T>>();
        let buffer = self.0.new_buffer_with_data(
            unsafe { std::mem::transmute(data.as_ptr()) },
            (data.len() * std::mem::size_of::<T>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        vec![Tensor {
            data: Box::new(buffer),
        }]
    }
}

/// Copy a tensor from the GPU
#[derive(LuminalEq, LuminalPrint, Clone)]
pub struct MetalCopyFromDevice<T>(Device, PhantomData<T>);

impl<T> MetalCopyFromDevice<T> {
    fn new(dev: Device) -> Self {
        Self(dev, Default::default())
    }
}

impl<T: MetalFloat + 'static + Copy> Operator for MetalCopyFromDevice<T> {
    fn process(&self, mut inp: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        if inp[0].0.borrowed().data.as_any().is::<Vec<f32>>() {
            // Already off device
            return vec![inp.pop().unwrap().0.cloned()];
        }
        let buffer = get_buffer_from_tensor(&inp[0].0);
        let mut data = vec![0.0; buffer.length() as usize / std::mem::size_of::<T>()];
        if let MTLStorageMode::Managed = buffer.storage_mode() {
            buffer.did_modify_range(NSRange::new(0, buffer.length()));
        }
        let ptr = buffer.contents() as *mut T;
        for (i, d) in data.iter_mut().enumerate() {
            *d = unsafe { *ptr.add(i) }.to_f32();
        }

        vec![Tensor {
            data: Box::new(data),
        }]
    }
}

#[derive(LuminalEq, LuminalPrint, Clone)]
pub struct MetalConstant<T>(pub T, Device);

impl<T: MetalFloat + 'static> Operator for MetalConstant<T> {
    fn process(&self, _: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        vec![Tensor {
            data: Box::new(self.1.new_buffer_with_data(
                &self.0 as *const T as *const _,
                std::mem::size_of::<T>() as u64,
                MTLResourceOptions::StorageModeShared,
            )),
        }]
    }
}

#[derive(LuminalEq, LuminalPrint, Clone)]
pub struct MetalContiguous<T>(
    ComputePipelineState,
    Device,
    ShapeTracker,
    PhantomData<T>,
    *const HashMap<char, usize>,
);

impl<T: MetalFloat> MetalContiguous<T> {
    fn new(
        shape: ShapeTracker,
        dev: Device,
        kernels: &mut HashMap<String, ComputePipelineState>,
        dyn_map: *const HashMap<char, usize>,
    ) -> Self {
        let (idx_exp, valid_exp) = get_idx_valid_exps(shape);
        let mut code = format!("
#include <metal_stdlib>
using namespace metal;
kernel void mkernel(device {} *inp [[buffer(0)]], device {} *out [[buffer(1)]], device uint& n_elements [[buffer(2)]], uint idx [[thread_position_in_grid]]{}) {{
    if (idx < n_elements && ({valid_exp} != 0)) {{
        out[idx] = inp[{idx_exp}];
    }}
}}
", T::type_name(), T::type_name(), render_dyn_dim_inputs(&[shape], 3),
        );
        let name = format!("kernel_{}", hash(&code));
        code = code.replace("mkernel", &name);

        if !kernels.contains_key(&name) {
            kernels.insert(name.clone(), compile_function(&name, &code, &dev));
        }
        Self(
            kernels[&name].clone(),
            dev,
            shape,
            Default::default(),
            dyn_map,
        )
    }
}

impl<T> MetalKernelForward for MetalContiguous<T> {
    fn metal_forward(
        &self,
        inputs: &[(&Buffer, ShapeTracker)],
        dev: &Device,
        command_buffer: &CommandBufferRef,
    ) -> Vec<Buffer> {
        if inputs[0].1.is_contiguous() && !inputs[0].1.is_sliced() && !inputs[0].1.is_padded() {
            return vec![inputs[0].0.to_owned()];
        }
        let inp_size = inputs[0].1.contiguous().n_elements();
        let out = dev.new_buffer(
            (inp_size * std::mem::size_of::<T>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let encoder =
            command_buffer.compute_command_encoder_with_descriptor(ComputePassDescriptor::new());
        encoder.set_compute_pipeline_state(&self.0);

        // Set inputs
        encoder.set_buffer(0, Some(inputs[0].0), 0);
        encoder.set_buffer(1, Some(&out), 0);
        encoder.set_int(2, inp_size as u32);
        input_dyn_dims(&[self.2], unsafe { self.4.as_ref().unwrap() }, encoder, 3);

        // Execute
        encoder.dispatch_1d(inp_size);
        encoder.end_encoding();

        vec![out]
    }
}

impl<T: 'static + Copy> Operator for MetalContiguous<T> {
    fn process(&self, tensors: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        autoreleasepool(|| {
            // Setup command queue / command buffer / encoder
            let command_queue = self.1.new_command_queue();
            let command_buffer = command_queue.new_command_buffer();

            let out = self
                .metal_forward(
                    &[(get_buffer_from_tensor(&tensors[0].0), tensors[0].1)],
                    &self.1,
                    command_buffer,
                )
                .pop()
                .unwrap();

            command_buffer.commit();
            command_buffer.wait_until_completed();

            vec![Tensor {
                data: Box::new(out),
            }]
        })
    }

    fn custom(&self, key: &str) -> Option<Box<dyn Any>> {
        if key == "metal" {
            return Some(Box::new(MetalKernelWrapper(Arc::new(Box::new(
                self.clone(),
            )))));
        }
        None
    }
}

#[derive(LuminalEq, LuminalPrint, Clone)]
pub struct MetalLog2<T>(
    pub ComputePipelineState,
    CommandQueue,
    Device,
    PhantomData<T>,
);

impl<T: MetalFloat> MetalLog2<T> {
    fn new(
        dev: Device,
        queue: CommandQueue,
        kernels: &mut HashMap<String, ComputePipelineState>,
    ) -> Self {
        let mut code = format!("
#include <metal_stdlib>
using namespace metal;
kernel void mkernel(device {} *inp [[buffer(0)]], device {} *out [[buffer(1)]], device uint& n_elements [[buffer(2)]], uint idx [[thread_position_in_grid]]) {{
    if (idx < n_elements) {{
        out[idx] = log2(inp[idx]);
    }}
}}", T::type_name(), T::type_name());
        let name = format!("kernel_{}", hash(&code));
        code = code.replace("mkernel", &name);

        if !kernels.contains_key(&name) {
            kernels.insert(name.clone(), compile_function(&name, &code, &dev));
        }
        Self(kernels[&name].clone(), queue, dev, Default::default())
    }
}

impl<T> MetalKernelForward for MetalLog2<T> {
    fn metal_forward(
        &self,
        inputs: &[(&Buffer, ShapeTracker)],
        dev: &Device,
        command_buffer: &CommandBufferRef,
    ) -> Vec<Buffer> {
        let inp_size = inputs[0].1.n_physical_elements();
        let out = dev.new_buffer(
            (inp_size * std::mem::size_of::<T>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let encoder =
            command_buffer.compute_command_encoder_with_descriptor(ComputePassDescriptor::new());
        encoder.set_compute_pipeline_state(&self.0);

        // Set function inputs
        encoder.set_buffer(0, Some(inputs[0].0), 0);
        encoder.set_buffer(1, Some(&out), 0);
        encoder.set_int(2, inp_size as u32);

        // Execute
        encoder.dispatch_1d(inp_size);
        encoder.end_encoding();

        vec![out]
    }
}

impl<T: 'static + Copy> Operator for MetalLog2<T> {
    fn process(&self, tensors: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        autoreleasepool(|| {
            let command_buffer = self.1.new_command_buffer();

            let out = self
                .metal_forward(
                    &[(get_buffer_from_tensor(&tensors[0].0), tensors[0].1)],
                    &self.2,
                    command_buffer,
                )
                .pop()
                .unwrap();

            command_buffer.commit();
            command_buffer.wait_until_completed();

            vec![Tensor {
                data: Box::new(out),
            }]
        })
    }

    fn custom(&self, key: &str) -> Option<Box<dyn Any>> {
        if key == "metal" {
            return Some(Box::new(MetalKernelWrapper(Arc::new(Box::new(
                self.clone(),
            )))));
        }
        None
    }
}

#[derive(LuminalEq, LuminalPrint, Clone)]
pub struct MetalExp2<T>(
    pub ComputePipelineState,
    CommandQueue,
    Device,
    PhantomData<T>,
);

impl<T: MetalFloat> MetalExp2<T> {
    fn new(
        dev: Device,
        queue: CommandQueue,
        kernels: &mut HashMap<String, ComputePipelineState>,
    ) -> Self {
        let mut code = format!("
#include <metal_stdlib>
using namespace metal;
kernel void mkernel(device {} *inp [[buffer(0)]], device {} *out [[buffer(1)]], device uint& n_elements [[buffer(2)]], uint idx [[thread_position_in_grid]]) {{
    if (idx < n_elements) {{
        out[idx] = exp2(inp[idx]);
    }}
}}", T::type_name(), T::type_name());
        let name = format!("kernel_{}", hash(&code));
        code = code.replace("mkernel", &name);

        if !kernels.contains_key(&name) {
            kernels.insert(name.clone(), compile_function(&name, &code, &dev));
        }
        Self(kernels[&name].clone(), queue, dev, Default::default())
    }
}
impl<T> MetalKernelForward for MetalExp2<T> {
    fn metal_forward(
        &self,
        inputs: &[(&Buffer, ShapeTracker)],
        dev: &Device,
        command_buffer: &CommandBufferRef,
    ) -> Vec<Buffer> {
        let inp_size = inputs[0].1.n_physical_elements();
        let out = dev.new_buffer(
            (inp_size * std::mem::size_of::<T>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let encoder =
            command_buffer.compute_command_encoder_with_descriptor(ComputePassDescriptor::new());
        encoder.set_compute_pipeline_state(&self.0);

        // Set function inputs
        encoder.set_buffer(0, Some(inputs[0].0), 0);
        encoder.set_buffer(1, Some(&out), 0);
        encoder.set_int(2, inp_size as u32);

        // Execute
        encoder.dispatch_1d(inp_size);
        encoder.end_encoding();

        vec![out]
    }
}

impl<T: 'static + Copy> Operator for MetalExp2<T> {
    fn process(&self, tensors: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        autoreleasepool(|| {
            let command_buffer = self.1.new_command_buffer();

            let out = self
                .metal_forward(
                    &[(get_buffer_from_tensor(&tensors[0].0), tensors[0].1)],
                    &self.2,
                    command_buffer,
                )
                .pop()
                .unwrap();

            command_buffer.commit();
            command_buffer.wait_until_completed();

            vec![Tensor {
                data: Box::new(out),
            }]
        })
    }

    fn custom(&self, key: &str) -> Option<Box<dyn Any>> {
        if key == "metal" {
            return Some(Box::new(MetalKernelWrapper(Arc::new(Box::new(
                self.clone(),
            )))));
        }
        None
    }
}

#[derive(LuminalEq, LuminalPrint, Clone)]
pub struct MetalSin<T>(
    pub ComputePipelineState,
    CommandQueue,
    Device,
    PhantomData<T>,
);

impl<T: MetalFloat> MetalSin<T> {
    fn new(
        dev: Device,
        queue: CommandQueue,
        kernels: &mut HashMap<String, ComputePipelineState>,
    ) -> Self {
        let mut code = format!("#include <metal_stdlib>
using namespace metal;
kernel void mkernel(device {} *inp [[buffer(0)]], device {} *out [[buffer(1)]], device uint& n_elements [[buffer(2)]], uint idx [[thread_position_in_grid]]) {{
    if (idx < n_elements) {{
        out[idx] = ({})sin((float)inp[idx]);
    }}
}}", T::type_name(), T::type_name(), T::type_name());
        let name = format!("kernel_{}", hash(&code));
        code = code.replace("mkernel", &name);

        if !kernels.contains_key(&name) {
            kernels.insert(name.clone(), compile_function(&name, &code, &dev));
        }
        Self(kernels[&name].clone(), queue, dev, Default::default())
    }
}
impl<T> MetalKernelForward for MetalSin<T> {
    fn metal_forward(
        &self,
        inputs: &[(&Buffer, ShapeTracker)],
        dev: &Device,
        command_buffer: &CommandBufferRef,
    ) -> Vec<Buffer> {
        let inp_size = inputs[0].1.n_physical_elements();
        let out = dev.new_buffer(
            (inp_size * std::mem::size_of::<T>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let encoder =
            command_buffer.compute_command_encoder_with_descriptor(ComputePassDescriptor::new());
        encoder.set_compute_pipeline_state(&self.0);

        // Set function inputs
        encoder.set_buffer(0, Some(inputs[0].0), 0);
        encoder.set_buffer(1, Some(&out), 0);
        encoder.set_int(2, inp_size as u32);

        // Execute
        encoder.dispatch_1d(inp_size);
        encoder.end_encoding();

        vec![out]
    }
}

impl<T: 'static + Copy> Operator for MetalSin<T> {
    fn process(&self, tensors: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        autoreleasepool(|| {
            let command_buffer = self.1.new_command_buffer();

            let out = self
                .metal_forward(
                    &[(get_buffer_from_tensor(&tensors[0].0), tensors[0].1)],
                    &self.2,
                    command_buffer,
                )
                .pop()
                .unwrap();

            command_buffer.commit();
            command_buffer.wait_until_completed();

            vec![Tensor {
                data: Box::new(out),
            }]
        })
    }

    fn custom(&self, key: &str) -> Option<Box<dyn Any>> {
        if key == "metal" {
            return Some(Box::new(MetalKernelWrapper(Arc::new(Box::new(
                self.clone(),
            )))));
        }
        None
    }
}

#[derive(LuminalEq, LuminalPrint, Clone)]
pub struct MetalSqrt<T>(
    pub ComputePipelineState,
    CommandQueue,
    Device,
    PhantomData<T>,
);

impl<T: MetalFloat> MetalSqrt<T> {
    fn new(
        dev: Device,
        queue: CommandQueue,
        kernels: &mut HashMap<String, ComputePipelineState>,
    ) -> Self {
        let mut code = format!("#include <metal_stdlib>
using namespace metal;
kernel void mkernel(device {} *inp [[buffer(0)]], device {} *out [[buffer(1)]], device uint& n_elements [[buffer(2)]], uint idx [[thread_position_in_grid]]) {{
    if (idx < n_elements) {{
        out[idx] = sqrt(inp[idx]);
    }}
}}", T::type_name(), T::type_name());
        let name = format!("kernel_{}", hash(&code));
        code = code.replace("mkernel", &name);

        if !kernels.contains_key(&name) {
            kernels.insert(name.clone(), compile_function(&name, &code, &dev));
        }
        Self(kernels[&name].clone(), queue, dev, Default::default())
    }
}
impl<T> MetalKernelForward for MetalSqrt<T> {
    fn metal_forward(
        &self,
        inputs: &[(&Buffer, ShapeTracker)],
        dev: &Device,
        command_buffer: &CommandBufferRef,
    ) -> Vec<Buffer> {
        let inp_size = inputs[0].1.n_physical_elements();
        let out = dev.new_buffer(
            (inp_size * std::mem::size_of::<T>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let encoder =
            command_buffer.compute_command_encoder_with_descriptor(ComputePassDescriptor::new());
        encoder.set_compute_pipeline_state(&self.0);

        // Set function inputs
        encoder.set_buffer(0, Some(inputs[0].0), 0);
        encoder.set_buffer(1, Some(&out), 0);
        encoder.set_int(2, inp_size as u32);

        // Execute
        encoder.dispatch_1d(inp_size);
        encoder.end_encoding();

        vec![out]
    }
}

impl<T: 'static + Copy> Operator for MetalSqrt<T> {
    fn process(&self, tensors: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        autoreleasepool(|| {
            let command_buffer = self.1.new_command_buffer();

            let out = self
                .metal_forward(
                    &[(get_buffer_from_tensor(&tensors[0].0), tensors[0].1)],
                    &self.2,
                    command_buffer,
                )
                .pop()
                .unwrap();

            command_buffer.commit();
            command_buffer.wait_until_completed();

            vec![Tensor {
                data: Box::new(out),
            }]
        })
    }

    fn custom(&self, key: &str) -> Option<Box<dyn Any>> {
        if key == "metal" {
            return Some(Box::new(MetalKernelWrapper(Arc::new(Box::new(
                self.clone(),
            )))));
        }
        None
    }
}

#[derive(LuminalEq, LuminalPrint, Clone)]
pub struct MetalRecip<T>(
    pub ComputePipelineState,
    CommandQueue,
    Device,
    PhantomData<T>,
);

impl<T: MetalFloat> MetalRecip<T> {
    fn new(
        dev: Device,
        queue: CommandQueue,
        kernels: &mut HashMap<String, ComputePipelineState>,
    ) -> Self {
        let mut code = format!("#include <metal_stdlib>
using namespace metal;
kernel void mkernel(device {} *inp [[buffer(0)]], device {} *out [[buffer(1)]], device uint& n_elements [[buffer(2)]], uint idx [[thread_position_in_grid]]) {{
    if (idx < n_elements) {{
        out[idx] = 1.0 / inp[idx];
    }}
}}", T::type_name(), T::type_name());
        let name = format!("kernel_{}", hash(&code));
        code = code.replace("mkernel", &name);

        if !kernels.contains_key(&name) {
            kernels.insert(name.clone(), compile_function(&name, &code, &dev));
        }
        Self(kernels[&name].clone(), queue, dev, Default::default())
    }
}
impl<T> MetalKernelForward for MetalRecip<T> {
    fn metal_forward(
        &self,
        inputs: &[(&Buffer, ShapeTracker)],
        dev: &Device,
        command_buffer: &CommandBufferRef,
    ) -> Vec<Buffer> {
        let inp_size = inputs[0].1.n_physical_elements();
        let out = dev.new_buffer(
            (inp_size * std::mem::size_of::<T>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let encoder =
            command_buffer.compute_command_encoder_with_descriptor(ComputePassDescriptor::new());
        encoder.set_compute_pipeline_state(&self.0);

        // Set function inputs
        encoder.set_buffer(0, Some(inputs[0].0), 0);
        encoder.set_buffer(1, Some(&out), 0);
        encoder.set_int(2, inp_size as u32);

        // Execute
        encoder.dispatch_1d(inp_size);
        encoder.end_encoding();

        vec![out]
    }
}

impl<T: 'static + Copy> Operator for MetalRecip<T> {
    fn process(&self, tensors: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        autoreleasepool(|| {
            let command_buffer = self.1.new_command_buffer();

            let out = self
                .metal_forward(
                    &[(get_buffer_from_tensor(&tensors[0].0), tensors[0].1)],
                    &self.2,
                    command_buffer,
                )
                .pop()
                .unwrap();

            command_buffer.commit();
            command_buffer.wait_until_completed();

            vec![Tensor {
                data: Box::new(out),
            }]
        })
    }

    fn custom(&self, key: &str) -> Option<Box<dyn Any>> {
        if key == "metal" {
            return Some(Box::new(MetalKernelWrapper(Arc::new(Box::new(
                self.clone(),
            )))));
        }
        None
    }
}

#[derive(LuminalEq, LuminalPrint, Clone)]
pub struct MetalAdd<T>(
    ComputePipelineState,
    CommandQueue,
    Device,
    ShapeTracker,
    ShapeTracker,
    PhantomData<T>,
    *const HashMap<char, usize>,
);

impl<T: MetalFloat> MetalAdd<T> {
    fn new(
        a_shape: ShapeTracker,
        b_shape: ShapeTracker,
        dev: Device,
        queue: CommandQueue,
        kernels: &mut HashMap<String, ComputePipelineState>,
        dyn_map: *const HashMap<char, usize>,
    ) -> Self {
        let (a_idx_exp, a_valid_exp) = get_idx_valid_exps(a_shape);
        let (b_idx_exp, b_valid_exp) = get_idx_valid_exps(b_shape);
        let mut code = format!(
            "
#include <metal_stdlib>
using namespace metal;
kernel void mkernel(device {} *inp_a [[buffer(0)]], device {} *inp_b [[buffer(1)]], device {} *out [[buffer(2)]], device uint& n_elements [[buffer(3)]], uint idx [[thread_position_in_grid]]{}) {{
    if (idx < n_elements) {{
        out[idx] = 
            (({a_valid_exp}) == 0 ? 0.0h : inp_a[{a_idx_exp}]) 
            + (({b_valid_exp}) == 0 ? 0.0h : inp_b[{b_idx_exp}]);
    }}
}}
", T::type_name(), T::type_name(), T::type_name(), render_dyn_dim_inputs(&[a_shape, b_shape], 4),
        );
        let name = format!("kernel_{}", hash(&code));
        code = code.replace("mkernel", &name);

        if !kernels.contains_key(&name) {
            kernels.insert(name.clone(), compile_function(&name, &code, &dev));
        }
        Self(
            kernels[&name].clone(),
            queue,
            dev,
            a_shape,
            b_shape,
            Default::default(),
            dyn_map,
        )
    }
}

impl<T> MetalKernelForward for MetalAdd<T> {
    fn metal_forward(
        &self,
        inputs: &[(&Buffer, ShapeTracker)],
        dev: &Device,
        command_buffer: &CommandBufferRef,
    ) -> Vec<Buffer> {
        let inp_size = inputs[0].1.n_elements();
        let out = dev.new_buffer(
            (inp_size * std::mem::size_of::<T>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let encoder =
            command_buffer.compute_command_encoder_with_descriptor(ComputePassDescriptor::new());
        encoder.set_compute_pipeline_state(&self.0);

        // Set inputs
        encoder.set_buffer(0, Some(inputs[0].0), 0);
        encoder.set_buffer(1, Some(inputs[1].0), 0);
        encoder.set_buffer(2, Some(&out), 0);
        encoder.set_int(3, inp_size as u32);
        input_dyn_dims(
            &[self.3, self.4],
            unsafe { self.6.as_ref().unwrap() },
            encoder,
            4,
        );

        // Execute
        encoder.dispatch_1d(inp_size);
        encoder.end_encoding();

        vec![out]
    }
}

impl<T: 'static + Copy> Operator for MetalAdd<T> {
    fn process(&self, tensors: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        autoreleasepool(|| {
            let command_buffer = self.1.new_command_buffer();

            let out = self
                .metal_forward(
                    &[
                        (get_buffer_from_tensor(&tensors[0].0), tensors[0].1),
                        (get_buffer_from_tensor(&tensors[1].0), tensors[1].1),
                    ],
                    &self.2,
                    command_buffer,
                )
                .pop()
                .unwrap();

            command_buffer.commit();
            command_buffer.wait_until_completed();

            vec![Tensor {
                data: Box::new(out),
            }]
        })
    }

    fn custom(&self, key: &str) -> Option<Box<dyn Any>> {
        if key == "metal" {
            return Some(Box::new(MetalKernelWrapper(Arc::new(Box::new(
                self.clone(),
            )))));
        }
        None
    }
}

#[derive(LuminalEq, LuminalPrint, Clone)]
pub struct MetalMul<T>(
    ComputePipelineState,
    CommandQueue,
    Device,
    ShapeTracker,
    ShapeTracker,
    PhantomData<T>,
    *const HashMap<char, usize>,
);

impl<T: MetalFloat> MetalMul<T> {
    fn new(
        a_shape: ShapeTracker,
        b_shape: ShapeTracker,
        dev: Device,
        queue: CommandQueue,
        kernels: &mut HashMap<String, ComputePipelineState>,
        dyn_map: *const HashMap<char, usize>,
    ) -> Self {
        let (a_idx_exp, a_valid_exp) = get_idx_valid_exps(a_shape);
        let (b_idx_exp, b_valid_exp) = get_idx_valid_exps(b_shape);
        let mut code = format!(
            "
#include <metal_stdlib>
using namespace metal;
kernel void mkernel(device {} *inp_a [[buffer(0)]], device {} *inp_b [[buffer(1)]], device {} *out [[buffer(2)]], device uint& n_elements [[buffer(3)]], uint idx [[thread_position_in_grid]]{}) {{
    if (idx < n_elements) {{
        out[idx] = 
            (({a_valid_exp}) == 0 ? 0.0h : inp_a[{a_idx_exp}]) 
            * (({b_valid_exp}) == 0 ? 0.0h : inp_b[{b_idx_exp}]);
    }}
}}
", T::type_name(), T::type_name(), T::type_name(), render_dyn_dim_inputs(&[a_shape, b_shape], 4),
        );
        let name = format!("kernel_{}", hash(&code));
        code = code.replace("mkernel", &name);

        if !kernels.contains_key(&name) {
            kernels.insert(name.clone(), compile_function(&name, &code, &dev));
        }
        Self(
            kernels[&name].clone(),
            queue,
            dev,
            a_shape,
            b_shape,
            Default::default(),
            dyn_map,
        )
    }
}
impl<T> MetalKernelForward for MetalMul<T> {
    fn metal_forward(
        &self,
        inputs: &[(&Buffer, ShapeTracker)],
        dev: &Device,
        command_buffer: &CommandBufferRef,
    ) -> Vec<Buffer> {
        let inp_size = inputs[0].1.n_elements();
        let out = dev.new_buffer(
            (inp_size * std::mem::size_of::<T>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let encoder =
            command_buffer.compute_command_encoder_with_descriptor(ComputePassDescriptor::new());
        encoder.set_compute_pipeline_state(&self.0);

        // Set inputs
        encoder.set_buffer(0, Some(inputs[0].0), 0);
        encoder.set_buffer(1, Some(inputs[1].0), 0);
        encoder.set_buffer(2, Some(&out), 0);
        encoder.set_int(3, inp_size as u32);
        input_dyn_dims(
            &[self.3, self.4],
            unsafe { self.6.as_ref().unwrap() },
            encoder,
            4,
        );

        // Execute
        encoder.dispatch_1d(inp_size);
        encoder.end_encoding();

        vec![out]
    }
}

impl<T: 'static + Copy> Operator for MetalMul<T> {
    fn process(&self, tensors: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        autoreleasepool(|| {
            let command_buffer = self.1.new_command_buffer();

            let out = self
                .metal_forward(
                    &[
                        (get_buffer_from_tensor(&tensors[0].0), tensors[0].1),
                        (get_buffer_from_tensor(&tensors[1].0), tensors[1].1),
                    ],
                    &self.2,
                    command_buffer,
                )
                .pop()
                .unwrap();

            command_buffer.commit();
            command_buffer.wait_until_completed();

            vec![Tensor {
                data: Box::new(out),
            }]
        })
    }

    fn custom(&self, key: &str) -> Option<Box<dyn Any>> {
        if key == "metal" {
            return Some(Box::new(MetalKernelWrapper(Arc::new(Box::new(
                self.clone(),
            )))));
        }
        None
    }
}

#[derive(LuminalEq, LuminalPrint, Clone)]
pub struct MetalLessThan<T>(
    ComputePipelineState,
    CommandQueue,
    Device,
    ShapeTracker,
    ShapeTracker,
    PhantomData<T>,
    *const HashMap<char, usize>,
);

impl<T: MetalFloat> MetalLessThan<T> {
    fn new(
        a_shape: ShapeTracker,
        b_shape: ShapeTracker,
        dev: Device,
        queue: CommandQueue,
        kernels: &mut HashMap<String, ComputePipelineState>,
        dyn_map: *const HashMap<char, usize>,
    ) -> Self {
        let (a_idx_exp, a_valid_exp) = get_idx_valid_exps(a_shape);
        let (b_idx_exp, b_valid_exp) = get_idx_valid_exps(b_shape);
        let type_name = T::type_name();
        let mut code = format!(
            "
#include <metal_stdlib>
using namespace metal;
kernel void mkernel(device {type_name} *inp_a [[buffer(0)]], device {type_name} *inp_b [[buffer(1)]], device {type_name} *out [[buffer(2)]], device uint& n_elements [[buffer(3)]], uint idx [[thread_position_in_grid]]{}) {{
    if (idx < n_elements) {{
        {type_name} a_t = 0.0h;
        {type_name} b_t = 0.0h;
        if (({a_valid_exp}) != 0) {{
            a_t = inp_a[{a_idx_exp}];
        }}
        if (({b_valid_exp}) != 0) {{
            b_t = inp_b[{b_idx_exp}];
        }}
        if (a_t < b_t) {{
            out[idx] = {};
        }} else {{
            out[idx] = {};
        }}
    }}
}}
", render_dyn_dim_inputs(&[a_shape, b_shape], 4), if T::is_f32() {"1.0"} else {"1.0h"},if T::is_f32() {"0.0"} else {"0.0h"},
        );
        let name = format!("kernel_{}", hash(&code));
        code = code.replace("mkernel", &name);

        if !kernels.contains_key(&name) {
            kernels.insert(name.clone(), compile_function(&name, &code, &dev));
        }
        Self(
            kernels[&name].clone(),
            queue,
            dev,
            a_shape,
            b_shape,
            Default::default(),
            dyn_map,
        )
    }
}

impl<T> MetalKernelForward for MetalLessThan<T> {
    fn metal_forward(
        &self,
        inputs: &[(&Buffer, ShapeTracker)],
        dev: &Device,
        command_buffer: &CommandBufferRef,
    ) -> Vec<Buffer> {
        let inp_size = inputs[0].1.n_elements();
        let out = dev.new_buffer(
            (inp_size * std::mem::size_of::<T>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let encoder =
            command_buffer.compute_command_encoder_with_descriptor(ComputePassDescriptor::new());
        encoder.set_compute_pipeline_state(&self.0);

        // Set inputs
        encoder.set_buffer(0, Some(inputs[0].0), 0);
        encoder.set_buffer(1, Some(inputs[1].0), 0);
        encoder.set_buffer(2, Some(&out), 0);
        encoder.set_int(3, inp_size as u32);
        input_dyn_dims(
            &[self.3, self.4],
            unsafe { self.6.as_ref().unwrap() },
            encoder,
            4,
        );

        // Execute
        encoder.dispatch_1d(inp_size);
        encoder.end_encoding();

        vec![out]
    }
}

impl<T: 'static + Copy> Operator for MetalLessThan<T> {
    fn process(&self, tensors: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        autoreleasepool(|| {
            let command_buffer = self.1.new_command_buffer();

            let out = self
                .metal_forward(
                    &[
                        (get_buffer_from_tensor(&tensors[0].0), tensors[0].1),
                        (get_buffer_from_tensor(&tensors[1].0), tensors[1].1),
                    ],
                    &self.2,
                    command_buffer,
                )
                .pop()
                .unwrap();

            command_buffer.commit();
            command_buffer.wait_until_completed();

            vec![Tensor {
                data: Box::new(out),
            }]
        })
    }

    fn custom(&self, key: &str) -> Option<Box<dyn Any>> {
        if key == "metal" {
            return Some(Box::new(MetalKernelWrapper(Arc::new(Box::new(
                self.clone(),
            )))));
        }
        None
    }
}

#[derive(LuminalEq, LuminalPrint, Clone)]
pub struct MetalMod<T>(
    ComputePipelineState,
    CommandQueue,
    Device,
    ShapeTracker,
    ShapeTracker,
    PhantomData<T>,
    *const HashMap<char, usize>,
);

impl<T: MetalFloat> MetalMod<T> {
    fn new(
        a_shape: ShapeTracker,
        b_shape: ShapeTracker,
        dev: Device,
        queue: CommandQueue,
        kernels: &mut HashMap<String, ComputePipelineState>,
        dyn_map: *const HashMap<char, usize>,
    ) -> Self {
        let (a_idx_exp, a_valid_exp) = get_idx_valid_exps(a_shape);
        let (b_idx_exp, b_valid_exp) = get_idx_valid_exps(b_shape);
        let mut code = format!(
            "
#include <metal_stdlib>
using namespace metal;
kernel void mkernel(device {} *inp_a [[buffer(0)]], device {} *inp_b [[buffer(1)]], device {} *out [[buffer(2)]], device uint& n_elements [[buffer(3)]], uint idx [[thread_position_in_grid]]{}) {{
    if (idx < n_elements) {{
        out[idx] = fmod(({a_valid_exp}) == 0 ? 0.0 : inp_a[{a_idx_exp}], ({b_valid_exp}) == 0 ? 0.0 : inp_b[{b_idx_exp}]);
    }}
}}
", T::type_name(), T::type_name(), T::type_name(), render_dyn_dim_inputs(&[a_shape, b_shape], 4),
        );
        let name = format!("kernel_{}", hash(&code));
        code = code.replace("mkernel", &name);

        if !kernels.contains_key(&name) {
            kernels.insert(name.clone(), compile_function(&name, &code, &dev));
        }
        Self(
            kernels[&name].clone(),
            queue,
            dev,
            a_shape,
            b_shape,
            Default::default(),
            dyn_map,
        )
    }
}
impl<T> MetalKernelForward for MetalMod<T> {
    fn metal_forward(
        &self,
        inputs: &[(&Buffer, ShapeTracker)],
        dev: &Device,
        command_buffer: &CommandBufferRef,
    ) -> Vec<Buffer> {
        let inp_size = inputs[0].1.n_elements();
        let out = dev.new_buffer(
            (inp_size * std::mem::size_of::<T>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let encoder =
            command_buffer.compute_command_encoder_with_descriptor(ComputePassDescriptor::new());
        encoder.set_compute_pipeline_state(&self.0);

        // Set inputs
        encoder.set_buffer(0, Some(inputs[0].0), 0);
        encoder.set_buffer(1, Some(inputs[1].0), 0);
        encoder.set_buffer(2, Some(&out), 0);
        encoder.set_int(3, inp_size as u32);
        input_dyn_dims(
            &[self.3, self.4],
            unsafe { self.6.as_ref().unwrap() },
            encoder,
            4,
        );

        // Execute
        encoder.dispatch_1d(inp_size);
        encoder.end_encoding();

        vec![out]
    }
}

impl<T: 'static + Copy> Operator for MetalMod<T> {
    fn process(&self, tensors: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        autoreleasepool(|| {
            let command_buffer = self.1.new_command_buffer();

            let out = self
                .metal_forward(
                    &[
                        (get_buffer_from_tensor(&tensors[0].0), tensors[0].1),
                        (get_buffer_from_tensor(&tensors[1].0), tensors[1].1),
                    ],
                    &self.2,
                    command_buffer,
                )
                .pop()
                .unwrap();

            command_buffer.commit();
            command_buffer.wait_until_completed();

            vec![Tensor {
                data: Box::new(out),
            }]
        })
    }

    fn custom(&self, key: &str) -> Option<Box<dyn Any>> {
        if key == "metal" {
            return Some(Box::new(MetalKernelWrapper(Arc::new(Box::new(
                self.clone(),
            )))));
        }
        None
    }
}

#[derive(LuminalEq, LuminalPrint, Clone)]
pub struct MetalSumReduce<T>(
    ComputePipelineState,
    CommandQueue,
    Device,
    pub usize,
    ShapeTracker,
    PhantomData<T>,
    *const HashMap<char, usize>,
);

impl<T: MetalFloat> MetalSumReduce<T> {
    fn new(
        shape: ShapeTracker,
        dim: usize,
        dev: Device,
        queue: CommandQueue,
        kernels: &mut HashMap<String, ComputePipelineState>,
        dyn_map: *const HashMap<char, usize>,
    ) -> Self {
        let (idx_exp, valid_exp) = get_idx_valid_exps(shape);
        let mut code = format!(
            "
#include <metal_stdlib>
using namespace metal;
kernel void mkernel(device {} *inp [[buffer(0)]], device {} *out [[buffer(1)]], device uint& n_elements [[buffer(2)]], device uint& front_size [[buffer(3)]], device uint& back_size [[buffer(4)]], device uint& dim_size [[buffer(5)]], uint i_ [[thread_position_in_grid]]{}) {{
    if (i_ < n_elements) {{
        uint a_ = i_ / back_size;
        uint b_ = i_ % back_size;
        {} reduce_value = 0.0;
        for (uint c_ = 0; c_ < dim_size; c_++) {{
            uint idx = a_ * dim_size * back_size + c_ * back_size + b_;
            if (({valid_exp}) != 0) {{
                reduce_value += inp[{idx_exp}];
            }}
        }}
        out[i_] = reduce_value;
    }}
}}
", T::type_name(), T::type_name(), render_dyn_dim_inputs(&[shape], 6), T::type_name(),
        );
        let name = format!("kernel_{}", hash(&code));
        code = code.replace("mkernel", &name);

        if !kernels.contains_key(&name) {
            kernels.insert(name.clone(), compile_function(&name, &code, &dev));
        }
        Self(
            kernels[&name].clone(),
            queue,
            dev,
            dim,
            shape,
            Default::default(),
            dyn_map,
        )
    }
}

impl<T> MetalKernelForward for MetalSumReduce<T> {
    fn metal_forward(
        &self,
        inputs: &[(&Buffer, ShapeTracker)],
        dev: &Device,
        command_buffer: &CommandBufferRef,
    ) -> Vec<Buffer> {
        let mut sh = inputs[0].1;
        sh.remove_dim(self.3);
        let inp_size = sh.contiguous().n_elements();

        let out = dev.new_buffer(
            (inp_size * std::mem::size_of::<T>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let front_size: usize = inputs[0]
            .1
            .shape()
            .iter()
            .take(self.3)
            .map(|i| i.to_usize().unwrap())
            .product();
        let back_size: usize = inputs[0]
            .1
            .shape()
            .iter()
            .skip(self.3 + 1)
            .map(|i| i.to_usize().unwrap())
            .product();
        let dim_size = inputs[0].1.shape()[self.3].to_usize().unwrap();

        let encoder =
            command_buffer.compute_command_encoder_with_descriptor(ComputePassDescriptor::new());
        encoder.set_compute_pipeline_state(&self.0);

        // Set inputs
        encoder.set_buffer(0, Some(inputs[0].0), 0);
        encoder.set_buffer(1, Some(&out), 0);
        encoder.set_int(2, inp_size as u32);
        encoder.set_int(3, front_size as u32);
        encoder.set_int(4, back_size as u32);
        encoder.set_int(5, dim_size as u32);
        input_dyn_dims(&[self.4], unsafe { self.6.as_ref().unwrap() }, encoder, 6);

        // Execute
        encoder.dispatch_1d(inp_size);
        encoder.end_encoding();

        vec![out]
    }
}

impl<T: 'static + Copy> Operator for MetalSumReduce<T> {
    fn process(&self, tensors: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        autoreleasepool(|| {
            // Setup command queue / command buffer / encoder
            let command_buffer = self.1.new_command_buffer();

            let out = self
                .metal_forward(
                    &[(get_buffer_from_tensor(&tensors[0].0), tensors[0].1)],
                    &self.2,
                    command_buffer,
                )
                .pop()
                .unwrap();

            command_buffer.commit();
            command_buffer.wait_until_completed();

            vec![Tensor {
                data: Box::new(out),
            }]
        })
    }

    fn custom(&self, key: &str) -> Option<Box<dyn Any>> {
        if key == "metal" {
            return Some(Box::new(MetalKernelWrapper(Arc::new(Box::new(
                self.clone(),
            )))));
        }
        None
    }
}

#[derive(LuminalEq, LuminalPrint, Clone)]
pub struct MetalMaxReduce<T>(
    ComputePipelineState,
    CommandQueue,
    Device,
    usize,
    ShapeTracker,
    PhantomData<T>,
    *const HashMap<char, usize>,
);

impl<T: MetalFloat> MetalMaxReduce<T> {
    fn new(
        shape: ShapeTracker,
        dim: usize,
        dev: Device,
        queue: CommandQueue,
        kernels: &mut HashMap<String, ComputePipelineState>,
        dyn_map: *const HashMap<char, usize>,
    ) -> Self {
        let (idx_exp, valid_exp) = get_idx_valid_exps(shape);
        let mut code = format!(
            "
#include <metal_stdlib>
using namespace metal;
kernel void mkernel(device {} *inp [[buffer(0)]], device {} *out [[buffer(1)]], device uint& n_elements [[buffer(2)]], device uint& front_size [[buffer(3)]], device uint& back_size [[buffer(4)]], device uint& dim_size [[buffer(5)]], uint i_ [[thread_position_in_grid]]{}) {{
    if (i_ < n_elements) {{
        uint a_ = i_ / back_size;
        uint b_ = i_ % back_size;
        {} reduce_value = -{};
        for (uint c_ = 0; c_ < dim_size; c_++) {{
            uint idx = a_ * dim_size * back_size + c_ * back_size + b_;
            if (({valid_exp}) != 0) {{
                int a_idx = {idx_exp};
                reduce_value = max(reduce_value, inp[a_idx]);
            }}
        }}
        out[i_] = reduce_value;
    }}
}}
", T::type_name(), T::type_name(), render_dyn_dim_inputs(&[shape], 6), T::type_name(), if T::is_f32() {"(float)0x7f800000"} else {"MAXHALF"},
        );
        let name = format!("kernel_{}", hash(&code));
        code = code.replace("mkernel", &name);

        if !kernels.contains_key(&name) {
            kernels.insert(name.clone(), compile_function(&name, &code, &dev));
        }
        Self(
            kernels[&name].clone(),
            queue,
            dev,
            dim,
            shape,
            Default::default(),
            dyn_map,
        )
    }
}
impl<T> MetalKernelForward for MetalMaxReduce<T> {
    fn metal_forward(
        &self,
        inputs: &[(&Buffer, ShapeTracker)],
        dev: &Device,
        command_buffer: &CommandBufferRef,
    ) -> Vec<Buffer> {
        let mut sh = inputs[0].1;
        sh.remove_dim(self.3);
        let inp_size = sh.contiguous().n_elements();

        let out = dev.new_buffer(
            (inp_size * std::mem::size_of::<T>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let front_size: usize = inputs[0]
            .1
            .shape()
            .iter()
            .take(self.3)
            .map(|i| i.to_usize().unwrap())
            .product();
        let back_size: usize = inputs[0]
            .1
            .shape()
            .iter()
            .skip(self.3 + 1)
            .map(|i| i.to_usize().unwrap())
            .product();
        let dim_size = inputs[0].1.shape()[self.3].to_usize().unwrap();

        let encoder =
            command_buffer.compute_command_encoder_with_descriptor(ComputePassDescriptor::new());
        encoder.set_compute_pipeline_state(&self.0);

        // Set inputs
        encoder.set_buffer(0, Some(inputs[0].0), 0);
        encoder.set_buffer(1, Some(&out), 0);
        encoder.set_int(2, inp_size as u32);
        encoder.set_int(3, front_size as u32);
        encoder.set_int(4, back_size as u32);
        encoder.set_int(5, dim_size as u32);
        input_dyn_dims(&[self.4], unsafe { self.6.as_ref().unwrap() }, encoder, 6);

        // Execute
        encoder.dispatch_1d(inp_size);
        encoder.end_encoding();

        vec![out]
    }
}

impl<T: 'static + Copy> Operator for MetalMaxReduce<T> {
    fn process(&self, tensors: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        autoreleasepool(|| {
            let a = tensors[0]
                .0
                .borrowed()
                .data
                .as_any()
                .downcast_ref::<Buffer>()
                .unwrap();

            // Setup command queue / command buffer / encoder
            let command_buffer = self.1.new_command_buffer();

            let out = self
                .metal_forward(&[(a, tensors[0].1)], &self.2, command_buffer)
                .pop()
                .unwrap();

            command_buffer.commit();
            command_buffer.wait_until_completed();

            vec![Tensor {
                data: Box::new(out),
            }]
        })
    }

    fn custom(&self, key: &str) -> Option<Box<dyn Any>> {
        if key == "metal" {
            return Some(Box::new(MetalKernelWrapper(Arc::new(Box::new(
                self.clone(),
            )))));
        }
        None
    }
}
