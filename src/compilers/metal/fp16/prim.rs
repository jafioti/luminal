use std::{collections::HashMap, fmt::Debug, sync::Arc};

use crate::{
    compilers::metal::*,
    op::{
        Add, Constant, Contiguous, Exp2, Function as LFunction, InputTensor, LessThan, Log2,
        MaxReduce, Mod, Mul, Operator, Print, Recip, Sin, Sqrt, SumReduce,
    },
    prelude::*,
};
use half::f16;
use itertools::Itertools;
use metal_rs::{objc::rc::autoreleasepool, *};
use num_traits::FromPrimitive;
use petgraph::{stable_graph::NodeIndex, visit::EdgeRef};

pub trait MetalKernelForward: Debug {
    fn metal_forward(
        &self,
        inputs: &[(&Buffer, ShapeTracker)],
        dev: &Device,
        command_buffer: &CommandBufferRef,
    ) -> Vec<Buffer>;
}

#[derive(Clone)]
pub struct MetalKernelWrapper(pub Arc<Box<dyn MetalKernelForward>>);

impl PartialEq for MetalKernelWrapper {
    fn eq(&self, _: &Self) -> bool {
        false
    }
}

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
            .unwrap()
            .iter()
            .copied()
            .map(f16::from_f32)
            .collect::<Vec<_>>();
        let buffer = self.0.new_buffer_with_data(
            unsafe { std::mem::transmute(data.as_ptr()) },
            (data.len() * std::mem::size_of::<f16>()) as u64,
            MTLResourceOptions::StorageModeShared,
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
        if inp[0].0.borrowed().data.as_any().is::<Vec<f32>>() {
            // Already off device
            return vec![inp.pop().unwrap().0.cloned()];
        }
        let buffer = get_buffer_from_tensor(&inp[0].0);
        let mut data = vec![0.0; buffer.length() as usize / std::mem::size_of::<f16>()];
        let ptr = buffer.contents() as *mut f16;
        for (i, d) in data.iter_mut().enumerate() {
            *d = unsafe { *ptr.add(i) }.to_f32();
        }
        vec![Tensor {
            data: Box::new(data),
        }]
    }
}

#[derive(Debug, Clone)]
pub struct MetalConstant(pub f16, Device);
impl PartialEq for MetalConstant {
    fn eq(&self, _: &Self) -> bool {
        false
    }
}

impl Operator for MetalConstant {
    fn process(&self, _: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        vec![Tensor {
            data: Box::new(self.1.new_buffer_with_data(
                &self.0 as *const f16 as *const _,
                std::mem::size_of::<f16>() as u64,
                MTLResourceOptions::StorageModeShared,
            )),
        }]
    }
}

#[derive(Debug, Clone)]
pub struct MetalContiguous(ComputePipelineState, Device, ShapeTracker);

impl PartialEq for MetalContiguous {
    fn eq(&self, _: &Self) -> bool {
        false
    }
}

impl MetalContiguous {
    fn new(
        shape: ShapeTracker,
        dev: Device,
        kernels: &mut HashMap<String, ComputePipelineState>,
    ) -> Self {
        let (idx_exp, valid_exp) = get_idx_valid_exps(shape);
        let mut code = format!(
            "
#include <metal_stdlib>
using namespace metal;
kernel void mkernel(device half *inp [[buffer(0)]], device half *out [[buffer(1)]], device uint& n_elements [[buffer(2)]], uint idx [[thread_position_in_grid]]{}) {{
    if (idx < n_elements && ({valid_exp} != 0)) {{
        out[idx] = inp[{idx_exp}];
    }}
}}
", render_dyn_dim_inputs(&[shape], 3),
        );
        let name = format!("kernel_{}", hash(&code));
        code = code.replace("mkernel", &name);

        if !kernels.contains_key(&name) {
            kernels.insert(name.clone(), compile_function(&name, &code, &dev));
        }
        Self(kernels[&name].clone(), dev, shape)
    }
}

impl MetalKernelForward for MetalContiguous {
    fn metal_forward(
        &self,
        inputs: &[(&Buffer, ShapeTracker)],
        dev: &Device,
        command_buffer: &CommandBufferRef,
    ) -> Vec<Buffer> {
        let res_shape = inputs[0].1.contiguous();
        let inp_size = res_shape.n_elements();
        let out = dev.new_buffer(
            (inp_size * std::mem::size_of::<f16>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let encoder =
            command_buffer.compute_command_encoder_with_descriptor(ComputePassDescriptor::new());
        encoder.set_compute_pipeline_state(&self.0);

        // Set inputs
        encoder.set_buffer(0, Some(inputs[0].0), 0);
        encoder.set_buffer(1, Some(&out), 0);
        encoder.set_int(2, inp_size as u32);
        input_dyn_dims(&[(self.2, inputs[0].1)], encoder, 3);

        // Execute
        encoder.dispatch_n_elements(inp_size);
        encoder.end_encoding();

        vec![out]
    }
}

impl Operator for MetalContiguous {
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

#[derive(Debug, Clone)]
pub struct MetalLog2(pub ComputePipelineState, CommandQueue, Device);
impl PartialEq for MetalLog2 {
    fn eq(&self, _: &Self) -> bool {
        false
    }
}

impl MetalLog2 {
    fn new(
        dev: Device,
        queue: CommandQueue,
        kernels: &mut HashMap<String, ComputePipelineState>,
    ) -> Self {
        let mut code = "#include <metal_stdlib>
using namespace metal;
kernel void mkernel(device half *inp [[buffer(0)]], device half *out [[buffer(1)]], device uint& n_elements [[buffer(2)]], uint idx [[thread_position_in_grid]]) {{
    if (idx < n_elements) {{
        out[idx] = log2(inp[idx]);
    }}
}}
".to_string();
        let name = format!("kernel_{}", hash(&code));
        code = code.replace("mkernel", &name);

        if !kernels.contains_key(&name) {
            kernels.insert(name.clone(), compile_function(&name, &code, &dev));
        }
        Self(kernels[&name].clone(), queue, dev)
    }
}

impl MetalKernelForward for MetalLog2 {
    fn metal_forward(
        &self,
        inputs: &[(&Buffer, ShapeTracker)],
        dev: &Device,
        command_buffer: &CommandBufferRef,
    ) -> Vec<Buffer> {
        let inp_size = inputs[0].1.n_physical_elements();
        let out = dev.new_buffer(
            (inp_size * std::mem::size_of::<f16>()) as u64,
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
        encoder.dispatch_n_elements(inp_size);
        encoder.end_encoding();

        vec![out]
    }
}

impl Operator for MetalLog2 {
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

#[derive(Debug, Clone)]
pub struct MetalExp2(pub ComputePipelineState, CommandQueue, Device);
impl PartialEq for MetalExp2 {
    fn eq(&self, _: &Self) -> bool {
        false
    }
}

impl MetalExp2 {
    fn new(
        dev: Device,
        queue: CommandQueue,
        kernels: &mut HashMap<String, ComputePipelineState>,
    ) -> Self {
        let mut code = "#include <metal_stdlib>
using namespace metal;
kernel void mkernel(device half *inp [[buffer(0)]], device half *out [[buffer(1)]], device uint& n_elements [[buffer(2)]], uint idx [[thread_position_in_grid]]) {{
    if (idx < n_elements) {{
        out[idx] = exp2(inp[idx]);
    }}
}}
".to_string();
        let name = format!("kernel_{}", hash(&code));
        code = code.replace("mkernel", &name);

        if !kernels.contains_key(&name) {
            kernels.insert(name.clone(), compile_function(&name, &code, &dev));
        }
        Self(kernels[&name].clone(), queue, dev)
    }
}
impl MetalKernelForward for MetalExp2 {
    fn metal_forward(
        &self,
        inputs: &[(&Buffer, ShapeTracker)],
        dev: &Device,
        command_buffer: &CommandBufferRef,
    ) -> Vec<Buffer> {
        let inp_size = inputs[0].1.n_physical_elements();
        let out = dev.new_buffer(
            (inp_size * std::mem::size_of::<f16>()) as u64,
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
        encoder.dispatch_n_elements(inp_size);
        encoder.end_encoding();

        vec![out]
    }
}

impl Operator for MetalExp2 {
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

#[derive(Debug, Clone)]
pub struct MetalSin(pub ComputePipelineState, CommandQueue, Device);
impl PartialEq for MetalSin {
    fn eq(&self, _: &Self) -> bool {
        false
    }
}

impl MetalSin {
    fn new(
        dev: Device,
        queue: CommandQueue,
        kernels: &mut HashMap<String, ComputePipelineState>,
    ) -> Self {
        let mut code = "#include <metal_stdlib>
using namespace metal;
kernel void mkernel(device half *inp [[buffer(0)]], device half *out [[buffer(1)]], device uint& n_elements [[buffer(2)]], uint idx [[thread_position_in_grid]]) {{
    if (idx < n_elements) {{
        out[idx] = (half)sin((float)inp[idx]);
    }}
}}
".to_string();
        let name = format!("kernel_{}", hash(&code));
        code = code.replace("mkernel", &name);

        if !kernels.contains_key(&name) {
            kernels.insert(name.clone(), compile_function(&name, &code, &dev));
        }
        Self(kernels[&name].clone(), queue, dev)
    }
}
impl MetalKernelForward for MetalSin {
    fn metal_forward(
        &self,
        inputs: &[(&Buffer, ShapeTracker)],
        dev: &Device,
        command_buffer: &CommandBufferRef,
    ) -> Vec<Buffer> {
        let inp_size = inputs[0].1.n_physical_elements();
        let out = dev.new_buffer(
            (inp_size * std::mem::size_of::<f16>()) as u64,
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
        encoder.dispatch_n_elements(inp_size);
        encoder.end_encoding();

        vec![out]
    }
}

impl Operator for MetalSin {
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

#[derive(Debug, Clone)]
pub struct MetalSqrt(pub ComputePipelineState, CommandQueue, Device);
impl PartialEq for MetalSqrt {
    fn eq(&self, _: &Self) -> bool {
        false
    }
}

impl MetalSqrt {
    fn new(
        dev: Device,
        queue: CommandQueue,
        kernels: &mut HashMap<String, ComputePipelineState>,
    ) -> Self {
        let mut code = "#include <metal_stdlib>
using namespace metal;
kernel void mkernel(device half *inp [[buffer(0)]], device half *out [[buffer(1)]], device uint& n_elements [[buffer(2)]], uint idx [[thread_position_in_grid]]) {{
    if (idx < n_elements) {{
        out[idx] = sqrt(inp[idx]);
    }}
}}
".to_string();
        let name = format!("kernel_{}", hash(&code));
        code = code.replace("mkernel", &name);

        if !kernels.contains_key(&name) {
            kernels.insert(name.clone(), compile_function(&name, &code, &dev));
        }
        Self(kernels[&name].clone(), queue, dev)
    }
}
impl MetalKernelForward for MetalSqrt {
    fn metal_forward(
        &self,
        inputs: &[(&Buffer, ShapeTracker)],
        dev: &Device,
        command_buffer: &CommandBufferRef,
    ) -> Vec<Buffer> {
        let inp_size = inputs[0].1.n_physical_elements();
        let out = dev.new_buffer(
            (inp_size * std::mem::size_of::<f16>()) as u64,
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
        encoder.dispatch_n_elements(inp_size);
        encoder.end_encoding();

        vec![out]
    }
}

impl Operator for MetalSqrt {
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

#[derive(Debug, Clone)]
pub struct MetalRecip(pub ComputePipelineState, CommandQueue, Device);
impl PartialEq for MetalRecip {
    fn eq(&self, _: &Self) -> bool {
        false
    }
}

impl MetalRecip {
    fn new(
        dev: Device,
        queue: CommandQueue,
        kernels: &mut HashMap<String, ComputePipelineState>,
    ) -> Self {
        let mut code = "#include <metal_stdlib>
using namespace metal;
kernel void mkernel(device half *inp [[buffer(0)]], device half *out [[buffer(1)]], device uint& n_elements [[buffer(2)]], uint idx [[thread_position_in_grid]]) {{
    if (idx < n_elements) {{
        out[idx] = 1.0 / inp[idx];
    }}
}}
".to_string();
        let name = format!("kernel_{}", hash(&code));
        code = code.replace("mkernel", &name);

        if !kernels.contains_key(&name) {
            kernels.insert(name.clone(), compile_function(&name, &code, &dev));
        }
        Self(kernels[&name].clone(), queue, dev)
    }
}
impl MetalKernelForward for MetalRecip {
    fn metal_forward(
        &self,
        inputs: &[(&Buffer, ShapeTracker)],
        dev: &Device,
        command_buffer: &CommandBufferRef,
    ) -> Vec<Buffer> {
        let inp_size = inputs[0].1.n_physical_elements();
        let out = dev.new_buffer(
            (inp_size * std::mem::size_of::<f16>()) as u64,
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
        encoder.dispatch_n_elements(inp_size);
        encoder.end_encoding();

        vec![out]
    }
}

impl Operator for MetalRecip {
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

#[derive(Debug, Clone)]
pub struct MetalAdd(
    ComputePipelineState,
    CommandQueue,
    Device,
    ShapeTracker,
    ShapeTracker,
);

impl PartialEq for MetalAdd {
    fn eq(&self, _: &Self) -> bool {
        false
    }
}

impl MetalAdd {
    fn new(
        a_shape: ShapeTracker,
        b_shape: ShapeTracker,
        dev: Device,
        queue: CommandQueue,
        kernels: &mut HashMap<String, ComputePipelineState>,
    ) -> Self {
        let (a_idx_exp, a_valid_exp) = get_idx_valid_exps(a_shape);
        let (b_idx_exp, b_valid_exp) = get_idx_valid_exps(b_shape);
        let mut code = format!(
            "
#include <metal_stdlib>
using namespace metal;
kernel void mkernel(device half *inp_a [[buffer(0)]], device half *inp_b [[buffer(1)]], device half *out [[buffer(2)]], device uint& n_elements [[buffer(3)]], uint idx [[thread_position_in_grid]]{}) {{
    if (idx < n_elements) {{
        out[idx] = 
            (({a_valid_exp}) == 0 ? 0.0h : inp_a[{a_idx_exp}]) 
            + (({b_valid_exp}) == 0 ? 0.0h : inp_b[{b_idx_exp}]);
    }}
}}
", render_dyn_dim_inputs(&[a_shape, b_shape], 4),
        );
        let name = format!("kernel_{}", hash(&code));
        code = code.replace("mkernel", &name);

        if !kernels.contains_key(&name) {
            kernels.insert(name.clone(), compile_function(&name, &code, &dev));
        }
        Self(kernels[&name].clone(), queue, dev, a_shape, b_shape)
    }
}

impl MetalKernelForward for MetalAdd {
    fn metal_forward(
        &self,
        inputs: &[(&Buffer, ShapeTracker)],
        dev: &Device,
        command_buffer: &CommandBufferRef,
    ) -> Vec<Buffer> {
        let inp_size = inputs[0].1.n_elements();
        let out = dev.new_buffer(
            (inp_size * std::mem::size_of::<f16>()) as u64,
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
        input_dyn_dims(&[(self.3, inputs[0].1), (self.4, inputs[1].1)], encoder, 4);

        // Execute
        encoder.dispatch_n_elements(inp_size);
        encoder.end_encoding();

        vec![out]
    }
}

impl Operator for MetalAdd {
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

#[derive(Debug, Clone)]
pub struct MetalMul(
    ComputePipelineState,
    CommandQueue,
    Device,
    ShapeTracker,
    ShapeTracker,
);

impl PartialEq for MetalMul {
    fn eq(&self, _: &Self) -> bool {
        false
    }
}

impl MetalMul {
    fn new(
        a_shape: ShapeTracker,
        b_shape: ShapeTracker,
        dev: Device,
        queue: CommandQueue,
        kernels: &mut HashMap<String, ComputePipelineState>,
    ) -> Self {
        let (a_idx_exp, a_valid_exp) = get_idx_valid_exps(a_shape);
        let (b_idx_exp, b_valid_exp) = get_idx_valid_exps(b_shape);
        let mut code = format!(
            "
#include <metal_stdlib>
using namespace metal;
kernel void mkernel(device half *inp_a [[buffer(0)]], device half *inp_b [[buffer(1)]], device half *out [[buffer(2)]], device uint& n_elements [[buffer(3)]], uint idx [[thread_position_in_grid]]{}) {{
    if (idx < n_elements) {{
        out[idx] = 
            (({a_valid_exp}) == 0 ? 0.0h : inp_a[{a_idx_exp}]) 
            * (({b_valid_exp}) == 0 ? 0.0h : inp_b[{b_idx_exp}]);
    }}
}}
", render_dyn_dim_inputs(&[a_shape, b_shape], 4),
        );
        let name = format!("kernel_{}", hash(&code));
        code = code.replace("mkernel", &name);

        if !kernels.contains_key(&name) {
            kernels.insert(name.clone(), compile_function(&name, &code, &dev));
        }
        Self(kernels[&name].clone(), queue, dev, a_shape, b_shape)
    }
}
impl MetalKernelForward for MetalMul {
    fn metal_forward(
        &self,
        inputs: &[(&Buffer, ShapeTracker)],
        dev: &Device,
        command_buffer: &CommandBufferRef,
    ) -> Vec<Buffer> {
        let inp_size = inputs[0].1.n_elements();
        let out = dev.new_buffer(
            (inp_size * std::mem::size_of::<f16>()) as u64,
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
        input_dyn_dims(&[(self.3, inputs[0].1), (self.4, inputs[1].1)], encoder, 4);

        // Execute
        encoder.dispatch_n_elements(inp_size);
        encoder.end_encoding();

        vec![out]
    }
}

impl Operator for MetalMul {
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

#[derive(Debug, Clone)]
pub struct MetalLessThan(
    ComputePipelineState,
    CommandQueue,
    Device,
    ShapeTracker,
    ShapeTracker,
);

impl PartialEq for MetalLessThan {
    fn eq(&self, _: &Self) -> bool {
        false
    }
}

impl MetalLessThan {
    fn new(
        a_shape: ShapeTracker,
        b_shape: ShapeTracker,
        dev: Device,
        queue: CommandQueue,
        kernels: &mut HashMap<String, ComputePipelineState>,
    ) -> Self {
        let (a_idx_exp, a_valid_exp) = get_idx_valid_exps(a_shape);
        let (b_idx_exp, b_valid_exp) = get_idx_valid_exps(b_shape);
        let mut code = format!(
            "
#include <metal_stdlib>
using namespace metal;
kernel void mkernel(device half *inp_a [[buffer(0)]], device half *inp_b [[buffer(1)]], device half *out [[buffer(2)]], device uint& n_elements [[buffer(3)]], uint idx [[thread_position_in_grid]]{}) {{
    if (idx < n_elements) {{
        half a_t = 0.0h;
        half b_t = 0.0h;
        if (({a_valid_exp}) != 0) {{
            a_t = inp_a[{a_idx_exp}];
        }}
        if (({b_valid_exp}) != 0) {{
            b_t = inp_b[{b_idx_exp}];
        }}
        if (a_t < b_t) {{
            out[idx] = 1.0h;
        }} else {{
            out[idx] = 0.0h;
        }}
    }}
}}
", render_dyn_dim_inputs(&[a_shape, b_shape], 4),
        );
        let name = format!("kernel_{}", hash(&code));
        code = code.replace("mkernel", &name);

        if !kernels.contains_key(&name) {
            kernels.insert(name.clone(), compile_function(&name, &code, &dev));
        }
        Self(kernels[&name].clone(), queue, dev, a_shape, b_shape)
    }
}

impl MetalKernelForward for MetalLessThan {
    fn metal_forward(
        &self,
        inputs: &[(&Buffer, ShapeTracker)],
        dev: &Device,
        command_buffer: &CommandBufferRef,
    ) -> Vec<Buffer> {
        let inp_size = inputs[0].1.n_elements();
        let out = dev.new_buffer(
            (inp_size * std::mem::size_of::<f16>()) as u64,
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
        input_dyn_dims(&[(self.3, inputs[0].1), (self.4, inputs[1].1)], encoder, 4);

        // Execute
        encoder.dispatch_n_elements(inp_size);
        encoder.end_encoding();

        vec![out]
    }
}

impl Operator for MetalLessThan {
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

#[derive(Debug, Clone)]
pub struct MetalMod(
    ComputePipelineState,
    CommandQueue,
    Device,
    ShapeTracker,
    ShapeTracker,
);

impl PartialEq for MetalMod {
    fn eq(&self, _: &Self) -> bool {
        false
    }
}

impl MetalMod {
    fn new(
        a_shape: ShapeTracker,
        b_shape: ShapeTracker,
        dev: Device,
        queue: CommandQueue,
        kernels: &mut HashMap<String, ComputePipelineState>,
    ) -> Self {
        let (a_idx_exp, a_valid_exp) = get_idx_valid_exps(a_shape);
        let (b_idx_exp, b_valid_exp) = get_idx_valid_exps(b_shape);
        let mut code = format!(
            "
#include <metal_stdlib>
using namespace metal;
kernel void mkernel(device half *inp_a [[buffer(0)]], device half *inp_b [[buffer(1)]], device half *out [[buffer(2)]], device uint& n_elements [[buffer(3)]], uint idx [[thread_position_in_grid]]{}) {{
    if (idx < n_elements) {{
        out[idx] = fmod(({a_valid_exp}) == 0 ? 0.0 : inp_a[{a_idx_exp}], ({b_valid_exp}) == 0 ? 0.0 : inp_b[{b_idx_exp}]);
    }}
}}
", render_dyn_dim_inputs(&[a_shape, b_shape], 4),
        );
        let name = format!("kernel_{}", hash(&code));
        code = code.replace("mkernel", &name);

        if !kernels.contains_key(&name) {
            kernels.insert(name.clone(), compile_function(&name, &code, &dev));
        }
        Self(kernels[&name].clone(), queue, dev, a_shape, b_shape)
    }
}
impl MetalKernelForward for MetalMod {
    fn metal_forward(
        &self,
        inputs: &[(&Buffer, ShapeTracker)],
        dev: &Device,
        command_buffer: &CommandBufferRef,
    ) -> Vec<Buffer> {
        let inp_size = inputs[0].1.n_elements();
        let out = dev.new_buffer(
            (inp_size * std::mem::size_of::<f16>()) as u64,
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
        input_dyn_dims(&[(self.3, inputs[0].1), (self.4, inputs[1].1)], encoder, 4);

        // Execute
        encoder.dispatch_n_elements(inp_size);
        encoder.end_encoding();

        vec![out]
    }
}

impl Operator for MetalMod {
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

#[derive(Debug, Clone)]
pub struct MetalSumReduce(
    ComputePipelineState,
    CommandQueue,
    Device,
    pub usize,
    ShapeTracker,
);

impl PartialEq for MetalSumReduce {
    fn eq(&self, _: &Self) -> bool {
        false
    }
}

impl MetalSumReduce {
    fn new(
        shape: ShapeTracker,
        dim: usize,
        dev: Device,
        queue: CommandQueue,
        kernels: &mut HashMap<String, ComputePipelineState>,
    ) -> Self {
        let (idx_exp, valid_exp) = get_idx_valid_exps(shape);
        let mut code = format!(
            "
#include <metal_stdlib>
using namespace metal;
kernel void mkernel(device half *inp [[buffer(0)]], device half *out [[buffer(1)]], device uint& n_elements [[buffer(2)]], device uint& front_size [[buffer(3)]], device uint& back_size [[buffer(4)]], device uint& dim_size [[buffer(5)]], uint i_ [[thread_position_in_grid]]{}) {{
    if (i_ < n_elements) {{
        uint a_ = i_ / back_size;
        uint b_ = i_ % back_size;
        half reduce_value = 0.0;
        for (uint c_ = 0; c_ < dim_size; c_++) {{
            uint idx = a_ * dim_size * back_size + c_ * back_size + b_;
            if (({valid_exp}) != 0) {{
                reduce_value += inp[{idx_exp}];
            }}
        }}
        out[i_] = reduce_value;
    }}
}}
", render_dyn_dim_inputs(&[shape], 6),
        );
        let name = format!("kernel_{}", hash(&code));
        code = code.replace("mkernel", &name);

        if !kernels.contains_key(&name) {
            kernels.insert(name.clone(), compile_function(&name, &code, &dev));
        }
        Self(kernels[&name].clone(), queue, dev, dim, shape)
    }
}

impl MetalKernelForward for MetalSumReduce {
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
            (inp_size * std::mem::size_of::<f16>()) as u64,
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
        input_dyn_dims(&[(self.4, inputs[0].1)], encoder, 6);

        // Execute
        encoder.dispatch_n_elements(inp_size);
        encoder.end_encoding();

        vec![out]
    }
}

impl Operator for MetalSumReduce {
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

#[derive(Debug, Clone)]
pub struct MetalMaxReduce(
    ComputePipelineState,
    CommandQueue,
    Device,
    usize,
    ShapeTracker,
);

impl PartialEq for MetalMaxReduce {
    fn eq(&self, _: &Self) -> bool {
        false
    }
}

impl MetalMaxReduce {
    fn new(
        shape: ShapeTracker,
        dim: usize,
        dev: Device,
        queue: CommandQueue,
        kernels: &mut HashMap<String, ComputePipelineState>,
    ) -> Self {
        let (idx_exp, valid_exp) = get_idx_valid_exps(shape);
        let mut code = format!(
            "
#include <metal_stdlib>
using namespace metal;
kernel void mkernel(device half *inp [[buffer(0)]], device half *out [[buffer(1)]], device uint& n_elements [[buffer(2)]], device uint& front_size [[buffer(3)]], device uint& back_size [[buffer(4)]], device uint& dim_size [[buffer(5)]], uint i_ [[thread_position_in_grid]]{}) {{
    if (i_ < n_elements) {{
        uint a_ = i_ / back_size;
        uint b_ = i_ % back_size;
        half reduce_value = -MAXHALF;
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
", render_dyn_dim_inputs(&[shape], 6),
        );
        let name = format!("kernel_{}", hash(&code));
        code = code.replace("mkernel", &name);

        if !kernels.contains_key(&name) {
            kernels.insert(name.clone(), compile_function(&name, &code, &dev));
        }
        Self(kernels[&name].clone(), queue, dev, dim, shape)
    }
}
impl MetalKernelForward for MetalMaxReduce {
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
            (inp_size * std::mem::size_of::<f16>()) as u64,
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
        input_dyn_dims(&[(self.4, inputs[0].1)], encoder, 6);

        // Execute
        encoder.dispatch_n_elements(inp_size);
        encoder.end_encoding();

        vec![out]
    }
}

impl Operator for MetalMaxReduce {
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

#[derive(Default, Debug)]
pub struct PrimitiveCompiler;

impl Compiler for PrimitiveCompiler {
    fn compile(&self, graph: &mut Graph) {
        let dev = Device::system_default().unwrap();
        let queue = dev.new_command_queue();
        // Go through the graph and insert copy ops
        // Copy function output to device and input from device
        for function_node in graph
            .graph
            .node_indices()
            .filter(|n| {
                graph
                    .graph
                    .node_weight(*n)
                    .unwrap()
                    .as_any()
                    .is::<LFunction>()
            })
            .collect::<Vec<_>>()
        {
            if graph
                .graph
                .node_weight(function_node)
                .unwrap()
                .as_any()
                .downcast_ref::<crate::op::Function>()
                .unwrap()
                .2
                == std::any::TypeId::of::<Vec<f32>>()
            {
                // Create copy node
                let copy_node = graph
                    .add_op(MetalCopyToDevice(dev.clone()))
                    .input(function_node, 0, ShapeTracker::new(&[]))
                    .finish();

                // Switch outgoing edges from input to copy_node
                for (edge_id, weight, dest) in graph
                    .graph
                    .edges_directed(function_node, petgraph::Direction::Outgoing)
                    .map(|e| (e.id(), *e.weight(), e.target()))
                    .filter(|(_, _, trg)| *trg != copy_node)
                    .collect::<Vec<_>>()
                {
                    graph.graph.add_edge(copy_node, dest, weight);
                    graph.graph.remove_edge(edge_id);
                }

                if graph.to_retrieve.contains(&function_node) {
                    graph.to_retrieve.insert(copy_node);
                }

                // If there are inputs to this function remap the function to the copy node
                // if graph
                //     .graph
                //     .edges_directed(function_node, petgraph::Direction::Incoming)
                //     .count()
                //     != 0
                // {
                move_references(
                    &mut graph.id_remap,
                    &mut graph.no_delete,
                    &mut graph.to_retrieve,
                    function_node,
                    copy_node,
                );
                // }
            }

            // Insert copy from device for function inputs
            for (source, edge, edge_weight) in graph
                .graph
                .edges_directed(function_node, petgraph::Direction::Incoming)
                .map(|e| (e.source(), e.id(), *e.weight()))
                .collect::<Vec<_>>()
            {
                let copy_from_node = graph
                    .add_op(MetalCopyFromDevice(dev.clone()))
                    .input(source, 0, ShapeTracker::new(&[]))
                    .finish();
                graph
                    .graph
                    .add_edge(copy_from_node, function_node, edge_weight);
                graph.graph.remove_edge(edge);
            }
        }

        // Copy to_retrieve from device
        for (output_node, output_shape) in graph
            .to_retrieve
            .iter()
            // Filter to non-functions
            .filter(|n| {
                !graph
                    .graph
                    .node_weight(**n)
                    .unwrap()
                    .as_any()
                    .is::<LFunction>()
            })
            .map(|n| {
                (
                    *n,
                    graph
                        .graph
                        .edges_directed(*n, petgraph::Direction::Incoming)
                        .filter_map(|e| e.weight().as_data())
                        .map(|i| i.2)
                        .max_by_key(|s| s.n_physical_elements())
                        .unwrap(),
                )
            })
            .collect::<Vec<_>>()
        {
            // Create copy node
            let copy_node = graph
                .add_op(MetalCopyFromDevice(dev.clone()))
                .input(output_node, 0, output_shape)
                .finish();

            move_references(
                &mut graph.id_remap,
                &mut graph.no_delete,
                &mut graph.to_retrieve,
                output_node,
                copy_node,
            );
        }

        // Copy prints from device
        for (output_node, edge) in graph
            .graph
            .node_indices()
            // Filter non-functions
            .filter(|n| graph.graph.node_weight(*n).unwrap().as_any().is::<Print>())
            .map(|n| {
                (
                    n,
                    graph
                        .graph
                        .edges_directed(n, petgraph::Direction::Incoming)
                        .find(|e| !e.weight().is_schedule())
                        .unwrap()
                        .id(),
                )
            })
            .collect::<Vec<_>>()
        {
            // Create copy node
            let (source, shape) = (
                graph.graph.edge_endpoints(edge).unwrap().0,
                graph.graph.edge_weight(edge).unwrap().as_data().unwrap().2,
            );
            let copy_node = graph
                .add_op(MetalCopyFromDevice(dev.clone()))
                .input(source, 0, shape)
                .finish();
            graph.graph.add_edge(
                copy_node,
                output_node,
                Dependency::Data {
                    input_order: 0,
                    output_order: 0,
                    shape,
                },
            );
            graph.graph.remove_edge(edge);
        }

        // Swap primitive ops
        let mut kernels = HashMap::new();
        for id in graph.graph.node_indices().collect::<Vec<_>>() {
            let src_shapes = graph
                .graph
                .edges_directed(id, petgraph::Direction::Incoming)
                .filter_map(|e| e.weight().as_data())
                .sorted_by_key(|e| e.0)
                .map(|e| e.2)
                .collect::<Vec<_>>();
            let op = graph.graph.node_weight(id).unwrap().as_any().type_id();
            let op_ref = graph.graph.node_weight_mut(id).unwrap();
            if is::<Log2>(op) {
                *op_ref = Box::new(MetalLog2::new(dev.clone(), queue.clone(), &mut kernels));
            } else if is::<Exp2>(op) {
                *op_ref = Box::new(MetalExp2::new(dev.clone(), queue.clone(), &mut kernels));
            } else if let Some(c) = op_ref.as_any().downcast_ref::<Constant>() {
                *op_ref = Box::new(MetalConstant(f16::from_f32(c.0), dev.clone()));
            } else if is::<Sin>(op) {
                *op_ref = Box::new(MetalSin::new(dev.clone(), queue.clone(), &mut kernels));
            } else if is::<Sqrt>(op) {
                *op_ref = Box::new(MetalSqrt::new(dev.clone(), queue.clone(), &mut kernels));
            } else if is::<Recip>(op) {
                *op_ref = Box::new(MetalRecip::new(dev.clone(), queue.clone(), &mut kernels));
            } else if is::<Add>(op) {
                *op_ref = Box::new(MetalAdd::new(
                    src_shapes[0],
                    src_shapes[1],
                    dev.clone(),
                    queue.clone(),
                    &mut kernels,
                ));
            } else if is::<Mul>(op) {
                *op_ref = Box::new(MetalMul::new(
                    src_shapes[0],
                    src_shapes[1],
                    dev.clone(),
                    queue.clone(),
                    &mut kernels,
                ));
            } else if is::<LessThan>(op) {
                *op_ref = Box::new(MetalLessThan::new(
                    src_shapes[0],
                    src_shapes[1],
                    dev.clone(),
                    queue.clone(),
                    &mut kernels,
                ));
            } else if is::<Mod>(op) {
                *op_ref = Box::new(MetalMod::new(
                    src_shapes[0],
                    src_shapes[1],
                    dev.clone(),
                    queue.clone(),
                    &mut kernels,
                ));
            } else if let Some(SumReduce(dim)) = op_ref.as_any().downcast_ref() {
                *op_ref = Box::new(MetalSumReduce::new(
                    src_shapes[0],
                    *dim,
                    dev.clone(),
                    queue.clone(),
                    &mut kernels,
                ));
            } else if let Some(MaxReduce(dim)) = op_ref.as_any().downcast_ref() {
                *op_ref = Box::new(MetalMaxReduce::new(
                    src_shapes[0],
                    *dim,
                    dev.clone(),
                    queue.clone(),
                    &mut kernels,
                ));
            } else if is::<Contiguous>(op) {
                *op_ref = Box::new(MetalContiguous::new(
                    src_shapes[0],
                    dev.clone(),
                    &mut kernels,
                ));
            }
        }
    }
}

/// In 16 bit, summing above 2048 doesn't work. This precludes the .expand(Dim).sum_reduce() pattern to get a dim size in a tensor, so we need to replace these fake reductions with an elementwise mul
#[derive(Debug, Default)]
pub struct FakeReductionCompiler;

impl Compiler for FakeReductionCompiler {
    fn compile(&self, graph: &mut Graph) {
        let mut sum_reduce = NodeIndex::default();
        let s = SelectEdge::new(
            SelectOp::new().check(|o, _| {
                if let Some(c) = o.as_any().downcast_ref::<MetalConstant>() {
                    c.0 == f16::ONE
                } else {
                    false
                }
            }),
            SelectOp::new()
                .ty::<MetalSumReduce>()
                .check(|o, shapes| {
                    if let Some(o) = o.as_any().downcast_ref::<MetalSumReduce>() {
                        shapes[0].fake[shapes[0].indexes[o.3]] // Ensure dimension we are reducing is fake
                    } else {
                        false
                    }
                })
                .ptr(&mut sum_reduce),
        );
        let mut compiled = None;
        for _ in s.search(graph) {
            let op_ref = graph.graph.node_weight_mut(sum_reduce).unwrap();
            let sum_reduce = op_ref.as_any().downcast_ref::<MetalSumReduce>().unwrap();
            if compiled.is_none() {
                compiled = Some(FakeSumReduce::compile(sum_reduce.2.clone()));
            }
            *op_ref = Box::new(FakeSumReduce(
                compiled.clone().unwrap(),
                sum_reduce.2.clone(),
                sum_reduce.3,
            ));
        }
    }
}

#[derive(Debug, Clone)]
pub struct FakeSumReduce(ComputePipelineState, Device, pub usize);
impl PartialEq for FakeSumReduce {
    fn eq(&self, _: &Self) -> bool {
        false
    }
}

impl FakeSumReduce {
    pub fn compile(dev: Device) -> ComputePipelineState {
        let mut code = "#include <metal_stdlib>
using namespace metal;
kernel void mkernel(device half *inp [[buffer(0)]], device half *out [[buffer(1)]], device uint& n_elements [[buffer(2)]], uint idx [[thread_position_in_grid]], device half& mul_factor [[buffer(3)]]) {{
    if (idx < n_elements) {{
        out[idx] = inp[idx] * mul_factor;
    }}
}}
".to_string();
        let name = format!("kernel_{}", hash(&code));
        code = code.replace("mkernel", &name);

        compile_function(&name, &code, &dev)
    }
}

impl MetalKernelForward for FakeSumReduce {
    fn metal_forward(
        &self,
        inputs: &[(&Buffer, ShapeTracker)],
        dev: &Device,
        command_buffer: &CommandBufferRef,
    ) -> Vec<Buffer> {
        let dim_size = f16::from_usize(inputs[0].1.shape()[self.2].to_usize().unwrap()).unwrap();
        let inp_size = inputs[0].1.n_physical_elements();
        let out = dev.new_buffer(
            (inp_size * std::mem::size_of::<f16>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let encoder =
            command_buffer.compute_command_encoder_with_descriptor(ComputePassDescriptor::new());
        encoder.set_compute_pipeline_state(&self.0);

        // Set inputs
        encoder.set_buffer(0, Some(inputs[0].0), 0);
        encoder.set_buffer(1, Some(&out), 0);
        encoder.set_int(2, inp_size as u32);
        encoder.set_bytes(
            3,
            std::mem::size_of::<f16>() as u64,
            &dim_size as *const f16 as *const _,
        );

        // Execute
        encoder.dispatch_n_elements(inp_size);
        encoder.end_encoding();

        vec![out]
    }
}

impl Operator for FakeSumReduce {
    fn process(&self, tensors: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        autoreleasepool(|| {
            let inp = tensors[0]
                .0
                .borrowed()
                .data
                .as_any()
                .downcast_ref::<Buffer>()
                .unwrap();

            // Setup command queue / command buffer / encoder
            let command_queue = self.1.new_command_queue();
            let command_buffer = command_queue.new_command_buffer();

            let out = self
                .metal_forward(&[(inp, tensors[0].1)], &self.1, command_buffer)
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

/// Sometimes CopyTo -> CopyFrom and CopyFrom -> CopyTo patterns remain, so let's clean them up
#[derive(Debug, Default)]
pub struct CopyCompiler;

impl Compiler for CopyCompiler {
    fn compile(&self, graph: &mut Graph) {
        for (first, second) in graph
            .graph
            .edge_indices()
            .filter_map(|e| graph.graph.edge_endpoints(e))
            .filter(|(a, b)| {
                (graph
                    .graph
                    .node_weight(*a)
                    .unwrap()
                    .as_any()
                    .is::<MetalCopyToDevice>()
                    && graph
                        .graph
                        .node_weight(*b)
                        .unwrap()
                        .as_any()
                        .is::<MetalCopyFromDevice>())
                    || (graph
                        .graph
                        .node_weight(*a)
                        .unwrap()
                        .as_any()
                        .is::<MetalCopyFromDevice>()
                        && graph
                            .graph
                            .node_weight(*b)
                            .unwrap()
                            .as_any()
                            .is::<MetalCopyToDevice>())
            })
            .unique_by(|n| n.0)
            .unique_by(|n| n.1)
            .collect::<Vec<_>>()
        {
            if graph
                .graph
                .edges_directed(first, petgraph::Direction::Outgoing)
                .filter(|e| graph.graph.contains_node(e.target()))
                .filter(|e| {
                    !graph
                        .graph
                        .node_weight(e.target())
                        .unwrap()
                        .as_any()
                        .is::<MetalCopyFromDevice>()
                        && !graph
                            .graph
                            .node_weight(e.target())
                            .unwrap()
                            .as_any()
                            .is::<MetalCopyToDevice>()
                })
                .count()
                > 0
                || graph.no_delete.contains(&first)
            {
                continue;
            }
            let source = graph.get_sources(first)[0];
            move_outgoing_edge(second, source.0, &mut graph.graph);
            move_references(
                &mut graph.id_remap,
                &mut graph.no_delete,
                &mut graph.to_retrieve,
                second,
                source.0,
            );
            graph.graph.remove_node(second);
            for dest in graph
                .get_dests(first)
                .iter()
                .map(|(i, _)| *i)
                .collect::<Vec<_>>()
            {
                move_outgoing_edge(dest, source.0, &mut graph.graph);
                move_references(
                    &mut graph.id_remap,
                    &mut graph.no_delete,
                    &mut graph.to_retrieve,
                    dest,
                    source.0,
                );
                graph.graph.remove_node(dest);
            }
            graph.graph.remove_node(first);
        }
    }
}

fn get_buffer_from_tensor<'a>(tensor: &'a InputTensor) -> &'a Buffer {
    tensor
        .borrowed()
        .data
        .as_any()
        .downcast_ref::<Buffer>()
        .unwrap()
}
