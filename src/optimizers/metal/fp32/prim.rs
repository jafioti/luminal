use std::{
    any::{Any, TypeId},
    collections::{hash_map::DefaultHasher, HashMap, HashSet},
    hash::{Hash, Hasher},
};

use crate::{
    op::{
        Add, Contiguous, Exp2, Function as LFunction, InputTensor, LessThan, Log2, MaxReduce, Mod,
        Mul, Operator, Print, Recip, Sin, Sqrt, SumReduce,
    },
    prelude::*,
};
use itertools::Itertools;
use metal_rs::{objc::rc::autoreleasepool, *};
use petgraph::visit::EdgeRef;
use regex::Regex;

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
        if inp[0].0.borrowed().data.as_any().is::<Vec<f32>>() {
            // Already off device
            return vec![inp.pop().unwrap().0.cloned()];
        }
        let buffer = inp[0]
            .0
            .borrowed()
            .data
            .as_any()
            .downcast_ref::<Buffer>()
            .unwrap();
        let mut data = vec![0.0; buffer.length() as usize / std::mem::size_of::<f32>()];
        let ptr = buffer.contents() as *mut f32;
        for (i, d) in data.iter_mut().enumerate() {
            *d = unsafe { *ptr.add(i) };
        }
        vec![Tensor {
            data: Box::new(data),
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
kernel void mkernel(device float *inp [[buffer(0)]], device float *out [[buffer(1)]], device uint& n_elements [[buffer(2)]], uint idx [[thread_position_in_grid]]{}) {{
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
impl Operator for MetalContiguous {
    fn process(&self, tensors: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        autoreleasepool(|| {
            let res_shape = tensors[0].1.contiguous();
            let inp_size = res_shape.n_elements();

            // Setup buffers
            let a = tensors[0]
                .0
                .borrowed()
                .data
                .as_any()
                .downcast_ref::<Buffer>()
                .unwrap();
            let out = self.1.new_buffer(
                (inp_size * std::mem::size_of::<f32>()) as u64,
                MTLResourceOptions::CPUCacheModeDefaultCache,
            );

            // Setup command queue / command buffer / encoder
            let command_queue = self.1.new_command_queue();
            let command_buffer = command_queue.new_command_buffer();
            let encoder = command_buffer
                .compute_command_encoder_with_descriptor(ComputePassDescriptor::new());
            encoder.set_compute_pipeline_state(&self.0);

            // Set inputs
            encoder.set_buffer(0, Some(a), 0);
            encoder.set_buffer(1, Some(&out), 0);
            encoder.set_int(2, inp_size as u32);
            input_dyn_dims(&[(self.2, tensors[0].1)], encoder, 3);

            // Execute
            encoder.dispatch_n_elements(inp_size);
            encoder.end_encoding();
            command_buffer.commit();
            command_buffer.wait_until_completed();

            vec![Tensor {
                data: Box::new(out),
            }]
        })
    }
}

#[derive(Debug, Clone)]
pub struct MetalLog2(ComputePipelineState, Device);
impl PartialEq for MetalLog2 {
    fn eq(&self, _: &Self) -> bool {
        false
    }
}

impl MetalLog2 {
    fn new(dev: Device, kernels: &mut HashMap<String, ComputePipelineState>) -> Self {
        let mut code = "#include <metal_stdlib>
using namespace metal;
kernel void mkernel(device float *inp [[buffer(0)]], device float *out [[buffer(1)]], device uint& n_elements [[buffer(2)]], uint idx [[thread_position_in_grid]]) {{
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
        Self(kernels[&name].clone(), dev)
    }
}
impl Operator for MetalLog2 {
    fn process(&self, tensors: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        autoreleasepool(|| {
            let inp_size = tensors[0].1.n_physical_elements();

            // Setup buffers
            let a = tensors[0]
                .0
                .borrowed()
                .data
                .as_any()
                .downcast_ref::<Buffer>()
                .unwrap();
            let out = self.1.new_buffer(
                (inp_size * std::mem::size_of::<f32>()) as u64,
                MTLResourceOptions::CPUCacheModeDefaultCache,
            );

            // Setup command queue / command buffer / encoder
            let command_queue = self.1.new_command_queue();
            let command_buffer = command_queue.new_command_buffer();
            let encoder = command_buffer
                .compute_command_encoder_with_descriptor(ComputePassDescriptor::new());
            encoder.set_compute_pipeline_state(&self.0);

            // Set inputs
            encoder.set_buffer(0, Some(a), 0);
            encoder.set_buffer(1, Some(&out), 0);
            encoder.set_int(2, inp_size as u32);

            // Execute
            encoder.dispatch_n_elements(inp_size);
            encoder.end_encoding();
            command_buffer.commit();
            command_buffer.wait_until_completed();

            vec![Tensor {
                data: Box::new(out),
            }]
        })
    }
}

#[derive(Debug, Clone)]
pub struct MetalExp2(ComputePipelineState, Device);
impl PartialEq for MetalExp2 {
    fn eq(&self, _: &Self) -> bool {
        false
    }
}

impl MetalExp2 {
    fn new(dev: Device, kernels: &mut HashMap<String, ComputePipelineState>) -> Self {
        let mut code = "#include <metal_stdlib>
using namespace metal;
kernel void mkernel(device float *inp [[buffer(0)]], device float *out [[buffer(1)]], device uint& n_elements [[buffer(2)]], uint idx [[thread_position_in_grid]]) {{
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
        Self(kernels[&name].clone(), dev)
    }
}
impl Operator for MetalExp2 {
    fn process(&self, tensors: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        autoreleasepool(|| {
            let inp_size = tensors[0].1.n_physical_elements();

            // Setup buffers
            let a = tensors[0]
                .0
                .borrowed()
                .data
                .as_any()
                .downcast_ref::<Buffer>()
                .unwrap();
            let out = self.1.new_buffer(
                (inp_size * std::mem::size_of::<f32>()) as u64,
                MTLResourceOptions::CPUCacheModeDefaultCache,
            );

            // Setup command queue / command buffer / encoder
            let command_queue = self.1.new_command_queue();
            let command_buffer = command_queue.new_command_buffer();
            let encoder = command_buffer
                .compute_command_encoder_with_descriptor(ComputePassDescriptor::new());
            encoder.set_compute_pipeline_state(&self.0);

            // Set inputs
            encoder.set_buffer(0, Some(a), 0);
            encoder.set_buffer(1, Some(&out), 0);
            encoder.set_int(2, inp_size as u32);

            // Execute
            encoder.dispatch_n_elements(inp_size);
            encoder.end_encoding();
            command_buffer.commit();
            command_buffer.wait_until_completed();

            vec![Tensor {
                data: Box::new(out),
            }]
        })
    }
}

#[derive(Debug, Clone)]
pub struct MetalSin(ComputePipelineState, Device);
impl PartialEq for MetalSin {
    fn eq(&self, _: &Self) -> bool {
        false
    }
}

impl MetalSin {
    fn new(dev: Device, kernels: &mut HashMap<String, ComputePipelineState>) -> Self {
        let mut code = "#include <metal_stdlib>
using namespace metal;
kernel void mkernel(device float *inp [[buffer(0)]], device float *out [[buffer(1)]], device uint& n_elements [[buffer(2)]], uint idx [[thread_position_in_grid]]) {{
    if (idx < n_elements) {{
        out[idx] = sin(inp[idx]);
    }}
}}
".to_string();
        let name = format!("kernel_{}", hash(&code));
        code = code.replace("mkernel", &name);

        if !kernels.contains_key(&name) {
            kernels.insert(name.clone(), compile_function(&name, &code, &dev));
        }
        Self(kernels[&name].clone(), dev)
    }
}
impl Operator for MetalSin {
    fn process(&self, tensors: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        autoreleasepool(|| {
            let inp_size = tensors[0].1.n_physical_elements();

            // Setup buffers
            let a = tensors[0]
                .0
                .borrowed()
                .data
                .as_any()
                .downcast_ref::<Buffer>()
                .unwrap();
            let out = self.1.new_buffer(
                (inp_size * std::mem::size_of::<f32>()) as u64,
                MTLResourceOptions::CPUCacheModeDefaultCache,
            );

            // Setup command queue / command buffer / encoder
            let command_queue = self.1.new_command_queue();
            let command_buffer = command_queue.new_command_buffer();
            let encoder = command_buffer
                .compute_command_encoder_with_descriptor(ComputePassDescriptor::new());
            encoder.set_compute_pipeline_state(&self.0);

            // Set inputs
            encoder.set_buffer(0, Some(a), 0);
            encoder.set_buffer(1, Some(&out), 0);
            encoder.set_int(2, inp_size as u32);

            // Execute
            encoder.dispatch_n_elements(inp_size);
            encoder.end_encoding();
            command_buffer.commit();
            command_buffer.wait_until_completed();

            vec![Tensor {
                data: Box::new(out),
            }]
        })
    }
}

#[derive(Debug, Clone)]
pub struct MetalSqrt(ComputePipelineState, Device);
impl PartialEq for MetalSqrt {
    fn eq(&self, _: &Self) -> bool {
        false
    }
}

impl MetalSqrt {
    fn new(dev: Device, kernels: &mut HashMap<String, ComputePipelineState>) -> Self {
        let mut code = "#include <metal_stdlib>
using namespace metal;
kernel void mkernel(device float *inp [[buffer(0)]], device float *out [[buffer(1)]], device uint& n_elements [[buffer(2)]], uint idx [[thread_position_in_grid]]) {{
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
        Self(kernels[&name].clone(), dev)
    }
}
impl Operator for MetalSqrt {
    fn process(&self, tensors: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        autoreleasepool(|| {
            let inp_size = tensors[0].1.n_physical_elements();

            // Setup buffers
            let a = tensors[0]
                .0
                .borrowed()
                .data
                .as_any()
                .downcast_ref::<Buffer>()
                .unwrap();
            let out = self.1.new_buffer(
                (inp_size * std::mem::size_of::<f32>()) as u64,
                MTLResourceOptions::CPUCacheModeDefaultCache,
            );

            // Setup command queue / command buffer / encoder
            let command_queue = self.1.new_command_queue();
            let command_buffer = command_queue.new_command_buffer();
            let encoder = command_buffer
                .compute_command_encoder_with_descriptor(ComputePassDescriptor::new());
            encoder.set_compute_pipeline_state(&self.0);

            // Set inputs
            encoder.set_buffer(0, Some(a), 0);
            encoder.set_buffer(1, Some(&out), 0);
            encoder.set_int(2, inp_size as u32);

            // Execute
            encoder.dispatch_n_elements(inp_size);
            encoder.end_encoding();
            command_buffer.commit();
            command_buffer.wait_until_completed();

            vec![Tensor {
                data: Box::new(out),
            }]
        })
    }
}

#[derive(Debug, Clone)]
pub struct MetalRecip(ComputePipelineState, Device);
impl PartialEq for MetalRecip {
    fn eq(&self, _: &Self) -> bool {
        false
    }
}

impl MetalRecip {
    fn new(dev: Device, kernels: &mut HashMap<String, ComputePipelineState>) -> Self {
        let mut code = "#include <metal_stdlib>
using namespace metal;
kernel void mkernel(device float *inp [[buffer(0)]], device float *out [[buffer(1)]], device uint& n_elements [[buffer(2)]], uint idx [[thread_position_in_grid]]) {{
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
        Self(kernels[&name].clone(), dev)
    }
}
impl Operator for MetalRecip {
    fn process(&self, tensors: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        autoreleasepool(|| {
            let inp_size = tensors[0].1.n_physical_elements();

            // Setup buffers
            let a = tensors[0]
                .0
                .borrowed()
                .data
                .as_any()
                .downcast_ref::<Buffer>()
                .unwrap();
            let out = self.1.new_buffer(
                (inp_size * std::mem::size_of::<f32>()) as u64,
                MTLResourceOptions::CPUCacheModeDefaultCache,
            );

            // Setup command queue / command buffer / encoder
            let command_queue = self.1.new_command_queue();
            let command_buffer = command_queue.new_command_buffer();
            let encoder = command_buffer
                .compute_command_encoder_with_descriptor(ComputePassDescriptor::new());
            encoder.set_compute_pipeline_state(&self.0);

            // Set inputs
            encoder.set_buffer(0, Some(a), 0);
            encoder.set_buffer(1, Some(&out), 0);
            encoder.set_int(2, inp_size as u32);

            // Execute
            encoder.dispatch_n_elements(inp_size);
            encoder.end_encoding();
            command_buffer.commit();
            command_buffer.wait_until_completed();

            vec![Tensor {
                data: Box::new(out),
            }]
        })
    }
}

#[derive(Debug, Clone)]
pub struct MetalAdd(ComputePipelineState, Device, ShapeTracker, ShapeTracker);

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
        kernels: &mut HashMap<String, ComputePipelineState>,
    ) -> Self {
        let (a_idx_exp, a_valid_exp) = get_idx_valid_exps(a_shape);
        let (b_idx_exp, b_valid_exp) = get_idx_valid_exps(b_shape);
        let mut code = format!(
            "
#include <metal_stdlib>
using namespace metal;
kernel void mkernel(device float *inp_a [[buffer(0)]], device float *inp_b [[buffer(1)]], device float *out [[buffer(2)]], device uint& n_elements [[buffer(3)]], uint idx [[thread_position_in_grid]]{}) {{
    if (idx < n_elements) {{
        out[idx] = 
            (({a_valid_exp}) == 0 ? 0.0 : inp_a[{a_idx_exp}]) 
            + (({b_valid_exp}) == 0 ? 0.0 : inp_b[{b_idx_exp}]);
    }}
}}
", render_dyn_dim_inputs(&[a_shape, b_shape], 4),
        );
        let name = format!("kernel_{}", hash(&code));
        code = code.replace("mkernel", &name);

        if !kernels.contains_key(&name) {
            kernels.insert(name.clone(), compile_function(&name, &code, &dev));
        }
        Self(kernels[&name].clone(), dev, a_shape, b_shape)
    }
}
impl Operator for MetalAdd {
    fn process(&self, tensors: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        autoreleasepool(|| {
            let inp_size = tensors[0].1.n_elements();

            // Setup buffers
            let a = tensors[0]
                .0
                .borrowed()
                .data
                .as_any()
                .downcast_ref::<Buffer>()
                .unwrap();
            let b = tensors[1]
                .0
                .borrowed()
                .data
                .as_any()
                .downcast_ref::<Buffer>()
                .unwrap();
            let out = self.1.new_buffer(
                (inp_size * std::mem::size_of::<f32>()) as u64,
                MTLResourceOptions::CPUCacheModeDefaultCache,
            );

            // Setup command queue / command buffer / encoder
            let command_queue = self.1.new_command_queue();
            let command_buffer = command_queue.new_command_buffer();
            let encoder = command_buffer
                .compute_command_encoder_with_descriptor(ComputePassDescriptor::new());
            encoder.set_compute_pipeline_state(&self.0);

            // Set inputs
            encoder.set_buffer(0, Some(a), 0);
            encoder.set_buffer(1, Some(b), 0);
            encoder.set_buffer(2, Some(&out), 0);
            encoder.set_int(3, inp_size as u32);
            input_dyn_dims(
                &[(self.2, tensors[0].1), (self.3, tensors[1].1)],
                encoder,
                4,
            );

            // Execute
            encoder.dispatch_n_elements(inp_size);
            encoder.end_encoding();
            command_buffer.commit();
            command_buffer.wait_until_completed();

            vec![Tensor {
                data: Box::new(out),
            }]
        })
    }
}

#[derive(Debug, Clone)]
pub struct MetalMul(ComputePipelineState, Device, ShapeTracker, ShapeTracker);

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
        kernels: &mut HashMap<String, ComputePipelineState>,
    ) -> Self {
        let (a_idx_exp, a_valid_exp) = get_idx_valid_exps(a_shape);
        let (b_idx_exp, b_valid_exp) = get_idx_valid_exps(b_shape);
        let mut code = format!(
            "
#include <metal_stdlib>
using namespace metal;
kernel void mkernel(device float *inp_a [[buffer(0)]], device float *inp_b [[buffer(1)]], device float *out [[buffer(2)]], device uint& n_elements [[buffer(3)]], uint idx [[thread_position_in_grid]]{}) {{
    if (idx < n_elements) {{
        out[idx] = 
            (({a_valid_exp}) == 0 ? 0.0 : inp_a[{a_idx_exp}]) 
            * (({b_valid_exp}) == 0 ? 0.0 : inp_b[{b_idx_exp}]);
    }}
}}
", render_dyn_dim_inputs(&[a_shape, b_shape], 4),
        );
        let name = format!("kernel_{}", hash(&code));
        code = code.replace("mkernel", &name);

        if !kernels.contains_key(&name) {
            kernels.insert(name.clone(), compile_function(&name, &code, &dev));
        }
        Self(kernels[&name].clone(), dev, a_shape, b_shape)
    }
}
impl Operator for MetalMul {
    fn process(&self, tensors: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        autoreleasepool(|| {
            let inp_size = tensors[0].1.n_elements();

            // Setup buffers
            let a = tensors[0]
                .0
                .borrowed()
                .data
                .as_any()
                .downcast_ref::<Buffer>()
                .unwrap();
            let b = tensors[1]
                .0
                .borrowed()
                .data
                .as_any()
                .downcast_ref::<Buffer>()
                .unwrap();
            let out = self.1.new_buffer(
                (inp_size * std::mem::size_of::<f32>()) as u64,
                MTLResourceOptions::CPUCacheModeDefaultCache,
            );

            // Setup command queue / command buffer / encoder
            let command_queue = self.1.new_command_queue();
            let command_buffer = command_queue.new_command_buffer();
            let encoder = command_buffer
                .compute_command_encoder_with_descriptor(ComputePassDescriptor::new());
            encoder.set_compute_pipeline_state(&self.0);

            // Set inputs
            encoder.set_buffer(0, Some(a), 0);
            encoder.set_buffer(1, Some(b), 0);
            encoder.set_buffer(2, Some(&out), 0);
            encoder.set_int(3, inp_size as u32);
            input_dyn_dims(
                &[(self.2, tensors[0].1), (self.3, tensors[1].1)],
                encoder,
                4,
            );

            // Execute
            encoder.dispatch_n_elements(inp_size);
            encoder.end_encoding();
            command_buffer.commit();
            command_buffer.wait_until_completed();

            vec![Tensor {
                data: Box::new(out),
            }]
        })
    }
}

#[derive(Debug, Clone)]
pub struct MetalLessThan(ComputePipelineState, Device, ShapeTracker, ShapeTracker);

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
        kernels: &mut HashMap<String, ComputePipelineState>,
    ) -> Self {
        let (a_idx_exp, a_valid_exp) = get_idx_valid_exps(a_shape);
        let (b_idx_exp, b_valid_exp) = get_idx_valid_exps(b_shape);
        let mut code = format!(
            "
#include <metal_stdlib>
using namespace metal;
kernel void mkernel(device float *inp_a [[buffer(0)]], device float *inp_b [[buffer(1)]], device float *out [[buffer(2)]], device uint& n_elements [[buffer(3)]], uint idx [[thread_position_in_grid]]{}) {{
    if (idx < n_elements) {{
        float a_t = 0.0;
        float b_t = 0.0;
        if (({a_valid_exp}) != 0) {{
            a_t = inp_a[{a_idx_exp}];
        }}
        if (({b_valid_exp}) != 0) {{
            b_t = inp_b[{b_idx_exp}];
        }}
        if (a_t < b_t) {{
            out[idx] = 1.0;
        }} else {{
            out[idx] = 0.0;
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
        Self(kernels[&name].clone(), dev, a_shape, b_shape)
    }
}
impl Operator for MetalLessThan {
    fn process(&self, tensors: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        autoreleasepool(|| {
            let inp_size = tensors[0].1.n_elements();

            // Setup buffers
            let a = tensors[0]
                .0
                .borrowed()
                .data
                .as_any()
                .downcast_ref::<Buffer>()
                .unwrap();
            let b = tensors[1]
                .0
                .borrowed()
                .data
                .as_any()
                .downcast_ref::<Buffer>()
                .unwrap();
            let out = self.1.new_buffer(
                (inp_size * std::mem::size_of::<f32>()) as u64,
                MTLResourceOptions::CPUCacheModeDefaultCache,
            );

            // Setup command queue / command buffer / encoder
            let command_queue = self.1.new_command_queue();
            let command_buffer = command_queue.new_command_buffer();
            let encoder = command_buffer
                .compute_command_encoder_with_descriptor(ComputePassDescriptor::new());
            encoder.set_compute_pipeline_state(&self.0);

            // Set inputs
            encoder.set_buffer(0, Some(a), 0);
            encoder.set_buffer(1, Some(b), 0);
            encoder.set_buffer(2, Some(&out), 0);
            encoder.set_int(3, inp_size as u32);
            input_dyn_dims(
                &[(self.2, tensors[0].1), (self.3, tensors[1].1)],
                encoder,
                4,
            );

            // Execute
            encoder.dispatch_n_elements(inp_size);
            encoder.end_encoding();
            command_buffer.commit();
            command_buffer.wait_until_completed();

            vec![Tensor {
                data: Box::new(out),
            }]
        })
    }
}

#[derive(Debug, Clone)]
pub struct MetalMod(ComputePipelineState, Device, ShapeTracker, ShapeTracker);

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
        kernels: &mut HashMap<String, ComputePipelineState>,
    ) -> Self {
        let (a_idx_exp, a_valid_exp) = get_idx_valid_exps(a_shape);
        let (b_idx_exp, b_valid_exp) = get_idx_valid_exps(b_shape);
        let mut code = format!(
            "
#include <metal_stdlib>
using namespace metal;
kernel void mkernel(device float *inp_a [[buffer(0)]], device float *inp_b [[buffer(1)]], device float *out [[buffer(2)]], device uint& n_elements [[buffer(3)]], uint idx [[thread_position_in_grid]]{}) {{
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
        Self(kernels[&name].clone(), dev, a_shape, b_shape)
    }
}
impl Operator for MetalMod {
    fn process(&self, tensors: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        autoreleasepool(|| {
            let inp_size = tensors[0].1.n_elements();

            // Setup buffers
            let a = tensors[0]
                .0
                .borrowed()
                .data
                .as_any()
                .downcast_ref::<Buffer>()
                .unwrap();
            let b = tensors[1]
                .0
                .borrowed()
                .data
                .as_any()
                .downcast_ref::<Buffer>()
                .unwrap();
            let out = self.1.new_buffer(
                (inp_size * std::mem::size_of::<f32>()) as u64,
                MTLResourceOptions::CPUCacheModeDefaultCache,
            );

            // Setup command queue / command buffer / encoder
            let command_queue = self.1.new_command_queue();
            let command_buffer = command_queue.new_command_buffer();
            let encoder = command_buffer
                .compute_command_encoder_with_descriptor(ComputePassDescriptor::new());
            encoder.set_compute_pipeline_state(&self.0);

            // Set inputs
            encoder.set_buffer(0, Some(a), 0);
            encoder.set_buffer(1, Some(b), 0);
            encoder.set_buffer(2, Some(&out), 0);
            encoder.set_int(3, inp_size as u32);
            input_dyn_dims(
                &[(self.2, tensors[0].1), (self.3, tensors[1].1)],
                encoder,
                4,
            );

            // Execute
            encoder.dispatch_n_elements(inp_size);
            encoder.end_encoding();
            command_buffer.commit();
            command_buffer.wait_until_completed();

            vec![Tensor {
                data: Box::new(out),
            }]
        })
    }
}

#[derive(Debug, Clone)]
pub struct MetalSumReduce(ComputePipelineState, Device, usize, ShapeTracker);

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
        kernels: &mut HashMap<String, ComputePipelineState>,
    ) -> Self {
        let (idx_exp, valid_exp) = get_idx_valid_exps(shape);
        let mut code = format!(
            "
#include <metal_stdlib>
using namespace metal;
kernel void mkernel(device float *inp [[buffer(0)]], device float *out [[buffer(1)]], device uint& n_elements [[buffer(2)]], device uint& front_size [[buffer(3)]], device uint& back_size [[buffer(4)]], device uint& dim_size [[buffer(5)]], uint i_ [[thread_position_in_grid]]{}) {{
    if (i_ < n_elements) {{
        uint a_ = i_ / back_size;
        uint b_ = i_ % back_size;
        float reduce_value = 0.0;
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
        Self(kernels[&name].clone(), dev, dim, shape)
    }
}
impl Operator for MetalSumReduce {
    fn process(&self, tensors: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        autoreleasepool(|| {
            let mut sh = tensors[0].1;
            sh.remove_dim(self.2);
            let inp_size = sh.contiguous().n_elements();

            // Setup buffers
            let a = tensors[0]
                .0
                .borrowed()
                .data
                .as_any()
                .downcast_ref::<Buffer>()
                .unwrap();
            let out = self.1.new_buffer(
                (inp_size * std::mem::size_of::<f32>()) as u64,
                MTLResourceOptions::CPUCacheModeDefaultCache,
            );
            let front_size: usize = tensors[0]
                .1
                .shape()
                .iter()
                .take(self.2)
                .map(|i| i.to_usize().unwrap())
                .product();
            let back_size: usize = tensors[0]
                .1
                .shape()
                .iter()
                .skip(self.2 + 1)
                .map(|i| i.to_usize().unwrap())
                .product();
            let dim_size = tensors[0].1.shape()[self.2].to_usize().unwrap();

            // Setup command queue / command buffer / encoder
            let command_queue = self.1.new_command_queue();
            let command_buffer = command_queue.new_command_buffer();
            let encoder = command_buffer
                .compute_command_encoder_with_descriptor(ComputePassDescriptor::new());
            encoder.set_compute_pipeline_state(&self.0);

            // Set inputs
            encoder.set_buffer(0, Some(a), 0);
            encoder.set_buffer(1, Some(&out), 0);
            encoder.set_int(2, inp_size as u32);
            encoder.set_int(3, front_size as u32);
            encoder.set_int(4, back_size as u32);
            encoder.set_int(5, dim_size as u32);
            input_dyn_dims(&[(self.3, tensors[0].1)], encoder, 6);

            // Execute
            encoder.dispatch_n_elements(inp_size);
            encoder.end_encoding();
            command_buffer.commit();
            command_buffer.wait_until_completed();

            vec![Tensor {
                data: Box::new(out),
            }]
        })
    }
}

#[derive(Debug, Clone)]
pub struct MetalMaxReduce(ComputePipelineState, Device, usize, ShapeTracker);

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
        kernels: &mut HashMap<String, ComputePipelineState>,
    ) -> Self {
        let (idx_exp, valid_exp) = get_idx_valid_exps(shape);
        let mut code = format!(
            "
#include <metal_stdlib>
using namespace metal;
kernel void mkernel(device float *inp [[buffer(0)]], device float *out [[buffer(1)]], device uint& n_elements [[buffer(2)]], device uint& front_size [[buffer(3)]], device uint& back_size [[buffer(4)]], device uint& dim_size [[buffer(5)]], uint i_ [[thread_position_in_grid]]{}) {{
    if (i_ < n_elements) {{
        uint a_ = i_ / back_size;
        uint b_ = i_ % back_size;
        float reduce_value = -(float)0x7f800000;
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
        Self(kernels[&name].clone(), dev, dim, shape)
    }
}
impl Operator for MetalMaxReduce {
    fn process(&self, tensors: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        autoreleasepool(|| {
            let mut sh = tensors[0].1;
            sh.remove_dim(self.2);
            let inp_size = sh.contiguous().n_elements();

            // Setup buffers
            let a = tensors[0]
                .0
                .borrowed()
                .data
                .as_any()
                .downcast_ref::<Buffer>()
                .unwrap();
            let out = self.1.new_buffer(
                (inp_size * std::mem::size_of::<f32>()) as u64,
                MTLResourceOptions::CPUCacheModeDefaultCache,
            );
            let front_size: usize = tensors[0]
                .1
                .shape()
                .iter()
                .take(self.2)
                .map(|i| i.to_usize().unwrap())
                .product();
            let back_size: usize = tensors[0]
                .1
                .shape()
                .iter()
                .skip(self.2 + 1)
                .map(|i| i.to_usize().unwrap())
                .product();
            let dim_size = tensors[0].1.shape()[self.2].to_usize().unwrap();

            // Setup command queue / command buffer / encoder
            let command_queue = self.1.new_command_queue();
            let command_buffer = command_queue.new_command_buffer();
            let encoder = command_buffer
                .compute_command_encoder_with_descriptor(ComputePassDescriptor::new());
            encoder.set_compute_pipeline_state(&self.0);

            // Set inputs
            encoder.set_buffer(0, Some(a), 0);
            encoder.set_buffer(1, Some(&out), 0);
            encoder.set_int(2, inp_size as u32);
            encoder.set_int(3, front_size as u32);
            encoder.set_int(4, back_size as u32);
            encoder.set_int(5, dim_size as u32);
            input_dyn_dims(&[(self.3, tensors[0].1)], encoder, 6);

            // Execute
            encoder.dispatch_n_elements(inp_size);
            encoder.end_encoding();
            command_buffer.commit();
            command_buffer.wait_until_completed();

            vec![Tensor {
                data: Box::new(out),
            }]
        })
    }
}

#[derive(Default)]
pub struct PrimitiveOptimizer;

impl GraphOptimizer for PrimitiveOptimizer {
    fn optimize(&self, graph: &mut Graph) {
        let dev = Device::system_default().unwrap();
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
            if graph
                .graph
                .edges_directed(function_node, petgraph::Direction::Incoming)
                .count()
                != 0
            {
                move_references(
                    &mut graph.id_remap,
                    &mut graph.no_delete,
                    &mut graph.to_retrieve,
                    function_node,
                    copy_node,
                );
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
                        .map(|i| i.weight().2)
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
                        .next()
                        .unwrap()
                        .id(),
                )
            })
            .collect::<Vec<_>>()
        {
            // Create copy node
            let (source, shape) = (
                graph.graph.edge_endpoints(edge).unwrap().0,
                graph.graph.edge_weight(edge).unwrap().2,
            );
            let copy_node = graph
                .add_op(MetalCopyFromDevice(dev.clone()))
                .input(source, 0, shape)
                .finish();
            graph.graph.add_edge(copy_node, output_node, (0, 0, shape));
            graph.graph.remove_edge(edge);
        }

        // Swap primitive ops
        let mut kernels = HashMap::new();
        for id in graph.graph.node_indices().collect::<Vec<_>>() {
            let src_shapes = graph
                .graph
                .edges_directed(id, petgraph::Direction::Incoming)
                .sorted_by_key(|e| e.weight().0)
                .map(|e| e.weight().2)
                .collect::<Vec<_>>();
            let op = graph.graph.node_weight(id).unwrap().as_any().type_id();
            let op_ref = graph.graph.node_weight_mut(id).unwrap();
            if is::<Log2>(op) {
                *op_ref = Box::new(MetalLog2::new(dev.clone(), &mut kernels));
            } else if is::<Exp2>(op) {
                *op_ref = Box::new(MetalExp2::new(dev.clone(), &mut kernels));
            } else if is::<Sin>(op) {
                *op_ref = Box::new(MetalSin::new(dev.clone(), &mut kernels));
            } else if is::<Sqrt>(op) {
                *op_ref = Box::new(MetalSqrt::new(dev.clone(), &mut kernels));
            } else if is::<Recip>(op) {
                *op_ref = Box::new(MetalRecip::new(dev.clone(), &mut kernels));
            } else if is::<Add>(op) {
                *op_ref = Box::new(MetalAdd::new(
                    src_shapes[0],
                    src_shapes[1],
                    dev.clone(),
                    &mut kernels,
                ));
            } else if is::<Mul>(op) {
                *op_ref = Box::new(MetalMul::new(
                    src_shapes[0],
                    src_shapes[1],
                    dev.clone(),
                    &mut kernels,
                ));
            } else if is::<LessThan>(op) {
                *op_ref = Box::new(MetalLessThan::new(
                    src_shapes[0],
                    src_shapes[1],
                    dev.clone(),
                    &mut kernels,
                ));
            } else if is::<Mod>(op) {
                *op_ref = Box::new(MetalMod::new(
                    src_shapes[0],
                    src_shapes[1],
                    dev.clone(),
                    &mut kernels,
                ));
            } else if let Some(SumReduce(dim)) = op_ref.as_any().downcast_ref() {
                *op_ref = Box::new(MetalSumReduce::new(
                    src_shapes[0],
                    *dim,
                    dev.clone(),
                    &mut kernels,
                ));
            } else if let Some(MaxReduce(dim)) = op_ref.as_any().downcast_ref() {
                *op_ref = Box::new(MetalMaxReduce::new(
                    src_shapes[0],
                    *dim,
                    dev.clone(),
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
    fn dispatch_n_elements(&self, n: usize);
}

impl DispatchNElements for ComputeCommandEncoderRef {
    fn dispatch_n_elements(&self, n: usize) {
        let n_threads = (n + 32) / 32;
        self.dispatch_thread_groups(
            MTLSize {
                width: n_threads as NSUInteger,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 32,
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
    shapes: &[(ShapeTracker, ShapeTracker)],
    encoder: &ComputeCommandEncoderRef,
    offset: usize,
) {
    let mut added = HashSet::new();
    for (d1, d2) in shapes
        .iter()
        .flat_map(|(a, b)| a.shape().into_iter().zip(b.shape()))
    {
        if let Dim::Unknown(c) = d1 {
            if !added.contains(&c) {
                encoder.set_int(offset + added.len(), d2.to_usize().unwrap() as u32);
                added.insert(c);
            }
        }
    }
}

fn render_dyn_dim_inputs(shapes: &[ShapeTracker], offset: usize) -> String {
    shapes
        .iter()
        .flat_map(|st| st.shape())
        .filter_map(|d| {
            if let Dim::Unknown(c) = d {
                Some(c)
            } else {
                None
            }
        })
        .unique()
        .enumerate()
        .map(|(i, c)| format!(", device uint& {c} [[buffer({})]]", i + offset))
        .collect::<String>()
}

fn get_idx_valid_exps(shape: ShapeTracker) -> (String, String) {
    let min_re = Regex::new(r"min\((.*?), (.*?)\)").unwrap();
    let max_re = Regex::new(r"max\((.*?), (.*?)\)").unwrap();
    let idx_exp = shape.index_expression().to_string();
    let idx_exp = max_re.replace_all(&idx_exp, "max((uint)($1), (uint)($2))");
    let idx_exp = min_re.replace_all(&idx_exp, "min((uint)($1), (uint)($2))");
    let val_exp = shape.valid_expression().to_string();
    let val_exp = max_re.replace_all(&val_exp, "max((uint)($1), (uint)($2))");
    let val_exp = min_re.replace_all(&val_exp, "min((uint)($1), (uint)($2))");
    (idx_exp.to_string(), val_exp.to_string())
}
