use std::{any::Any, fmt::Debug, marker::PhantomData, mem::size_of, sync::Arc};

use super::*;
use metal_rs::*;
use objc::rc::autoreleasepool;
use petgraph::visit::EdgeRef;
use rustc_hash::FxHashMap;

use luminal::{
    op::{Function as LFunction, *},
    prelude::*,
};

/// Copy a tensor to the GPU
#[derive(LuminalEqTrue, LuminalPrint, Clone)]
pub struct MetalCopyToDevice<T>(Device, PhantomData<T>);

impl<T> MetalCopyToDevice<T> {
    pub fn new(dev: Device) -> Self {
        Self(dev, Default::default())
    }
}

impl<T: MetalFloat> Operator for MetalCopyToDevice<T> {
    fn process(&mut self, mut inp: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        if inp[0].0.borrowed().data.as_any().is::<MetalBuffer>() {
            // Already on device
            return vec![inp.pop().unwrap().0.cloned()];
        }
        let mut data = inp[0]
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
        if data.len() == 0 {
            data.push(T::from_f32(0.0));
        }
        let buffer = self.0.new_buffer_with_bytes_no_copy(
            data.as_ptr() as *mut _,
            (data.len() * std::mem::size_of::<T>()) as u64,
            MTLResourceOptions::StorageModeShared,
            None,
        );
        data.leak();
        vec![Tensor::new(MetalBuffer(buffer))]
    }
}

/// Copy a tensor from the GPU
#[derive(LuminalEqTrue, LuminalPrint, Clone)]
pub struct MetalCopyFromDevice<T>(Device, PhantomData<T>);

impl<T> MetalCopyFromDevice<T> {
    pub fn new(dev: Device) -> Self {
        Self(dev, Default::default())
    }
}

impl<T: MetalFloat> Operator for MetalCopyFromDevice<T> {
    fn process(&mut self, mut inp: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        if inp[0].0.borrowed().data.as_any().is::<Vec<f32>>() {
            // Already off device
            return vec![inp.pop().unwrap().0.cloned()];
        }
        let buffer = get_buffer_from_tensor(&inp[0].0);
        let mut data = vec![0.0; buffer.length() as usize / std::mem::size_of::<T>()];
        let ptr = buffer.contents() as *mut T;
        for (i, d) in data.iter_mut().enumerate() {
            *d = unsafe { *ptr.add(i) }.to_f32();
        }

        vec![Tensor {
            data: Box::new(data),
        }]
    }

    fn custom(&mut self, key: &str, _: Box<dyn Any>) -> Option<Box<dyn Any>> {
        // This op can accept non contiguous inputs
        if key == "non_contiguous" {
            return Some(Box::new(()));
        }
        None
    }
}

#[derive(Clone)]
pub struct MetalConstant<T>(
    pub ConstantValue,
    pub Device,
    pub *const FxHashMap<char, usize>,
    pub PhantomData<T>,
);

impl<T> PartialEq for MetalConstant<T> {
    fn eq(&self, other: &Self) -> bool {
        self.0.eq(&other.0)
    }
}

impl<T> Debug for MetalConstant<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "MetalConstant({:?})", self.0)
    }
}

impl<T: MetalFloat> Operator for MetalConstant<T> {
    fn process(&mut self, _: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        let val = T::from_f32(match &self.0 {
            ConstantValue::Expression(e) => {
                e.exec(unsafe { self.2.as_ref().unwrap() }).unwrap() as f32
            }
            ConstantValue::Float(f) => *f,
        });
        vec![Tensor {
            data: Box::new(MetalBuffer(self.1.new_buffer_with_data(
                &val as *const T as *const _,
                std::mem::size_of::<T>() as u64,
                MTLResourceOptions::StorageModeShared,
            ))),
        }]
    }

    fn custom(&mut self, key: &str, _: Box<dyn Any>) -> Option<Box<dyn Any>> {
        if key == "elementwise" {
            if let ConstantValue::Float(f) = self.0 {
                return Some(Box::new(f.to_string()));
            }
        }
        None
    }
}

#[derive(LuminalEqTrue, LuminalPrint, Clone)]
pub struct MetalContiguous<T> {
    pipeline: ComputePipelineState,
    queue: CommandQueue,
    device: Device,
    dyn_symbols: Vec<char>,
    _phantom: PhantomData<T>,
    dyn_map: *const FxHashMap<char, usize>,
}

impl<T: MetalFloat> MetalContiguous<T> {
    pub fn new(
        shape: ShapeTracker,
        device: Device,
        queue: CommandQueue,
        dyn_map: *const FxHashMap<char, usize>,
    ) -> Self {
        let (idx_exp, valid_exp) = get_idx_valid_exps(shape);
        let (dyn_symbols, rendered) = render_dyn_dim_inputs(&[shape], 3);
        let type_name = T::type_name();
        let code = format!("
#include <metal_stdlib>
using namespace metal;
kernel void mkernel(device {type_name} *inp [[buffer(0)]], device {type_name} *out [[buffer(1)]], device int& n_elements [[buffer(2)]], uint idx [[thread_position_in_grid]]{rendered}) {{
    if (idx < n_elements && {valid_exp} != 0) {{
        out[idx] = inp[{idx_exp}];
    }}
}}
");
        Self {
            pipeline: compile_function("mkernel", &code, &device),
            queue,
            device,
            dyn_symbols,
            _phantom: Default::default(),
            dyn_map,
        }
    }
}

impl<T> MetalKernel for MetalContiguous<T> {
    fn output_buffer_sizes(&self, input_shapes: &[ShapeTracker]) -> Vec<BigExpression> {
        vec![input_shapes[0].contiguous().n_elements() * size_of::<T>()]
    }
    fn metal_forward(
        &self,
        inputs: &[(&Buffer, ShapeTracker)],
        command_buffer: &CommandBufferRef,
        _: &[&Buffer],
        output_buffers: &[&Buffer],
    ) {
        let inp_size = inputs[0].1.n_elements().to_usize().unwrap();
        let encoder =
            command_buffer.compute_command_encoder_with_descriptor(ComputePassDescriptor::new());
        encoder.set_compute_pipeline_state(&self.pipeline);

        // Set inputs
        encoder.set_buffer(0, Some(inputs[0].0), 0);
        encoder.set_buffer(1, Some(output_buffers[0]), 0);
        encoder.set_u32(2, inp_size as u32);
        input_dyn_dims(
            &self.dyn_symbols,
            unsafe { self.dyn_map.as_ref().unwrap() },
            encoder,
            3,
        );

        // Execute
        encoder.dispatch_1d(inp_size);
        encoder.end_encoding();
    }
}

impl<T: MetalFloat> Operator for MetalContiguous<T> {
    fn process(&mut self, tensors: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        autoreleasepool(|| {
            // Setup command buffer and output buffer
            let command_buffer = self.queue.new_command_buffer();
            let inp_size = tensors[0].1.n_elements().to_usize().unwrap();
            let out = self.device.new_buffer(
                (inp_size * size_of::<T>()) as u64,
                MTLResourceOptions::StorageModeShared,
            );

            // Schedule op on the command buffer
            self.metal_forward(
                &[(get_buffer_from_tensor(&tensors[0].0), tensors[0].1)],
                command_buffer,
                &[],
                &[&out],
            );

            // Run the command buffer
            command_buffer.commit();
            command_buffer.wait_until_completed();

            vec![Tensor::new(MetalBuffer(out))]
        })
    }

    fn custom(&mut self, key: &str, input: Box<dyn Any>) -> Option<Box<dyn Any>> {
        if key == "metal" {
            return Some(Box::new(MetalKernelWrapper(Arc::new(Box::new(
                self.clone(),
            )))));
        }
        // This op can accept non contiguous inputs
        if key == "non_contiguous" {
            return Some(Box::new(()));
        }
        if key == "recompile_shapes" {
            if let Some(input_shapes) = input.downcast_ref::<Vec<ShapeTracker>>() {
                *self = Self::new(
                    input_shapes[0],
                    self.device.clone(),
                    self.queue.clone(),
                    self.dyn_map,
                )
            }
        }
        if key == "elementwise" {
            return Some(Box::new("input0".to_string()));
        }
        None
    }
}

#[derive(LuminalEqTrue, LuminalPrint, Clone)]
pub struct MetalLog2<T> {
    pipeline: ComputePipelineState,
    queue: CommandQueue,
    device: Device,
    _phantom: PhantomData<T>,
}

impl<T: MetalFloat> MetalLog2<T> {
    pub fn new(device: Device, queue: CommandQueue) -> Self {
        let type_name = T::type_name();
        let code = format!("
#include <metal_stdlib>
using namespace metal;
kernel void mkernel(device {type_name} *inp [[buffer(0)]], device {type_name} *out [[buffer(1)]], device int& n_elements [[buffer(2)]], uint idx [[thread_position_in_grid]]) {{
    if (idx < n_elements) {{
        out[idx] = log2(inp[idx]);
    }}
}}");
        Self {
            pipeline: compile_function("mkernel", &code, &device),
            queue,
            device,
            _phantom: Default::default(),
        }
    }
}

impl<T> MetalKernel for MetalLog2<T> {
    fn output_buffer_sizes(&self, input_shapes: &[ShapeTracker]) -> Vec<BigExpression> {
        vec![input_shapes[0].n_physical_elements() * size_of::<T>()]
    }
    fn metal_forward(
        &self,
        inputs: &[(&Buffer, ShapeTracker)],
        command_buffer: &CommandBufferRef,
        _: &[&Buffer],
        output_buffers: &[&Buffer],
    ) {
        let inp_size = inputs[0].1.n_physical_elements().to_usize().unwrap();
        let encoder =
            command_buffer.compute_command_encoder_with_descriptor(ComputePassDescriptor::new());
        encoder.set_compute_pipeline_state(&self.pipeline);

        // Set function inputs
        encoder.set_buffer(0, Some(inputs[0].0), 0);
        encoder.set_buffer(1, Some(output_buffers[0]), 0);
        encoder.set_u32(2, inp_size as u32);

        // Execute
        encoder.dispatch_1d(inp_size);
        encoder.end_encoding();
    }
}

impl<T: MetalFloat> Operator for MetalLog2<T> {
    fn process(&mut self, tensors: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        autoreleasepool(|| {
            let command_buffer = self.queue.new_command_buffer();
            let inp_size = tensors[0].1.n_physical_elements().to_usize().unwrap();
            let out = self.device.new_buffer(
                (inp_size * std::mem::size_of::<T>()) as u64,
                MTLResourceOptions::StorageModeShared,
            );

            self.metal_forward(
                &[(get_buffer_from_tensor(&tensors[0].0), tensors[0].1)],
                command_buffer,
                &[],
                &[&out],
            );

            command_buffer.commit();
            command_buffer.wait_until_completed();

            vec![Tensor::new(MetalBuffer(out))]
        })
    }

    fn custom(&mut self, key: &str, _: Box<dyn Any>) -> Option<Box<dyn Any>> {
        if key == "metal" {
            return Some(Box::new(MetalKernelWrapper(Arc::new(Box::new(
                self.clone(),
            )))));
        }
        if key == "elementwise" {
            return Some(Box::new("log2(input0)".to_string()));
        }
        // This op can accept non contiguous inputs
        if key == "non_contiguous" {
            return Some(Box::new(()));
        }
        None
    }
}

#[derive(LuminalEqTrue, LuminalPrint, Clone)]
pub struct MetalExp2<T> {
    pipeline: ComputePipelineState,
    queue: CommandQueue,
    device: Device,
    _phantom: PhantomData<T>,
}

impl<T: MetalFloat> MetalExp2<T> {
    pub fn new(device: Device, queue: CommandQueue) -> Self {
        let type_name = T::type_name();
        let code = format!("
#include <metal_stdlib>
using namespace metal;
kernel void mkernel(device {type_name} *inp [[buffer(0)]], device {type_name} *out [[buffer(1)]], device int& n_elements [[buffer(2)]], uint idx [[thread_position_in_grid]]) {{
    if (idx < n_elements) {{
        out[idx] = exp2(inp[idx]);
    }}
}}");
        Self {
            pipeline: compile_function("mkernel", &code, &device),
            queue,
            device,
            _phantom: Default::default(),
        }
    }
}
impl<T> MetalKernel for MetalExp2<T> {
    fn output_buffer_sizes(&self, input_shapes: &[ShapeTracker]) -> Vec<BigExpression> {
        vec![input_shapes[0].n_physical_elements() * size_of::<T>()]
    }
    fn metal_forward(
        &self,
        inputs: &[(&Buffer, ShapeTracker)],
        command_buffer: &CommandBufferRef,
        _: &[&Buffer],
        output_buffers: &[&Buffer],
    ) {
        let inp_size = inputs[0].1.n_physical_elements().to_usize().unwrap();
        let encoder =
            command_buffer.compute_command_encoder_with_descriptor(ComputePassDescriptor::new());
        encoder.set_compute_pipeline_state(&self.pipeline);

        // Set function inputs
        encoder.set_buffer(0, Some(inputs[0].0), 0);
        encoder.set_buffer(1, Some(output_buffers[0]), 0);
        encoder.set_u32(2, inp_size as u32);

        // Execute
        encoder.dispatch_1d(inp_size);
        encoder.end_encoding();
    }
}

impl<T: MetalFloat> Operator for MetalExp2<T> {
    fn process(&mut self, tensors: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        autoreleasepool(|| {
            let command_buffer = self.queue.new_command_buffer();

            let inp_size = tensors[0].1.n_physical_elements().to_usize().unwrap();
            let out = self.device.new_buffer(
                (inp_size * std::mem::size_of::<T>()) as u64,
                MTLResourceOptions::StorageModeShared,
            );
            self.metal_forward(
                &[(get_buffer_from_tensor(&tensors[0].0), tensors[0].1)],
                command_buffer,
                &[],
                &[&out],
            );

            command_buffer.commit();
            command_buffer.wait_until_completed();

            vec![Tensor::new(MetalBuffer(out))]
        })
    }

    fn custom(&mut self, key: &str, _: Box<dyn Any>) -> Option<Box<dyn Any>> {
        if key == "metal" {
            return Some(Box::new(MetalKernelWrapper(Arc::new(Box::new(
                self.clone(),
            )))));
        }
        if key == "elementwise" {
            return Some(Box::new("exp2(input0)".to_string()));
        }
        // This op can accept non contiguous inputs
        if key == "non_contiguous" {
            return Some(Box::new(()));
        }
        None
    }
}

#[derive(LuminalEqTrue, LuminalPrint, Clone)]
pub struct MetalSin<T> {
    pipeline: ComputePipelineState,
    queue: CommandQueue,
    device: Device,
    _phantom: PhantomData<T>,
}

impl<T: MetalFloat> MetalSin<T> {
    pub fn new(device: Device, queue: CommandQueue) -> Self {
        let type_name = T::type_name();
        let code = format!("#include <metal_stdlib>
using namespace metal;
kernel void mkernel(device {type_name} *inp [[buffer(0)]], device {type_name} *out [[buffer(1)]], device int& n_elements [[buffer(2)]], uint idx [[thread_position_in_grid]]) {{
    if (idx < n_elements) {{
        out[idx] = ({type_name})sin((float)inp[idx]);
    }}
}}");
        Self {
            pipeline: compile_function("mkernel", &code, &device),
            queue,
            device,
            _phantom: Default::default(),
        }
    }
}
impl<T> MetalKernel for MetalSin<T> {
    fn output_buffer_sizes(&self, input_shapes: &[ShapeTracker]) -> Vec<BigExpression> {
        vec![input_shapes[0].n_physical_elements() * size_of::<T>()]
    }
    fn metal_forward(
        &self,
        inputs: &[(&Buffer, ShapeTracker)],
        command_buffer: &CommandBufferRef,
        _: &[&Buffer],
        output_buffers: &[&Buffer],
    ) {
        let inp_size = inputs[0].1.n_physical_elements().to_usize().unwrap();

        let encoder =
            command_buffer.compute_command_encoder_with_descriptor(ComputePassDescriptor::new());
        encoder.set_compute_pipeline_state(&self.pipeline);

        // Set function inputs
        encoder.set_buffer(0, Some(inputs[0].0), 0);
        encoder.set_buffer(1, Some(output_buffers[0]), 0);
        encoder.set_u32(2, inp_size as u32);

        // Execute
        encoder.dispatch_1d(inp_size);
        encoder.end_encoding();
    }
}

impl<T: MetalFloat> Operator for MetalSin<T> {
    fn process(&mut self, tensors: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        autoreleasepool(|| {
            let command_buffer = self.queue.new_command_buffer();
            let inp_size = tensors[0].1.n_physical_elements().to_usize().unwrap();
            let out = self.device.new_buffer(
                (inp_size * std::mem::size_of::<T>()) as u64,
                MTLResourceOptions::StorageModeShared,
            );

            self.metal_forward(
                &[(get_buffer_from_tensor(&tensors[0].0), tensors[0].1)],
                command_buffer,
                &[],
                &[&out],
            );

            command_buffer.commit();
            command_buffer.wait_until_completed();

            vec![Tensor::new(MetalBuffer(out))]
        })
    }

    fn custom(&mut self, key: &str, _: Box<dyn Any>) -> Option<Box<dyn Any>> {
        if key == "metal" {
            return Some(Box::new(MetalKernelWrapper(Arc::new(Box::new(
                self.clone(),
            )))));
        }
        if key == "elementwise" {
            return Some(Box::new("sin(input0)".to_string()));
        }
        // This op can accept non contiguous inputs
        if key == "non_contiguous" {
            return Some(Box::new(()));
        }
        None
    }
}

#[derive(LuminalEqTrue, LuminalPrint, Clone)]
pub struct MetalSqrt<T> {
    pipeline: ComputePipelineState,
    queue: CommandQueue,
    device: Device,
    _phantom: PhantomData<T>,
}

impl<T: MetalFloat> MetalSqrt<T> {
    pub fn new(device: Device, queue: CommandQueue) -> Self {
        let type_name = T::type_name();
        let code = format!("
#include <metal_stdlib>
using namespace metal;
kernel void mkernel(device {type_name} *inp [[buffer(0)]], device {type_name} *out [[buffer(1)]], device int& n_elements [[buffer(2)]], uint idx [[thread_position_in_grid]]) {{
    if (idx < n_elements) {{
        out[idx] = sqrt(inp[idx]);
    }}
}}");
        Self {
            pipeline: compile_function("mkernel", &code, &device),
            queue,
            device,
            _phantom: Default::default(),
        }
    }
}
impl<T> MetalKernel for MetalSqrt<T> {
    fn output_buffer_sizes(&self, input_shapes: &[ShapeTracker]) -> Vec<BigExpression> {
        vec![input_shapes[0].n_physical_elements() * size_of::<T>()]
    }
    fn metal_forward(
        &self,
        inputs: &[(&Buffer, ShapeTracker)],
        command_buffer: &CommandBufferRef,
        _: &[&Buffer],
        output_buffers: &[&Buffer],
    ) {
        let inp_size = inputs[0].1.n_physical_elements().to_usize().unwrap();
        let encoder =
            command_buffer.compute_command_encoder_with_descriptor(ComputePassDescriptor::new());
        encoder.set_compute_pipeline_state(&self.pipeline);

        // Set function inputs
        encoder.set_buffer(0, Some(inputs[0].0), 0);
        encoder.set_buffer(1, Some(output_buffers[0]), 0);
        encoder.set_u32(2, inp_size as u32);

        // Execute
        encoder.dispatch_1d(inp_size);
        encoder.end_encoding();
    }
}

impl<T: MetalFloat> Operator for MetalSqrt<T> {
    fn process(&mut self, tensors: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        autoreleasepool(|| {
            let command_buffer = self.queue.new_command_buffer();
            let inp_size = tensors[0].1.n_physical_elements().to_usize().unwrap();
            let out = self.device.new_buffer(
                (inp_size * std::mem::size_of::<T>()) as u64,
                MTLResourceOptions::StorageModeShared,
            );

            self.metal_forward(
                &[(get_buffer_from_tensor(&tensors[0].0), tensors[0].1)],
                command_buffer,
                &[],
                &[&out],
            );

            command_buffer.commit();
            command_buffer.wait_until_completed();

            vec![Tensor::new(MetalBuffer(out))]
        })
    }

    fn custom(&mut self, key: &str, _: Box<dyn Any>) -> Option<Box<dyn Any>> {
        if key == "metal" {
            return Some(Box::new(MetalKernelWrapper(Arc::new(Box::new(
                self.clone(),
            )))));
        }
        if key == "elementwise" {
            return Some(Box::new("sqrt(input0)".to_string()));
        }
        // This op can accept non contiguous inputs
        if key == "non_contiguous" {
            return Some(Box::new(()));
        }
        None
    }
}

#[derive(LuminalEqTrue, LuminalPrint, Clone)]
pub struct MetalRecip<T> {
    pipeline: ComputePipelineState,
    queue: CommandQueue,
    device: Device,
    _phantom: PhantomData<T>,
}

impl<T: MetalFloat> MetalRecip<T> {
    pub fn new(device: Device, queue: CommandQueue) -> Self {
        let type_name = T::type_name();
        let code = format!("
#include <metal_stdlib>
using namespace metal;
kernel void mkernel(device {type_name} *inp [[buffer(0)]], device {type_name} *out [[buffer(1)]], device int& n_elements [[buffer(2)]], uint idx [[thread_position_in_grid]]) {{
    if (idx < n_elements) {{
        out[idx] = 1.0 / inp[idx];
    }}
}}");
        Self {
            pipeline: compile_function("mkernel", &code, &device),
            queue,
            device,
            _phantom: Default::default(),
        }
    }
}
impl<T> MetalKernel for MetalRecip<T> {
    fn output_buffer_sizes(&self, input_shapes: &[ShapeTracker]) -> Vec<BigExpression> {
        vec![input_shapes[0].n_physical_elements() * size_of::<T>()]
    }
    fn metal_forward(
        &self,
        inputs: &[(&Buffer, ShapeTracker)],
        command_buffer: &CommandBufferRef,
        _: &[&Buffer],
        output_buffers: &[&Buffer],
    ) {
        let inp_size = inputs[0].1.n_physical_elements().to_usize().unwrap();
        let encoder =
            command_buffer.compute_command_encoder_with_descriptor(ComputePassDescriptor::new());
        encoder.set_compute_pipeline_state(&self.pipeline);

        // Set function inputs
        encoder.set_buffer(0, Some(inputs[0].0), 0);
        encoder.set_buffer(1, Some(output_buffers[0]), 0);
        encoder.set_u32(2, inp_size as u32);

        // Execute
        encoder.dispatch_1d(inp_size);
        encoder.end_encoding();
    }
}

impl<T: MetalFloat> Operator for MetalRecip<T> {
    fn process(&mut self, tensors: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        autoreleasepool(|| {
            let command_buffer = self.queue.new_command_buffer();
            let inp_size = tensors[0].1.n_physical_elements().to_usize().unwrap();
            let out = self.device.new_buffer(
                (inp_size * std::mem::size_of::<T>()) as u64,
                MTLResourceOptions::StorageModeShared,
            );

            self.metal_forward(
                &[(get_buffer_from_tensor(&tensors[0].0), tensors[0].1)],
                command_buffer,
                &[],
                &[&out],
            );

            command_buffer.commit();
            command_buffer.wait_until_completed();

            vec![Tensor::new(MetalBuffer(out))]
        })
    }

    fn custom(&mut self, key: &str, _: Box<dyn Any>) -> Option<Box<dyn Any>> {
        if key == "metal" {
            return Some(Box::new(MetalKernelWrapper(Arc::new(Box::new(
                self.clone(),
            )))));
        }
        if key == "elementwise" {
            return Some(Box::new("1.0 / input0".to_string()));
        }
        // This op can accept non contiguous inputs
        if key == "non_contiguous" {
            return Some(Box::new(()));
        }
        None
    }
}

#[derive(LuminalPrint, LuminalEqTrue, Clone)]
pub struct MetalAdd<T> {
    pipeline: ComputePipelineState,
    queue: CommandQueue,
    device: Device,
    _phantom: PhantomData<T>,
    dyn_symbols: Vec<char>,
    dyn_map: *const FxHashMap<char, usize>,
}

impl<T: MetalFloat> MetalAdd<T> {
    pub fn new(
        a_shape: ShapeTracker,
        b_shape: ShapeTracker,
        device: Device,
        queue: CommandQueue,
        dyn_map: *const FxHashMap<char, usize>,
    ) -> Self {
        let (a_idx_exp, a_valid_exp) = get_idx_valid_exps(a_shape);
        let (b_idx_exp, b_valid_exp) = get_idx_valid_exps(b_shape);
        let (dyn_symbols, rendered) = render_dyn_dim_inputs(&[a_shape, b_shape], 4);
        let type_name = T::type_name();
        let code = format!("
#include <metal_stdlib>
using namespace metal;
kernel void mkernel(device {type_name} *inp_a [[buffer(0)]], device {type_name} *inp_b [[buffer(1)]], device {type_name} *out [[buffer(2)]], device int& n_elements [[buffer(3)]], uint idx [[thread_position_in_grid]]{rendered}) {{
    if (idx < n_elements) {{
        out[idx] =
            (({a_valid_exp}) == 0 ? 0.0h : inp_a[{a_idx_exp}])
            + (({b_valid_exp}) == 0 ? 0.0h : inp_b[{b_idx_exp}]);
    }}
}}
");
        Self {
            pipeline: compile_function("mkernel", &code, &device),
            queue,
            device,
            dyn_symbols,
            _phantom: Default::default(),
            dyn_map,
        }
    }
}

impl<T> MetalKernel for MetalAdd<T> {
    fn output_buffer_sizes(&self, input_shapes: &[ShapeTracker]) -> Vec<BigExpression> {
        vec![input_shapes[0].n_elements() * size_of::<T>()]
    }
    fn metal_forward(
        &self,
        inputs: &[(&Buffer, ShapeTracker)],
        command_buffer: &CommandBufferRef,
        _: &[&Buffer],
        output_buffers: &[&Buffer],
    ) {
        let inp_size = inputs[0].1.n_elements().to_usize().unwrap();
        let encoder =
            command_buffer.compute_command_encoder_with_descriptor(ComputePassDescriptor::new());
        encoder.set_compute_pipeline_state(&self.pipeline);

        // Set inputs
        encoder.set_buffer(0, Some(inputs[0].0), 0);
        encoder.set_buffer(1, Some(inputs[1].0), 0);
        encoder.set_buffer(2, Some(output_buffers[0]), 0);
        encoder.set_u32(3, inp_size as u32);
        input_dyn_dims(
            &self.dyn_symbols,
            unsafe { self.dyn_map.as_ref().unwrap() },
            &encoder,
            4,
        );
        // Execute
        encoder.dispatch_1d(inp_size);
        encoder.end_encoding();
    }
}

impl<T: MetalFloat> Operator for MetalAdd<T> {
    fn process(&mut self, tensors: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        autoreleasepool(|| {
            let command_buffer = self.queue.new_command_buffer();
            let inp_size = tensors[0].1.n_elements().to_usize().unwrap();
            let out = self.device.new_buffer(
                (inp_size * std::mem::size_of::<T>()) as u64,
                MTLResourceOptions::StorageModeShared,
            );

            self.metal_forward(
                &[
                    (get_buffer_from_tensor(&tensors[0].0), tensors[0].1),
                    (get_buffer_from_tensor(&tensors[1].0), tensors[1].1),
                ],
                command_buffer,
                &[],
                &[&out],
            );

            command_buffer.commit();
            command_buffer.wait_until_completed();

            vec![Tensor::new(MetalBuffer(out))]
        })
    }

    fn custom(&mut self, key: &str, input: Box<dyn Any>) -> Option<Box<dyn Any>> {
        if key == "metal" {
            return Some(Box::new(MetalKernelWrapper(Arc::new(Box::new(
                self.clone(),
            )))));
        }
        // This op can accept non contiguous inputs
        if key == "non_contiguous" {
            return Some(Box::new(()));
        }
        if key == "recompile_shapes" {
            if let Some(input_shapes) = input.downcast_ref::<Vec<ShapeTracker>>() {
                *self = Self::new(
                    input_shapes[0],
                    input_shapes[1],
                    self.device.clone(),
                    self.queue.clone(),
                    self.dyn_map,
                )
            }
        }
        if key == "elementwise" {
            return Some(Box::new("input0 + input1".to_string()));
        }
        None
    }
}

#[derive(LuminalPrint, LuminalEqTrue, Clone)]
pub struct MetalMul<T> {
    pipeline: ComputePipelineState,
    queue: CommandQueue,
    device: Device,
    dyn_symbols: Vec<char>,
    _phantom: PhantomData<T>,
    dyn_map: *const FxHashMap<char, usize>,
}

impl<T: MetalFloat> MetalMul<T> {
    pub fn new(
        a_shape: ShapeTracker,
        b_shape: ShapeTracker,
        device: Device,
        queue: CommandQueue,
        dyn_map: *const FxHashMap<char, usize>,
    ) -> Self {
        let (a_idx_exp, a_valid_exp) = get_idx_valid_exps(a_shape);
        let (b_idx_exp, b_valid_exp) = get_idx_valid_exps(b_shape);
        let (dyn_symbols, rendered) = render_dyn_dim_inputs(&[a_shape, b_shape], 4);
        let type_name = T::type_name();
        let code = format!("
#include <metal_stdlib>
using namespace metal;
kernel void mkernel(device {type_name} *inp_a [[buffer(0)]], device {type_name} *inp_b [[buffer(1)]], device {type_name} *out [[buffer(2)]], device int& n_elements [[buffer(3)]], uint idx [[thread_position_in_grid]]{rendered}) {{
    if (idx < n_elements) {{
        out[idx] =
            (({a_valid_exp}) == 0 ? 0.0h : inp_a[{a_idx_exp}])
            * (({b_valid_exp}) == 0 ? 0.0h : inp_b[{b_idx_exp}]);
    }}
}}
");
        Self {
            pipeline: compile_function("mkernel", &code, &device),
            queue,
            device,
            dyn_symbols,
            _phantom: Default::default(),
            dyn_map,
        }
    }
}
impl<T> MetalKernel for MetalMul<T> {
    fn output_buffer_sizes(&self, input_shapes: &[ShapeTracker]) -> Vec<BigExpression> {
        vec![input_shapes[0].n_elements() * size_of::<T>()]
    }
    fn metal_forward(
        &self,
        inputs: &[(&Buffer, ShapeTracker)],
        command_buffer: &CommandBufferRef,
        _: &[&Buffer],
        output_buffers: &[&Buffer],
    ) {
        let inp_size = inputs[0].1.n_elements().to_usize().unwrap();

        let encoder =
            command_buffer.compute_command_encoder_with_descriptor(ComputePassDescriptor::new());
        encoder.set_compute_pipeline_state(&self.pipeline);

        // Set inputs
        encoder.set_buffer(0, Some(inputs[0].0), 0);
        encoder.set_buffer(1, Some(inputs[1].0), 0);
        encoder.set_buffer(2, Some(output_buffers[0]), 0);
        encoder.set_u32(3, inp_size as u32);
        input_dyn_dims(
            &self.dyn_symbols,
            unsafe { self.dyn_map.as_ref().unwrap() },
            &encoder,
            4,
        );

        // Execute
        encoder.dispatch_1d(inp_size);
        encoder.end_encoding();
    }
}

impl<T: MetalFloat> Operator for MetalMul<T> {
    fn process(&mut self, tensors: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        autoreleasepool(|| {
            let command_buffer = self.queue.new_command_buffer();
            let inp_size = tensors[0].1.n_elements().to_usize().unwrap();
            let out = self.device.new_buffer(
                (inp_size * std::mem::size_of::<T>()) as u64,
                MTLResourceOptions::StorageModeShared,
            );

            self.metal_forward(
                &[
                    (get_buffer_from_tensor(&tensors[0].0), tensors[0].1),
                    (get_buffer_from_tensor(&tensors[1].0), tensors[1].1),
                ],
                command_buffer,
                &[],
                &[&out],
            );

            command_buffer.commit();
            command_buffer.wait_until_completed();

            vec![Tensor::new(MetalBuffer(out))]
        })
    }

    fn custom(&mut self, key: &str, input: Box<dyn Any>) -> Option<Box<dyn Any>> {
        if key == "metal" {
            return Some(Box::new(MetalKernelWrapper(Arc::new(Box::new(
                self.clone(),
            )))));
        }
        // This op can accept non contiguous inputs
        if key == "non_contiguous" {
            return Some(Box::new(()));
        }
        if key == "recompile_shapes" {
            if let Some(input_shapes) = input.downcast_ref::<Vec<ShapeTracker>>() {
                *self = Self::new(
                    input_shapes[0],
                    input_shapes[1],
                    self.device.clone(),
                    self.queue.clone(),
                    self.dyn_map,
                )
            }
        }
        if key == "elementwise" {
            return Some(Box::new("input0 * input1".to_string()));
        }
        None
    }
}

#[derive(LuminalEqTrue, LuminalPrint, Clone)]
pub struct MetalLessThan<T> {
    pipeline: ComputePipelineState,
    queue: CommandQueue,
    device: Device,
    dyn_symbols: Vec<char>,
    _phantom: PhantomData<T>,
    dyn_map: *const FxHashMap<char, usize>,
}

impl<T: MetalFloat> MetalLessThan<T> {
    pub fn new(
        a_shape: ShapeTracker,
        b_shape: ShapeTracker,
        device: Device,
        queue: CommandQueue,
        dyn_map: *const FxHashMap<char, usize>,
    ) -> Self {
        let (a_idx_exp, a_valid_exp) = get_idx_valid_exps(a_shape);
        let (b_idx_exp, b_valid_exp) = get_idx_valid_exps(b_shape);
        let type_name = T::type_name();
        let (dyn_symbols, rendered) = render_dyn_dim_inputs(&[a_shape, b_shape], 4);
        let code = format!("
#include <metal_stdlib>
using namespace metal;
kernel void mkernel(device {type_name} *inp_a [[buffer(0)]], device {type_name} *inp_b [[buffer(1)]], device {type_name} *out [[buffer(2)]], device int& n_elements [[buffer(3)]], uint idx [[thread_position_in_grid]]{rendered}) {{
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
", if T::is_f32() {"1.0"} else {"1.0h"},if T::is_f32() {"0.0"} else {"0.0h"},
        );
        Self {
            pipeline: compile_function("mkernel", &code, &device),
            queue,
            device,
            dyn_symbols,
            _phantom: Default::default(),
            dyn_map,
        }
    }
}

impl<T> MetalKernel for MetalLessThan<T> {
    fn output_buffer_sizes(&self, input_shapes: &[ShapeTracker]) -> Vec<BigExpression> {
        vec![input_shapes[0].n_elements() * size_of::<T>()]
    }
    fn metal_forward(
        &self,
        inputs: &[(&Buffer, ShapeTracker)],
        command_buffer: &CommandBufferRef,
        _: &[&Buffer],
        output_buffers: &[&Buffer],
    ) {
        let inp_size = inputs[0].1.n_elements().to_usize().unwrap();

        let encoder =
            command_buffer.compute_command_encoder_with_descriptor(ComputePassDescriptor::new());
        encoder.set_compute_pipeline_state(&self.pipeline);

        // Set inputs
        encoder.set_buffer(0, Some(inputs[0].0), 0);
        encoder.set_buffer(1, Some(inputs[1].0), 0);
        encoder.set_buffer(2, Some(output_buffers[0]), 0);
        encoder.set_u32(3, inp_size as u32);
        input_dyn_dims(
            &self.dyn_symbols,
            unsafe { self.dyn_map.as_ref().unwrap() },
            &encoder,
            4,
        );

        // Execute
        encoder.dispatch_1d(inp_size);
        encoder.end_encoding();
    }
}

impl<T: MetalFloat> Operator for MetalLessThan<T> {
    fn process(&mut self, tensors: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        autoreleasepool(|| {
            let command_buffer = self.queue.new_command_buffer();
            let inp_size = tensors[0].1.n_elements().to_usize().unwrap();
            let out = self.device.new_buffer(
                (inp_size * std::mem::size_of::<T>()) as u64,
                MTLResourceOptions::StorageModeShared,
            );

            self.metal_forward(
                &[
                    (get_buffer_from_tensor(&tensors[0].0), tensors[0].1),
                    (get_buffer_from_tensor(&tensors[1].0), tensors[1].1),
                ],
                command_buffer,
                &[],
                &[&out],
            );

            command_buffer.commit();
            command_buffer.wait_until_completed();

            vec![Tensor::new(MetalBuffer(out))]
        })
    }

    fn custom(&mut self, key: &str, input: Box<dyn Any>) -> Option<Box<dyn Any>> {
        if key == "metal" {
            return Some(Box::new(MetalKernelWrapper(Arc::new(Box::new(
                self.clone(),
            )))));
        }
        // This op can accept non contiguous inputs
        if key == "non_contiguous" {
            return Some(Box::new(()));
        }
        if key == "recompile_shapes" {
            if let Some(input_shapes) = input.downcast_ref::<Vec<ShapeTracker>>() {
                *self = Self::new(
                    input_shapes[0],
                    input_shapes[1],
                    self.device.clone(),
                    self.queue.clone(),
                    self.dyn_map,
                )
            }
        }
        if key == "elementwise" {
            return Some(Box::new("(float)(input0 < input1)".to_string()));
        }
        None
    }
}

#[derive(LuminalEqTrue, LuminalPrint, Clone)]
pub struct MetalMod<T> {
    pipeline: ComputePipelineState,
    queue: CommandQueue,
    device: Device,
    dyn_symbols: Vec<char>,
    _phantom: PhantomData<T>,
    dyn_map: *const FxHashMap<char, usize>,
}

impl<T: MetalFloat> MetalMod<T> {
    pub fn new(
        a_shape: ShapeTracker,
        b_shape: ShapeTracker,
        device: Device,
        queue: CommandQueue,
        dyn_map: *const FxHashMap<char, usize>,
    ) -> Self {
        let (a_idx_exp, a_valid_exp) = get_idx_valid_exps(a_shape);
        let (b_idx_exp, b_valid_exp) = get_idx_valid_exps(b_shape);
        let (dyn_symbols, rendered) = render_dyn_dim_inputs(&[a_shape, b_shape], 4);
        let type_name = T::type_name();
        let code = format!(
            "
#include <metal_stdlib>
using namespace metal;
kernel void mkernel(device {type_name} *inp_a [[buffer(0)]], device {type_name} *inp_b [[buffer(1)]], device {type_name} *out [[buffer(2)]], device int& n_elements [[buffer(3)]], uint idx [[thread_position_in_grid]]{rendered}) {{
    if (idx < n_elements) {{
        out[idx] = fmod(({a_valid_exp}) == 0 ? 0.0 : inp_a[{a_idx_exp}], ({b_valid_exp}) == 0 ? 0.0 : inp_b[{b_idx_exp}]);
    }}
}}
");
        Self {
            pipeline: compile_function("mkernel", &code, &device),
            queue,
            device,
            dyn_symbols,
            _phantom: Default::default(),
            dyn_map,
        }
    }
}
impl<T> MetalKernel for MetalMod<T> {
    fn output_buffer_sizes(&self, input_shapes: &[ShapeTracker]) -> Vec<BigExpression> {
        vec![input_shapes[0].n_elements() * size_of::<T>()]
    }
    fn metal_forward(
        &self,
        inputs: &[(&Buffer, ShapeTracker)],
        command_buffer: &CommandBufferRef,
        _: &[&Buffer],
        output_buffers: &[&Buffer],
    ) {
        let inp_size = inputs[0].1.n_elements().to_usize().unwrap();

        let encoder =
            command_buffer.compute_command_encoder_with_descriptor(ComputePassDescriptor::new());
        encoder.set_compute_pipeline_state(&self.pipeline);

        // Set inputs
        encoder.set_buffer(0, Some(inputs[0].0), 0);
        encoder.set_buffer(1, Some(inputs[1].0), 0);
        encoder.set_buffer(2, Some(output_buffers[0]), 0);
        encoder.set_u32(3, inp_size as u32);
        input_dyn_dims(
            &self.dyn_symbols,
            unsafe { self.dyn_map.as_ref().unwrap() },
            &encoder,
            4,
        );
        // Execute
        encoder.dispatch_1d(inp_size);
        encoder.end_encoding();
    }
}

impl<T: MetalFloat> Operator for MetalMod<T> {
    fn process(&mut self, tensors: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        autoreleasepool(|| {
            let command_buffer = self.queue.new_command_buffer();
            let inp_size = tensors[0].1.n_elements().to_usize().unwrap();
            let out = self.device.new_buffer(
                (inp_size * std::mem::size_of::<T>()) as u64,
                MTLResourceOptions::StorageModeShared,
            );

            self.metal_forward(
                &[
                    (get_buffer_from_tensor(&tensors[0].0), tensors[0].1),
                    (get_buffer_from_tensor(&tensors[1].0), tensors[1].1),
                ],
                command_buffer,
                &[],
                &[&out],
            );

            command_buffer.commit();
            command_buffer.wait_until_completed();

            vec![Tensor::new(MetalBuffer(out))]
        })
    }

    fn custom(&mut self, key: &str, input: Box<dyn Any>) -> Option<Box<dyn Any>> {
        if key == "metal" {
            return Some(Box::new(MetalKernelWrapper(Arc::new(Box::new(
                self.clone(),
            )))));
        }
        // This op can accept non contiguous inputs
        if key == "non_contiguous" {
            return Some(Box::new(()));
        }
        if key == "recompile_shapes" {
            if let Some(input_shapes) = input.downcast_ref::<Vec<ShapeTracker>>() {
                *self = Self::new(
                    input_shapes[0],
                    input_shapes[1],
                    self.device.clone(),
                    self.queue.clone(),
                    self.dyn_map,
                )
            }
        }
        if key == "elementwise" {
            return Some(Box::new("fmod(input0, input1)".to_string()));
        }
        None
    }
}

#[derive(LuminalPrint, Clone)]
pub struct MetalSumReduce<T> {
    pipeline: ComputePipelineState,
    queue: CommandQueue,
    device: Device,
    pub dim: usize,
    dyn_symbols: Vec<char>,
    _phantom: PhantomData<T>,
    dyn_map: *const FxHashMap<char, usize>,
}

impl<T> PartialEq for MetalSumReduce<T> {
    fn eq(&self, other: &Self) -> bool {
        self.dim == other.dim
    }
}

impl<T: MetalFloat> MetalSumReduce<T> {
    pub fn new(
        shape: ShapeTracker,
        dim: usize,
        device: Device,
        queue: CommandQueue,
        dyn_map: *const FxHashMap<char, usize>,
    ) -> Self {
        let (idx_exp, valid_exp) = get_idx_valid_exps(shape);
        let (dyn_symbols, rendered) = render_dyn_dim_inputs(&[shape], 6);
        let type_name = T::type_name();
        let code = format!("
#include <metal_stdlib>
using namespace metal;
kernel void mkernel(device {type_name} *inp [[buffer(0)]], device {type_name} *out [[buffer(1)]], device int& n_elements [[buffer(2)]], device int& front_size [[buffer(3)]], device int& back_size [[buffer(4)]], device int& dim_size [[buffer(5)]], uint i_ [[thread_position_in_grid]]{rendered}) {{
    if (i_ < n_elements) {{
        int a_ = i_ / back_size;
        int b_ = i_ % back_size;
        {type_name} reduce_value = 0.0;
        for (int c_ = 0; c_ < dim_size; c_++) {{
            uint idx = a_ * dim_size * back_size + c_ * back_size + b_;
            if (({valid_exp}) != 0) {{
                reduce_value += inp[{idx_exp}];
            }}
        }}
        out[i_] = reduce_value;
    }}
}}
");
        Self {
            pipeline: compile_function("mkernel", &code, &device),
            queue,
            device,
            dim,
            dyn_symbols,
            _phantom: Default::default(),
            dyn_map,
        }
    }
}

impl<T> MetalKernel for MetalSumReduce<T> {
    fn output_buffer_sizes(&self, input_shapes: &[ShapeTracker]) -> Vec<BigExpression> {
        let mut sh = input_shapes[0];
        sh.remove_dim(self.dim);
        vec![sh.n_elements() * size_of::<T>()]
    }
    fn metal_forward(
        &self,
        inputs: &[(&Buffer, ShapeTracker)],
        command_buffer: &CommandBufferRef,
        _: &[&Buffer],
        output_buffers: &[&Buffer],
    ) {
        let mut sh = inputs[0].1;
        sh.remove_dim(self.dim);
        let inp_size = sh.n_elements().to_usize().unwrap();
        let front_size: usize = inputs[0]
            .1
            .shape()
            .iter()
            .take(self.dim)
            .map(|i| i.to_usize().unwrap())
            .product();
        let back_size: usize = inputs[0]
            .1
            .shape()
            .iter()
            .skip(self.dim + 1)
            .map(|i| i.to_usize().unwrap())
            .product();
        let dim_size = inputs[0].1.shape()[self.dim].to_usize().unwrap();

        let encoder =
            command_buffer.compute_command_encoder_with_descriptor(ComputePassDescriptor::new());
        encoder.set_compute_pipeline_state(&self.pipeline);

        // Set inputs
        encoder.set_buffer(0, Some(inputs[0].0), 0);
        encoder.set_buffer(1, Some(output_buffers[0]), 0);
        encoder.set_u32(2, inp_size as u32);
        encoder.set_u32(3, front_size as u32);
        encoder.set_u32(4, back_size as u32);
        encoder.set_u32(5, dim_size as u32);
        input_dyn_dims(
            &self.dyn_symbols,
            unsafe { self.dyn_map.as_ref().unwrap() },
            &encoder,
            6,
        );

        // Execute
        encoder.dispatch_1d(inp_size);
        encoder.end_encoding();
    }
}

impl<T: MetalFloat> Operator for MetalSumReduce<T> {
    fn process(&mut self, tensors: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        autoreleasepool(|| {
            // Setup command queue / command buffer / encoder
            let command_buffer = self.queue.new_command_buffer();
            let mut sh = tensors[0].1;
            sh.remove_dim(self.dim);
            let inp_size = sh.n_elements().to_usize().unwrap();
            let out = self.device.new_buffer(
                (inp_size * std::mem::size_of::<T>()) as u64,
                MTLResourceOptions::StorageModeShared,
            );

            self.metal_forward(
                &[(get_buffer_from_tensor(&tensors[0].0), tensors[0].1)],
                command_buffer,
                &[],
                &[&out],
            );

            command_buffer.commit();
            command_buffer.wait_until_completed();

            vec![Tensor::new(MetalBuffer(out))]
        })
    }

    fn custom(&mut self, key: &str, input: Box<dyn Any>) -> Option<Box<dyn Any>> {
        if key == "metal" {
            return Some(Box::new(MetalKernelWrapper(Arc::new(Box::new(
                self.clone(),
            )))));
        }
        // This op can accept non contiguous inputs
        if key == "non_contiguous" {
            return Some(Box::new(()));
        }
        if key == "recompile_shapes" {
            if let Some(input_shapes) = input.downcast_ref::<Vec<ShapeTracker>>() {
                *self = Self::new(
                    input_shapes[0],
                    self.dim,
                    self.device.clone(),
                    self.queue.clone(),
                    self.dyn_map,
                )
            }
        }
        None
    }
}

#[derive(LuminalPrint, Clone)]
pub struct MetalMaxReduce<T> {
    pipeline: ComputePipelineState,
    queue: CommandQueue,
    device: Device,
    dim: usize,
    dyn_symbols: Vec<char>,
    _phantom: PhantomData<T>,
    dyn_map: *const FxHashMap<char, usize>,
}

impl<T> PartialEq for MetalMaxReduce<T> {
    fn eq(&self, other: &Self) -> bool {
        self.dim == other.dim
    }
}

impl<T: MetalFloat> MetalMaxReduce<T> {
    pub fn new(
        shape: ShapeTracker,
        dim: usize,
        device: Device,
        queue: CommandQueue,
        dyn_map: *const FxHashMap<char, usize>,
    ) -> Self {
        let (idx_exp, valid_exp) = get_idx_valid_exps(shape);
        let type_name = T::type_name();
        let (dyn_symbols, rendered) = render_dyn_dim_inputs(&[shape], 6);
        let code = format!("
#include <metal_stdlib>
using namespace metal;
kernel void mkernel(device {type_name} *inp [[buffer(0)]], device {type_name} *out [[buffer(1)]], device int& n_elements [[buffer(2)]], device int& front_size [[buffer(3)]], device int& back_size [[buffer(4)]], device int& dim_size [[buffer(5)]], uint i_ [[thread_position_in_grid]]{rendered}) {{
    if (i_ < n_elements) {{
        int a_ = i_ / back_size;
        int b_ = i_ % back_size;
        {type_name} reduce_value = -{};
        for (int c_ = 0; c_ < dim_size; c_++) {{
            uint idx = a_ * dim_size * back_size + c_ * back_size + b_;
            if (({valid_exp}) != 0) {{
                int a_idx = {idx_exp};
                reduce_value = max(reduce_value, inp[a_idx]);
            }}
        }}
        out[i_] = reduce_value;
    }}
}}
", if T::is_f32() {"(float)0x7f800000"} else {"MAXHALF"},
        );
        Self {
            pipeline: compile_function("mkernel", &code, &device),
            queue,
            device,
            dim,
            dyn_symbols,
            _phantom: Default::default(),
            dyn_map,
        }
    }
}
impl<T> MetalKernel for MetalMaxReduce<T> {
    fn output_buffer_sizes(&self, input_shapes: &[ShapeTracker]) -> Vec<BigExpression> {
        let mut sh = input_shapes[0];
        sh.remove_dim(self.dim);
        vec![sh.n_elements() * size_of::<T>()]
    }
    fn metal_forward(
        &self,
        inputs: &[(&Buffer, ShapeTracker)],
        command_buffer: &CommandBufferRef,
        _: &[&Buffer],
        output_buffers: &[&Buffer],
    ) {
        let mut sh = inputs[0].1;
        sh.remove_dim(self.dim);
        let inp_size = sh.contiguous().n_elements().to_usize().unwrap();
        let front_size: usize = inputs[0]
            .1
            .shape()
            .iter()
            .take(self.dim)
            .map(|i| i.to_usize().unwrap())
            .product();
        let back_size: usize = inputs[0]
            .1
            .shape()
            .iter()
            .skip(self.dim + 1)
            .map(|i| i.to_usize().unwrap())
            .product();
        let dim_size = inputs[0].1.shape()[self.dim].to_usize().unwrap();

        let encoder =
            command_buffer.compute_command_encoder_with_descriptor(ComputePassDescriptor::new());
        encoder.set_compute_pipeline_state(&self.pipeline);

        // Set inputs
        encoder.set_buffer(0, Some(inputs[0].0), 0);
        encoder.set_buffer(1, Some(output_buffers[0]), 0);
        encoder.set_u32(2, inp_size as u32);
        encoder.set_u32(3, front_size as u32);
        encoder.set_u32(4, back_size as u32);
        encoder.set_u32(5, dim_size as u32);
        input_dyn_dims(
            &self.dyn_symbols,
            unsafe { self.dyn_map.as_ref().unwrap() },
            &encoder,
            6,
        );

        // Execute
        encoder.dispatch_1d(inp_size);
        encoder.end_encoding();
    }
}

impl<T: MetalFloat> Operator for MetalMaxReduce<T> {
    fn process(&mut self, tensors: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        autoreleasepool(|| {
            let a = tensors[0]
                .0
                .borrowed()
                .data
                .as_any()
                .downcast_ref::<Buffer>()
                .unwrap();

            // Setup command queue / command buffer / encoder
            let command_buffer = self.queue.new_command_buffer();
            let mut sh = tensors[0].1;
            sh.remove_dim(self.dim);
            let inp_size = sh.n_elements().to_usize().unwrap();
            let out = self.device.new_buffer(
                (inp_size * std::mem::size_of::<T>()) as u64,
                MTLResourceOptions::StorageModeShared,
            );

            self.metal_forward(&[(a, tensors[0].1)], command_buffer, &[], &[&out]);

            command_buffer.commit();
            command_buffer.wait_until_completed();

            vec![Tensor::new(MetalBuffer(out))]
        })
    }

    fn custom(&mut self, key: &str, input: Box<dyn Any>) -> Option<Box<dyn Any>> {
        if key == "metal" {
            return Some(Box::new(MetalKernelWrapper(Arc::new(Box::new(
                self.clone(),
            )))));
        }
        // This op can accept non contiguous inputs
        if key == "non_contiguous" {
            return Some(Box::new(()));
        }
        if key == "recompile_shapes" {
            if let Some(input_shapes) = input.downcast_ref::<Vec<ShapeTracker>>() {
                *self = Self::new(
                    input_shapes[0],
                    self.dim,
                    self.device.clone(),
                    self.queue.clone(),
                    self.dyn_map,
                )
            }
        }
        None
    }
}

#[derive(Default, LuminalPrint)]
pub struct PrimitiveCompiler<T>(PhantomData<T>);

impl<T: MetalFloat + 'static> Compiler for PrimitiveCompiler<T> {
    fn compile<To: ToIdsMut>(&self, graph: &mut Graph, mut remap: To) {
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
            // Create copy node
            let copy_node = graph
                .add_op(MetalCopyToDevice::<T>::new(dev.clone()))
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

            // Insert copy from device for function inputs
            for (source, edge, edge_weight) in graph
                .graph
                .edges_directed(function_node, petgraph::Direction::Incoming)
                .map(|e| (e.source(), e.id(), *e.weight()))
                .collect::<Vec<_>>()
            {
                let copy_from_node = graph
                    .add_op(MetalCopyFromDevice::<T>::new(dev.clone()))
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
                        .max_by_key(|s| s.n_physical_elements().to_usize().unwrap_or_default())
                        .unwrap(),
                )
            })
            .collect::<Vec<_>>()
        {
            if graph
                .graph
                .node_weight(output_node)
                .unwrap()
                .as_any()
                .is::<MetalCopyToDevice<T>>()
            {
                // This output is already a copy to, instead of adding a copy from, let's remap back to the source
                let src = graph
                    .graph
                    .neighbors_directed(output_node, petgraph::Direction::Incoming)
                    .next()
                    .unwrap();
                graph.no_delete.remove(&output_node);
                graph.to_retrieve.remove(&output_node);
                graph.no_delete.insert(src);
                graph.to_retrieve.insert(src);
            } else {
                // Create copy node
                let copy_node = graph
                    .add_op(MetalCopyFromDevice::<T>::new(dev.clone()))
                    .input(output_node, 0, output_shape)
                    .finish();

                move_references(
                    &mut remap,
                    &mut graph.no_delete,
                    &mut graph.to_retrieve,
                    output_node,
                    copy_node,
                );
            }
        }

        // Copy prints and diffs from device
        for (output_node, edge) in graph
            .graph
            .node_indices()
            // Filter non-functions
            .filter(|n| {
                graph.graph.node_weight(*n).unwrap().as_any().is::<Print>()
                    || graph.graph.node_weight(*n).unwrap().as_any().is::<Diff>()
            })
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
                .add_op(MetalCopyFromDevice::<T>::new(dev.clone()))
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
                *op_ref = Box::new(MetalLog2::<T>::new(dev.clone(), queue.clone()));
            } else if is::<Exp2>(op) {
                *op_ref = Box::new(MetalExp2::<T>::new(dev.clone(), queue.clone()));
            } else if let Some(c) = op_ref.as_any().downcast_ref::<Constant>() {
                *op_ref = Box::new(MetalConstant::<T>(
                    c.0.clone(),
                    dev.clone(),
                    c.1,
                    Default::default(),
                ));
            } else if is::<Sin>(op) {
                *op_ref = Box::new(MetalSin::<T>::new(dev.clone(), queue.clone()));
            } else if is::<Sqrt>(op) {
                *op_ref = Box::new(MetalSqrt::<T>::new(dev.clone(), queue.clone()));
            } else if is::<Recip>(op) {
                *op_ref = Box::new(MetalRecip::<T>::new(dev.clone(), queue.clone()));
            } else if is::<Add>(op) {
                *op_ref = Box::new(MetalAdd::<T>::new(
                    src_shapes[0],
                    src_shapes[1],
                    dev.clone(),
                    queue.clone(),
                    &graph.dyn_map,
                ));
            } else if is::<Mul>(op) {
                *op_ref = Box::new(MetalMul::<T>::new(
                    src_shapes[0],
                    src_shapes[1],
                    dev.clone(),
                    queue.clone(),
                    &graph.dyn_map,
                ));
            } else if is::<LessThan>(op) {
                *op_ref = Box::new(MetalLessThan::<T>::new(
                    src_shapes[0],
                    src_shapes[1],
                    dev.clone(),
                    queue.clone(),
                    &graph.dyn_map,
                ));
            } else if is::<Mod>(op) {
                *op_ref = Box::new(MetalMod::<T>::new(
                    src_shapes[0],
                    src_shapes[1],
                    dev.clone(),
                    queue.clone(),
                    &graph.dyn_map,
                ));
            } else if let Some(SumReduce(dim)) = op_ref.as_any().downcast_ref() {
                *op_ref = Box::new(MetalSumReduce::<T>::new(
                    src_shapes[0],
                    *dim,
                    dev.clone(),
                    queue.clone(),
                    &graph.dyn_map,
                ));
            } else if let Some(MaxReduce(dim)) = op_ref.as_any().downcast_ref() {
                *op_ref = Box::new(MetalMaxReduce::<T>::new(
                    src_shapes[0],
                    *dim,
                    dev.clone(),
                    queue.clone(),
                    &graph.dyn_map,
                ));
            } else if is::<Contiguous>(op) {
                *op_ref = Box::new(MetalContiguous::<T>::new(
                    src_shapes[0],
                    dev.clone(),
                    queue.clone(),
                    &graph.dyn_map,
                ));
            }
        }
    }
}
