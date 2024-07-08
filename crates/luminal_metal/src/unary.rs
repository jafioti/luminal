use num_traits::float::FloatConst;
use rustc_hash::FxHashMap;
use std::{any::Any, marker::PhantomData, mem::size_of, sync::Arc};

use petgraph::visit::EdgeRef;

use luminal::{
    op::{ConstantValue, InputTensor, Operator},
    prelude::*,
};

use metal_rs::{objc::rc::autoreleasepool, *};

use crate::{
    compile_function, constant, get_buffer_from_tensor, get_idx_valid_exps, input_dyn_dims,
    prim::*, render_dyn_dim_inputs, DispatchNElements, MetalBuffer, MetalFloat, MetalKernel,
    MetalKernelWrapper, SetInt,
};

use super::binary::MetalSub;

/// Special kernel for efficient mean reduction
#[derive(Clone)]
pub struct MetalMeanReduce<T>(
    ComputePipelineState,
    CommandQueue,
    Device,
    pub usize,
    Vec<char>,
    *const FxHashMap<char, usize>,
    PhantomData<T>,
    pub ShapeTracker,
);
crate::debug_type!(MetalMeanReduce);

impl<T> PartialEq for MetalMeanReduce<T> {
    fn eq(&self, other: &Self) -> bool {
        self.3 == other.3
    }
}

impl<T: MetalFloat> MetalMeanReduce<T> {
    pub fn new(
        dev: Device,
        queue: CommandQueue,
        dim: usize,
        shape: ShapeTracker,
        dyn_map: *const FxHashMap<char, usize>,
    ) -> Self {
        let (idx_exp, valid_exp) = get_idx_valid_exps(shape);
        let (dyn_symbols, rendered) = render_dyn_dim_inputs(&[shape], 6);
        let type_name = T::type_name();
        let mut code = format!("
#include <metal_stdlib>
using namespace metal;
kernel void mkernel(device {type_name} *inp [[buffer(0)]], device {type_name} *out [[buffer(1)]], device int& n_elements [[buffer(2)]], device int& front_size [[buffer(3)]], device int& back_size [[buffer(4)]], device int& dim_size [[buffer(5)]], uint i_ [[thread_position_in_grid]]{rendered}) {{
    if (i_ < n_elements) {{
        int a_ = i_ / back_size;
        int b_ = i_ % back_size;
        float reduce_value = 0.0;
        for (int c_ = 0; c_ < dim_size; c_++) {{
            int idx = a_ * dim_size * back_size + c_ * back_size + b_;
            if (({valid_exp}) != 0) {{
                reduce_value += (float)inp[{idx_exp}];
            }}
        }}
        out[i_] = ({type_name})(reduce_value / (float)dim_size);
    }}
}}");
        code = code.replace("mkernel", "kernel_mean_reduce");

        Self(
            compile_function("kernel_mean_reduce", &code, &dev),
            queue,
            dev,
            dim,
            dyn_symbols,
            dyn_map,
            Default::default(),
            shape,
        )
    }
}

impl<T> MetalKernel for MetalMeanReduce<T> {
    fn output_buffer_sizes(&self, input_shapes: &[ShapeTracker]) -> Vec<BigExpression> {
        let mut sh = input_shapes[0];
        sh.remove_dim(self.3);
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
        sh.remove_dim(self.3);
        let inp_size = sh.n_elements().to_usize().unwrap();

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
        encoder.set_buffer(1, Some(output_buffers[0]), 0);
        encoder.set_u32(2, inp_size as u32);
        encoder.set_u32(3, front_size as u32);
        encoder.set_u32(4, back_size as u32);
        encoder.set_u32(5, dim_size as u32);
        input_dyn_dims(&self.4, unsafe { self.5.as_ref().unwrap() }, encoder, 6);

        // Execute
        encoder.dispatch_1d(inp_size);
        encoder.end_encoding();
    }
}

impl<T: MetalFloat> Operator for MetalMeanReduce<T> {
    fn process(&mut self, tensors: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        autoreleasepool(|| {
            // Setup buffers
            let mut sh = tensors[0].1;
            sh.remove_dim(self.3);
            let inp_size = sh.n_elements().to_usize().unwrap();
            let out = self.2.new_buffer(
                (inp_size * std::mem::size_of::<T>()) as u64,
                MTLResourceOptions::StorageModeShared,
            );

            // Setup command queue / command buffer / encoder
            let command_buffer = self.1.new_command_buffer();

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
            #[allow(clippy::arc_with_non_send_sync)]
            return Some(Box::new(MetalKernelWrapper(Arc::new(Box::new(
                self.clone(),
            )))));
        }
        None
    }
}

/// Replace the mean reduce pattern with a special kernel. This is meant to be ran **after** the FakeSumReduceCompiler.
#[derive(Default, Debug)]
pub struct MeanReduceCompiler<T>(PhantomData<T>);

impl<T: MetalFloat> Compiler for MeanReduceCompiler<T> {
    type Output = ();
    fn compile<To: ToIdsMut>(&self, graph: &mut Graph, mut ids: To) {
        let dev = Device::system_default().unwrap();
        let queue = dev.new_command_queue();
        // Look for the mean-reduce pattern
        // mul(recip(fake_sum_reduce(const_ones)), sum_reduce(x))
        let fake_sum_reduce = op::<MetalConstant<T>>();
        let sum_reduce = op::<MetalSumReduce<T>>();
        let mul = binary::<MetalMul<T>>(
            sum_reduce.clone(),
            unary::<MetalRecip<T>>(fake_sum_reduce.clone()),
        );
        let mut s = mul.clone().search(graph);
        while s.next_match() {
            if s.check_no_delete(&[mul.id]) {
                // An intermediate node can't be deleted
                continue;
            }
            let (sum_reduce, mul) = (s.get(&sum_reduce), s.get(&mul));
            let dim = graph.get_op::<MetalSumReduce<T>>(sum_reduce).dim;
            // Insert MeanReduce op
            let src = graph.get_sources(sum_reduce)[0];
            let mean_reduce = graph
                .add_op(MetalMeanReduce::<T>::new(
                    dev.clone(),
                    queue.clone(),
                    dim,
                    src.2,
                    &graph.dyn_map,
                ))
                .input(src.0, 0, src.2)
                .finish();

            // Create edges to dests
            move_outgoing_edge(mul, mean_reduce, graph);
            remap(mul, mean_reduce, &mut ids, graph);

            // Remove the old ops
            graph.remove_node(mul);
            s.try_delete();
        }
    }
}

/// Special kernel for efficient std norming
#[derive(Clone)]
pub struct MetalStdNorm<T> {
    pipeline: ComputePipelineState,
    device: Device,
    queue: CommandQueue,
    pub epsilon: f32, // Epsilon
    _phantom: PhantomData<T>,
}
crate::debug_type!(MetalStdNorm);

impl<T> PartialEq for MetalStdNorm<T> {
    fn eq(&self, other: &Self) -> bool {
        self.epsilon == other.epsilon
    }
}

impl<T: MetalFloat> MetalStdNorm<T> {
    pub fn new(epsilon: f32, device: Device, queue: CommandQueue) -> Self {
        let type_name = T::type_name();
        let kernel_code = format!("#include <metal_stdlib>
#define SIMD_WIDTH 32

using namespace metal;
kernel void kernel_std_norm(
        device const  {type_name} * src0 [[buffer(0)]],
        device       {type_name} * dst [[buffer(1)]],
        constant   int64_t & row_size [[buffer(2)]],
        constant     float & eps [[buffer(3)]],
        threadgroup float  * buf [[threadgroup(0)]],
        uint threadgroup_position_in_grid[[threadgroup_position_in_grid]],
        uint thread_position_in_threadgroup[[thread_position_in_threadgroup]],
        uint simdgroup_index_in_threadgroup[[simdgroup_index_in_threadgroup]],
        uint thread_index_in_simdgroup[[thread_index_in_simdgroup]],
        uint threads_per_threadgroup[[threads_per_threadgroup]]) {{
    device const {type_name}4 * x = (device const {type_name}4 *) (src0 + threadgroup_position_in_grid * row_size);

    float4 sumf = 0;

    // parallel sum
    for (int i = thread_position_in_threadgroup; i < row_size/4; i += threads_per_threadgroup) {{
        sumf += (float4)x[i] * (float4)x[i];
    }}
    float all_sum = sumf[0] + sumf[1] + sumf[2] + sumf[3];
    all_sum = simd_sum(all_sum);

    if (threads_per_threadgroup > SIMD_WIDTH) {{
        if (simdgroup_index_in_threadgroup == 0) {{
            buf[thread_index_in_simdgroup] = 0.0f;
        }}

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (thread_index_in_simdgroup == 0) {{
            buf[simdgroup_index_in_threadgroup] = all_sum;
        }}

        threadgroup_barrier(mem_flags::mem_threadgroup);

        all_sum = buf[thread_index_in_simdgroup];
        all_sum = simd_sum(all_sum);
    }}

    const float mean  = all_sum / row_size;
    const float scale = rsqrt(mean + eps);

    device {type_name}4 * y = (device {type_name}4 *) (dst + threadgroup_position_in_grid * row_size);
    for (int i = thread_position_in_threadgroup; i < row_size/4; i += threads_per_threadgroup) {{
        y[i] = ({type_name}4)(x[i] * scale);
    }}
}}");

        Self {
            pipeline: compile_function("kernel_std_norm", &kernel_code, &device),
            device,
            queue,
            epsilon,
            _phantom: Default::default(),
        }
    }
}

impl<T> MetalKernel for MetalStdNorm<T> {
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
        let encoder =
            command_buffer.compute_command_encoder_with_descriptor(ComputePassDescriptor::new());
        encoder.set_compute_pipeline_state(&self.pipeline);
        let row_size = inputs[0].1.shape().last().unwrap().to_usize().unwrap();

        // Set inputs
        encoder.set_buffer(0, Some(inputs[0].0), 0);
        encoder.set_buffer(1, Some(output_buffers[0]), 0);
        encoder.set_i64(2, row_size as i64);
        encoder.set_f32(3, self.epsilon);
        let batch_size = inputs[0]
            .1
            .shape()
            .into_iter()
            .take(inputs[0].1.len() - 1)
            .map(|i| i.to_usize().unwrap())
            .product::<usize>();
        let mut nth = 32; // SIMD width
        while nth < row_size / 4 && nth < 1024 {
            nth *= 2;
        }
        encoder.set_threadgroup_memory_length(0, 32 * size_of::<f32>() as u64);
        encoder.dispatch_thread_groups(
            MTLSize {
                width: batch_size as u64,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: nth as u64,
                height: 1,
                depth: 1,
            },
        );
        encoder.end_encoding();
    }
}

impl<T: 'static + Clone> Operator for MetalStdNorm<T> {
    fn process(&mut self, tensors: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        autoreleasepool(|| {
            let command_buffer = self.queue.new_command_buffer();
            let a = get_buffer_from_tensor(&tensors[0].0);
            let out = self.device.new_buffer(
                (tensors[0].1.n_elements().to_usize().unwrap() * size_of::<T>()) as u64,
                MTLResourceOptions::StorageModeShared,
            );

            self.metal_forward(&[(a, tensors[0].1)], command_buffer, &[], &[&out]);

            command_buffer.commit();
            command_buffer.wait_until_completed();

            vec![Tensor::new(MetalBuffer(out))]
        })
    }

    fn custom(&mut self, key: &str, _: Box<dyn Any>) -> Option<Box<dyn Any>> {
        if key == "metal" {
            #[allow(clippy::arc_with_non_send_sync)]
            return Some(Box::new(MetalKernelWrapper(Arc::new(Box::new(
                self.clone(),
            )))));
        }
        None
    }
}

/// Replace the mean reduce pattern with a special kernel. This is meant to be ran **after** the FakeSumReduceCompiler.
#[derive(Default, Debug)]
pub struct StdNormCompiler<T>(PhantomData<T>);

impl<T: MetalFloat> Compiler for StdNormCompiler<T> {
    type Output = ();
    fn compile<To: ToIdsMut>(&self, graph: &mut Graph, mut ids: To) {
        let dev = Device::system_default().unwrap();
        let queue = dev.new_command_queue();
        // Look for the RMSNorm pattern
        // mul(recip(sqrt(add(mean_reduce(mul(x, x)), 1e-6))), x)

        let mut eps = op::<MetalConstant<T>>();
        eps.check(|op, _| {
            if let Some(c) = op.as_any().downcast_ref::<MetalConstant<T>>() {
                if let ConstantValue::Float(v) = c.0 {
                    v <= 1e-2 && v > 0.0
                } else {
                    false
                }
            } else {
                false
            }
        });
        let inp = node();
        let square = unary::<MetalMul<T>>(inp.clone()); // This should check both inputs! For some reason doesn't work
        let mean = unary::<MetalMeanReduce<T>>(square.clone());
        let add = binary::<MetalAdd<T>>(mean.clone(), eps.clone());
        let mul = unary::<MetalMul<T>>(unary::<MetalRecip<T>>(unary::<MetalSqrt<T>>(add.clone())));

        let mut s = mul.clone().search(graph);
        while s.next_match() {
            if s.check_no_delete(&[mul.id]) {
                // An intermediate node can't be deleted
                continue;
            }
            let ConstantValue::Float(epsilon_num) = graph.get_op::<MetalConstant<T>>(s.get(&eps)).0
            else {
                continue;
            };
            let (mut x, _, mut sh) = graph.get_sources(s.get(&square))[0];
            if let Some(mean_reduce) = graph.try_get_op::<MetalMeanReduce<T>>(s.get(&mean)) {
                if mean_reduce.3 != sh.len() - 1 {
                    continue;
                }
            }
            if sh
                .shape()
                .last()
                .unwrap()
                .to_usize()
                .map(|i| i % 32 != 0 || i < 32)
                .unwrap_or(true)
            {
                continue;
            }
            if !graph
                .get_sources(s.get(&mul))
                .iter()
                .any(|(i, _, _)| *i == x)
            {
                continue;
            }

            // Input must be contiguous
            if sh.is_reshaped() {
                x = graph
                    .add_op(MetalContiguous::<T>::new(
                        sh,
                        dev.clone(),
                        queue.clone(),
                        &graph.dyn_map,
                    ))
                    .input(x, 0, sh)
                    .finish();
                sh = sh.contiguous();
            }

            // Insert RMSNorm op
            let rms_norm = graph
                .add_op(MetalStdNorm::<T>::new(
                    epsilon_num,
                    dev.clone(),
                    queue.clone(),
                ))
                .input(x, 0, sh)
                .finish();

            // Create edges to dests
            let mul = s.get(&mul);
            move_outgoing_edge(mul, rms_norm, graph);
            remap(mul, rms_norm, &mut ids, graph);

            // Remove the old ops
            graph.remove_node(mul);
            s.try_delete();
        }
    }
}

#[derive(Clone)]
pub struct MetalExp<T> {
    pipeline: ComputePipelineState,
    device: Device,
    queue: CommandQueue,
    _phantom: PhantomData<T>,
}
crate::debug_type!(MetalExp);

impl<T: MetalFloat> MetalExp<T> {
    fn new(device: Device, queue: CommandQueue) -> Self {
        let type_name = T::type_name();
        Self {
            pipeline: compile_function("kernel_metal_exp", &format!("
#include <metal_stdlib>
using namespace metal;
kernel void kernel_metal_exp(device {type_name} *inp [[buffer(0)]], device {type_name} *out [[buffer(1)]], device int& n_elements [[buffer(2)]], uint i_ [[thread_position_in_grid]]) {{
    if (i_ < n_elements) {{
        out[i_] = exp(inp[i_]);
    }}
}}
"), &device),
            device,
            queue,
            _phantom: Default::default(),
        }
    }
}

impl<T: MetalFloat> MetalKernel for MetalExp<T> {
    fn output_buffer_sizes(&self, input_shapes: &[ShapeTracker]) -> Vec<BigExpression> {
        vec![input_shapes[0].n_elements() * std::mem::size_of::<T>()]
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

        // Set inputs
        encoder.set_buffer(0, Some(inputs[0].0), 0);
        encoder.set_buffer(1, Some(output_buffers[0]), 0);
        encoder.set_u32(2, inp_size as u32);

        // Execute
        encoder.dispatch_1d(inp_size);
        encoder.end_encoding();
    }
}

impl<T: MetalFloat> Operator for MetalExp<T> {
    fn process(&mut self, tensors: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        autoreleasepool(|| {
            // Setup buffers
            let a_inp = get_buffer_from_tensor(&tensors[0].0);
            let inp_size = tensors[0].1.n_physical_elements().to_usize().unwrap();
            let out = self.device.new_buffer(
                (inp_size * std::mem::size_of::<T>()) as u64,
                MTLResourceOptions::StorageModeShared,
            );

            // Setup command queue / command buffer / encoder
            let command_buffer = self.queue.new_command_buffer();

            self.metal_forward(&[(a_inp, tensors[0].1)], command_buffer, &[], &[&out]);

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
            return Some(Box::new("exp(input0)".to_string()));
        }
        None
    }
}

#[derive(Default, Debug)]
pub struct MetalExpCompiler<T: MetalFloat>(PhantomData<T>);

impl<T: MetalFloat> Compiler for MetalExpCompiler<T> {
    type Output = ();
    fn compile<To: ToIdsMut>(&self, graph: &mut Graph, mut ids: To) {
        let dev = Device::system_default().unwrap();
        let queue = dev.new_command_queue();
        // Look for the exp pattern
        // exp2(mul(x, const))

        let inp = node();
        let mul = binary::<MetalMul<T>>(inp.clone(), constant::<T>(1.0 / f32::ln(2.)));
        let exp2 = unary::<MetalExp2<T>>(mul.clone());
        let mut s = exp2.clone().search(graph);
        while s.next_match() {
            if s.check_no_delete(&[exp2.id]) {
                // An intermediate node can't be deleted
                continue;
            }

            // Insert exp op
            let (_, _, src_shape) = graph
                .edges_connecting(s.get(&inp), s.get(&mul))
                .next()
                .unwrap()
                .weight()
                .as_data()
                .unwrap();
            let exp = graph
                .add_op(MetalExp::<T>::new(dev.clone(), queue.clone()))
                .input(s.get(&inp), 0, src_shape)
                .finish();

            // Create edges to dests
            let exp2 = s.get(&exp2);
            move_outgoing_edge(exp2, exp, graph);
            remap(exp2, exp, &mut ids, graph);

            // Remove the old ops
            graph.remove_node(exp2);
            s.try_delete();
        }
    }
}

/// Special kernel for cos
#[derive(Clone)]
pub struct MetalCos<T> {
    pipeline: ComputePipelineState,
    device: Device,
    queue: CommandQueue,
    _phantom: PhantomData<T>,
}
crate::debug_type!(MetalCos);

impl<T: MetalFloat> MetalCos<T> {
    fn new(device: Device, queue: CommandQueue) -> Self {
        let type_name = T::type_name();
        Self {
            pipeline: compile_function("kernel_metal_cos", &format!("#include <metal_stdlib>
using namespace metal;
kernel void kernel_metal_cos(device {type_name} *inp [[buffer(0)]], device {type_name} *out [[buffer(1)]], device int& n_elements [[buffer(2)]], uint idx [[thread_position_in_grid]]) {{
    if (idx < n_elements) {{
        out[idx] = cos(inp[idx]);
    }}
}}"), &device),
            device,
            queue,
            _phantom: Default::default()
        }
    }
}

impl<T: MetalFloat> MetalKernel for MetalCos<T> {
    fn output_buffer_sizes(&self, input_shapes: &[ShapeTracker]) -> Vec<BigExpression> {
        vec![input_shapes[0].n_elements() * std::mem::size_of::<T>()]
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

        // Set inputs
        encoder.set_buffer(0, Some(inputs[0].0), 0);
        encoder.set_buffer(1, Some(output_buffers[0]), 0);
        encoder.set_u32(2, inp_size as u32);

        // Execute
        encoder.dispatch_1d(inp_size);
        encoder.end_encoding();
    }
}

impl<T: MetalFloat> Operator for MetalCos<T> {
    fn process(&mut self, tensors: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        autoreleasepool(|| {
            let a = get_buffer_from_tensor(&tensors[0].0);
            let inp_size = tensors[0].1.n_physical_elements().to_usize().unwrap();
            let out = self.device.new_buffer(
                (inp_size * std::mem::size_of::<T>()) as u64,
                MTLResourceOptions::StorageModeShared,
            );
            // Setup command queue / command buffer / encoder
            let command_buffer = self.queue.new_command_buffer();

            self.metal_forward(&[(a, tensors[0].1)], command_buffer, &[], &[&out]);

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
            return Some(Box::new("cos(input0)".to_string()));
        }
        None
    }
}

#[derive(Default, Debug)]
pub struct MetalCosCompiler<T>(PhantomData<T>);

impl<T: MetalFloat> Compiler for MetalCosCompiler<T> {
    type Output = ();
    fn compile<To: ToIdsMut>(&self, graph: &mut Graph, mut ids: To) {
        let dev = Device::system_default().unwrap();
        let queue = dev.new_command_queue();
        // Look for the cos pattern
        // sin(add(mul(const_neg_one, x), const_pi_over_2))

        let const_pi = constant::<T>(f32::PI() / 2.);
        let inp = node();
        let sub = binary::<MetalSub<T>>(inp.clone(), const_pi.clone());
        let sin = unary::<MetalSin<T>>(sub.clone());
        let mut s = sin.clone().search(graph);
        while s.next_match() {
            if s.check_no_delete(&[sin.id]) {
                // An intermediate node can't be deleted
                continue;
            }

            // Insert cos op
            let shape = graph
                .edges_directed(s.get(&sub), petgraph::Direction::Incoming)
                .filter(|e| !e.weight().is_schedule())
                .find(|e| e.source() != s.get(&const_pi))
                .unwrap()
                .weight()
                .as_data()
                .unwrap()
                .2;
            let cos = graph
                .add_op(MetalCos::<T>::new(dev.clone(), queue.clone()))
                .input(s.get(&inp), 0, shape)
                .finish();

            // Create edges to dests
            let sin = s.get(&sin);
            move_outgoing_edge(sin, cos, graph);
            remap(sin, cos, &mut ids, graph);

            // Remove the old ops
            graph.remove_node(sin);
            s.try_delete();
        }
    }
}

#[cfg(test)]
mod tests {
    use luminal::prelude::*;

    use crate::tests::assert_op_in_graph;

    use super::{MetalMeanReduce, MetalStdNorm};
    #[test]
    fn test_norms() {
        let mut cx = Graph::new();
        let a = cx.tensor(32).set([0.; 32]);
        let mut b = a.layer_norm(0, 1e-5).retrieve();

        cx.compile(
            <(
                GenericCompiler,
                crate::prim::PrimitiveCompiler<f16>,
                crate::SpecialOpsCompiler<f16>,
            )>::default(),
            &mut b,
        );

        assert_op_in_graph::<MetalStdNorm<f16>>(&cx);
        assert_op_in_graph::<MetalMeanReduce<f16>>(&cx);
    }
}
