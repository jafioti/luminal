use num_traits::FloatConst;
use std::{marker::PhantomData, mem::size_of, sync::Arc};

use petgraph::{stable_graph::NodeIndex, visit::EdgeRef};

use crate::{
    compilers::metal::{prim::*, *},
    op::{ConstantValue, InputTensor, Operator},
    prelude::*,
    select_const, select_ty,
};

use metal_rs::{objc::rc::autoreleasepool, *};

use super::binary::MetalSub;

/// Special kernel for efficient mean reduction
#[derive(LuminalPrint, Clone)]
pub struct MetalMeanReduce<T>(
    ComputePipelineState,
    CommandQueue,
    Device,
    pub usize,
    Vec<char>,
    *const FxHashMap<char, usize>,
    PhantomData<T>,
);

impl<T> PartialEq for MetalMeanReduce<T> {
    fn eq(&self, other: &Self) -> bool {
        self.3 == other.3
    }
}

impl<T: MetalFloat> MetalMeanReduce<T> {
    fn new(
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

            vec![Tensor::new(out)]
        })
    }

    fn custom(&mut self, key: &str, input: Box<dyn Any>) -> Option<Box<dyn Any>> {
        if key == "metal" {
            #[allow(clippy::arc_with_non_send_sync)]
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
                *self = MetalMeanReduce::<T>::new(
                    self.2.clone(),
                    self.1.clone(),
                    self.3,
                    input_shapes[0],
                    self.5,
                );
            }
        }
        None
    }
}

/// Replace the mean reduce pattern with a special kernel. This is meant to be ran **after** the FakeSumReduceCompiler.
#[derive(Default, Debug)]
pub struct MeanReduceCompiler<T>(PhantomData<T>);

impl<T: MetalFloat> Compiler for MeanReduceCompiler<T> {
    fn compile<To: ToIdsMut>(&self, graph: &mut Graph, mut remap: To) {
        let dev = Device::system_default().unwrap();
        let queue = dev.new_command_queue();
        // Look for the mean-reduce pattern
        // mul(recip(fake_sum_reduce(const_ones)), sum_reduce(x))
        let (mut fake_sum_reduce, mut recip, mut mul, mut sum_reduce) = (
            NodeIndex::default(),
            NodeIndex::default(),
            NodeIndex::default(),
            NodeIndex::default(),
        );

        let s = SelectOp::new()
            .ty::<MetalSumReduce<T>>()
            .ptr(&mut sum_reduce)
            .edge(
                SelectOp::new()
                    .ty::<MetalConstant<T>>()
                    .ptr(&mut fake_sum_reduce)
                    .edge(SelectOp::new().ty::<MetalRecip<T>>().ptr(&mut recip))
                    .edge(SelectOp::new().ty::<MetalMul<T>>().ptr(&mut mul)),
            );

        let mut searcher = s.search(graph);
        while searcher.next_match() {
            if graph.no_delete.contains(&sum_reduce)
                || graph.no_delete.contains(&fake_sum_reduce)
                || graph.no_delete.contains(&recip)
            {
                // An intermediate node can't be deleted
                continue;
            }
            let dim = graph
                .graph
                .node_weight(sum_reduce)
                .unwrap()
                .as_any()
                .downcast_ref::<MetalSumReduce<T>>()
                .unwrap()
                .dim;
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
            move_outgoing_edge(mul, mean_reduce, &mut graph.graph);
            move_references(
                &mut remap,
                &mut graph.no_delete,
                &mut graph.to_retrieve,
                mul,
                mean_reduce,
            );

            // Remove the old ops
            graph.graph.remove_node(mul);
            graph.safe_remove_node(recip, 0);
            graph.safe_remove_node(fake_sum_reduce, 0);
            graph.safe_remove_node(sum_reduce, 0);
        }
    }
}

/// Special kernel for efficient std norming
#[derive(LuminalPrint, Clone)]
pub struct MetalStdNorm<T> {
    pipeline: ComputePipelineState,
    device: Device,
    queue: CommandQueue,
    epsilon: f32, // Epsilon
    _phantom: PhantomData<T>,
}

impl<T> PartialEq for MetalStdNorm<T> {
    fn eq(&self, other: &Self) -> bool {
        self.epsilon == other.epsilon
    }
}

impl<T: MetalFloat> MetalStdNorm<T> {
    fn new(epsilon: f32, device: Device, queue: CommandQueue) -> Self {
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
            pipeline: compile_function(&"kernel_std_norm", &kernel_code, &device),
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
            let a = tensors[0]
                .0
                .borrowed()
                .data
                .as_any()
                .downcast_ref::<Buffer>()
                .unwrap();
            let out = self.device.new_buffer(
                (tensors[0].1.n_elements().to_usize().unwrap() * size_of::<T>()) as u64,
                MTLResourceOptions::StorageModeShared,
            );

            self.metal_forward(&[(a, tensors[0].1)], command_buffer, &[], &[&out]);

            command_buffer.commit();
            command_buffer.wait_until_completed();

            vec![Tensor::new(out)]
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
    fn compile<To: ToIdsMut>(&self, graph: &mut Graph, mut remap: To) {
        let dev = Device::system_default().unwrap();
        let queue = dev.new_command_queue();
        // Look for the RMSNorm pattern
        // mul(recip(sqrt(add(mean_reduce(mul(x, x)), 1e-6))), x)
        let (mut square, mut mean, mut add, mut sqrt, mut recip, mut mul, mut epsilon) = (
            NodeIndex::default(),
            NodeIndex::default(),
            NodeIndex::default(),
            NodeIndex::default(),
            NodeIndex::default(),
            NodeIndex::default(),
            NodeIndex::default(),
        );

        let s = SelectOp::new()
            .ty::<MetalMul<T>>()
            .ptr(&mut square)
            .edge(SelectOp::new().ty::<MetalMeanReduce<T>>().ptr(&mut mean))
            .edge(
                SelectOp::new()
                    .check(|op, _| {
                        if let Some(c) = op.as_any().downcast_ref::<MetalConstant<T>>() {
                            if let ConstantValue::Float(v) = c.0 {
                                v <= 1e-3 && v > 0.0
                            } else {
                                false
                            }
                        } else {
                            false
                        }
                    })
                    .ptr(&mut epsilon)
                    .edge(SelectOp::new().ty::<MetalAdd<T>>().ptr(&mut add)),
            )
            .edge(SelectOp::new().ty::<MetalSqrt<T>>().ptr(&mut sqrt))
            .edge(SelectOp::new().ty::<MetalRecip<T>>().ptr(&mut recip))
            .edge(SelectOp::new().ty::<MetalMul<T>>().ptr(&mut mul));

        let mut searcher = s.search(graph);
        while searcher.next_match() {
            if check_no_delete(graph, &[add, sqrt, recip, mul, epsilon, square, mean]) {
                // An intermediate node can't be deleted
                continue;
            }
            let ConstantValue::Float(epsilon_num) = graph
                .graph
                .node_weight(epsilon)
                .unwrap()
                .as_any()
                .downcast_ref::<MetalConstant<T>>()
                .unwrap()
                .0
            else {
                continue;
            };
            let (mut x, _, mut sh) = graph.get_sources(square)[0];
            if let Some(mean_reduce) = graph
                .graph
                .node_weight(mean)
                .unwrap()
                .as_any()
                .downcast_ref::<MetalMeanReduce<T>>()
            {
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
            if !graph.get_sources(square).iter().all(|(i, _, _)| *i == x) {
                continue;
            }
            if !graph.get_sources(mul).iter().any(|(i, _, _)| *i == x) {
                continue;
            }

            // Input must be contiguous
            if !sh.is_contiguous() || sh.is_sliced() || sh.is_padded() {
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
            move_outgoing_edge(mul, rms_norm, &mut graph.graph);
            move_references(
                &mut remap,
                &mut graph.no_delete,
                &mut graph.to_retrieve,
                mul,
                rms_norm,
            );

            // Remove the old ops
            graph.graph.remove_node(mul);
            graph.safe_remove_node(recip, 0);
            graph.safe_remove_node(sqrt, 0);
            graph.safe_remove_node(add, 0);
            graph.safe_remove_node(epsilon, 0);
            graph.safe_remove_node(mean, 0);
            graph.safe_remove_node(square, 0);
        }
    }
}

#[derive(LuminalEqTrue, LuminalPrint, Clone)]
pub struct MetalExp<T: MetalFloat> {
    pipeline: ComputePipelineState,
    device: Device,
    queue: CommandQueue,
    _phantom: PhantomData<T>,
}

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
            let a_inp = tensors[0]
                .0
                .borrowed()
                .data
                .as_any()
                .downcast_ref::<Buffer>()
                .unwrap();
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

            vec![Tensor::new(out)]
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
        // This op can accept non contiguous inputs
        if key == "non_contiguous" {
            return Some(Box::new(()));
        }
        None
    }
}

#[derive(Default, Debug)]
pub struct MetalExpCompiler<T: MetalFloat>(PhantomData<T>);

impl<T: MetalFloat> Compiler for MetalExpCompiler<T> {
    fn compile<To: ToIdsMut>(&self, graph: &mut Graph, mut remap: To) {
        let dev = Device::system_default().unwrap();
        let queue = dev.new_command_queue();
        // Look for the exp pattern
        // exp2(mul(x, const))
        let (mut constant, mut mul, mut exp2) = (
            NodeIndex::default(),
            NodeIndex::default(),
            NodeIndex::default(),
        );

        let s = select_const!(1.0 / f32::ln(2.), T)
            .ptr(&mut constant)
            .edge(SelectOp::new().ty::<MetalMul<T>>().ptr(&mut mul))
            .edge(SelectOp::new().ty::<MetalExp2<T>>().ptr(&mut exp2));

        let mut searcher = s.search(graph);
        while searcher.next_match() {
            if graph.no_delete.contains(&constant)
                || graph.no_delete.contains(&mul)
                || graph.no_delete.contains(&exp2)
            {
                // An intermediate node can't be deleted
                continue;
            }

            // Insert exp op
            let src = graph
                .get_sources(mul)
                .into_iter()
                .find(|(i, _, _)| *i != constant)
                .unwrap();
            let exp = graph
                .add_op(MetalExp::<T>::new(dev.clone(), queue.clone()))
                .input(src.0, 0, src.2)
                .finish();

            // Create edges to dests
            move_outgoing_edge(exp2, exp, &mut graph.graph);
            move_references(
                &mut remap,
                &mut graph.no_delete,
                &mut graph.to_retrieve,
                exp2,
                exp,
            );

            // Remove the old ops
            graph.graph.remove_node(exp2);
            graph.safe_remove_node(mul, 0);
            graph.safe_remove_node(constant, 0);
        }
    }
}

/// Special kernel for cos
#[derive(LuminalEqTrue, LuminalPrint, Clone)]
pub struct MetalCos<T: MetalFloat> {
    pipeline: ComputePipelineState,
    device: Device,
    queue: CommandQueue,
    _phantom: PhantomData<T>,
}

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
            let a = tensors[0]
                .0
                .borrowed()
                .data
                .as_any()
                .downcast_ref::<Buffer>()
                .unwrap();
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

            vec![Tensor::new(out)]
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
        // This op can accept non contiguous inputs
        if key == "non_contiguous" {
            return Some(Box::new(()));
        }
        None
    }
}

#[derive(Default, Debug)]
pub struct MetalCosCompiler<T>(PhantomData<T>);

impl<T: MetalFloat> Compiler for MetalCosCompiler<T> {
    fn compile<To: ToIdsMut>(&self, graph: &mut Graph, mut remap: To) {
        let dev = Device::system_default().unwrap();
        let queue = dev.new_command_queue();
        // Look for the cos pattern
        // sin(add(mul(const_neg_one, x), const_pi_over_2))
        let (mut const_pi, mut sub, mut sin, mut x) = (
            NodeIndex::default(),
            NodeIndex::default(),
            NodeIndex::default(),
            NodeIndex::default(),
        );

        let s = SelectOp::new()
            .ptr(&mut x)
            .edge(
                select_const!(f32::PI() / 2., T)
                    .ptr(&mut const_pi)
                    .edge(select_ty!(MetalSub<T>).ptr(&mut sub)),
            )
            .edge(SelectOp::new().ty::<MetalSin<T>>().ptr(&mut sin));

        let mut searcher = s.search(graph);
        while searcher.next_match() {
            if check_no_delete(graph, &[const_pi, sub]) {
                // An intermediate node can't be deleted
                continue;
            }

            // Insert cos op
            let shape = graph
                .graph
                .edges_directed(sub, petgraph::Direction::Incoming)
                .filter(|e| !e.weight().is_schedule())
                .find(|e| e.source() != const_pi)
                .unwrap()
                .weight()
                .as_data()
                .unwrap()
                .2;
            let cos = graph
                .add_op(MetalCos::<T>::new(dev.clone(), queue.clone()))
                .input(x, 0, shape)
                .finish();

            // Create edges to dests
            move_outgoing_edge(sin, cos, &mut graph.graph);
            move_references(
                &mut remap,
                &mut graph.no_delete,
                &mut graph.to_retrieve,
                sin,
                cos,
            );

            // Remove the old ops
            graph.graph.remove_node(sin);
            graph.safe_remove_node(sub, 0);
            graph.safe_remove_node(const_pi, 0);
        }
    }
}

/// Special kernel for efficient softmax. Currently only works on the last dim
#[derive(LuminalPrint, LuminalEqTrue, Clone)]
pub struct MetalSoftmax<T> {
    single_row_pipeline: ComputePipelineState,
    looped_pipeline: ComputePipelineState,
    queue: CommandQueue,
    device: Device,
    _phantom: PhantomData<T>,
}

const SOFTMAX_N_READS: usize = 4;
const SOFTMAX_LOOPED_LIMIT: usize = 4096;
const SIMD_SIZE: usize = 32;
impl<T> MetalKernel for MetalSoftmax<T> {
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
        let batch_size = inputs[0]
            .1
            .shape()
            .iter()
            .take(inputs[0].1.len() - 1)
            .map(|i| i.to_usize().unwrap())
            .product::<usize>()
            .max(1);
        let axis_size = inputs[0].1.shape().last().unwrap().to_usize().unwrap();

        let encoder =
            command_buffer.compute_command_encoder_with_descriptor(ComputePassDescriptor::new());
        encoder.set_buffer(0, Some(inputs[0].0), 0);
        encoder.set_buffer(1, Some(output_buffers[0]), 0);
        encoder.set_i32(2, axis_size as i32);
        encoder.set_threadgroup_memory_length(0, (SIMD_SIZE * std::mem::size_of::<u32>()) as u64);
        if axis_size <= SOFTMAX_LOOPED_LIMIT {
            encoder.set_compute_pipeline_state(&self.single_row_pipeline);
            let threadgroup_needed = (axis_size + SOFTMAX_N_READS - 1) / SOFTMAX_N_READS;
            let simds_needed = (threadgroup_needed + SIMD_SIZE - 1) / SIMD_SIZE;
            let threadgroup_size = SIMD_SIZE * simds_needed;
            let n_threads = batch_size * threadgroup_size;
            encoder.dispatch_threads(
                MTLSize::new(n_threads as u64, 1, 1),
                MTLSize::new(threadgroup_size as u64, 1, 1),
            );
        } else {
            encoder.set_compute_pipeline_state(&self.looped_pipeline);
            encoder.dispatch_1d(batch_size * axis_size);
        }
        encoder.end_encoding();
    }
}

impl<T: MetalFloat> Operator for MetalSoftmax<T> {
    fn process(&mut self, tensors: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        autoreleasepool(|| {
            // Setup buffers
            let inp_size = tensors[0].1.n_elements().to_usize().unwrap();
            let out = self
                .device
                .new_buffer(inp_size as u64, MTLResourceOptions::StorageModeShared);

            // Setup command queue / command buffer / encoder
            let command_buffer = self.queue.new_command_buffer();

            self.metal_forward(
                &[(get_buffer_from_tensor(&tensors[0].0), tensors[0].1)],
                command_buffer,
                &[],
                &[&out],
            );

            command_buffer.commit();
            command_buffer.wait_until_completed();

            vec![Tensor::new(out)]
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

/// Replace the softmax pattern with a special kernel.
#[derive(Default, Debug)]
pub struct SoftmaxCompiler<T>(PhantomData<T>);

impl<T: MetalFloat> Compiler for SoftmaxCompiler<T> {
    fn compile<To: ToIdsMut>(&self, graph: &mut Graph, mut remap: To) {
        let dev = Device::system_default().unwrap();
        let queue = dev.new_command_queue();
        // Look for the mean-reduce pattern
        // mul(recip(fake_sum_reduce(const_ones)), sum_reduce(x))
        let (mut max_reduce, mut sub, mut exp, mut sum_reduce, mut recip, mut mul) = (
            NodeIndex::default(),
            NodeIndex::default(),
            NodeIndex::default(),
            NodeIndex::default(),
            NodeIndex::default(),
            NodeIndex::default(),
        );

        let s = SelectOp::new()
            .ty::<MetalMaxReduce<T>>()
            .ptr(&mut max_reduce)
            .edge(SelectOp::new().ty::<MetalSub<T>>().ptr(&mut sub))
            .edge(SelectOp::new().ty::<MetalExp<T>>().ptr(&mut exp))
            .edge(
                SelectOp::new()
                    .ty::<MetalSumReduce<T>>()
                    .ptr(&mut sum_reduce),
            )
            .edge(SelectOp::new().ty::<MetalRecip<T>>().ptr(&mut recip))
            .edge(SelectOp::new().ty::<MetalMul<T>>().ptr(&mut mul));

        let lib = compile_lib(&dev, include_str!("kernels/softmax.metal"));
        let type_name = if T::is_f32() { "float32" } else { "float16" };
        let mut searcher = s.search(graph);
        while searcher.next_match() {
            if check_no_delete(graph, &[max_reduce, sub, exp, sum_reduce, recip]) {
                // An intermediate node can't be deleted
                continue;
            }
            // Insert Softmax op
            let src = graph.get_sources(max_reduce)[0];
            let mean_reduce = graph
                .add_op(MetalSoftmax::<T> {
                    device: dev.clone(),
                    queue: queue.clone(),
                    _phantom: Default::default(),
                    single_row_pipeline: select_function_from_lib(
                        &lib,
                        &format!("softmax_{type_name}"),
                        &dev,
                    ),
                    looped_pipeline: select_function_from_lib(
                        &lib,
                        &format!("softmax_looped_{type_name}"),
                        &dev,
                    ),
                })
                .input(src.0, 0, src.2)
                .finish();

            // Create edges to dests
            move_outgoing_edge(mul, mean_reduce, &mut graph.graph);
            move_references(
                &mut remap,
                &mut graph.no_delete,
                &mut graph.to_retrieve,
                mul,
                mean_reduce,
            );

            // Remove the old ops
            graph.graph.remove_node(mul);
            graph.safe_remove_node(recip, 0);
            graph.safe_remove_node(sum_reduce, 0);
            graph.safe_remove_node(exp, 0);
            graph.safe_remove_node(sub, 0);
            graph.safe_remove_node(max_reduce, 0);
        }
    }
}

/// Special kernel for rotating. Probably shouldn't exist, seeing as it's only for rotary embeddings
#[derive(LuminalPrint, LuminalEqTrue, Clone)]
pub struct MetalRope<T> {
    pipeline: ComputePipelineState,
    axis_size: usize,
    seq_offset: BigExpression,
    queue: CommandQueue,
    device: Device,
    dyn_symbols: Vec<char>,
    dyn_map: *const FxHashMap<char, usize>,
    _phantom: PhantomData<T>,
}

impl<T: MetalFloat> MetalRope<T> {
    fn new(
        axis_size: usize,
        seq_offset: BigExpression,
        shape: ShapeTracker,
        device: Device,
        queue: CommandQueue,
        dyn_map: *const FxHashMap<char, usize>,
    ) -> Self {
        let type_name = T::type_name();
        let (index, valid) = get_idx_valid_exps(shape);
        let (dyn_symbols, rendered) = render_dyn_dim_inputs(&[shape], 3);
        Self {
            pipeline: compile_function(
                "mkernel",
                &format!(
                    "
#include <metal_stdlib>
using namespace metal;
kernel void mkernel(
    device {type_name} *inp [[buffer(0)]],
    device {type_name} *out [[buffer(1)]],
    device uint& seq_offset [[buffer(2)]],
    uint thread_index_in_threadgroup [[thread_index_in_threadgroup]],
    uint3 threads_per_threadgroup [[threads_per_threadgroup]],
    uint3 threadgroups_per_grid [[threadgroups_per_grid]],
    uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]]
    {rendered}
) {{
    const uint seq_len = threadgroups_per_grid.y;
    const uint size = threads_per_threadgroup.x * 2;
    const uint local_seq_pos = threadgroup_position_in_grid.y;
    const uint head_pos = threadgroup_position_in_grid.x;
    const uint heads = threadgroups_per_grid.x;
    const uint global_seq_pos = local_seq_pos + seq_offset;
    const uint vec_pos = thread_index_in_threadgroup * 2;
    const float theta = (float)global_seq_pos * pow(1000000.0, -(float)vec_pos / (float)size);
    const float sin_theta = sin(theta);
    const float cos_theta = cos(theta);

    uint idx = threadgroup_position_in_grid.z * heads * seq_len * size + head_pos * seq_len * size + local_seq_pos * size + vec_pos;
    float x0 = ({valid} == 0 ? 0.0 : (float)inp[{index}]);
    idx += 1;
    float x1 = ({valid} == 0 ? 0.0 : (float)inp[{index}]);
    out[idx - 1] = ({type_name})(x0 * cos_theta - x1 * sin_theta);
    out[idx] = ({type_name})(x0 * sin_theta + x1 * cos_theta);
}}"
                ),
                &device,
            ),
            device,
            queue,
            dyn_symbols,
            axis_size,
            seq_offset,
            dyn_map,
            _phantom: Default::default(),
        }
    }
}

impl<T> MetalKernel for MetalRope<T> {
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
        encoder.set_buffer(0, Some(inputs[0].0), 0);
        encoder.set_buffer(1, Some(output_buffers[0]), 0);
        encoder.set_u32(
            2,
            self.seq_offset
                .exec(unsafe { self.dyn_map.as_ref().unwrap() })
                .unwrap() as u32,
        );
        input_dyn_dims(
            &self.dyn_symbols,
            unsafe { self.dyn_map.as_ref().unwrap() },
            &encoder,
            3,
        );
        encoder.set_compute_pipeline_state(&self.pipeline);
        let sh = inputs[0].1.shape();
        encoder.dispatch_thread_groups(
            MTLSize {
                width: sh[1].to_usize().unwrap() as u64,
                height: sh[2].to_usize().unwrap() as u64,
                depth: sh[0].to_usize().unwrap() as u64,
            },
            MTLSize::new((sh[3].to_usize().unwrap() / 2) as u64, 1, 1),
        );
        encoder.end_encoding();
    }
}

impl<T: MetalFloat> Operator for MetalRope<T> {
    fn process(&mut self, tensors: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        autoreleasepool(|| {
            // Setup buffers
            let inp_size = tensors[0].1.n_physical_elements().to_usize().unwrap();
            let out = self
                .device
                .new_buffer(inp_size as u64, MTLResourceOptions::StorageModeShared);

            // Setup command queue / command buffer / encoder
            let command_buffer = self.queue.new_command_buffer();

            self.metal_forward(
                &[(get_buffer_from_tensor(&tensors[0].0), tensors[0].1)],
                command_buffer,
                &[],
                &[&out],
            );

            command_buffer.commit();
            command_buffer.wait_until_completed();

            vec![Tensor::new(out)]
        })
    }

    fn custom(&mut self, key: &str, input: Box<dyn Any>) -> Option<Box<dyn Any>> {
        if key == "metal" {
            #[allow(clippy::arc_with_non_send_sync)]
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
                    self.axis_size,
                    self.seq_offset.clone(),
                    input_shapes[0],
                    self.device.clone(),
                    self.queue.clone(),
                    self.dyn_map,
                );
            }
        }
        None
    }
}

/// Replace the rotate pattern with a special kernel.
#[derive(Default, Debug)]
pub struct RopeCompiler<T>(PhantomData<T>);

impl<T: MetalFloat> Compiler for RopeCompiler<T> {
    fn compile<To: ToIdsMut>(&self, graph: &mut Graph, mut remap: To) {
        let dev = Device::system_default().unwrap();
        let queue = dev.new_command_queue();
        let mut head_dim_arange = NodeIndex::default();
        let mut two = NodeIndex::default();
        let mut mul_2 = NodeIndex::default();
        let mut inv_head_dim = NodeIndex::default();
        let mut head_dim_mul = NodeIndex::default();
        let mut theta = NodeIndex::default();
        let mut theta_mul = NodeIndex::default();
        let mut exp = NodeIndex::default();
        let mut recip = NodeIndex::default();
        let mut seq_arange = NodeIndex::default();
        let mut seq_expr = NodeIndex::default();
        let mut seq_add = NodeIndex::default();
        let mut freq_seq_mul = NodeIndex::default();
        let mut input = NodeIndex::default();
        let mut split_contig1 = NodeIndex::default();
        let mut split_contig2 = NodeIndex::default();
        let mut split_contig3 = NodeIndex::default();
        let mut sin1 = NodeIndex::default();
        let mut sin2 = NodeIndex::default();
        let mut cos1 = NodeIndex::default();
        let mut cos2 = NodeIndex::default();
        let mut out_add = NodeIndex::default();
        let mut out_sub = NodeIndex::default();
        let mut final_add = NodeIndex::default();
        let mut out_mul1 = NodeIndex::default();
        let mut out_mul2 = NodeIndex::default();
        let mut out_mul3 = NodeIndex::default();
        let mut out_mul4 = NodeIndex::default();

        let freqs = select_const!(1000000.0_f32.ln(), T)
            .ptr(&mut theta)
            .edge(
                select_ty!(MetalConstant<T>)
                    .ptr(&mut inv_head_dim)
                    .edge(
                        select_ty!(MetalConstant<T>)
                            .ptr(&mut two)
                            .edge(
                                select_ty!(crate::compilers::metal::other::MetalARange<T>)
                                    .ptr(&mut head_dim_arange)
                                    .edge(select_ty!(MetalMul<T>).ptr(&mut mul_2)),
                            )
                            .edge(select_ty!(MetalMul<T>).ptr(&mut head_dim_mul)),
                    )
                    .edge(select_ty!(MetalMul<T>).ptr(&mut theta_mul)),
            )
            .edge(select_ty!(MetalExp<T>).ptr(&mut exp))
            .edge(select_ty!(MetalRecip<T>).ptr(&mut recip));
        let seq = select_ty!(MetalConstant<T>).ptr(&mut seq_expr).edge(
            select_ty!(crate::compilers::metal::other::MetalARange<T>)
                .ptr(&mut seq_arange)
                .edge(select_ty!(MetalAdd<T>).ptr(&mut seq_add)),
        );
        let emb = freqs.edge(seq.edge(select_ty!(MetalMul<T>).ptr(&mut freq_seq_mul)));
        let split = SelectOp::new()
            .ptr(&mut input)
            .edge(select_ty!(MetalContiguous<T>).ptr(&mut split_contig1));
        let x0 = split
            .clone()
            .edge(select_ty!(MetalContiguous<T>).ptr(&mut split_contig2));
        let x1 = split.edge(select_ty!(MetalContiguous<T>).ptr(&mut split_contig3));
        let x0_sin = emb
            .clone()
            .edge(select_ty!(MetalSin<T>).ptr(&mut sin1))
            .edge(x0.clone().edge(select_ty!(MetalMul<T>).ptr(&mut out_mul1)));
        let x0_cos = emb
            .clone()
            .edge(select_ty!(MetalCos<T>).ptr(&mut cos1))
            .edge(x0.edge(select_ty!(MetalMul<T>).ptr(&mut out_mul2)));
        let x1_sin = emb
            .clone()
            .edge(select_ty!(MetalSin<T>).ptr(&mut sin2))
            .edge(x1.clone().edge(select_ty!(MetalMul<T>).ptr(&mut out_mul3)));
        let x1_cos = emb
            .clone()
            .edge(select_ty!(MetalCos<T>).ptr(&mut cos2))
            .edge(x1.edge(select_ty!(MetalMul<T>).ptr(&mut out_mul4)));
        let x0_out = x1_sin.edge(x0_cos.edge(select_ty!(MetalSub<T>).ptr(&mut out_sub)));
        let x1_out = x0_sin.edge(x1_cos.edge(select_ty!(MetalAdd<T>).ptr(&mut out_add)));
        let mut searcher = x1_out
            .edge(x0_out.edge(select_ty!(MetalAdd<T>).ptr(&mut final_add)))
            .search(graph);

        while searcher.next_match() {
            if check_no_delete(
                graph,
                &[
                    head_dim_arange,
                    two,
                    mul_2,
                    inv_head_dim,
                    head_dim_mul,
                    theta,
                    theta_mul,
                    exp,
                    recip,
                    seq_arange,
                    seq_expr,
                    seq_add,
                    freq_seq_mul,
                    input,
                    split_contig1,
                    split_contig2,
                    split_contig3,
                    sin1,
                    sin2,
                    cos1,
                    cos2,
                    out_add,
                    out_sub,
                    final_add,
                    out_mul1,
                    out_mul2,
                    out_mul3,
                    out_mul4,
                ],
            ) {
                continue;
            }

            let shape = graph
                .graph
                .edges_connecting(input, split_contig1)
                .next()
                .unwrap()
                .weight()
                .as_data()
                .unwrap()
                .2;
            let Some(MetalConstant(ConstantValue::Expression(e), ..)) = graph
                .graph
                .node_weight(seq_expr)
                .unwrap()
                .as_any()
                .downcast_ref::<MetalConstant<T>>()
            else {
                continue;
            };
            let rope_op = graph
                .add_op(MetalRope::<T>::new(
                    shape.shape()[3].to_usize().unwrap(),
                    e.clone(),
                    shape,
                    dev.clone(),
                    queue.clone(),
                    &graph.dyn_map,
                ))
                .input(input, 0, shape)
                .finish();
            move_outgoing_edge(final_add, rope_op, &mut graph.graph);

            // Delete old ops
            graph.graph.remove_node(final_add);
            graph.safe_remove_node(out_add, 0);
            graph.safe_remove_node(out_sub, 0);
            graph.safe_remove_node(out_mul1, 0);
            graph.safe_remove_node(out_mul2, 0);
            graph.safe_remove_node(out_mul3, 0);
            graph.safe_remove_node(out_mul4, 0);
            graph.safe_remove_node(sin1, 0);
            graph.safe_remove_node(sin2, 0);
            graph.safe_remove_node(cos1, 0);
            graph.safe_remove_node(cos2, 0);
            graph.safe_remove_node(split_contig3, 0);
            graph.safe_remove_node(split_contig2, 0);
            graph.safe_remove_node(split_contig1, 0);
            graph.safe_remove_node(freq_seq_mul, 0);
            graph.safe_remove_node(seq_add, 0);
            graph.safe_remove_node(seq_arange, 0);
            graph.safe_remove_node(seq_expr, 0);
            graph.safe_remove_node(recip, 0);
            graph.safe_remove_node(exp, 0);
            graph.safe_remove_node(theta_mul, 0);
            graph.safe_remove_node(head_dim_mul, 0);
            graph.safe_remove_node(mul_2, 0);
            graph.safe_remove_node(head_dim_arange, 0);
            graph.safe_remove_node(two, 0);
            graph.safe_remove_node(inv_head_dim, 0);
            graph.safe_remove_node(theta, 0);
        }
        // graph.display();
    }
}
