use num_traits::FloatConst;
use std::{marker::PhantomData, mem::size_of, sync::Arc};

use half::f16;
use petgraph::{stable_graph::NodeIndex, visit::EdgeRef};

use crate::{
    compilers::metal::{prim::*, *},
    constant_select_op,
    op::{ConstantValue, InputTensor, Operator},
    prelude::*,
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
    *const HashMap<char, usize>,
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
        dyn_map: *const HashMap<char, usize>,
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
                (inp_size * std::mem::size_of::<f16>()) as u64,
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
        uint threadgroup_pos[[threadgroup_position_in_grid]],
        uint simdgroup_pos[[thread_index_in_simdgroup]]) {{
    device const {type_name}4 * x = (device const {type_name}4 *) (src0 + threadgroup_pos * row_size);

    float4 sumf = 0;

    // parallel sum
    for (int i = simdgroup_pos; i < row_size/4; i += SIMD_WIDTH) {{
        sumf += (float4)x[i] * (float4)x[i];
    }}
    float all_sum = sumf[0] + sumf[1] + sumf[2] + sumf[3];
    all_sum = simd_sum(all_sum);
    const float mean  = all_sum/row_size;
    const float scale = 1.0f/sqrt(mean + eps);

    device half4 * y = (device half4 *) (dst + threadgroup_pos * row_size);
    for (int i = simdgroup_pos; i < row_size/4; i += SIMD_WIDTH) {{
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
        encoder.set_threadgroup_memory_length(0, 32 * size_of::<f32>() as u64);
        encoder.dispatch_thread_groups(
            MTLSize {
                width: batch_size as u64,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 32.min(row_size / 4) as u64,
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

        let s = constant_select_op!(1.0 / f32::ln(2.), T)
            .ptr(&mut constant)
            .edge(SelectOp::new().ty::<MetalMul<f16>>().ptr(&mut mul))
            .edge(SelectOp::new().ty::<MetalExp2<f16>>().ptr(&mut exp2));

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
                constant_select_op!(f32::PI() / 2., T)
                    .ptr(&mut const_pi)
                    .edge(SelectOp::new().ty::<MetalSub<T>>().ptr(&mut sub)),
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
            graph.graph.remove_node(sub);
            graph.graph.remove_node(const_pi);
            graph.graph.remove_node(sin);
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
pub struct MetalRotate<T> {
    pipeline: ComputePipelineState,
    queue: CommandQueue,
    device: Device,
    _phantom: PhantomData<T>,
    axis_size: usize,
}

impl<T: MetalFloat> MetalRotate<T> {
    fn new(axis_size: usize, device: Device, queue: CommandQueue) -> Self {
        let half_size = axis_size / 2;
        let type_name = T::type_name();
        Self {
            pipeline: compile_function("mkernel", &format!("
#include <metal_stdlib>
using namespace metal;
kernel void mkernel(device {type_name} *inp [[buffer(0)]], device {type_name} *out [[buffer(1)]], device uint& n_elements [[buffer(2)]], uint idx [[thread_position_in_grid]]) {{
    if (idx < n_elements) {{
        if ((idx % {axis_size}) < {half_size}) {{
            out[idx] = ({type_name})-(float)inp[idx + {half_size}];
        }} else {{
            out[idx] = inp[idx - {half_size}];
        }}
    }}
}}
"), &device),
            device,
            queue,
            axis_size,
            _phantom: Default::default()
        }
    }
}

impl<T> MetalKernel for MetalRotate<T> {
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
        let n_elements = inputs[0].1.n_physical_elements().to_usize().unwrap();
        let encoder =
            command_buffer.compute_command_encoder_with_descriptor(ComputePassDescriptor::new());
        encoder.set_buffer(0, Some(inputs[0].0), 0);
        encoder.set_buffer(1, Some(output_buffers[0]), 0);
        encoder.set_u32(2, n_elements as u32);
        encoder.set_compute_pipeline_state(&self.pipeline);
        encoder.dispatch_1d(n_elements);
        encoder.end_encoding();
    }
}

impl<T: MetalFloat> Operator for MetalRotate<T> {
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

/// Replace the rotate pattern with a special kernel.
#[derive(Default, Debug)]
pub struct RotateCompiler<T>(PhantomData<T>);

impl<T: MetalFloat> Compiler for RotateCompiler<T> {
    fn compile<To: ToIdsMut>(&self, graph: &mut Graph, mut remap: To) {
        let dev = Device::system_default().unwrap();
        let queue = dev.new_command_queue();
        let (mut neg_one, mut add, mut mul, mut contig) = (
            NodeIndex::default(),
            NodeIndex::default(),
            NodeIndex::default(),
            NodeIndex::default(),
        );

        let mut searcher = constant_select_op!(-1.0, T)
            .ptr(&mut neg_one)
            .edge(SelectOp::new().ty::<MetalMul<T>>().ptr(&mut mul))
            .edge(
                SelectOp::new()
                    .ty::<MetalContiguous<T>>()
                    .ptr(&mut contig)
                    .edge(SelectOp::new().ty::<MetalAdd<T>>().ptr(&mut add)),
            )
            .search(graph);
        while searcher.next_match() {
            if check_no_delete(graph, &[mul]) {
                // An intermediate node can't be deleted
                continue;
            }
            // Check shapes
            let a_shape_first = graph
                .graph
                .edges_directed(mul, petgraph::Direction::Incoming)
                .find(|e| e.source() != neg_one)
                .unwrap()
                .weight()
                .as_data()
                .unwrap()
                .2;
            let Some(axis_size) = a_shape_first.dims[a_shape_first.indexes[3]].to_usize() else {
                continue;
            };
            let b_shape_first = graph
                .graph
                .edges_directed(contig, petgraph::Direction::Incoming)
                .next()
                .unwrap()
                .weight()
                .as_data()
                .unwrap()
                .2;
            let a_shape_last = graph
                .graph
                .edges_connecting(mul, add)
                .next()
                .unwrap()
                .weight()
                .as_data()
                .unwrap()
                .2;
            let b_shape_last = graph
                .graph
                .edges_connecting(contig, add)
                .next()
                .unwrap()
                .weight()
                .as_data()
                .unwrap()
                .2;
            if a_shape_first.len() != 4
                || a_shape_first.slices[a_shape_first.indexes[3]]
                    .0
                    .to_usize()
                    .map(|i| i != axis_size / 2)
                    .unwrap_or(true)
                || b_shape_first.slices[b_shape_first.indexes[3]]
                    .1
                    .to_usize()
                    .map(|i| i != axis_size / 2)
                    .unwrap_or(true)
                || a_shape_last.padding[a_shape_last.indexes[3]]
                    .1
                    .to_usize()
                    .map(|i| i != axis_size / 2)
                    .unwrap_or(true)
                || b_shape_last.padding[b_shape_last.indexes[3]]
                    .0
                    .to_usize()
                    .map(|i| i != axis_size / 2)
                    .unwrap_or(true)
            {
                continue;
            }
            let mut a = graph
                .graph
                .edges_directed(mul, petgraph::Direction::Incoming)
                .find(|e| e.source() != neg_one)
                .map(|e| (e.source(), e.weight().as_data().unwrap()))
                .unwrap();
            for i in 0..a.1 .2.len() {
                a.1 .2.padding[i].0 = 0.into();
                a.1 .2.padding[i].1 = 0.into();
                a.1 .2.slices[i].0 = 0.into();
                a.1 .2.slices[i].1 = i32::MAX.into();
            }
            // Insert op
            let rotate = graph
                .add_op(MetalRotate::<T>::new(axis_size, dev.clone(), queue.clone()))
                .input(a.0, 0, a.1 .2)
                .finish();

            // Create edges to dests
            move_outgoing_edge(add, rotate, &mut graph.graph);
            move_references(
                &mut remap,
                &mut graph.no_delete,
                &mut graph.to_retrieve,
                add,
                rotate,
            );

            // Remove the old ops
            graph.graph.remove_node(add);
            graph.safe_remove_node(mul, 0);
            graph.safe_remove_node(neg_one, 0);
            graph.safe_remove_node(contig, 0);
        }
    }
}
