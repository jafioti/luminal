use std::{mem::size_of, sync::Arc};

use half::f16;
use num_traits::FloatConst;
use petgraph::{stable_graph::NodeIndex, visit::EdgeRef};

use crate::{
    compilers::metal::{prim::*, *},
    constant_select_op,
    op::{ConstantValue, InputTensor, Operator},
    prelude::{metal::binary::MetalSub, *},
};

use metal_rs::{objc::rc::autoreleasepool, *};

/// Special kernel for cos
#[derive(LuminalEq, LuminalPrint, Clone)]
pub struct MetalCos {
    pipeline: ComputePipelineState,
    device: Device,
    queue: CommandQueue,
}

impl MetalCos {
    fn new(device: Device, queue: CommandQueue) -> Self {
        Self {
            pipeline: compile_function("kernel_metal_cos", "#include <metal_stdlib>
using namespace metal;
kernel void kernel_metal_cos(device half *inp [[buffer(0)]], device half *out [[buffer(1)]], device int& n_elements [[buffer(2)]], uint i_ [[thread_position_in_grid]]) {
    if (i_ < n_elements) {
        out[i_] = cos(inp[i_]);
    }
}", &device),
            device,
            queue,
        }
    }
}

impl MetalKernel for MetalCos {
    fn output_buffer_sizes(&self, input_shapes: &[ShapeTracker]) -> Vec<BigExpression> {
        vec![input_shapes[0].n_elements() * size_of::<f16>()]
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
        encoder.set_int(2, inp_size as u32);

        // Execute
        encoder.dispatch_1d(inp_size);
        encoder.end_encoding();
    }
}

impl Operator for MetalCos {
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
                (inp_size * std::mem::size_of::<f16>()) as u64,
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
        // This op can accept non contiguous inputs
        if key == "non_contiguous" {
            return Some(Box::new(()));
        }
        None
    }
}

#[derive(Default, Debug)]
pub struct MetalCosCompiler;

impl Compiler for MetalCosCompiler {
    fn compile<T: ToIdsMut>(&self, graph: &mut Graph, mut remap: T) {
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
                SelectOp::new()
                    .check(|op, _| {
                        if let Some(c) = op.as_any().downcast_ref::<MetalConstant<f16>>() {
                            if let ConstantValue::Float(v) = c.0 {
                                v == f32::PI() / 2.
                            } else {
                                false
                            }
                        } else {
                            false
                        }
                    })
                    .ptr(&mut const_pi)
                    .edge(SelectOp::new().ty::<MetalSub<f16>>().ptr(&mut sub)),
            )
            .edge(SelectOp::new().ty::<MetalSin<f16>>().ptr(&mut sin));

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
                .add_op(MetalCos::new(dev.clone(), queue.clone()))
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

#[derive(LuminalEq, LuminalPrint, Clone)]
pub struct MetalExp {
    pipeline: ComputePipelineState,
    device: Device,
    queue: CommandQueue,
}

impl MetalExp {
    fn new(device: Device, queue: CommandQueue) -> Self {
        let mut code =
            "
#include <metal_stdlib>
using namespace metal;
kernel void mkernel(device half *inp [[buffer(0)]], device half *out [[buffer(1)]], device int& n_elements [[buffer(2)]], uint i_ [[thread_position_in_grid]]) {{
    if (i_ < n_elements) {{
        out[i_] = exp(inp[i_]);
    }}
}}
".to_string();
        code = code.replace("mkernel", "kernel_metal_exp");

        Self {
            pipeline: compile_function("kernel_metal_exp", &code, &device),
            device,
            queue,
        }
    }
}

impl MetalKernel for MetalExp {
    fn output_buffer_sizes(&self, input_shapes: &[ShapeTracker]) -> Vec<BigExpression> {
        vec![input_shapes[0].n_elements() * size_of::<f16>()]
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
        encoder.set_int(2, inp_size as u32);

        // Execute
        encoder.dispatch_1d(inp_size);
        encoder.end_encoding();
    }
}

impl Operator for MetalExp {
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
                (inp_size * std::mem::size_of::<f16>()) as u64,
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
        // This op can accept non contiguous inputs
        if key == "non_contiguous" {
            return Some(Box::new(()));
        }
        None
    }
}

/// Replace the mean reduce pattern with a special kernel. This is meant to be ran **after** the FakeSumReduceCompiler.
#[derive(Default, Debug)]
pub struct MetalExpCompiler;

impl Compiler for MetalExpCompiler {
    fn compile<T: ToIdsMut>(&self, graph: &mut Graph, mut remap: T) {
        let dev = Device::system_default().unwrap();
        let queue = dev.new_command_queue();
        // Look for the exp pattern
        // exp2(mul(x, const))
        let (mut constant, mut mul, mut exp2) = (
            NodeIndex::default(),
            NodeIndex::default(),
            NodeIndex::default(),
        );

        let s = SelectOp::new()
            .check(|op, _| {
                if let Some(c) = op.as_any().downcast_ref::<MetalConstant<f16>>() {
                    if let ConstantValue::Float(v) = c.0 {
                        v == 1.0 / f32::ln(2.)
                    } else {
                        false
                    }
                } else {
                    false
                }
            })
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
                .add_op(MetalExp::new(dev.clone(), queue.clone()))
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
            graph.graph.remove_node(mul);
            graph.graph.remove_node(constant);
            graph.graph.remove_node(exp2);
        }
    }
}

/// Special kernel for cos
#[derive(LuminalEq, LuminalPrint, Clone)]
pub struct MetalSwish {
    pipeline: ComputePipelineState,
    device: Device,
    queue: CommandQueue,
}

impl MetalSwish {
    fn new(device: Device, queue: CommandQueue) -> Self {
        Self {
            pipeline: compile_function("swish_kernel", "#include <metal_stdlib>
using namespace metal;
kernel void swish_kernel(device half *inp [[buffer(0)]], device half *out [[buffer(1)]], device int& n_elements [[buffer(2)]], uint idx [[thread_position_in_grid]]) {
    if (idx < n_elements) {
        out[idx] = inp[idx] * (1.0h / (1.0h + exp(-inp[idx])));
    }
}", &device),
            device,
            queue,
        }
    }
}

impl MetalKernel for MetalSwish {
    fn output_buffer_sizes(&self, input_shapes: &[ShapeTracker]) -> Vec<BigExpression> {
        vec![input_shapes[0].n_physical_elements() * size_of::<f16>()]
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
        encoder.set_int(2, inp_size as u32);

        // Execute
        encoder.dispatch_1d(inp_size);
        encoder.end_encoding();
    }
}

impl Operator for MetalSwish {
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
                (inp_size * std::mem::size_of::<f16>()) as u64,
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
        // This op can accept non contiguous inputs
        if key == "non_contiguous" {
            return Some(Box::new(()));
        }
        None
    }
}

#[derive(Default, Debug)]
pub struct MetalSwishCompiler;

impl Compiler for MetalSwishCompiler {
    fn compile<T: ToIdsMut>(&self, graph: &mut Graph, _: T) {
        let dev = Device::system_default().unwrap();
        let queue = dev.new_command_queue();
        // Look for the swish pattern
        let (
            mut x,
            mut neg_one,
            mut mul1,
            mut mul2,
            mut mul3,
            mut exp,
            mut one,
            mut one2,
            mut add,
            mut recip,
        ) = (
            NodeIndex::default(),
            NodeIndex::default(),
            NodeIndex::default(),
            NodeIndex::default(),
            NodeIndex::default(),
            NodeIndex::default(),
            NodeIndex::default(),
            NodeIndex::default(),
            NodeIndex::default(),
            NodeIndex::default(),
        );
        let mut searcher = constant_select_op!(1.0, f16)
            .ptr(&mut one)
            .edge(
                constant_select_op!(1.0, f16)
                    .ptr(&mut one2)
                    .edge(
                        SelectOp::new()
                            .ptr(&mut x)
                            .edge(
                                constant_select_op!(-1.0, f16)
                                    .ptr(&mut neg_one)
                                    .edge(SelectOp::new().ty::<MetalMul<f16>>().ptr(&mut mul1)),
                            )
                            .edge(SelectOp::new().ty::<MetalExp>().ptr(&mut exp))
                            .edge(SelectOp::new().ty::<MetalAdd<f16>>().ptr(&mut add)),
                    )
                    .edge(SelectOp::new().ty::<MetalRecip<f16>>().ptr(&mut recip))
                    .edge(SelectOp::new().ty::<MetalMul<f16>>().ptr(&mut mul2)),
            )
            .edge(SelectOp::new().ty::<MetalMul<f16>>().ptr(&mut mul3))
            .search(graph);

        while searcher.next_match() {
            if check_no_delete(graph, &[neg_one, mul1, mul2, mul3, exp, one, add, recip])
                || one != one2
            {
                // An intermediate node can't be deleted
                continue;
            }

            // Insert swish op
            let shape = graph
                .graph
                .edges_connecting(x, mul1)
                .find_map(|e| e.weight().as_data())
                .unwrap()
                .2;
            let swish = graph
                .add_op(MetalSwish::new(dev.clone(), queue.clone()))
                .input(x, 0, shape)
                .finish();

            // Create edges to dests
            move_outgoing_edge(mul3, swish, &mut graph.graph);

            // Remove the old ops
            graph.graph.remove_node(mul1);
            graph.graph.remove_node(mul2);
            graph.graph.remove_node(mul3);
            graph.graph.remove_node(neg_one);
            graph.graph.remove_node(exp);
            graph.graph.remove_node(one);
            graph.graph.remove_node(add);
            graph.graph.remove_node(recip);
        }
    }
}
