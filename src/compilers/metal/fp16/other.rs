use std::{sync::Arc, mem::size_of};

use half::f16;
use num_traits::FloatConst;
use petgraph::stable_graph::NodeIndex;

use crate::{
    op::{InputTensor, Operator, ConstantValue},
    compilers::metal::{*, prim::*},
    prelude::*,
};

use metal_rs::{objc::rc::autoreleasepool, *};

/// Special kernel for cos
#[derive(LuminalEq, LuminalPrint, Clone)]
pub struct MetalCos(ComputePipelineState, Device);

impl MetalCos {
    fn new(dev: Device) -> Self {
        let mut code = 
            "
#include <metal_stdlib>
using namespace metal;
kernel void mkernel(device half *inp [[buffer(0)]], device half *out [[buffer(1)]], device int& n_elements [[buffer(2)]], uint i_ [[thread_position_in_grid]]) {{
    if (i_ < n_elements) {{
        out[i_] = cos(inp[i_]);
    }}
}}
".to_string();
        code = code.replace("mkernel", "kernel_metal_cos");

        Self(compile_function("kernel_metal_cos", &code, &dev), dev)
    }
}

impl MetalKernelForward for MetalCos {
    fn output_buffer_sizes(&self, input_shapes: &[ShapeTracker]) -> Vec<BigExpression> {
        vec![input_shapes[0].n_elements() * size_of::<f16>()]
    }
    fn metal_forward(
        &self,
        inputs: &[(&Buffer, ShapeTracker)],
        _: &Device,
        command_buffer: &CommandBufferRef,
        _: &[&Buffer],
        output_buffers: &[&Buffer],
    ) {
        let inp_size = inputs[0].1.n_physical_elements().to_usize().unwrap();
        let encoder = command_buffer
            .compute_command_encoder_with_descriptor(ComputePassDescriptor::new());
        encoder.set_compute_pipeline_state(&self.0);

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
    fn process(&self, tensors: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        autoreleasepool(|| {
            let a = tensors[0]
                .0
                .borrowed()
                .data
                .as_any()
                .downcast_ref::<Buffer>()
                .unwrap();
            let inp_size = tensors[0].1.n_physical_elements().to_usize().unwrap();
            let out = self.1.new_buffer(
                (inp_size * std::mem::size_of::<f16>()) as u64,
                MTLResourceOptions::StorageModeShared,
            );
            // Setup command queue / command buffer / encoder
            let command_queue = self.1.new_command_queue();
            let command_buffer = command_queue.new_command_buffer();

            self.metal_forward(&[(a, tensors[0].1)], &self.1, command_buffer, &[], &[&out]);
            
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
pub struct MetalCosCompiler;

impl Compiler for MetalCosCompiler {
    fn compile(&self, graph: &mut Graph) {
        let dev = Device::system_default().unwrap();
        // Look for the cos pattern
        // sin(add(mul(const_neg_one, x), const_pi_over_2))
        let (
            mut const_neg_one,
            mut const_pi,
            mut mul,
            mut add,
            mut sin,
            mut x,
        ) = (
            NodeIndex::default(),
            NodeIndex::default(),
            NodeIndex::default(),
            NodeIndex::default(),
            NodeIndex::default(),
            NodeIndex::default(),
        );

        let s = SelectEdge::new(
            SelectEdge::new(
                SelectOp::new().check(|op, _| if let Some(c) = op.as_any().downcast_ref::<MetalConstant<f16>>() {
                    if let ConstantValue::Float(v) = c.0 {
                        v == f32::PI() / 2.
                    } else {
                        false
                    }
                } else {false}).ptr(&mut const_pi),
                SelectEdge::new(
                    SelectEdge::new(
                        SelectOp::new().ptr(&mut x),
                        SelectEdge::new(
                                SelectOp::new().check(|op, _| if let Some(c) = op.as_any().downcast_ref::<MetalConstant<f16>>() {
                                    if let ConstantValue::Float(v) = c.0 {
                                        v == -1.0
                                    } else {
                                        false
                                    }
                                } else {false}).ptr(&mut const_neg_one),
                            SelectOp::new().ty::<MetalMul<f16>>().ptr(&mut mul),
                        ),
                    ),
                    SelectOp::new().ty::<MetalAdd<f16>>().ptr(&mut add),
                ),
            ),
            SelectOp::new().ty::<MetalSin<f16>>().ptr(&mut sin),
        );
        for _ in s.search(graph) {
            if graph.no_delete.contains(&const_neg_one)
                || graph.no_delete.contains(&const_pi)
                || graph.no_delete.contains(&mul)
                || graph.no_delete.contains(&add)
            {
                // An intermediate node can't be deleted
                continue;
            }

            // Insert cos op
            let shape = graph.graph.edges_directed(mul, petgraph::Direction::Incoming).find_map(|e| e.weight().as_data()).unwrap().2;
            let cos = graph
                .add_op(MetalCos::new(dev.clone()))
                .input(x, 0, shape)
                .finish();

            // Create edges to dests
            move_outgoing_edge(sin, cos, &mut graph.graph);
            move_references(
                &mut graph.id_remap,
                &mut graph.no_delete,
                &mut graph.to_retrieve,
                sin,
                cos,
            );

            // Remove the old ops
            graph.graph.remove_node(mul);
            graph.graph.remove_node(add);
            graph.graph.remove_node(const_neg_one);
            graph.graph.remove_node(const_pi);
            graph.graph.remove_node(sin);
        }
    }
}

#[derive(LuminalEq, LuminalPrint, Clone)]
pub struct MetalExp(ComputePipelineState, Device);

impl MetalExp {
    fn new(dev: Device) -> Self {
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

        Self(compile_function("kernel_metal_exp", &code, &dev), dev)
    }
}

impl MetalKernelForward for MetalExp {
    fn output_buffer_sizes(&self, input_shapes: &[ShapeTracker]) -> Vec<BigExpression> {
        vec![input_shapes[0].n_elements() * size_of::<f16>()]
    }
    fn metal_forward(
        &self,
        inputs: &[(&Buffer, ShapeTracker)],
        _: &Device,
        command_buffer: &CommandBufferRef,
        _: &[&Buffer],
        output_buffers: &[&Buffer],
    ) {
        let inp_size = inputs[0].1.n_physical_elements().to_usize().unwrap();

        let encoder = command_buffer
                .compute_command_encoder_with_descriptor(ComputePassDescriptor::new());
            encoder.set_compute_pipeline_state(&self.0);

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
    fn process(&self, tensors: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
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
            let out = self.1.new_buffer(
                (inp_size * std::mem::size_of::<f16>()) as u64,
                MTLResourceOptions::StorageModeShared,
            );
            

            // Setup command queue / command buffer / encoder
            let command_queue = self.1.new_command_queue();
            let command_buffer = command_queue.new_command_buffer();

            self.metal_forward(&[(a_inp, tensors[0].1)], &self.1, command_buffer, &[], &[&out]);
            
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

/// Replace the mean reduce pattern with a special kernel. This is meant to be ran **after** the FakeSumReduceCompiler.
#[derive(Default, Debug)]
pub struct MetalExpCompiler;

impl Compiler for MetalExpCompiler {
    fn compile(&self, graph: &mut Graph) {
        let dev = Device::system_default().unwrap();
        // Look for the exp pattern
        // exp2(mul(x, const))
        let (
            mut constant,
            mut mul,
            mut exp2
        ) = (
            NodeIndex::default(),
            NodeIndex::default(),
            NodeIndex::default(),
        );

        let s = 
        SelectEdge::new(
            SelectEdge::new(
                SelectOp::new()
                    .check(|op, _| if let Some(c) = op.as_any().downcast_ref::<MetalConstant<f16>>() {
                            if let ConstantValue::Float(v) = c.0 {
                                v == 1.0 / f32::ln(2.)
                            } else {
                                false
                            }
                        } else {false}
                    ).ptr(&mut constant), 
                SelectOp::new().ty::<MetalMul<f16>>().ptr(&mut mul)), 
            SelectOp::new().ty::<MetalExp2<f16>>().ptr(&mut exp2)
        );

        for _ in s.search(graph) {
            if graph.no_delete.contains(&constant)
                || graph.no_delete.contains(&mul)
                || graph.no_delete.contains(&exp2)
            {
                // An intermediate node can't be deleted
                continue;
            }

            // Insert exp op
            let src = graph.get_sources(mul).into_iter().find(|(i, _, _)| *i != constant).unwrap();
            let exp = graph
                .add_op(MetalExp::new(dev.clone()))
                .input(src.0, 0, src.2)
                .finish();

            // Create edges to dests
            move_outgoing_edge(exp2, exp, &mut graph.graph);
            move_references(
                &mut graph.id_remap,
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

#[derive(LuminalEq, LuminalPrint, Clone)]
pub struct MetalGather(ComputePipelineState, Device, usize);

impl MetalGather {
    fn new(dev: Device, embed_dim: usize) -> Self {
        Self(compile_function("metal_gather", "
#include <metal_stdlib>
using namespace metal;
kernel void metal_gather(device float *inp [[buffer(0)]], device half *weights [[buffer(1)]], device half *out [[buffer(2)]], device int& n_embeddings [[buffer(3)]], device int& embedding_dim [[buffer(4)]], int2 i_ [[thread_position_in_grid]]) {
    if (i_.x < n_embeddings && i_.y < embedding_dim) {
        out[i_.x * embedding_dim + i_.y] = weights[(int)inp[i_.x] * embedding_dim + i_.y];
    }
}
        ", &dev), dev, embed_dim)
    }
}

impl Operator for MetalGather {
    fn process(&self, tensors: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        autoreleasepool(|| {
            // Setup buffers
            let indexes = tensors[0]
                .0
                .borrowed()
                .data
                .as_any()
                .downcast_ref::<Vec<f32>>()
                .unwrap();
            let index_buffer = self.1.new_buffer_with_data(
                unsafe { std::mem::transmute(indexes.as_ptr()) },
                (indexes.len() * std::mem::size_of::<f32>()) as u64,
                MTLResourceOptions::StorageModeShared,
            );
            let b_inp = tensors[1]
                .0
                .borrowed()
                .data
                .as_any()
                .downcast_ref::<Buffer>()
                .unwrap();
            

            // Setup command queue / command buffer / encoder
            let command_queue = self.1.new_command_queue();
            let command_buffer = command_queue.new_command_buffer();

            // Input 0 is indexes, input 1 is embedding weights
            let n_embeddings = tensors[0].1.n_physical_elements().to_usize().unwrap();
            let out = self.1.new_buffer(
                (n_embeddings * self.2 * std::mem::size_of::<f16>()) as u64,
                MTLResourceOptions::StorageModeShared,
            );

            let encoder = command_buffer
                    .compute_command_encoder_with_descriptor(ComputePassDescriptor::new());
            encoder.set_compute_pipeline_state(&self.0);

            // Set inputs
            encoder.set_buffer(0, Some(&index_buffer), 0);
            encoder.set_buffer(1, Some(b_inp), 0);
            encoder.set_buffer(2, Some(&out), 0);
            encoder.set_int(3, n_embeddings as u32);
            encoder.set_int(4, self.2 as u32);

            // Execute
            encoder.dispatch_threads(MTLSize { width: n_embeddings as u64, height: self.2 as u64, depth: 1 }, MTLSize { width: 16, height: 16, depth: 1 });
            encoder.end_encoding();

            
            command_buffer.commit();
            command_buffer.wait_until_completed();

            vec![Tensor {
                data: Box::new(out),
            }]
        })
    }
}

/// Replace the mean reduce pattern with a special kernel. This is meant to be ran **after** the FakeSumReduceCompiler.
#[derive(Default, Debug)]
pub struct MetalGatherCompiler;

impl Compiler for MetalGatherCompiler {
    fn compile(&self, graph: &mut Graph) {
        let dev = Device::system_default().unwrap();
        // Look for the exp pattern
        // exp2(mul(x, const))
        let mut gather = NodeIndex::default();

        let s: SelectEdge = SelectOp::new().check(|op, _| if let Some(op) = op.as_any().downcast_ref::<crate::op::Function>() {
            op.0 == "Gather"
        } else {false}).ptr(&mut gather).into();
        for _ in s.search(graph) {
            let srcs = graph.get_sources(gather);
            let (indexes, weights_copy_from) = (srcs[0], srcs[1]);
            let copy_to = graph.get_dests(gather)[0].0;
            if graph.no_delete.contains(&weights_copy_from.0) || graph.get_dests(gather).len() > 1 {
                // An intermediate node can't be deleted
                continue;
            }

            // Insert gather op
            let weight_src = graph.get_sources(weights_copy_from.0)[0];
            let new_gather = graph
                .add_op(MetalGather::new(dev.clone(), weights_copy_from.2.shape()[1].to_usize().unwrap()))
                .input(indexes.0, indexes.1, indexes.2)
                .input(weight_src.0, weight_src.1, weights_copy_from.2)
                .finish();

            // Create edges to dests
            move_outgoing_edge(copy_to, new_gather, &mut graph.graph);
            move_references(
                &mut graph.id_remap,
                &mut graph.no_delete,
                &mut graph.to_retrieve,
                gather,
                new_gather,
            );
            move_references(
                &mut graph.id_remap,
                &mut graph.no_delete,
                &mut graph.to_retrieve,
                weights_copy_from.0,
                new_gather,
            );
            move_references(
                &mut graph.id_remap,
                &mut graph.no_delete,
                &mut graph.to_retrieve,
                copy_to,
                new_gather,
            );

            // Remove the old ops
            graph.graph.remove_node(weights_copy_from.0);
            graph.graph.remove_node(gather);
            graph.graph.remove_node(copy_to);
        }
    }
}