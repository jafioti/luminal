use std::marker::PhantomData;

use num_traits::FloatConst;
use objc::rc::autoreleasepool;
use petgraph::{stable_graph::NodeIndex, visit::EdgeRef, Direction};

use crate::{
    compilers::metal::{prim::*, *},
    constant_select_op,
    op::{ConstantValue, Operator},
    prelude::*,
};

use super::binary::MetalSub;

/// Sometimes CopyTo -> CopyFrom and CopyFrom -> CopyTo patterns remain, so let's clean them up
#[derive(LuminalPrint, Default)]
pub struct CopyCompiler<T>(PhantomData<T>);

impl<T: MetalFloat> Compiler for CopyCompiler<T> {
    fn compile<To: ToIdsMut>(&self, graph: &mut Graph, mut remap: To) {
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
                    .is::<MetalCopyToDevice<T>>()
                    && graph
                        .graph
                        .node_weight(*b)
                        .unwrap()
                        .as_any()
                        .is::<MetalCopyFromDevice<T>>())
                    || (graph
                        .graph
                        .node_weight(*a)
                        .unwrap()
                        .as_any()
                        .is::<MetalCopyFromDevice<T>>()
                        && graph
                            .graph
                            .node_weight(*b)
                            .unwrap()
                            .as_any()
                            .is::<MetalCopyToDevice<T>>())
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
                        .is::<MetalCopyFromDevice<T>>()
                        && !graph
                            .graph
                            .node_weight(e.target())
                            .unwrap()
                            .as_any()
                            .is::<MetalCopyToDevice<T>>()
                })
                .count()
                > 0
                || graph.no_delete.contains(&first)
            {
                continue;
            }
            let Some(source) = graph.get_sources(first).pop() else {
                continue;
            };
            move_outgoing_edge(second, source.0, &mut graph.graph);
            move_references(
                &mut remap,
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
                    &mut remap,
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

/// Special kernel for efficient mean reduction
#[derive(LuminalEq, LuminalPrint, Clone)]
pub struct MetalARange<T: MetalFloat>(
    ComputePipelineState,
    CommandQueue,
    Device,
    BigExpression,
    *const HashMap<char, usize>,
    PhantomData<T>,
);

impl<T: MetalFloat> MetalARange<T> {
    fn new(
        dev: Device,
        queue: CommandQueue,
        dim: BigExpression,
        dyn_map: *const HashMap<char, usize>,
    ) -> Self {
        Self(
            compile_function("metal_arange", &format!("
#include <metal_stdlib>
using namespace metal;
kernel void metal_arange(device {} *out [[buffer(0)]], device int& n_elements [[buffer(1)]], uint idx [[thread_position_in_grid]]) {{
    if (idx < n_elements) {{
        out[idx] = ({})idx;
    }}
}}", T::type_name(), T::type_name()), &dev),
            queue,
            dev,
            dim,
            dyn_map,
            Default::default(),
        )
    }
}

impl<T: MetalFloat> MetalKernel for MetalARange<T> {
    fn output_buffer_sizes(&self, _: &[ShapeTracker]) -> Vec<BigExpression> {
        vec![self.3.clone() * std::mem::size_of::<f16>()]
    }
    fn metal_forward(
        &self,
        _: &[(&Buffer, ShapeTracker)],
        command_buffer: &CommandBufferRef,
        _: &[&Buffer],
        output_buffers: &[&Buffer],
    ) {
        // Calculate size
        let size = self.3.exec(unsafe { self.4.as_ref().unwrap() }).unwrap();

        let encoder =
            command_buffer.compute_command_encoder_with_descriptor(ComputePassDescriptor::new());
        encoder.set_compute_pipeline_state(&self.0);

        // Set inputs
        encoder.set_buffer(0, Some(output_buffers[0]), 0);
        encoder.set_u32(1, size as u32);

        // Execute
        encoder.dispatch_1d(size);
        encoder.end_encoding();
    }
}

impl<T: MetalFloat> Operator for MetalARange<T> {
    fn process(&mut self, _: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        autoreleasepool(|| {
            // Setup command queue / command buffer / encoder
            let command_buffer = self.1.new_command_buffer();
            let size = self.3.exec(unsafe { self.4.as_ref().unwrap() }).unwrap();
            let out = self.2.new_buffer(
                (size * std::mem::size_of::<f16>()) as u64,
                MTLResourceOptions::StorageModeShared,
            );

            self.metal_forward(&[], command_buffer, &[], &[&out]);

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
#[derive(Default, LuminalPrint)]
pub struct ARangeCompiler<T: MetalFloat>(PhantomData<T>);

impl<T: MetalFloat> Compiler for ARangeCompiler<T> {
    fn compile<To: ToIdsMut>(&self, graph: &mut Graph, _: To) {
        let dev = Device::system_default().unwrap();
        let queue = dev.new_command_queue();
        let (
            mut one_const,
            mut contig1,
            mut contig2,
            mut contig3,
            mut contig4,
            mut sum_reduce,
            mut subtraction_constant,
            mut subtraction,
        ) = (
            NodeIndex::default(),
            NodeIndex::default(),
            NodeIndex::default(),
            NodeIndex::default(),
            NodeIndex::default(),
            NodeIndex::default(),
            NodeIndex::default(),
            NodeIndex::default(),
        );

        // TODO: Make sure this actually checks the shape transformations to ensure pooling happens
        let contig = SelectOp::new().ty::<MetalContiguous<T>>();
        let pre_sub_pattern = constant_select_op!(1.0, T)
            .ptr(&mut one_const)
            .edge(contig.clone().ptr(&mut contig1))
            .edge(contig.clone().ptr(&mut contig2))
            .edge(contig.clone().ptr(&mut contig3))
            .edge(contig.clone().ptr(&mut contig4))
            .edge(
                SelectOp::new()
                    .ty::<MetalSumReduce<T>>()
                    .ptr(&mut sum_reduce),
            );
        let mut s1 = pre_sub_pattern
            .clone()
            .edge(
                constant_select_op!(1.0, T)
                    .ptr(&mut subtraction_constant)
                    .edge(SelectOp::new().ty::<MetalSub<T>>().ptr(&mut subtraction)),
            )
            .search(graph);
        let mut s2 = pre_sub_pattern
            .edge(
                constant_select_op!(-1.0, T)
                    .ptr(&mut subtraction_constant)
                    .edge(SelectOp::new().ty::<MetalAdd<T>>().ptr(&mut subtraction)),
            )
            .search(graph);

        while s1.next_match() || s2.next_match() {
            let arange_amount = {
                let sh = graph
                    .graph
                    .edge_weight(
                        graph
                            .graph
                            .edges_connecting(one_const, contig1)
                            .next()
                            .unwrap()
                            .id(),
                    )
                    .unwrap()
                    .as_data()
                    .unwrap()
                    .2;
                sh.dims[sh.indexes[sh.len() - 1]]
            };
            let arange_op = graph
                .add_op(MetalARange::<T>::new(
                    dev.clone(),
                    queue.clone(),
                    arange_amount.into(),
                    &graph.dyn_map,
                ))
                .finish();
            move_outgoing_edge(subtraction, arange_op, &mut graph.graph);

            graph.graph.remove_node(one_const);
            graph.graph.remove_node(contig1);
            graph.graph.remove_node(contig2);
            graph.graph.remove_node(contig3);
            graph.graph.remove_node(contig4);
            graph.graph.remove_node(sum_reduce);
            graph.graph.remove_node(subtraction);
            graph.graph.remove_node(subtraction_constant);
        }
    }
}

#[derive(Debug, Default)]
pub struct ContiguousElimination<T>(PhantomData<T>);

impl<T: MetalFloat> Compiler for ContiguousElimination<T> {
    fn compile<To: ToIdsMut>(&self, graph: &mut Graph, mut remap: To) {
        // Look for contiguous calls going to ops that can accept non-contiguous inputs (marked non_contiguous)
        let (mut contig, mut op) = (NodeIndex::default(), NodeIndex::default());
        let pattern = SelectOp::new()
            .ty::<MetalContiguous<T>>()
            .ptr(&mut contig)
            .edge(
                SelectOp::new()
                    .check(|op, _| op.custom("non_contiguous", Box::new(())).is_some())
                    .ptr(&mut op),
            );
        let mut selector = pattern.search(graph);
        while selector.next_match() {
            if graph.no_delete.contains(&contig)
                || graph
                    .graph
                    .edges_directed(contig, Direction::Outgoing)
                    .count()
                    > 1
            {
                continue;
            }
            // Shape going from contig to op
            // let first_shape = graph
            //     .graph
            //     .edges_directed(contig, Direction::Incoming)
            //     .find_map(|e| e.weight().as_data())
            //     .unwrap()
            //     .2;
            let second_shape = graph
                .graph
                .edges_connecting(contig, op)
                .find_map(|e| e.weight().as_data())
                .unwrap()
                .2;
            // Here we should check if second shape and first shape are mergeable instead of just checking if second_shape is contiguous
            if second_shape.is_contiguous()
                && !second_shape.is_sliced()
                && !second_shape.is_padded()
            {
                let source = graph
                    .graph
                    .neighbors_directed(contig, petgraph::Direction::Incoming)
                    .next()
                    .unwrap();
                move_incoming_edge(contig, op, &mut graph.graph);
                move_references(
                    &mut remap,
                    &mut graph.no_delete,
                    &mut graph.to_retrieve,
                    contig,
                    source,
                );
                graph.graph.remove_node(contig);
                let new_shapes = graph
                    .get_sources(op)
                    .into_iter()
                    .map(|(_, _, s)| s)
                    .collect::<Vec<_>>();
                graph
                    .graph
                    .node_weight_mut(op)
                    .unwrap()
                    .custom("recompile_shapes", Box::new(new_shapes));
            }
        }
    }
}

// #[derive(LuminalEq, LuminalPrint, Clone)]
// pub struct MetalSoftmax<T: MetalFloat> {
//     pipeline: ComputePipelineState,
//     queue: CommandQueue,
//     device: Device,
//     _phantom: PhantomData<T>,
// }

// impl<T: MetalFloat> MetalSoftmax<T> {
//     fn new(device: Device, queue: CommandQueue) -> Self {
//         Self {
//             pipeline: compile_function(
//                 "metal_softmax",
//                 "
// #include <metal_stdlib>
// using namespace metal;
// kernel void metal_softmax(device half *out [[buffer(0)]], device int& n_elements [[buffer(1)]], uint idx [[thread_position_in_grid]]) {
// }",
//                 &device,
//             ),
//             queue,
//             device,
//             _phantom: Default::default(),
//         }
//     }
// }

// impl<T: MetalFloat> MetalKernel for MetalSoftmax<T> {
//     fn output_buffer_sizes(&self, _: &[ShapeTracker]) -> Vec<BigExpression> {
//         vec![BigExpression::from(0) * std::mem::size_of::<f16>()]
//     }
//     fn metal_forward(
//         &self,
//         _: &[(&Buffer, ShapeTracker)],
//         _: &CommandBufferRef,
//         _: &[&Buffer],
//         _: &[&Buffer],
//     ) {
//         // let encoder =
//         //     command_buffer.compute_command_encoder_with_descriptor(ComputePassDescriptor::new());
//         // encoder.set_compute_pipeline_state(&self.pipeline);

//         // // Set inputs
//         // encoder.set_buffer(0, Some(output_buffers[0]), 0);
//         // encoder.set_int(1, size as u32);

//         // // Execute
//         // encoder.dispatch_1d(size);
//         // encoder.end_encoding();
//     }
// }

// impl<T: MetalFloat> Operator for MetalSoftmax<T> {
//     fn process(&mut self, _: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
//         // autoreleasepool(|| {
//         //     // Setup command queue / command buffer / encoder
//         //     let command_buffer = self.1.new_command_buffer();
//         //     let size = self.3.exec(unsafe { self.4.as_ref().unwrap() }).unwrap();
//         //     let out = self.2.new_buffer(
//         //         (size * std::mem::size_of::<f16>()) as u64,
//         //         MTLResourceOptions::StorageModeShared,
//         //     );

//         //     self.metal_forward(&[], command_buffer, &[], &[&out]);

//         //     command_buffer.commit();
//         //     command_buffer.wait_until_completed();

//         //     vec![Tensor::new(out)]
//         // })
//         vec![]
//     }

//     fn custom(&mut self, key: &str, _: Box<dyn Any>) -> Option<Box<dyn Any>> {
//         if key == "metal" {
//             #[allow(clippy::arc_with_non_send_sync)]
//             return Some(Box::new(MetalKernelWrapper(Arc::new(Box::new(
//                 self.clone(),
//             )))));
//         }
//         None
//     }
// }

// /// Replace the mean reduce pattern with a special kernel. This is meant to be ran **after** the FakeSumReduceCompiler.
// #[derive(Default, LuminalPrint)]
// pub struct MetalSoftmaxCompiler<T: MetalFloat>(PhantomData<T>);

// impl<T: MetalFloat> Compiler for MetalSoftmaxCompiler<T> {
//     fn compile<To: ToIdsMut>(&self, graph: &mut Graph, _: To) {
//         let dev = Device::system_default().unwrap();
//         let queue = dev.new_command_queue();
//         let (mut x1, mut max_reduce, mut sub, mut exp, mut sum_reduce, mut recip, mut mul) = (
//             NodeIndex::default(),
//             NodeIndex::default(),
//             NodeIndex::default(),
//             NodeIndex::default(),
//             NodeIndex::default(),
//             NodeIndex::default(),
//             NodeIndex::default(),
//         );

//         let mut searcher = SelectOp::new()
//             .ptr(&mut x1)
//             .edge(
//                 SelectOp::new()
//                     .ty::<MetalMaxReduce<T>>()
//                     .ptr(&mut max_reduce),
//             )
//             .edge(SelectOp::new().ty::<MetalSub<T>>().ptr(&mut sub))
//             .edge(SelectOp::new().ty::<MetalExp<T>>().ptr(&mut exp))
//             .edge(
//                 SelectOp::new()
//                     .ty::<MetalSumReduce<T>>()
//                     .ptr(&mut sum_reduce),
//             )
//             .edge(SelectOp::new().ty::<MetalRecip<T>>().ptr(&mut recip))
//             .edge(SelectOp::new().ty::<MetalMul<T>>().ptr(&mut mul))
//             .search(graph);

//         while searcher.next_match() {
//             if graph.get_sources(mul).iter().any(|(i, _, _)| *i == exp)
//                 && graph.get_sources(sub).iter().any(|(i, _, _)| *i == x1)
//             {
//                 let softmax = graph
//                     .add_op(MetalSoftmax::<T>::new(dev.clone(), queue.clone()))
//                     .finish();
//                 move_outgoing_edge(mul, softmax, &mut graph.graph);

//                 graph.graph.remove_node(mul);
//                 graph.graph.remove_node(recip);
//                 graph.graph.remove_node(max_reduce);
//                 graph.graph.remove_node(sum_reduce);
//                 graph.graph.remove_node(sub);
//                 graph.graph.remove_node(exp);
//             }
//         }
//     }
// }

#[derive(LuminalEq, LuminalPrint, Clone)]
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
        // This op can accept non contiguous inputs
        if key == "non_contiguous" {
            return Some(Box::new(()));
        }
        None
    }
}

/// Replace the mean reduce pattern with a special kernel. This is meant to be ran **after** the FakeSumReduceCompiler.
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
            graph.graph.remove_node(mul);
            graph.graph.remove_node(constant);
            graph.graph.remove_node(exp2);
        }
    }
}

/// Special kernel for cos
#[derive(LuminalEq, LuminalPrint, Clone)]
pub struct MetalSwish<T: MetalFloat> {
    pipeline: ComputePipelineState,
    device: Device,
    queue: CommandQueue,
    _phantom: PhantomData<T>,
}

impl<T: MetalFloat> MetalSwish<T> {
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
            _phantom: Default::default(),
        }
    }
}

impl<T: MetalFloat> MetalKernel for MetalSwish<T> {
    fn output_buffer_sizes(&self, input_shapes: &[ShapeTracker]) -> Vec<BigExpression> {
        vec![input_shapes[0].n_physical_elements() * std::mem::size_of::<T>()]
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

impl<T: MetalFloat> Operator for MetalSwish<T> {
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
        // This op can accept non contiguous inputs
        if key == "non_contiguous" {
            return Some(Box::new(()));
        }
        None
    }
}

#[derive(Default, Debug)]
pub struct MetalSwishCompiler<T>(PhantomData<T>);

impl<T: MetalFloat> Compiler for MetalSwishCompiler<T> {
    fn compile<To: ToIdsMut>(&self, graph: &mut Graph, _: To) {
        let dev = Device::system_default().unwrap();
        let queue = dev.new_command_queue();
        // Look for the swish pattern
        let (mut neg_one, mut mul1, mut mul2, mut exp, mut one, mut add, mut recip) = (
            NodeIndex::default(),
            NodeIndex::default(),
            NodeIndex::default(),
            NodeIndex::default(),
            NodeIndex::default(),
            NodeIndex::default(),
            NodeIndex::default(),
        );

        let neg_one_node = constant_select_op!(-1.0, T).ptr(&mut neg_one);
        let mul1_node = SelectOp::new().ty::<MetalMul<T>>().ptr(&mut mul1);
        let mul2_node = SelectOp::new().ty::<MetalMul<T>>().ptr(&mut mul2);
        let exp_node = SelectOp::new().ty::<MetalExp<T>>().ptr(&mut exp);
        let recip_node = SelectOp::new().ty::<MetalRecip<T>>().ptr(&mut recip);
        let add_node = SelectOp::new().ty::<MetalAdd<T>>().ptr(&mut add);
        let mut searcher = neg_one_node
            .edge(mul1_node)
            .edge(exp_node)
            .edge(add_node)
            .edge(recip_node)
            .edge(mul2_node)
            .search(graph);

        while searcher.next_match() {
            if check_no_delete(graph, &[neg_one, mul1, mul2, exp, one, add, recip]) {
                // An intermediate node can't be deleted
                continue;
            }

            // Check the if input to add is one
            let add_sources = graph.get_sources(add);
            let (src1_index, _, _) = add_sources[0];
            let (src2_index, _, _) = add_sources[1];

            let src_index = if src1_index == exp {
                src2_index
            } else {
                src1_index
            };

            let test_op = graph.graph.node_weight(src_index).unwrap();

            // If test op is not 1, we continue
            let test_op = test_op.as_any().downcast_ref::<MetalConstant<T>>();
            if let Some(test_op) = test_op {
                if test_op.0 != ConstantValue::Float(1.0) {
                    continue;
                } else {
                    one = src_index;
                }
            } else {
                continue;
            }

            // Now we look for the input
            let mul1_sources = graph.get_sources(mul1);
            let (src1_index, _, shape1) = mul1_sources[0];
            let (src2_index, _, shape2) = mul1_sources[1];

            let (src_index, shape) = if src1_index == neg_one {
                (src2_index, shape2)
            } else {
                (src1_index, shape1)
            };

            // Insert swish op
            let swish = graph
                .add_op(MetalSwish::<T>::new(dev.clone(), queue.clone()))
                .input(src_index, 0, shape)
                .finish();

            // Create edges to dests
            move_outgoing_edge(mul2, swish, &mut graph.graph);

            // Remove the old ops
            graph.graph.remove_node(mul1);
            graph.graph.remove_node(mul2);
            graph.graph.remove_node(neg_one);
            graph.graph.remove_node(exp);
            graph.graph.remove_node(one);
            graph.graph.remove_node(add);
            graph.graph.remove_node(recip);
        }
    }
}

/// Special kernel for cos
#[derive(LuminalEq, LuminalPrint, Clone)]
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
kernel void kernel_metal_cos(device {type_name} *inp [[buffer(0)]], device {type_name} *out [[buffer(1)]], device int& n_elements [[buffer(2)]], uint i_ [[thread_position_in_grid]]) {{
    if (i_ < n_elements) {{
        out[i_] = cos(inp[i_]);
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
