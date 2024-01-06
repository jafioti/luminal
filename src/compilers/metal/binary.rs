use std::{marker::PhantomData, mem::size_of};

use objc::rc::autoreleasepool;
use petgraph::{stable_graph::NodeIndex, visit::EdgeRef, Direction};

use crate::{
    compilers::metal::{prim::*, *},
    op::{ConstantValue, Operator},
    prelude::*,
};

use super::other::MetalARange;

#[derive(LuminalEq, LuminalPrint, Clone)]
pub struct MetalSub<T>(
    ComputePipelineState,
    CommandQueue,
    Device,
    ShapeTracker,
    ShapeTracker,
    PhantomData<T>,
    *const HashMap<char, usize>,
);

impl<T: MetalFloat> MetalSub<T> {
    pub fn new(
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
kernel void mkernel(device {} *inp_a [[buffer(0)]], device {} *inp_b [[buffer(1)]], device {} *out [[buffer(2)]], device int& n_elements [[buffer(3)]], uint idx [[thread_position_in_grid]]{}) {{
    if (idx < n_elements) {{
        out[idx] =
            (({a_valid_exp}) == 0 ? 0.0 : inp_a[{a_idx_exp}])
            - (({b_valid_exp}) == 0 ? 0.0 : inp_b[{b_idx_exp}]);
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

impl<T> MetalKernel for MetalSub<T> {
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
        encoder.set_compute_pipeline_state(&self.0);

        // Set inputs
        encoder.set_buffer(0, Some(inputs[0].0), 0);
        encoder.set_buffer(1, Some(inputs[1].0), 0);
        encoder.set_buffer(2, Some(output_buffers[0]), 0);
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
    }
}

impl<T: MetalFloat> Operator for MetalSub<T> {
    fn process(&mut self, tensors: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        autoreleasepool(|| {
            let command_buffer = self.1.new_command_buffer();
            let inp_size = tensors[0].1.n_elements().to_usize().unwrap();
            let out = self.2.new_buffer(
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

            vec![Tensor {
                data: Box::new(out),
            }]
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
                    self.2.clone(),
                    self.1.clone(),
                    &mut HashMap::new(),
                    self.6,
                )
            }
        }
        None
    }
}

#[derive(LuminalPrint, Default)]
pub struct MetalSubtractionCompiler<T: MetalFloat>(PhantomData<T>);

impl<T: MetalFloat> Compiler for MetalSubtractionCompiler<T> {
    fn compile<To: ToIdsMut>(&self, graph: &mut Graph, _: To) {
        let dev = Device::system_default().unwrap();
        let queue = dev.new_command_queue();
        let (mut neg_one, mut mul, mut add) = (
            NodeIndex::default(),
            NodeIndex::default(),
            NodeIndex::default(),
        );
        let s = SelectOp::new()
            .check(|o, _| {
                if let Some(c) = o.as_any().downcast_ref::<MetalConstant<T>>() {
                    if let ConstantValue::Float(f) = c.0 {
                        f == -1.
                    } else {
                        false
                    }
                } else {
                    false
                }
            })
            .ptr(&mut neg_one)
            .edge(SelectOp::new().ty::<MetalMul<T>>().ptr(&mut mul))
            .edge(SelectOp::new().ty::<MetalAdd<T>>().ptr(&mut add));

        let mut searcher = s.search(graph);
        while searcher.next_match() {
            if check_no_delete(graph, &[neg_one, mul, add]) {
                continue;
            }
            let (a, a_edge) = graph
                .graph
                .edges_directed(add, petgraph::Direction::Incoming)
                .find(|e| e.source() != mul)
                .map(|e| (e.source(), e.weight().as_data().unwrap()))
                .unwrap();
            let (b, b_edge) = graph
                .graph
                .edges_directed(mul, petgraph::Direction::Incoming)
                .find(|e| e.source() != neg_one)
                .map(|e| (e.source(), e.weight().as_data().unwrap()))
                .unwrap();
            let b_final_shape = graph
                .graph
                .edges_connecting(mul, add)
                .next()
                .unwrap()
                .weight()
                .as_data()
                .unwrap()
                .2;
            if b_final_shape != b_edge.2.contiguous() {
                continue;
            }
            let sub = graph
                .add_op(MetalSub::<T>::new(
                    a_edge.2,
                    b_edge.2,
                    dev.clone(),
                    queue.clone(),
                    &mut HashMap::new(),
                    &graph.dyn_map,
                ))
                .input(a, a_edge.1, a_edge.2)
                .input(b, b_edge.1, b_edge.2)
                .finish();
            move_outgoing_edge(add, sub, &mut graph.graph);

            graph.graph.remove_node(neg_one);
            graph.graph.remove_node(mul);
            graph.graph.remove_node(add);
        }
    }
}

#[derive(LuminalEq, LuminalPrint, Clone)]
pub struct MetalEqual<T>(
    ComputePipelineState,
    CommandQueue,
    Device,
    ShapeTracker,
    ShapeTracker,
    PhantomData<T>,
    *const HashMap<char, usize>,
);

impl<T: MetalFloat> MetalEqual<T> {
    pub fn new(
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
kernel void mkernel(device {} *inp_a [[buffer(0)]], device {} *inp_b [[buffer(1)]], device {} *out [[buffer(2)]], device int& n_elements [[buffer(3)]], uint idx [[thread_position_in_grid]]{}) {{
    if (idx < n_elements) {{
        {} a_val = (({a_valid_exp}) == 0 ? 0.0 : inp_a[{a_idx_exp}]);
        {} b_val = (({b_valid_exp}) == 0 ? 0.0 : inp_b[{b_idx_exp}]);
        out[idx] = ({})(a_val == b_val);
    }}
}}
", T::type_name(), T::type_name(), T::type_name(), render_dyn_dim_inputs(&[a_shape, b_shape], 4), T::type_name(), T::type_name(), T::type_name(),
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

impl<T> MetalKernel for MetalEqual<T> {
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
        encoder.set_compute_pipeline_state(&self.0);

        // Set inputs
        encoder.set_buffer(0, Some(inputs[0].0), 0);
        encoder.set_buffer(1, Some(inputs[1].0), 0);
        encoder.set_buffer(2, Some(output_buffers[0]), 0);
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
    }
}

impl<T: MetalFloat> Operator for MetalEqual<T> {
    fn process(&mut self, tensors: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        autoreleasepool(|| {
            let command_buffer = self.1.new_command_buffer();
            let inp_size = tensors[0].1.n_elements().to_usize().unwrap();
            let out = self.2.new_buffer(
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

            vec![Tensor {
                data: Box::new(out),
            }]
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
                    self.2.clone(),
                    self.1.clone(),
                    &mut HashMap::new(),
                    self.6,
                )
            }
        }
        None
    }
}

#[derive(LuminalPrint, Default)]
pub struct MetalEqualCompiler<T: MetalFloat>(PhantomData<T>);

impl<T: MetalFloat> Compiler for MetalEqualCompiler<T> {
    fn compile<To: ToIdsMut>(&self, graph: &mut Graph, _: To) {
        let dev = Device::system_default().unwrap();
        let queue = dev.new_command_queue();
        let (mut less_than1, mut less_than2, mut add, mut one, mut sub) = (
            NodeIndex::default(),
            NodeIndex::default(),
            NodeIndex::default(),
            NodeIndex::default(),
            NodeIndex::default(),
        );
        let s = SelectOp::new()
            .check(|o, _| {
                if let Some(c) = o.as_any().downcast_ref::<MetalConstant<T>>() {
                    if let ConstantValue::Float(f) = c.0 {
                        f == 1.0
                    } else {
                        false
                    }
                } else {
                    false
                }
            })
            .ptr(&mut one)
            .edge(
                SelectOp::new()
                    .ty::<MetalLessThan<T>>()
                    .ptr(&mut less_than1)
                    .edge(
                        SelectOp::new()
                            .ty::<MetalLessThan<T>>()
                            .ptr(&mut less_than2)
                            .edge(SelectOp::new().ty::<MetalAdd<T>>().ptr(&mut add)),
                    )
                    .edge(SelectOp::new().ty::<MetalSub<T>>().ptr(&mut sub)),
            );

        let mut searcher = s.search(graph);
        while searcher.next_match() {
            let lt1_inputs = graph
                .graph
                .neighbors_directed(less_than1, Direction::Incoming)
                .sorted()
                .collect::<Vec<_>>();
            let lt2_inputs = graph
                .graph
                .neighbors_directed(less_than2, Direction::Incoming)
                .sorted()
                .collect::<Vec<_>>();
            if lt1_inputs != lt2_inputs {
                continue;
            }
            let inputs = graph
                .graph
                .edges_directed(less_than1, Direction::Incoming)
                .sorted_by_key(|e| e.weight().as_data().unwrap().0)
                .map(|e| e.source())
                .collect::<Vec<_>>();
            let (a, b) = (inputs[0], inputs[1]);
            if check_no_delete(graph, &[less_than1, less_than2, add, one, sub]) {
                continue;
            }
            let a_edge = graph
                .graph
                .edge_weight(
                    graph
                        .graph
                        .edges_connecting(a, less_than1)
                        .next()
                        .unwrap()
                        .id(),
                )
                .unwrap()
                .as_data()
                .unwrap();
            let b_edge = graph
                .graph
                .edge_weight(
                    graph
                        .graph
                        .edges_connecting(b, less_than1)
                        .next()
                        .unwrap()
                        .id(),
                )
                .unwrap()
                .as_data()
                .unwrap();
            let equals = graph
                .add_op(MetalEqual::<T>::new(
                    a_edge.2,
                    b_edge.2,
                    dev.clone(),
                    queue.clone(),
                    &mut HashMap::new(),
                    &graph.dyn_map,
                ))
                .input(a, a_edge.1, a_edge.2)
                .input(b, b_edge.1, b_edge.2)
                .finish();
            move_outgoing_edge(sub, equals, &mut graph.graph);

            graph.graph.remove_node(less_than1);
            graph.graph.remove_node(less_than2);
            graph.graph.remove_node(add);
            graph.graph.remove_node(one);
            graph.graph.remove_node(sub);
            searcher.clear_cached_results();
        }
    }
}

#[derive(LuminalEq, LuminalPrint, Clone)]
pub struct MetalGather<T>(ComputePipelineState, Device, usize, PhantomData<T>);

impl<T: MetalFloat> MetalGather<T> {
    fn new(dev: Device, embed_dim: usize) -> Self {
        Self(compile_function("metal_gather", &format!(
            "
#include <metal_stdlib>
using namespace metal;
kernel void metal_gather(device float *inp [[buffer(0)]], device {} *weights [[buffer(1)]], device {} *out [[buffer(2)]], device int& n_embeddings [[buffer(3)]], device int& embedding_dim [[buffer(4)]], uint2 i_ [[thread_position_in_grid]]) {{
    if (i_.x < n_embeddings && i_.y < embedding_dim) {{
        out[i_.x * embedding_dim + i_.y] = weights[(int)inp[i_.x] * embedding_dim + i_.y];
    }}
}}", T::type_name(), T::type_name()
        ), &dev), dev, embed_dim, Default::default())
    }
}

impl<T: MetalFloat> Operator for MetalGather<T> {
    fn process(&mut self, tensors: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
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

            let out = self.1.new_buffer(
                (indexes.len() * self.2 * std::mem::size_of::<f16>()) as u64,
                MTLResourceOptions::StorageModeShared,
            );

            let encoder = command_buffer
                .compute_command_encoder_with_descriptor(ComputePassDescriptor::new());
            encoder.set_compute_pipeline_state(&self.0);

            // Set inputs
            encoder.set_buffer(0, Some(&index_buffer), 0);
            encoder.set_buffer(1, Some(b_inp), 0);
            encoder.set_buffer(2, Some(&out), 0);
            encoder.set_int(3, indexes.len() as u32);
            encoder.set_int(4, self.2 as u32);

            // Execute
            encoder.dispatch_threads(
                MTLSize {
                    width: indexes.len() as u64,
                    height: self.2 as u64,
                    depth: 1,
                },
                MTLSize {
                    width: 16,
                    height: 16,
                    depth: 1,
                },
            );
            encoder.end_encoding();

            command_buffer.commit();
            command_buffer.wait_until_completed();

            vec![Tensor {
                data: Box::new(out),
            }]
        })
    }
}

#[derive(LuminalPrint, Default)]
pub struct MetalGatherCompiler<T: MetalFloat>(PhantomData<T>);

impl<T: MetalFloat> Compiler for MetalGatherCompiler<T> {
    fn compile<To: ToIdsMut>(&self, graph: &mut Graph, _: To) {
        let dev = Device::system_default().unwrap();
        let (mut ind_copy, mut arange, mut equal, mut mul, mut sum_reduce) = (
            NodeIndex::default(),
            NodeIndex::default(),
            NodeIndex::default(),
            NodeIndex::default(),
            NodeIndex::default(),
        );
        let s = SelectOp::new()
            .ty::<MetalARange<T>>()
            .ptr(&mut arange)
            .edge(
                SelectOp::new()
                    .ty::<MetalCopyToDevice<T>>()
                    .ptr(&mut ind_copy)
                    .edge(SelectOp::new().ty::<MetalEqual<T>>().ptr(&mut equal)),
            )
            .edge(SelectOp::new().ty::<MetalMul<T>>().ptr(&mut mul))
            .edge(
                SelectOp::new()
                    .ty::<MetalSumReduce<T>>()
                    .ptr(&mut sum_reduce),
            );
        let mut searcher = s.search(graph);
        while searcher.next_match() {
            if check_no_delete(graph, &[arange, equal, mul, sum_reduce]) {
                continue;
            }
            let embedding_dim = graph
                .graph
                .edges_directed(mul, Direction::Incoming)
                .find(|e| e.source() != equal && !e.weight().is_schedule())
                .unwrap()
                .weight()
                .as_data()
                .unwrap()
                .2
                .shape()[2]
                .to_usize()
                .unwrap();
            let gather = graph
                .add_op(MetalGather::<T>::new(dev.clone(), embedding_dim))
                .finish();
            move_incoming_edge(ind_copy, gather, &mut graph.graph);
            graph.graph.remove_node(equal);
            move_incoming_edge(mul, gather, &mut graph.graph);
            move_outgoing_edge(sum_reduce, gather, &mut graph.graph);
            graph.graph.remove_node(arange);
            graph.graph.remove_node(ind_copy);
            graph.graph.remove_node(mul);
            graph.graph.remove_node(sum_reduce);
        }
    }
}

#[cfg(test)]
mod tests {
    crate::test_imports!();
    #[test]
    fn test_subtraction() {
        let mut cx = Graph::new();
        let a = cx
            .tensor::<R1<10>>()
            .set(vec![1., 2., 3., 4., 5., 6., 7., 8., 9., 10.]);
        let b = cx.tensor::<R0>().set(vec![1.]);
        let mut c = (a - b.expand()).retrieve();
        let mut d = (-a + b.expand()).retrieve();

        cx.execute();

        let unopt_c = c.data();
        c.drop();
        let unopt_d = d.data();
        d.drop();

        cx.compile(MetalFp16Compiler::default(), (&mut c, &mut d));
        cx.execute();

        assert_close(&unopt_c, &c.data());
        assert_close(&unopt_d, &d.data());
    }
}
