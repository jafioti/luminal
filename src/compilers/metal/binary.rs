use std::marker::PhantomData;

use objc::rc::autoreleasepool;
use petgraph::{stable_graph::NodeIndex, visit::EdgeRef};

use crate::{
    compilers::metal::{prim::*, *},
    op::{ConstantValue, Operator},
    prelude::*,
};

pub type MetalBinaryCompilers<T> = (MetalSubtractionCompiler<T>, MetalEqualCompiler<T>);

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
kernel void mkernel(device {} *inp_a [[buffer(0)]], device {} *inp_b [[buffer(1)]], device {} *out [[buffer(2)]], device uint& n_elements [[buffer(3)]], uint idx [[thread_position_in_grid]]{}) {{
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

impl<T> MetalKernelForward for MetalSub<T> {
    fn metal_forward(
        &self,
        inputs: &[(&Buffer, ShapeTracker)],
        dev: &Device,
        command_buffer: &CommandBufferRef,
    ) -> Vec<Buffer> {
        let inp_size = inputs[0].1.n_elements();
        let out = dev.new_buffer(
            (inp_size * std::mem::size_of::<T>()) as u64,
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
        input_dyn_dims(
            &[self.3, self.4],
            unsafe { self.6.as_ref().unwrap() },
            encoder,
            4,
        );

        // Execute
        encoder.dispatch_1d(inp_size);
        encoder.end_encoding();

        vec![out]
    }
}

impl<T: MetalFloat> Operator for MetalSub<T> {
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

#[derive(Debug, Default)]
pub struct MetalSubtractionCompiler<T: MetalFloat>(PhantomData<T>);

impl<T: MetalFloat> Compiler for MetalSubtractionCompiler<T> {
    fn compile(&self, graph: &mut Graph) {
        let dev = Device::system_default().unwrap();
        let queue = dev.new_command_queue();
        let (mut a, mut b, mut neg_one, mut mul, mut add) = (
            NodeIndex::default(),
            NodeIndex::default(),
            NodeIndex::default(),
            NodeIndex::default(),
            NodeIndex::default(),
        );
        let s = SelectEdge::new(
            SelectOp::new().ptr(&mut a),
            SelectEdge::new(
                SelectEdge::new(
                    SelectOp::new()
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
                        .ptr(&mut neg_one),
                    SelectEdge::new(
                        SelectOp::new().ptr(&mut b),
                        SelectOp::new().ty::<MetalMul<T>>().ptr(&mut mul),
                    ),
                ),
                SelectOp::new().ty::<MetalAdd<T>>().ptr(&mut add),
            ),
        );

        for _ in s.search(graph) {
            if check_no_delete(graph, &[a, b, neg_one, mul, add]) {
                continue;
            }
            let a_edge = graph
                .graph
                .edge_weight(graph.graph.edges_connecting(a, add).next().unwrap().id())
                .unwrap()
                .as_data()
                .unwrap();
            let b_edge = graph
                .graph
                .edge_weight(graph.graph.edges_connecting(b, mul).next().unwrap().id())
                .unwrap()
                .as_data()
                .unwrap();
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
kernel void mkernel(device {} *inp_a [[buffer(0)]], device {} *inp_b [[buffer(1)]], device {} *out [[buffer(2)]], device uint& n_elements [[buffer(3)]], uint idx [[thread_position_in_grid]]{}) {{
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

impl<T> MetalKernelForward for MetalEqual<T> {
    fn metal_forward(
        &self,
        inputs: &[(&Buffer, ShapeTracker)],
        dev: &Device,
        command_buffer: &CommandBufferRef,
    ) -> Vec<Buffer> {
        let inp_size = inputs[0].1.n_elements();
        let out = dev.new_buffer(
            (inp_size * std::mem::size_of::<T>()) as u64,
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
        input_dyn_dims(
            &[self.3, self.4],
            unsafe { self.6.as_ref().unwrap() },
            encoder,
            4,
        );

        // Execute
        encoder.dispatch_1d(inp_size);
        encoder.end_encoding();

        vec![out]
    }
}

impl<T: MetalFloat> Operator for MetalEqual<T> {
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

#[derive(Debug, Default)]
pub struct MetalEqualCompiler<T: MetalFloat>(PhantomData<T>);

impl<T: MetalFloat> Compiler for MetalEqualCompiler<T> {
    fn compile(&self, graph: &mut Graph) {
        let dev = Device::system_default().unwrap();
        let queue = dev.new_command_queue();
        let (mut less_than1, mut less_than2, mut add, mut one, mut sub) = (
            NodeIndex::default(),
            NodeIndex::default(),
            NodeIndex::default(),
            NodeIndex::default(),
            NodeIndex::default(),
        );
        let s = SelectEdge::new(
            SelectEdge::new(
                SelectOp::new()
                    .ty::<MetalLessThan<T>>()
                    .ptr(&mut less_than1),
                SelectEdge::new(
                    SelectOp::new()
                        .ty::<MetalLessThan<T>>()
                        .ptr(&mut less_than2),
                    SelectOp::new().ty::<MetalAdd<T>>().ptr(&mut add),
                ),
            ),
            SelectEdge::new(
                SelectOp::new()
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
                    .ptr(&mut one),
                SelectOp::new().ty::<MetalSub<T>>().ptr(&mut sub),
            ),
        );

        for _ in s.search(graph) {
            let lt1_inputs = graph
                .graph
                .neighbors_directed(less_than1, petgraph::Direction::Incoming)
                .sorted()
                .collect::<Vec<_>>();
            let lt2_inputs = graph
                .graph
                .neighbors_directed(less_than2, petgraph::Direction::Incoming)
                .sorted()
                .collect::<Vec<_>>();
            if lt1_inputs != lt2_inputs {
                continue;
            }
            let (a, b) = (lt1_inputs[0], lt1_inputs[1]);
            if check_no_delete(graph, &[a, b, less_than1, less_than2, add, one, sub]) {
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
        }
    }
}
