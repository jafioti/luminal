use std::{collections::HashMap, fmt::Debug, sync::Arc};

use crate::{
    compilers::metal::*,
    op::{
        Add, Constant, Contiguous, Exp2, Function as LFunction, InputTensor, LessThan, Log2,
        MaxReduce, Mod, Mul, Operator, Print, Recip, Sin, Sqrt, SumReduce,
    },
    prelude::*,
};
use half::f16;
use itertools::Itertools;
use metal_rs::{objc::rc::autoreleasepool, *};
use num_traits::FromPrimitive;
use petgraph::{stable_graph::NodeIndex, visit::EdgeRef};

#[derive(Default, Debug)]
pub struct PrimitiveCompiler;

impl Compiler for PrimitiveCompiler {
    fn compile(&self, graph: &mut Graph) {
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
            if graph
                .graph
                .node_weight(function_node)
                .unwrap()
                .as_any()
                .downcast_ref::<crate::op::Function>()
                .unwrap()
                .2
                == std::any::TypeId::of::<Vec<f32>>()
            {
                // Create copy node
                let copy_node = graph
                    .add_op(MetalCopyToDevice::<f16>::new(dev.clone()))
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
                    .add_op(MetalCopyFromDevice::<f16>::new(dev.clone()))
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
                        .max_by_key(|s| s.n_physical_elements())
                        .unwrap(),
                )
            })
            .collect::<Vec<_>>()
        {
            // Create copy node
            let copy_node = graph
                .add_op(MetalCopyFromDevice::<f16>::new(dev.clone()))
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
                .add_op(MetalCopyFromDevice::<f16>::new(dev.clone()))
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
        let mut kernels = HashMap::new();
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
                *op_ref = Box::new(MetalLog2::<f16>::new(
                    dev.clone(),
                    queue.clone(),
                    &mut kernels,
                ));
            } else if is::<Exp2>(op) {
                *op_ref = Box::new(MetalExp2::<f16>::new(
                    dev.clone(),
                    queue.clone(),
                    &mut kernels,
                ));
            } else if let Some(c) = op_ref.as_any().downcast_ref::<Constant>() {
                *op_ref = Box::new(MetalConstant(f16::from_f32(c.0), dev.clone()));
            } else if is::<Sin>(op) {
                *op_ref = Box::new(MetalSin::<f16>::new(
                    dev.clone(),
                    queue.clone(),
                    &mut kernels,
                ));
            } else if is::<Sqrt>(op) {
                *op_ref = Box::new(MetalSqrt::<f16>::new(
                    dev.clone(),
                    queue.clone(),
                    &mut kernels,
                ));
            } else if is::<Recip>(op) {
                *op_ref = Box::new(MetalRecip::<f16>::new(
                    dev.clone(),
                    queue.clone(),
                    &mut kernels,
                ));
            } else if is::<Add>(op) {
                *op_ref = Box::new(MetalAdd::<f16>::new(
                    src_shapes[0],
                    src_shapes[1],
                    dev.clone(),
                    queue.clone(),
                    &mut kernels,
                    &graph.dyn_map,
                ));
            } else if is::<Mul>(op) {
                *op_ref = Box::new(MetalMul::<f16>::new(
                    src_shapes[0],
                    src_shapes[1],
                    dev.clone(),
                    queue.clone(),
                    &mut kernels,
                    &graph.dyn_map,
                ));
            } else if is::<LessThan>(op) {
                *op_ref = Box::new(MetalLessThan::<f16>::new(
                    src_shapes[0],
                    src_shapes[1],
                    dev.clone(),
                    queue.clone(),
                    &mut kernels,
                    &graph.dyn_map,
                ));
            } else if is::<Mod>(op) {
                *op_ref = Box::new(MetalMod::<f16>::new(
                    src_shapes[0],
                    src_shapes[1],
                    dev.clone(),
                    queue.clone(),
                    &mut kernels,
                    &graph.dyn_map,
                ));
            } else if let Some(SumReduce(dim)) = op_ref.as_any().downcast_ref() {
                *op_ref = Box::new(MetalSumReduce::<f16>::new(
                    src_shapes[0],
                    *dim,
                    dev.clone(),
                    queue.clone(),
                    &mut kernels,
                    &graph.dyn_map,
                ));
            } else if let Some(MaxReduce(dim)) = op_ref.as_any().downcast_ref() {
                *op_ref = Box::new(MetalMaxReduce::<f16>::new(
                    src_shapes[0],
                    *dim,
                    dev.clone(),
                    queue.clone(),
                    &mut kernels,
                    &graph.dyn_map,
                ));
            } else if is::<Contiguous>(op) {
                *op_ref = Box::new(MetalContiguous::<f16>::new(
                    src_shapes[0],
                    dev.clone(),
                    &mut kernels,
                    &graph.dyn_map,
                ));
            }
        }
    }
}

/// In 16 bit, summing above 2048 doesn't work. This precludes the .expand(Dim).sum_reduce() pattern to get a dim size in a tensor, so we need to replace these fake reductions with an elementwise mul
#[derive(Debug, Default)]
pub struct FakeReductionCompiler;

impl Compiler for FakeReductionCompiler {
    fn compile(&self, graph: &mut Graph) {
        let mut sum_reduce = NodeIndex::default();
        let s = SelectEdge::new(
            SelectOp::new().check(|o, _| {
                if let Some(c) = o.as_any().downcast_ref::<MetalConstant<f16>>() {
                    c.0 == f16::ONE
                } else {
                    false
                }
            }),
            SelectOp::new()
                .ty::<MetalSumReduce<f16>>()
                .check(|o, shapes| {
                    if let Some(o) = o.as_any().downcast_ref::<MetalSumReduce<f16>>() {
                        shapes[0].fake[shapes[0].indexes[o.3]] // Ensure dimension we are reducing is fake
                    } else {
                        false
                    }
                })
                .ptr(&mut sum_reduce),
        );
        let mut compiled = None;
        for _ in s.search(graph) {
            let op_ref = graph.graph.node_weight_mut(sum_reduce).unwrap();
            let sum_reduce = op_ref
                .as_any()
                .downcast_ref::<MetalSumReduce<f16>>()
                .unwrap();
            if compiled.is_none() {
                compiled = Some(FakeSumReduce::compile(sum_reduce.2.clone()));
            }
            *op_ref = Box::new(FakeSumReduce(
                compiled.clone().unwrap(),
                sum_reduce.2.clone(),
                sum_reduce.3,
            ));
        }
    }
}

#[derive(Debug, Clone)]
pub struct FakeSumReduce(ComputePipelineState, Device, pub usize);
impl PartialEq for FakeSumReduce {
    fn eq(&self, _: &Self) -> bool {
        false
    }
}

impl FakeSumReduce {
    pub fn compile(dev: Device) -> ComputePipelineState {
        let mut code = "#include <metal_stdlib>
using namespace metal;
kernel void mkernel(device half *inp [[buffer(0)]], device half *out [[buffer(1)]], device uint& n_elements [[buffer(2)]], uint idx [[thread_position_in_grid]], device half& mul_factor [[buffer(3)]]) {{
    if (idx < n_elements) {{
        out[idx] = inp[idx] * mul_factor;
    }}
}}
".to_string();
        let name = format!("kernel_{}", hash(&code));
        code = code.replace("mkernel", &name);

        compile_function(&name, &code, &dev)
    }
}

impl MetalKernelForward for FakeSumReduce {
    fn metal_forward(
        &self,
        inputs: &[(&Buffer, ShapeTracker)],
        dev: &Device,
        command_buffer: &CommandBufferRef,
    ) -> Vec<Buffer> {
        let dim_size = f16::from_usize(inputs[0].1.shape()[self.2].to_usize().unwrap()).unwrap();
        let inp_size = inputs[0].1.n_physical_elements();
        let out = dev.new_buffer(
            (inp_size * std::mem::size_of::<f16>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let encoder =
            command_buffer.compute_command_encoder_with_descriptor(ComputePassDescriptor::new());
        encoder.set_compute_pipeline_state(&self.0);

        // Set inputs
        encoder.set_buffer(0, Some(inputs[0].0), 0);
        encoder.set_buffer(1, Some(&out), 0);
        encoder.set_int(2, inp_size as u32);
        encoder.set_bytes(
            3,
            std::mem::size_of::<f16>() as u64,
            &dim_size as *const f16 as *const _,
        );

        // Execute
        encoder.dispatch_1d(inp_size);
        encoder.end_encoding();

        vec![out]
    }
}

impl Operator for FakeSumReduce {
    fn process(&self, tensors: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        autoreleasepool(|| {
            let inp = tensors[0]
                .0
                .borrowed()
                .data
                .as_any()
                .downcast_ref::<Buffer>()
                .unwrap();

            // Setup command queue / command buffer / encoder
            let command_queue = self.1.new_command_queue();
            let command_buffer = command_queue.new_command_buffer();

            let out = self
                .metal_forward(&[(inp, tensors[0].1)], &self.1, command_buffer)
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

/// Sometimes CopyTo -> CopyFrom and CopyFrom -> CopyTo patterns remain, so let's clean them up
#[derive(Debug, Default)]
pub struct CopyCompiler;

impl Compiler for CopyCompiler {
    fn compile(&self, graph: &mut Graph) {
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
                    .is::<MetalCopyToDevice<f16>>()
                    && graph
                        .graph
                        .node_weight(*b)
                        .unwrap()
                        .as_any()
                        .is::<MetalCopyFromDevice<f16>>())
                    || (graph
                        .graph
                        .node_weight(*a)
                        .unwrap()
                        .as_any()
                        .is::<MetalCopyFromDevice<f16>>()
                        && graph
                            .graph
                            .node_weight(*b)
                            .unwrap()
                            .as_any()
                            .is::<MetalCopyToDevice<f16>>())
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
                        .is::<MetalCopyFromDevice<f16>>()
                        && !graph
                            .graph
                            .node_weight(e.target())
                            .unwrap()
                            .as_any()
                            .is::<MetalCopyToDevice<f16>>()
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
                &mut graph.id_remap,
                &mut graph.no_delete,
                &mut graph.to_retrieve,
                second,
                source.0,
            );
            graph.graph.remove_node(second);
            graph.id_remap.retain(|_, v| *v != second);
            for dest in graph
                .get_dests(first)
                .iter()
                .map(|(i, _)| *i)
                .collect::<Vec<_>>()
            {
                move_outgoing_edge(dest, source.0, &mut graph.graph);
                move_references(
                    &mut graph.id_remap,
                    &mut graph.no_delete,
                    &mut graph.to_retrieve,
                    dest,
                    source.0,
                );
                graph.graph.remove_node(dest);
                graph.id_remap.retain(|_, v| *v != dest);
            }
            graph.graph.remove_node(first);
            graph.id_remap.retain(|_, v| *v != first);
        }
    }
}
