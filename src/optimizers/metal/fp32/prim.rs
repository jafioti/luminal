use std::{
    any::{Any, TypeId},
    collections::{hash_map::DefaultHasher, HashMap, HashSet},
    hash::{Hash, Hasher},
};

use crate::{
    op::{Contiguous, Function as LFunction, InputTensor, Operator, Print},
    prelude::*,
};
use itertools::Itertools;
use metal_rs::*;
use petgraph::visit::EdgeRef;

/// Copy a tensor to the GPU
#[derive(Debug, Clone)]
pub struct MetalCopyToDevice(Device);
impl PartialEq for MetalCopyToDevice {
    fn eq(&self, _: &Self) -> bool {
        false
    }
}

impl Operator for MetalCopyToDevice {
    fn process(&self, mut inp: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        if inp[0].0.borrowed().data.as_any().is::<Buffer>() {
            // Already on device
            return vec![inp.pop().unwrap().0.cloned()];
        }
        let data = inp[0]
            .0
            .borrowed()
            .data
            .as_any()
            .downcast_ref::<Vec<f32>>()
            .unwrap();
        let buffer = self.0.new_buffer_with_data(
            unsafe { std::mem::transmute(data.as_ptr()) },
            (data.len() * std::mem::size_of::<f32>()) as u64,
            MTLResourceOptions::CPUCacheModeDefaultCache,
        );
        vec![Tensor {
            data: Box::new(buffer),
        }]
    }
}

/// Copy a tensor from the GPU
#[derive(Debug, Clone)]
pub struct MetalCopyFromDevice(Device);
impl PartialEq for MetalCopyFromDevice {
    fn eq(&self, _: &Self) -> bool {
        false
    }
}

impl Operator for MetalCopyFromDevice {
    fn process(&self, mut inp: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        if inp[0].0.borrowed().data.as_any().is::<Vec<f32>>()
            || inp[0].0.borrowed().data.as_any().is::<Vec<usize>>()
        {
            // Already off device
            return vec![inp.pop().unwrap().0.cloned()];
        }
        let mut data = vec![0.0; inp[0].1.n_physical_elements()];
        let buffer = inp[0]
            .0
            .borrowed()
            .data
            .as_any()
            .downcast_ref::<Buffer>()
            .unwrap();
        let ptr = buffer.contents() as *mut f32;
        for (i, d) in data.iter_mut().enumerate() {
            *d = unsafe { *ptr.add(i) };
        }
        vec![Tensor {
            data: Box::new(data),
        }]
    }
}

#[derive(Debug, Clone)]
pub struct MetalContiguous(ComputePipelineState, Device, ShapeTracker);

impl PartialEq for MetalContiguous {
    fn eq(&self, _: &Self) -> bool {
        false
    }
}

impl MetalContiguous {
    fn new(
        shape: ShapeTracker,
        dev: Device,
        kernels: &mut HashMap<String, ComputePipelineState>,
    ) -> Self {
        let idx_exp = shape.index_expression();
        let valid_exp = shape.valid_expression();
        let mut code = format!(
            "
#include <metal_stdlib>
using namespace metal;
kernel void mkernel(device float *inp [[buffer(0)]], device float *out [[buffer(1)]], device int& n_elements [[buffer(2)]], uint idx [[thread_position_in_grid]]{}) {{
    if (idx < n_elements && ({valid_exp} != 0)) {{
        out[idx] = inp[{idx_exp}];
    }}
}}
",
            shape
                .shape()
                .into_iter()
                .filter_map(|d| if let Dim::Unknown(c) = d {
                    Some(c)
                } else {
                    None
                })
                .unique()
                .enumerate()
                .map(|(i, c)| format!(", device int& {c} [[buffer({})]]", i + 3))
                .collect::<String>()
        );
        let name = format!("kernel_{}", hash(&code));
        code = code.replace("mkernel", &name);

        if !kernels.contains_key(&name) {
            kernels.insert(name.clone(), compile_function(&name, &code, &dev));
        }
        Self(kernels[&name].clone(), dev, shape)
    }
}
impl Operator for MetalContiguous {
    fn process(&self, tensors: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        let res_shape = tensors[0].1.contiguous();
        let inp_size = res_shape.n_elements();
        let a = tensors[0]
            .0
            .borrowed()
            .data
            .as_any()
            .downcast_ref::<Buffer>()
            .unwrap();
        let out = self.1.new_buffer(
            (inp_size * std::mem::size_of::<f32>()) as u64,
            MTLResourceOptions::CPUCacheModeDefaultCache,
        );

        let command_queue = self.1.new_command_queue();
        let command_buffer = command_queue.new_command_buffer();
        let encoder =
            command_buffer.compute_command_encoder_with_descriptor(ComputePassDescriptor::new());
        encoder.set_compute_pipeline_state(&self.0);

        encoder.set_buffer(0, Some(a), 0);
        encoder.set_buffer(1, Some(&out), 0);
        encoder.set_bytes(
            2,
            std::mem::size_of::<i32>() as u64,
            &(inp_size as i32) as *const i32 as *const _,
        );

        let mut added = HashSet::new();
        let mut dims = [0; 10];
        for (d1, d2) in self.2.shape().into_iter().zip(tensors[0].1.shape()) {
            if let Dim::Unknown(c) = d1 {
                if !added.contains(&c) {
                    dims[added.len()] = d2.to_usize().unwrap() as i32;
                    added.insert(c);
                }
            }
        }
        #[allow(clippy::needless_range_loop)]
        for i in 0..added.len() {
            encoder.set_bytes(
                (3 + i) as u64,
                std::mem::size_of::<i32>() as u64,
                &dims[i] as *const i32 as *const _,
            );
        }

        let num_threads = self.0.thread_execution_width();
        encoder.dispatch_thread_groups(
            MTLSize {
                width: ((inp_size as NSUInteger + num_threads) / num_threads),
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: num_threads,
                height: 1,
                depth: 1,
            },
        );
        encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();

        vec![Tensor {
            data: Box::new(out),
        }]
    }
}

#[derive(Default)]
pub struct PrimitiveOptimizer;

impl GraphOptimizer for PrimitiveOptimizer {
    fn optimize(&self, graph: &mut Graph) {
        let dev = Device::system_default().unwrap();
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
                .add_op(MetalCopyToDevice(dev.clone()))
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

            // If there are inputs to this function remap the function to the copy node
            if graph
                .graph
                .edges_directed(function_node, petgraph::Direction::Incoming)
                .count()
                != 0
            {
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
                    .add_op(MetalCopyFromDevice(dev.clone()))
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
            // Filter non-functions
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
                        .next()
                        .unwrap()
                        .weight()
                        .2,
                )
            })
            .collect::<Vec<_>>()
        {
            // Create copy node
            let copy_node = graph
                .add_op(MetalCopyFromDevice(dev.clone()))
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
                        .next()
                        .unwrap()
                        .id(),
                )
            })
            .collect::<Vec<_>>()
        {
            // Create copy node
            let (source, shape) = (
                graph.graph.edge_endpoints(edge).unwrap().0,
                graph.graph.edge_weight(edge).unwrap().2,
            );
            let copy_node = graph
                .add_op(MetalCopyFromDevice(dev.clone()))
                .input(source, 0, shape)
                .finish();
            graph.graph.add_edge(copy_node, output_node, (0, 0, shape));
            graph.graph.remove_edge(edge);
        }

        // Swap primitive ops
        let mut kernels = HashMap::new();
        for id in graph.graph.node_indices().collect::<Vec<_>>() {
            let src_shapes = graph
                .graph
                .edges_directed(id, petgraph::Direction::Incoming)
                .sorted_by_key(|e| e.weight().0)
                .map(|e| e.weight().2)
                .collect::<Vec<_>>();
            let op = graph.graph.node_weight(id).unwrap().as_any().type_id();
            let op_ref = graph.graph.node_weight_mut(id).unwrap();
            if is::<Contiguous>(op) {
                *op_ref = Box::new(MetalContiguous::new(
                    src_shapes[0],
                    dev.clone(),
                    &mut kernels,
                ));
            }
        }
    }
}

fn compile_function(name: &str, code: &str, device: &Device) -> ComputePipelineState {
    let library = device
        .new_library_with_source(code, &CompileOptions::new())
        .unwrap();
    let pipeline_state_descriptor = ComputePipelineDescriptor::new();
    pipeline_state_descriptor
        .set_compute_function(Some(&library.get_function(name, None).unwrap()));

    device
        .new_compute_pipeline_state_with_function(
            pipeline_state_descriptor.compute_function().unwrap(),
        )
        .unwrap()
}

fn is<T: Any>(type_id: TypeId) -> bool {
    type_id == TypeId::of::<T>()
}

fn hash<T: Hash>(obj: T) -> u64 {
    let mut hasher = DefaultHasher::new();
    obj.hash(&mut hasher);
    hasher.finish()
}
