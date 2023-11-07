use std::{
    collections::HashSet,
    sync::{Arc, Mutex},
};

use metal_rs::{Buffer, CommandBufferRef, CommandQueue, Device};
use petgraph::{
    algo::is_cyclic_directed,
    stable_graph::NodeIndex,
    visit::EdgeRef,
    Direction::{self},
};

use crate::{
    op::{InputTensor, Operator},
    prelude::*,
};

use super::prim::MetalKernelWrapper;

#[derive(Default)]
pub struct CommonBufferCompiler;

impl Compiler for CommonBufferCompiler {
    fn compile(&self, graph: &mut Graph) {
        let dev = Device::system_default().unwrap();
        let mut already_added_nodes = HashSet::new();
        for node in graph.graph.node_indices().collect::<Vec<_>>() {
            if !already_added_nodes.contains(&node)
                && graph
                    .graph
                    .node_weight(node)
                    .unwrap()
                    .custom("metal")
                    .is_some()
            {
                // Start a set from this node
                let b = Arc::new(Mutex::new((
                    std::ptr::null::<CommandBufferRef>(),
                    dev.new_command_queue(),
                )));
                let setup = graph
                    .add_op(SetupMetalKernels { buffer: b.clone() })
                    .finish();
                let exec = graph
                    .add_op(ExecuteMetalKernels { buffer: b.clone() })
                    .finish();
                let mut current_set = HashSet::new();
                build_set_from_node(
                    node,
                    b.clone(),
                    &dev,
                    &mut current_set,
                    &mut already_added_nodes,
                    graph,
                    setup,
                    exec,
                );
                // Add execute op
                for node in current_set.clone() {
                    for outside_node in graph
                        .graph
                        .edges_directed(node, Direction::Outgoing)
                        .filter(|e| !e.weight().is_schedule())
                        .map(|e| e.target())
                        .filter(|n| !current_set.contains(n))
                        .collect::<Vec<_>>()
                    {
                        let edge = graph
                            .graph
                            .add_edge(exec, outside_node, Dependency::Schedule);
                        if is_cyclic_directed(&graph.graph) {
                            graph.graph.remove_edge(edge);
                        }
                    }
                }
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn build_set_from_node(
    node: NodeIndex,
    buffer: Arc<Mutex<(*const CommandBufferRef, CommandQueue)>>,
    dev: &Device,
    current_set: &mut HashSet<NodeIndex>,
    already_added_nodes: &mut HashSet<NodeIndex>,
    graph: &mut Graph,
    setup_node: NodeIndex,
    exec_node: NodeIndex,
) {
    let edge1 = graph.graph.add_edge(node, exec_node, Dependency::Schedule);
    let edge2 = graph.graph.add_edge(setup_node, node, Dependency::Schedule);
    if is_cyclic_directed(&graph.graph) {
        graph.graph.remove_edge(edge1);
        graph.graph.remove_edge(edge2);
        return;
    }
    // Wrap current node
    let wrapper = graph
        .graph
        .node_weight(node)
        .unwrap()
        .custom("metal")
        .unwrap()
        .downcast::<MetalKernelWrapper>()
        .unwrap();
    *graph.graph.node_weight_mut(node).unwrap() = Box::new(MetalKernelOperation {
        wrapper,
        dev: dev.clone(),
        buffer: buffer.clone(),
    });
    current_set.insert(node);
    already_added_nodes.insert(node);
    // Add incoming
    for node in graph
        .graph
        .edges_directed(node, Direction::Incoming)
        .filter(|e| !e.weight().is_schedule())
        .map(|e| e.source())
        .filter(|n| !already_added_nodes.contains(n))
        .collect::<Vec<_>>()
    {
        if graph
            .graph
            .node_weight(node)
            .unwrap()
            .custom("metal")
            .is_some()
        {
            build_set_from_node(
                node,
                buffer.clone(),
                dev,
                current_set,
                already_added_nodes,
                graph,
                setup_node,
                exec_node,
            );
        }
    }
    // Add outgoing
    for node in graph
        .graph
        .edges_directed(node, Direction::Outgoing)
        .filter(|e| !e.weight().is_schedule())
        .map(|e| e.target())
        .filter(|n| !already_added_nodes.contains(n))
        .collect::<Vec<_>>()
    {
        if graph
            .graph
            .node_weight(node)
            .unwrap()
            .custom("metal")
            .is_some()
        {
            build_set_from_node(
                node,
                buffer.clone(),
                dev,
                current_set,
                already_added_nodes,
                graph,
                setup_node,
                exec_node,
            );
        }
    }
}

#[derive(Debug)]
struct SetupMetalKernels {
    buffer: Arc<Mutex<(*const CommandBufferRef, CommandQueue)>>,
}

impl PartialEq for SetupMetalKernels {
    fn eq(&self, _: &Self) -> bool {
        false
    }
}

impl Operator for SetupMetalKernels {
    fn process(&self, _: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        let mut buffer = self.buffer.lock().unwrap();
        // buffer.2 = unsafe { objc_autoreleasePoolPush() };
        buffer.0 = buffer.1.new_command_buffer();
        vec![]
    }
}

#[derive(Debug)]
struct ExecuteMetalKernels {
    buffer: Arc<Mutex<(*const CommandBufferRef, CommandQueue)>>,
}

impl PartialEq for ExecuteMetalKernels {
    fn eq(&self, _: &Self) -> bool {
        false
    }
}

impl Operator for ExecuteMetalKernels {
    fn process(&self, _: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        let buffer = self.buffer.lock().unwrap();
        unsafe {
            buffer.0.as_ref().unwrap().commit();
            buffer.0.as_ref().unwrap().wait_until_completed();
            // objc_autoreleasePoolPop(buffer.2); // Don't know why, but this causes a segfault (I think it frees the command buffer before use or something)
        };
        vec![]
    }
}

#[derive(Debug)]
struct MetalKernelOperation {
    wrapper: Box<MetalKernelWrapper>,
    dev: Device,
    buffer: Arc<Mutex<(*const CommandBufferRef, CommandQueue)>>,
}

impl PartialEq for MetalKernelOperation {
    fn eq(&self, _: &Self) -> bool {
        false
    }
}

impl Operator for MetalKernelOperation {
    fn process(&self, inp: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        self.wrapper
            .0
            .metal_forward(
                &inp.iter()
                    .map(|(t, sh)| {
                        (
                            t.borrowed().data.as_any().downcast_ref::<Buffer>().unwrap(),
                            *sh,
                        )
                    })
                    .collect::<Vec<_>>(),
                &self.dev,
                unsafe { self.buffer.lock().unwrap().0.as_ref().unwrap() },
            )
            .into_iter()
            .map(|b| Tensor { data: Box::new(b) })
            .collect()
    }
}
