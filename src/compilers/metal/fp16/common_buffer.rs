use std::{
    collections::HashSet,
    sync::{Arc, Mutex},
};

use metal_rs::{Buffer, CommandBufferRef, CommandQueue, Device};
use petgraph::{
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
        let mut create = vec![];
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
                let mut current_set = HashSet::new();
                current_set.insert(node);
                build_set_from_node(
                    node,
                    b.clone(),
                    &dev,
                    &mut current_set,
                    &mut already_added_nodes,
                    graph,
                );
                // Add execute op
                let mut outside_nodes = HashSet::new();
                for node in &current_set {
                    outside_nodes.extend(
                        graph
                            .graph
                            .edges_directed(*node, Direction::Outgoing)
                            .map(|e| e.target())
                            .filter(|n| !current_set.contains(n)),
                    );
                }
                create.push((
                    SetupMetalKernels { buffer: b.clone() },
                    ExecuteMetalKernels { buffer: b },
                    current_set,
                    outside_nodes,
                ));
            }
        }
        for (setup, exec, current, outside) in create {
            let setup = graph.add_op(setup).finish();
            let exec = graph.add_op(exec).finish();
            for node in current {
                graph.add_schedule_dependency(setup, node);
                graph.add_schedule_dependency(node, exec);
            }
            for outside_node in outside {
                graph.add_schedule_dependency(exec, outside_node);
            }
        }
    }
}

fn build_set_from_node(
    node: NodeIndex,
    buffer: Arc<Mutex<(*const CommandBufferRef, CommandQueue)>>,
    dev: &Device,
    current_set: &mut HashSet<NodeIndex>,
    already_added_nodes: &mut HashSet<NodeIndex>,
    graph: &mut Graph,
) {
    // Check if this has any incoming or outgoing nodes that aren't metal nodes that eventually connect back to the set
    for node in graph
        .graph
        .edges_directed(node, Direction::Incoming)
        .map(|e| e.source())
    {
        if check_if_reaches(graph, node, current_set, Direction::Outgoing, false) {
            return;
        }
    }
    for node in graph
        .graph
        .edges_directed(node, Direction::Outgoing)
        .map(|e| e.target())
    {
        if check_if_reaches(graph, node, current_set, Direction::Incoming, false) {
            return;
        }
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
            );
        }
    }
    // Add outgoing
    for node in graph
        .graph
        .edges_directed(node, Direction::Outgoing)
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
            );
        }
    }
}

fn check_if_reaches(
    graph: &Graph,
    start: NodeIndex,
    set: &HashSet<NodeIndex>,
    dir: Direction,
    passed_non_metal: bool,
) -> bool {
    for e in graph.graph.edges_directed(start, dir) {
        let next_node = match dir {
            Direction::Incoming => e.source(),
            Direction::Outgoing => e.target(),
        };
        if set.contains(&next_node) {
            return passed_non_metal;
        }
        let non_metal = graph
            .graph
            .node_weight(next_node)
            .unwrap()
            .custom("metal")
            .is_none();
        if check_if_reaches(graph, next_node, set, dir, non_metal | passed_non_metal) {
            return true;
        }
    }
    false
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
        // println!("Setting up");
        let mut buffer = self.buffer.lock().unwrap();
        // buffer.2 = unsafe { objc_autoreleasePoolPush() };
        // buffer.1 = self.dev.new_command_queue();
        buffer.0 = buffer.1.new_command_buffer();
        // println!("Setup");
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
        let data = self.buffer.lock().unwrap();
        let buff = unsafe { data.0.as_ref().unwrap() };
        let inputs = inp
            .iter()
            .map(|(t, sh)| {
                (
                    t.borrowed().data.as_any().downcast_ref::<Buffer>().unwrap(),
                    *sh,
                )
            })
            .collect::<Vec<_>>();
        self.wrapper
            .0
            .metal_forward(&inputs, &self.dev, buff)
            .into_iter()
            .map(|b| Tensor { data: Box::new(b) })
            .collect()
    }
}
