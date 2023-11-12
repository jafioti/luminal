use std::{
    collections::HashSet,
    sync::{Arc, Mutex},
};

use metal_rs::{Buffer, CommandBuffer, CommandQueue, Device};
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

use super::{
    matmul::MetalBatchMatmul2D,
    prim::{MetalKernelWrapper, MetalMul},
};

#[derive(Default, Debug)]
pub struct CommonBufferCompiler;

impl Compiler for CommonBufferCompiler {
    fn compile(&self, graph: &mut Graph) {
        let dev = Device::system_default().unwrap();
        let mut already_added_nodes = HashSet::new();
        let mut queue = dev.new_command_queue();
        let mut num_buffers_on_queue = 0;
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
                if num_buffers_on_queue >= 63 {
                    num_buffers_on_queue = 0;
                    queue = dev.new_command_queue();
                } else {
                    num_buffers_on_queue += 1;
                }
                let b = Arc::new(Mutex::new(Some(queue.new_command_buffer().to_owned())));
                let exec = graph
                    .add_op(ExecuteMetalKernels {
                        queue: queue.clone(),
                        buffer: b.clone(),
                    })
                    .finish();
                let mut current_set = HashSet::new();
                build_set_from_node(
                    node,
                    b.clone(),
                    &dev,
                    &mut current_set,
                    &mut already_added_nodes,
                    graph,
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
            // if already_added_nodes.len() > 38 {
            //     break;
            // }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn build_set_from_node(
    node: NodeIndex,
    buffer: Arc<Mutex<Option<CommandBuffer>>>,
    dev: &Device,
    current_set: &mut HashSet<NodeIndex>,
    already_added_nodes: &mut HashSet<NodeIndex>,
    graph: &mut Graph,
    exec_node: NodeIndex,
) {
    if current_set.len() > 1 {
        return;
    }
    if graph
        .graph
        .node_weight(node)
        .unwrap()
        .as_any()
        .is::<MetalMul>()
        || graph
            .graph
            .node_weight(node)
            .unwrap()
            .as_any()
            .is::<MetalBatchMatmul2D>()
    {
        return;
    }
    let edge1 = graph.graph.add_edge(node, exec_node, Dependency::Schedule);
    if is_cyclic_directed(&graph.graph) {
        graph.graph.remove_edge(edge1);
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
                exec_node,
            );
        }
    }
}

#[derive(Debug)]
struct SetupMetalKernels {
    queue: CommandQueue,
    buffer: Arc<Mutex<Option<CommandBuffer>>>,
}

impl PartialEq for SetupMetalKernels {
    fn eq(&self, _: &Self) -> bool {
        false
    }
}

impl Operator for SetupMetalKernels {
    fn process(&self, _: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        let mut buffer = self.buffer.lock().unwrap();
        *buffer = Some(self.queue.new_command_buffer().to_owned());
        vec![]
    }
}

#[derive(Debug)]
struct ExecuteMetalKernels {
    queue: CommandQueue,
    buffer: Arc<Mutex<Option<CommandBuffer>>>,
}

impl PartialEq for ExecuteMetalKernels {
    fn eq(&self, _: &Self) -> bool {
        false
    }
}

impl Operator for ExecuteMetalKernels {
    fn process(&self, _: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        let mut buffer = self.buffer.lock().unwrap();
        buffer.as_ref().unwrap().commit();
        buffer.as_ref().unwrap().wait_until_completed();
        *buffer = Some(self.queue.new_command_buffer().to_owned());
        vec![]
    }
}

#[derive(Debug)]
struct MetalKernelOperation {
    wrapper: Box<MetalKernelWrapper>,
    dev: Device,
    buffer: Arc<Mutex<Option<CommandBuffer>>>,
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
                self.buffer.lock().unwrap().as_ref().unwrap(),
            )
            .into_iter()
            .map(|b| Tensor { data: Box::new(b) })
            .collect()
    }
}

#[test]
fn test_common_buffer() {
    crate::test_imports!();
    let mut cx = Graph::new();
    let a = cx.new_tensor::<R1<5>>("").set(random_vec(5)).keep();
    let b = cx.new_tensor::<R1<5>>("").set(random_vec(5)).keep();
    let c = cx.new_tensor::<R1<5>>("").set(random_vec(5)).keep();
    let d = ((a + b) * c).retrieve();

    cx.execute();
    let d_unopt = d.data();
    d.drop();

    cx.compile(MetalFp16Compiler::default());
    cx.execute();

    assert_close(&d.data(), &d_unopt);
}
