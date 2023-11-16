use std::{
    collections::{HashSet, VecDeque},
    sync::{Arc, Mutex},
};

use itertools::Itertools;
use metal_rs::{Buffer, CommandBuffer, CommandQueue, Device};
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

#[derive(Default, Debug)]
pub struct CommonBufferCompiler;

impl Compiler for CommonBufferCompiler {
    fn compile(&self, graph: &mut Graph) {
        let dev = Device::system_default().unwrap();
        let mut already_added_nodes = HashSet::new();
        let mut queue = dev.new_command_queue();
        let mut num_buffers_on_queue = 0;
        let mut is_metal: HashSet<NodeIndex> = graph
            .graph
            .node_indices()
            .filter(|i| {
                graph
                    .graph
                    .node_weight(*i)
                    .unwrap()
                    .custom("metal")
                    .is_some()
            })
            .collect();
        for node in graph.graph.node_indices().collect::<Vec<_>>() {
            if !already_added_nodes.contains(&node) && is_metal.contains(&node) {
                // Start a set from this node
                if num_buffers_on_queue >= 63 {
                    num_buffers_on_queue = 0;
                    queue = dev.new_command_queue();
                } else {
                    num_buffers_on_queue += 1;
                }
                let b = Arc::new(Mutex::new(queue.new_command_buffer().to_owned()));
                let exec = graph
                    .add_op(ExecuteMetalKernels {
                        queue: queue.clone(),
                        buffer: b.clone(),
                    })
                    .finish();
                let mut current_set = HashSet::new();
                build_set_from_node(
                    node,
                    b,
                    &dev,
                    &mut current_set,
                    &mut already_added_nodes,
                    graph,
                    exec,
                    &is_metal,
                    &get_lower_nodes(&graph.graph, node),
                );
                // Add deps from execute op to consumers
                for node in &current_set {
                    is_metal.remove(node);
                    for outside_node in graph
                        .graph
                        .edges_directed(*node, Direction::Outgoing)
                        .filter(|e| !e.weight().is_schedule())
                        .map(|e| e.target())
                        .filter(|n| !current_set.contains(n))
                        .collect::<Vec<_>>()
                    {
                        graph.add_schedule_dependency(exec, outside_node);
                    }
                }
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn build_set_from_node(
    node: NodeIndex,
    buffer: Arc<Mutex<CommandBuffer>>,
    dev: &Device,
    current_set: &mut HashSet<NodeIndex>,
    already_added_nodes: &mut HashSet<NodeIndex>,
    graph: &mut Graph,
    exec_node: NodeIndex,
    is_metal: &HashSet<NodeIndex>,
    lower_nodes: &HashSet<NodeIndex>,
) {
    graph.add_schedule_dependency(node, exec_node);
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
    // Add outgoing
    for node in graph
        .graph
        .edges_directed(node, Direction::Outgoing)
        .filter(|e| !e.weight().is_schedule())
        .map(|e| e.target())
        .unique()
        .collect::<Vec<_>>()
    {
        if !already_added_nodes.contains(&node)
            && is_metal.contains(&node)
            && !reverse_dfs(&graph.graph, is_metal, current_set, node, lower_nodes)
        {
            build_set_from_node(
                node,
                buffer.clone(),
                dev,
                current_set,
                already_added_nodes,
                graph,
                exec_node,
                is_metal,
                lower_nodes,
            );
        }
    }
}

fn get_lower_nodes(graph: &MainGraph, start_from: NodeIndex) -> HashSet<NodeIndex> {
    let mut stack = VecDeque::new();
    let mut seen = HashSet::new();

    stack.push_back(start_from);
    while let Some(node) = stack.pop_back() {
        if seen.contains(&node) {
            continue;
        }
        seen.insert(node);
        for neighbor in graph
            .edges_directed(node, Direction::Outgoing)
            .filter(|e| !e.weight().is_schedule())
            .map(|e| e.target())
        {
            stack.push_back(neighbor);
        }
    }
    seen
}

fn reverse_dfs(
    graph: &MainGraph,
    is_metal: &HashSet<NodeIndex>,
    set: &HashSet<NodeIndex>,
    start_from: NodeIndex,
    lower_nodes: &HashSet<NodeIndex>,
) -> bool {
    let mut stack = VecDeque::new();

    stack.push_back((start_from, false)); // second element in tuple indicates if we've passed a non-metal node

    while let Some((node, mut passed_non_metal)) = stack.pop_back() {
        if !is_metal.contains(&node) {
            // Mark that we've passed a non-metal node.
            passed_non_metal = true;
        }

        if set.contains(&node) {
            // Now if we passed a non-selected node and the current node is in `stop`, return true.
            if passed_non_metal {
                return true;
            }
        } else if lower_nodes.contains(&node) {
            // Iterate over all neighbors by incoming edges (reversed).
            for neighbor in graph
                .edges_directed(node, Direction::Incoming)
                .filter(|e| !e.weight().is_schedule())
                .map(|e| e.source())
            {
                stack.push_back((neighbor, passed_non_metal));
            }
        }
    }

    false
}

#[derive(Debug)]
struct ExecuteMetalKernels {
    queue: CommandQueue,
    buffer: Arc<Mutex<CommandBuffer>>,
}

impl PartialEq for ExecuteMetalKernels {
    fn eq(&self, _: &Self) -> bool {
        false
    }
}

impl Operator for ExecuteMetalKernels {
    fn process(&self, _: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        let mut buffer = self.buffer.lock().unwrap();
        buffer.commit();
        buffer.wait_until_completed();
        *buffer = self.queue.new_command_buffer().to_owned();
        vec![]
    }
}

struct MetalKernelOperation {
    wrapper: Box<MetalKernelWrapper>,
    dev: Device,
    buffer: Arc<Mutex<CommandBuffer>>,
}

impl std::fmt::Debug for MetalKernelOperation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "MetalKernel-{:?}", self.wrapper.0)
    }
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
                self.buffer.lock().unwrap().as_ref(),
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
