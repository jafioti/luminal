use std::{
    cell::UnsafeCell,
    collections::{HashMap, HashSet},
    fmt::Debug,
    sync::Arc,
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

use super::get_buffer_from_tensor;

#[derive(Default, LuminalPrint)]
pub struct CommandBufferCompiler;

impl Compiler for CommandBufferCompiler {
    fn compile(&self, graph: &mut Graph) {
        let is_metal: HashSet<NodeIndex> = graph
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
        // Do forward pass
        let mut forward_map: HashMap<NodeIndex, usize> = HashMap::new();
        for node in graph
            .graph
            .node_indices()
            .filter(|n| graph.graph.edges_directed(*n, Direction::Incoming).count() == 0)
            .sorted()
        {
            let mut stack = vec![node];
            while let Some(node) = stack.pop() {
                // Get rank as max of predecessors
                let rank = graph
                    .graph
                    .neighbors_directed(node, Direction::Incoming)
                    .filter_map(|i| forward_map.get(&i).map(|r| (i, *r)))
                    .map(|(node_index, rank)| {
                        if is_metal.contains(&node) != is_metal.contains(&node_index) {
                            rank + 1
                        } else {
                            rank
                        }
                    })
                    .max()
                    .unwrap_or_default();
                // Max it with the current entry in the map or insert
                if let Some(entry) = forward_map.get_mut(&node) {
                    if rank > *entry {
                        *entry = rank;
                        stack.extend(graph.graph.neighbors_directed(node, Direction::Outgoing));
                    }
                } else {
                    forward_map.insert(node, rank);
                    stack.extend(graph.graph.neighbors_directed(node, Direction::Outgoing));
                }
            }
        }

        // Do backward pass
        let mut backward_map: HashMap<NodeIndex, usize> = HashMap::new();
        for node in graph
            .graph
            .node_indices()
            .filter(|n| graph.graph.edges_directed(*n, Direction::Outgoing).count() == 0)
            .sorted()
        {
            let mut stack = vec![node];
            while let Some(node) = stack.pop() {
                // Get rank as max of successors
                let rank = graph
                    .graph
                    .neighbors_directed(node, Direction::Outgoing)
                    .filter_map(|i| backward_map.get(&i).map(|r| (i, *r)))
                    .map(|(node_index, rank)| {
                        if is_metal.contains(&node) != is_metal.contains(&node_index) {
                            rank + 1
                        } else {
                            rank
                        }
                    })
                    .max()
                    .unwrap_or_default();
                // Max it with the current entry in the map or insert
                if let Some(entry) = backward_map.get_mut(&node) {
                    if rank > *entry {
                        *entry = rank;
                        stack.extend(graph.graph.neighbors_directed(node, Direction::Incoming));
                    }
                } else {
                    backward_map.insert(node, rank);
                    stack.extend(graph.graph.neighbors_directed(node, Direction::Incoming));
                }
            }
        }
        // Get sets (Rank -> # of nodes with that rank)
        let forward_sets = forward_map
            .iter()
            .sorted_by_key(|(_, v)| **v)
            .group_by(|(_, v)| **v)
            .into_iter()
            .map(|(k, g)| (k, g.count()))
            .collect::<HashMap<_, _>>();
        let backward_sets = backward_map
            .iter()
            .sorted_by_key(|(_, v)| **v)
            .group_by(|(_, v)| **v)
            .into_iter()
            .map(|(k, g)| (k, g.count()))
            .collect::<HashMap<_, _>>();

        // Assign nodes to sets
        let mut node_sets: HashMap<(bool, usize), HashSet<NodeIndex>> = HashMap::new();
        for node in graph.graph.node_indices().filter(|i| is_metal.contains(i)) {
            let forward_bigger =
                forward_sets[&forward_map[&node]] >= backward_sets[&backward_map[&node]];
            node_sets
                .entry((
                    forward_bigger,
                    if forward_bigger {
                        forward_map[&node]
                    } else {
                        backward_map[&node]
                    },
                ))
                .and_modify(|set| {
                    set.insert(node);
                })
                .or_insert(HashSet::from([node]));
        }
        // Add sets to graph
        let dev = Device::system_default().unwrap();
        let mut queue = dev.new_command_queue();
        let mut num_buffers_on_queue = 0;
        for set in node_sets.values() {
            if num_buffers_on_queue >= 63 {
                num_buffers_on_queue = 0;
                queue = dev.new_command_queue();
            } else {
                num_buffers_on_queue += 1;
            }
            #[allow(clippy::arc_with_non_send_sync)]
            let buffer = Arc::new(UnsafeCell::new(queue.new_command_buffer().to_owned()));
            let exec = graph
                .add_op(ExecuteMetalKernels {
                    queue: queue.clone(),
                    buffer: buffer.clone(),
                })
                .finish();
            for node in set {
                // Create schedule dependency
                graph.add_schedule_dependency(*node, exec);
                // Wrap node in MetalKernelOperation
                let wrapper = graph
                    .graph
                    .node_weight(*node)
                    .unwrap()
                    .custom("metal")
                    .unwrap()
                    .downcast::<MetalKernelWrapper>()
                    .unwrap();
                *graph.graph.node_weight_mut(*node).unwrap() = Box::new(CommandBufferWrapper {
                    wrapper,
                    buffer: buffer.clone(),
                    dyn_map: &graph.dyn_map,
                });
                // Create schedule dependencies from exec to consumers
                for outside_node in graph
                    .graph
                    .edges_directed(*node, Direction::Outgoing)
                    .filter(|e| !e.weight().is_schedule())
                    .map(|e| e.target())
                    .filter(|n| !set.contains(n))
                    .collect::<Vec<_>>()
                {
                    graph.add_schedule_dependency(exec, outside_node);
                }
            }
        }
    }
}

#[derive(LuminalEq, LuminalPrint)]
struct ExecuteMetalKernels {
    queue: CommandQueue,
    buffer: Arc<UnsafeCell<CommandBuffer>>,
}

impl Operator for ExecuteMetalKernels {
    fn process(&mut self, _: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        let buffer = unsafe { &mut *self.buffer.get() };
        buffer.commit();
        buffer.wait_until_completed();
        *buffer = self.queue.new_command_buffer().to_owned();
        vec![]
    }
}

#[derive(LuminalEq, Clone)]
struct CommandBufferWrapper {
    wrapper: Box<MetalKernelWrapper>,
    buffer: Arc<UnsafeCell<CommandBuffer>>,
    dyn_map: *const HashMap<char, usize>,
}

impl std::fmt::Debug for CommandBufferWrapper {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "MetalKernel({:?})", self.wrapper.0)
    }
}

impl MetalKernel for CommandBufferWrapper {
    fn intermediate_buffer_sizes(
        &self,
        input_shapes: &[ShapeTracker],
    ) -> Vec<symbolic::BigExpression> {
        self.wrapper.0.intermediate_buffer_sizes(input_shapes)
    }
    fn output_buffer_sizes(&self, input_shapes: &[ShapeTracker]) -> Vec<symbolic::BigExpression> {
        self.wrapper.0.output_buffer_sizes(input_shapes)
    }
    fn metal_forward(
        &self,
        inputs: &[(&Buffer, ShapeTracker)],
        _: &metal_rs::CommandBufferRef,
        intermediate_buffers: &[&Buffer],
        output_buffers: &[&Buffer],
    ) {
        self.wrapper.0.metal_forward(
            inputs,
            unsafe { &*self.buffer.get() },
            intermediate_buffers,
            output_buffers,
        );
    }
    fn without_command_buffer(
        &self,
        inputs: &[(&Buffer, ShapeTracker)],
        intermediate_buffers: &[&Buffer],
        output_buffers: &[&Buffer],
    ) {
        self.metal_forward(
            inputs,
            unsafe { &*self.buffer.get() },
            intermediate_buffers,
            output_buffers,
        )
    }
}

impl Operator for CommandBufferWrapper {
    fn process(&mut self, inp: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        self.without_storage_buffers(
            &inp.iter()
                .map(|(t, sh)| (get_buffer_from_tensor(t), *sh))
                .collect::<Vec<_>>(),
            unsafe { &*self.buffer.get() },
            unsafe { self.dyn_map.as_ref().unwrap() },
        )
        .into_iter()
        .map(|b| Tensor { data: Box::new(b) })
        .collect()
    }

    fn custom(&self, key: &str) -> Option<Box<dyn std::any::Any>> {
        if key == "metal" {
            return Some(Box::new(MetalKernelWrapper(Arc::new(Box::new(
                self.clone(),
            )))));
        }
        None
    }
}

#[test]
fn test_common_buffer() {
    crate::test_imports!();
    let mut cx = Graph::new();
    let a = cx.tensor::<R1<5>>().set(random_vec(5)).keep();
    let b = cx.tensor::<R1<5>>().set(random_vec(5)).keep();
    let c = cx.tensor::<R1<5>>().set(random_vec(5)).keep();
    let d = ((a + b) * c).retrieve();

    cx.execute();
    let d_unopt = d.data();
    d.drop();

    cx.compile(MetalFp16Compiler::default());
    cx.execute();

    assert_close(&d.data(), &d_unopt);
}
