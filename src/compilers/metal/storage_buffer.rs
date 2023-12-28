use std::{
    cell::UnsafeCell,
    collections::{BTreeMap, BTreeSet, HashMap},
    fmt::Debug,
    sync::Arc,
};

use itertools::Itertools;
use metal_rs::{Buffer, Device, MTLResourceOptions};
use petgraph::{
    algo::toposort,
    stable_graph::NodeIndex,
    visit::{EdgeRef, IntoEdgesDirected},
    Direction::{self},
};

use crate::{
    op::{InputTensor, Operator},
    prelude::{symbolic::BigExpression, *},
};

#[derive(Default, LuminalPrint)]
pub struct StorageBufferCompiler;

impl Compiler for StorageBufferCompiler {
    fn compile(&self, graph: &mut Graph) {
        // First pass - get clear sets for each node
        #[allow(clippy::type_complexity)]
        let mut first_pass: HashMap<
            NodeIndex,
            (
                BTreeMap<NodeIndex, BTreeSet<NodeIndex>>,
                BTreeSet<NodeIndex>,
            ),
        > = HashMap::new();
        // Loop through starting nodes in graph
        for node in toposort(&graph.graph, None).unwrap() {
            // Run through parents to build new tenative set and clear set
            let (mut tenative_sets, mut clear_set) = (BTreeMap::default(), BTreeSet::default());
            for parent in graph
                .graph
                .edges_directed(node, Direction::Incoming)
                .filter(|e| !e.weight().is_schedule())
                .map(|e| e.source())
            {
                let parent_children = graph
                    .graph
                    .edges_directed(parent, Direction::Outgoing)
                    .filter(|e| !e.weight().is_schedule())
                    .map(|e| e.target())
                    .collect::<BTreeSet<_>>();
                tenative_sets.insert(parent, parent_children);
                if let Some((parent_tenative_set, parent_clear_set)) = first_pass.get(&parent) {
                    for (node_index, new_tenative_set) in
                        parent_tenative_set.iter().map(|(n, c)| {
                            let mut c = c.clone();
                            c.retain(|n| *n != parent);
                            (*n, c)
                        })
                    {
                        if let Some(set) = tenative_sets.get(&node_index) {
                            *tenative_sets.get_mut(&node_index).unwrap() =
                                btreeset_intersection(new_tenative_set, set);
                        } else {
                            tenative_sets.insert(node_index, new_tenative_set);
                        }
                    }
                    clear_set.extend(
                        tenative_sets
                            .iter()
                            .filter(|(_, v)| v.is_empty())
                            .map(|(n, _)| *n),
                    );
                    tenative_sets.retain(|_, v| !v.is_empty());
                    clear_set.extend(parent_clear_set);
                }
            }
            first_pass.insert(node, (tenative_sets, clear_set));
        }

        // Second pass - assign buffers
        let available_buffers = graph
            .graph
            .node_indices()
            .filter_map(|n| {
                if let Some(Ok(wrapper)) = graph
                    .graph
                    .node_weight(n)
                    .unwrap()
                    .custom("metal")
                    .map(|n| n.downcast::<MetalKernelWrapper>())
                {
                    Some((n, wrapper))
                } else {
                    None
                }
            })
            .map(|(n, wrapper)| {
                let input_shapes = graph
                    .get_sources(n)
                    .into_iter()
                    .map(|(_, _, i)| i)
                    .collect::<Vec<_>>();
                let output_buffers = wrapper.0.output_buffer_sizes(&input_shapes);
                let intermediate_buffers = wrapper.0.intermediate_buffer_sizes(&input_shapes);
                (n, (output_buffers, intermediate_buffers))
            })
            .collect::<HashMap<_, _>>();
        // Loop through nodes in graph
        let mut buffers = vec![];
        let mut buffer_map = HashMap::new();
        for node in toposort(&graph.graph, None).unwrap() {
            buffer_map.insert(node, (vec![], vec![]));
            let Some(Some(wrapper)) = graph
                .graph
                .node_weight(node)
                .unwrap()
                .custom("metal")
                .map(|e| e.downcast_ref::<MetalKernelWrapper>().cloned())
            else {
                continue;
            };
            let input_shapes = graph
                .get_sources(node)
                .into_iter()
                .map(|(_, _, i)| i)
                .collect::<Vec<_>>();
            // Assign output buffers
            for required_buffer in wrapper.0.output_buffer_sizes(&input_shapes) {
                // Find an applicable buffer
                if let Some((buffer_index, source_node, _)) = first_pass[&node]
                    .1
                    .iter()
                    .filter(|i| available_buffers.contains_key(i))
                    .flat_map(|i| {
                        available_buffers[i]
                            .0
                            .iter()
                            .cloned()
                            .enumerate()
                            .map(|(o, b)| (o, *i, b))
                    })
                    .find(|(_, _, size)| *size == required_buffer)
                {
                    let buffer = buffer_map.get(&source_node).unwrap().0[buffer_index];
                    buffer_map.get_mut(&node).unwrap().0.push(buffer);
                } else {
                    // Allocate new buffer
                    buffer_map.get_mut(&node).unwrap().0.push(buffers.len());
                    buffers.push(required_buffer);
                }
            }
            // Assign intermediate buffers
            for required_buffer in wrapper.0.intermediate_buffer_sizes(&input_shapes) {
                // Find an applicable buffer
                if let Some((buffer_index, source_node, _)) = first_pass[&node]
                    .1
                    .iter()
                    .filter(|i| available_buffers.contains_key(i))
                    .flat_map(|i| {
                        available_buffers[i]
                            .1
                            .iter()
                            .cloned()
                            .enumerate()
                            .map(|(o, b)| (o, *i, b))
                    })
                    .find(|(_, _, size)| *size == required_buffer)
                {
                    let buffer = buffer_map.get(&source_node).unwrap().1[buffer_index];
                    buffer_map.get_mut(&node).unwrap().1.push(buffer);
                } else {
                    // Allocate new buffer
                    buffer_map.get_mut(&node).unwrap().1.push(buffers.len());
                    buffers.push(required_buffer);
                }
            }
        }

        // We now have the buffers to allocate, and the buffers needed for each op.
        // Let's create the allocator op and wrap all the metal ops
        let dev = Device::system_default().unwrap();
        let shared_buffers = Arc::new(UnsafeCell::new(vec![]));
        let allocator = graph
            .add_op(AllocateMetalBuffers {
                dev: dev.clone(),
                dyn_map: &graph.dyn_map,
                buffer_sizes: buffers,
                buffers: shared_buffers.clone(),
            })
            .finish();
        // Ensure allocator is ran before any nodes that use the buffers
        for node in graph
            .graph
            .node_indices()
            // Starting node must have no incoming edges
            .filter(|e| {
                graph
                    .graph
                    .edges_directed(*e, Direction::Incoming)
                    .filter(|e| !e.weight().is_schedule())
                    .count()
                    == 0
            })
            // Starting node must have at least one outgoing edge
            .filter(|e| {
                graph
                    .graph
                    .edges_directed(*e, Direction::Outgoing)
                    .filter(|e| !e.weight().is_schedule())
                    .count()
                    > 0
            })
            .collect_vec()
        {
            graph.add_schedule_dependency(allocator, node);
        }
        // Wrap nodes in StorageBufferWrapper
        for (node, (output_buffers, intermediate_buffers)) in buffer_map
            .into_iter()
            .filter(|(_, b)| !b.0.is_empty() || !b.1.is_empty())
        {
            let wrapper = graph
                .graph
                .node_weight(node)
                .unwrap()
                .custom("metal")
                .unwrap()
                .downcast::<MetalKernelWrapper>()
                .unwrap();
            *graph.graph.node_weight_mut(node).unwrap() = Box::new(StorageBufferWrapper {
                wrapper,
                dev: dev.clone(),
                buffers: shared_buffers.clone(),
                output_buffers,
                intermediate_buffers,
            });
        }
    }
}

fn btreeset_intersection<T: Ord>(mut a: BTreeSet<T>, b: &BTreeSet<T>) -> BTreeSet<T> {
    a.retain(|i| b.contains(i));
    a
}

#[derive(LuminalEq, LuminalPrint)]
struct AllocateMetalBuffers {
    dev: Device,
    dyn_map: *const HashMap<char, usize>,
    buffer_sizes: Vec<BigExpression>,
    buffers: Arc<UnsafeCell<Vec<Buffer>>>,
}

impl Operator for AllocateMetalBuffers {
    fn process(&mut self, _: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        // Allocate all buffers
        *unsafe { &mut *self.buffers.get() } = self
            .buffer_sizes
            .iter()
            .map(|e| e.exec(unsafe { self.dyn_map.as_ref().unwrap() }).unwrap())
            .map(|i| {
                self.dev
                    .new_buffer(i as u64, MTLResourceOptions::StorageModeShared)
            })
            .collect();
        vec![]
    }
}

#[derive(LuminalEq)]
struct StorageBufferWrapper {
    wrapper: Box<MetalKernelWrapper>,
    dev: Device,
    buffers: Arc<UnsafeCell<Vec<Buffer>>>,
    intermediate_buffers: Vec<usize>,
    output_buffers: Vec<usize>,
}

impl std::fmt::Debug for StorageBufferWrapper {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.wrapper.0.fmt(f)
    }
}

impl Operator for StorageBufferWrapper {
    fn process(&mut self, inp: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        let intermediate_buffers = self
            .intermediate_buffers
            .iter()
            .map(|i| unsafe { &(*self.buffers.get())[*i] })
            .collect::<Vec<_>>();
        let output_buffers = self
            .output_buffers
            .iter()
            .map(|i| unsafe { &(*self.buffers.get())[*i] })
            .collect::<Vec<_>>();
        let queue = self.dev.new_command_queue();
        self.wrapper.0.metal_forward(
            &inp.iter()
                .map(|(t, sh)| {
                    (
                        t.borrowed().data.as_any().downcast_ref::<Buffer>().unwrap(),
                        *sh,
                    )
                })
                .collect::<Vec<_>>(),
            &self.dev,
            queue.new_command_buffer(),
            &intermediate_buffers,
            &output_buffers,
        );
        output_buffers
            .into_iter()
            .map(|b| Tensor {
                data: Box::new(b.clone()), // This is dubious. Is cloning cheap?
            })
            .collect()
    }
}

#[test]
fn test_shared_buffers() {
    crate::test_imports!();
    let mut cx = Graph::new();
    let a = cx.tensor::<R1<5>>().set(random_vec(5)).keep();
    let b = a.exp_2();
    let c = a.log_2() * b;
    let d = b.recip();
    let e = (c + d).retrieve();

    cx.execute();
    let e_unopt = e.data();
    e.drop();

    cx.compile(MetalFp16Compiler::default());
    cx.execute();

    assert_close_precision(&e.data(), &e_unopt, 2);
}
