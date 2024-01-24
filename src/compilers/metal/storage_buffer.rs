use std::{
    cell::UnsafeCell,
    collections::{BTreeMap, BTreeSet, HashMap, HashSet},
    sync::Arc,
};

use itertools::Itertools;
use metal_rs::{Buffer, Device, MTLResourceOptions};
use petgraph::{
    algo::toposort,
    stable_graph::NodeIndex,
    visit::EdgeRef,
    Direction::{self},
};

use crate::{
    op::{InputTensor, Operator},
    prelude::{symbolic::BigExpression, *},
};

use super::get_buffer_from_tensor;

#[derive(Default, LuminalPrint)]
pub struct StorageBufferCompiler;

impl Compiler for StorageBufferCompiler {
    fn compile<To: ToIdsMut>(&self, graph: &mut Graph, _: To) {
        // First pass - get clear sets for each node
        #[allow(clippy::type_complexity)]
        let mut first_pass: HashMap<
            NodeIndex,
            (
                BTreeMap<NodeIndex, BTreeSet<NodeIndex>>,
                BTreeSet<NodeIndex>,
            ),
        > = HashMap::new();
        let toposort = toposort(&graph.graph, None).unwrap();
        // Loop through nodes in graph
        for node in &toposort {
            // Run through parents to build new tenative set and clear set
            let (mut tenative_sets, mut clear_set) = (BTreeMap::default(), BTreeSet::default());
            for parent in graph
                .graph
                .edges_directed(*node, Direction::Incoming)
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
            first_pass.insert(*node, (tenative_sets, clear_set));
        }

        // Second pass - assign buffers
        let available_buffers = graph
            .graph
            .node_indices()
            .filter(|n| !graph.no_delete.contains(n))
            .collect::<Vec<_>>()
            .into_iter()
            .filter_map(|n| {
                if let Some(Ok(wrapper)) = graph
                    .graph
                    .node_weight_mut(n)
                    .unwrap()
                    .custom("metal", Box::new(()))
                    .map(|n| n.downcast::<MetalKernelWrapper>())
                {
                    Some((n, wrapper))
                } else {
                    None
                }
            })
            .collect::<Vec<_>>()
            .into_iter()
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
        let mut used = HashSet::<NodeIndex>::new();
        for node in &toposort {
            if graph.no_delete.contains(node) {
                continue;
            }
            let Some(Ok(wrapper)) = graph
                .graph
                .node_weight_mut(*node)
                .unwrap()
                .custom("metal", Box::new(()))
                .map(|e| e.downcast::<MetalKernelWrapper>())
            else {
                continue;
            };
            buffer_map.insert(*node, (vec![], vec![]));
            let input_shapes = graph
                .get_sources(*node)
                .into_iter()
                .map(|(_, _, i)| i)
                .collect::<Vec<_>>();
            // Assign output buffers
            for required_buffer in wrapper.0.output_buffer_sizes(&input_shapes) {
                // Find an applicable buffer
                if let Some((buffer_index, source_node, _)) = first_pass[&node]
                    .1
                    .iter()
                    .filter(|i| !used.contains(i))
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
                    buffer_map.get_mut(node).unwrap().0.push(buffer);
                    // Remove this buffer from first_pass so it can't be used again
                    used.insert(source_node);
                } else {
                    // Allocate new buffer
                    buffer_map.get_mut(node).unwrap().0.push(buffers.len());
                    buffers.push(required_buffer);
                }
            }
            // Assign intermediate buffers
            for required_buffer in wrapper.0.intermediate_buffer_sizes(&input_shapes) {
                // Find an applicable buffer
                if let Some((buffer_index, source_node, _)) = first_pass[&node]
                    .1
                    .iter()
                    .filter(|i| !used.contains(i))
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
                    buffer_map.get_mut(node).unwrap().1.push(buffer);
                    used.insert(source_node);
                } else {
                    // Allocate new buffer
                    buffer_map.get_mut(node).unwrap().1.push(buffers.len());
                    buffers.push(required_buffer);
                }
            }
        }

        // We now have the buffers to allocate, and the buffers needed for each op.
        // Let's create the allocator op and wrap all the metal ops
        let shared_buffers = Arc::new(UnsafeCell::new(vec![]));
        let allocator = graph
            .add_op(AllocateMetalBuffers {
                dev: Device::system_default().unwrap(),
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
                .node_weight_mut(node)
                .unwrap()
                .custom("metal", Box::new(()))
                .unwrap()
                .downcast::<MetalKernelWrapper>()
                .unwrap();
            *graph.graph.node_weight_mut(node).unwrap() = Box::new(StorageBufferWrapper {
                wrapper,
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

#[derive(LuminalEqFalse, LuminalPrint)]
struct AllocateMetalBuffers {
    dev: Device,
    dyn_map: *const HashMap<char, usize>,
    buffer_sizes: Vec<BigExpression>,
    buffers: Arc<UnsafeCell<Vec<Buffer>>>,
}

impl Operator for AllocateMetalBuffers {
    fn process(&mut self, _: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        let buffers = unsafe { &mut *self.buffers.get() };
        let dyn_map = unsafe { self.dyn_map.as_ref().unwrap() };
        // Allocate all buffers
        if buffers.is_empty() {
            *buffers = self
                .buffer_sizes
                .iter()
                .map(|e| {
                    self.dev.new_buffer(
                        e.exec(dyn_map).unwrap() as u64,
                        MTLResourceOptions::StorageModeShared,
                    )
                })
                .collect();
        } else {
            for (size, buffer) in self.buffer_sizes.iter().zip(buffers) {
                let size = size.exec(dyn_map).unwrap() as u64;
                if buffer.length() < size {
                    buffer.set_purgeable_state(metal_rs::MTLPurgeableState::Empty);
                    *buffer = self
                        .dev
                        .new_buffer(size, MTLResourceOptions::StorageModeShared);
                }
            }
        }
        vec![]
    }
}

#[derive(LuminalEqFalse)]
struct StorageBufferWrapper {
    wrapper: Box<MetalKernelWrapper>,
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
        self.wrapper.0.without_command_buffer(
            &inp.iter()
                .map(|(t, sh)| (get_buffer_from_tensor(t), *sh))
                .collect::<Vec<_>>(),
            &intermediate_buffers,
            &output_buffers,
        );
        self.output_buffers
            .iter()
            .map(|i| Tensor {
                data: Box::new(unsafe { (*self.buffers.get())[*i].clone() }),
            })
            .collect()
    }
}

#[test]
fn test_shared_buffers() {
    crate::test_imports!();
    let mut cx = Graph::new();
    let a = cx.tensor::<R1<5>>().set(random_vec(5)).keep();
    let b = a.exp2();
    let c = a.log2() * b;
    let d = b.recip();
    let mut e = (c + d).retrieve();

    cx.execute();
    let e_unopt = e.data();
    e.drop();

    cx.compile(MetalCompiler::<f16>::default(), &mut e);
    cx.execute();

    assert_close_precision(&e.data(), &e_unopt, 2);
}
