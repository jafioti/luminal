use std::{
    cell::UnsafeCell,
    collections::{BTreeMap, BTreeSet, HashMap, HashSet},
    fmt::Debug,
    sync::Arc,
};

use itertools::Itertools;
use metal_rs::{Buffer, CommandBuffer, CommandQueue, Device, MTLResourceOptions};
use petgraph::{
    algo::toposort,
    stable_graph::NodeIndex,
    visit::Bfs,
    Direction::{self},
};

use crate::{
    op::{InputTensor, Operator},
    prelude::*,
};

#[derive(Default, LuminalPrint)]
pub struct StorageBufferCompiler;

impl Compiler for StorageBufferCompiler {
    fn compile(&self, graph: &mut Graph) {
        // First pass - get clear sets for each node
        #[allow(clippy::type_complexity)]
        let mut first_pass: HashMap<
            NodeIndex,
            (BTreeMap<NodeIndex, Vec<NodeIndex>>, BTreeSet<NodeIndex>),
        > = HashMap::new();
        let starting_nodes = graph
            .graph
            .node_indices()
            .filter(|n| {
                graph
                    .graph
                    .neighbors_directed(*n, Direction::Incoming)
                    .count()
                    == 0
            })
            .collect_vec();
        // Loop through starting nodes in graph
        for node in &starting_nodes {
            // Breadth first search from starting nodes
            let mut bfs = Bfs::new(&graph.graph, *node);
            while let Some(node) = bfs.next(&graph.graph) {
                // Run through parents to build new tenative set and clear set
                let (mut tenative_set, mut clear_set) = (BTreeMap::default(), BTreeSet::default());
                for parent in graph.graph.neighbors_directed(node, Direction::Incoming) {
                    if let Some((parent_tenative_set, parent_clear_set)) = first_pass.get(&parent) {
                        let new_tenative_set = parent_tenative_set
                            .iter()
                            .map(|(n, c)| {
                                let mut c = c.clone();
                                c.retain(|n| *n != parent);
                                (*n, c)
                            })
                            .collect::<BTreeMap<_, _>>();
                        tenative_set.extend(new_tenative_set);
                        clear_set.extend(
                            tenative_set
                                .iter()
                                .filter(|(_, v)| v.is_empty())
                                .map(|(n, _)| *n),
                        );
                        tenative_set.retain(|_, v| !v.is_empty());
                        clear_set.extend(parent_clear_set);
                    }
                }
                first_pass.insert(node, (tenative_set, clear_set));
            }
        }

        // Second pass - assign buffers
        let available_buffers = graph
            .graph
            .node_indices()
            .map(|n| {
                let input_shapes = graph
                    .get_sources(n)
                    .into_iter()
                    .map(|(_, _, i)| i)
                    .collect::<Vec<_>>();
                let output_buffers = graph
                    .graph
                    .node_weight(n)
                    .unwrap()
                    .custom("metal")
                    .unwrap()
                    .downcast_ref::<MetalKernelWrapper>()
                    .unwrap()
                    .0
                    .output_buffer_sizes(&input_shapes);
                (n, output_buffers)
            })
            .collect::<HashMap<_, _>>();
        // Loop through starting nodes in graph
        for node in toposort(&graph.graph, None).unwrap() {
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
                if let Some((source_node, applicable_buffer)) = first_pass[&node]
                    .1
                    .iter()
                    .flat_map(|i| available_buffers[i].iter().cloned().map(|b| (*i, b)))
                    .find(|(_, size)| *size == required_buffer)
                {}
            }
            // Assing intermediate buffers
            for required_buffer in wrapper.0.intermediate_buffer_sizes(&input_shapes) {}
        }
    }
}

#[derive(LuminalEq, LuminalPrint)]
struct ExecuteMetalKernels {
    queue: CommandQueue,
    buffer: Arc<UnsafeCell<CommandBuffer>>,
}

impl Operator for ExecuteMetalKernels {
    fn process(&self, _: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        let buffer = unsafe { &mut *self.buffer.get() };
        buffer.commit();
        buffer.wait_until_completed();
        *buffer = self.queue.new_command_buffer().to_owned();
        vec![]
    }
}

#[derive(LuminalEq)]
struct MetalKernelOperation {
    wrapper: Box<MetalKernelWrapper>,
    dev: Device,
    buffer: Arc<UnsafeCell<CommandBuffer>>,
    dyn_map: *const HashMap<char, usize>,
}

impl std::fmt::Debug for MetalKernelOperation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "MetalKernel({:?})", self.wrapper.0)
    }
}

impl Operator for MetalKernelOperation {
    fn process(&self, inp: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        // For now let's allocate the required buffers here
        let inp_shapes = inp.iter().map(|(_, s)| *s).collect::<Vec<_>>();
        let intermediate_buffers = self
            .wrapper
            .0
            .intermediate_buffer_sizes(&inp_shapes)
            .into_iter()
            .map(|n| {
                self.dev.new_buffer(
                    n.exec(unsafe { self.dyn_map.as_ref().unwrap() }).unwrap() as u64,
                    MTLResourceOptions::StorageModeShared,
                )
            })
            .collect::<Vec<_>>();
        let intermediate_buffers_ref = intermediate_buffers.iter().collect::<Vec<_>>();
        let output_buffers = self
            .wrapper
            .0
            .output_buffer_sizes(&inp_shapes)
            .into_iter()
            .map(|n| {
                self.dev.new_buffer(
                    n.exec(unsafe { self.dyn_map.as_ref().unwrap() }).unwrap() as u64,
                    MTLResourceOptions::StorageModeShared,
                )
            })
            .collect::<Vec<_>>();
        let output_buffers_ref = output_buffers.iter().collect::<Vec<_>>();
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
            unsafe { &*self.buffer.get() },
            &intermediate_buffers_ref,
            &output_buffers_ref,
        );
        output_buffers
            .into_iter()
            .map(|b| Tensor { data: Box::new(b) })
            .collect()
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
