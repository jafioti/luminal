use std::{
    cell::UnsafeCell,
    collections::{HashMap, HashSet},
    fmt::Debug,
    sync::Arc,
};

use itertools::Itertools;
use metal_rs::{Buffer, CommandBuffer, CommandQueue, Device, MTLResourceOptions};
use petgraph::{
    stable_graph::NodeIndex,
    visit::{Bfs, EdgeRef},
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
        // First pass
        #[allow(clippy::type_complexity)]
        let mut first_pass: HashMap<
            NodeIndex,
            (Vec<(NodeIndex, Vec<NodeIndex>)>, Vec<NodeIndex>),
        > = HashMap::new();
        // Loop through starting nodes in graph
        for node in graph
            .graph
            .node_indices()
            .filter(|n| {
                graph
                    .graph
                    .neighbors_directed(*n, Direction::Incoming)
                    .count()
                    == 0
            })
            .collect_vec()
        {
            // Breadth first search from starting nodes
            let mut bfs = Bfs::new(&graph.graph, node);
            while let Some(node) = bfs.next(&graph.graph) {
                todo!();
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
