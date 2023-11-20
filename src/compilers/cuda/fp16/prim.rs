use std::{
    any::{Any, TypeId},
    collections::hash_map::DefaultHasher,
    fmt::Debug,
    hash::{Hash, Hasher},
    sync::Arc,
};

use cudarc::{
    driver::{CudaDevice, CudaFunction, CudaSlice, LaunchAsync, LaunchConfig},
    nvrtc::{compile_ptx_with_opts, CompileOptions},
};
use half::f16;
use itertools::Itertools;
use num_traits::FromPrimitive;
use petgraph::{stable_graph::NodeIndex, visit::EdgeRef};

use crate::{op::*, prelude::*};

// Unary Op (A -> A)

// Binary Ops

/// Convert all primitive ops to cuda primitive ops, and insert copy to and from device ops
#[derive(LuminalPrint, Default)]
pub struct CudaPrimitiveCompiler;

impl Compiler for CudaPrimitiveCompiler {
    fn compile(&self, graph: &mut Graph) {
        let dev = CudaDevice::new(0).unwrap();
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
                    .is::<Function>()
                    && graph.graph.edges(*n).count() != 0
            })
            .collect::<Vec<_>>()
        {
            if graph
                .graph
                .node_weight(function_node)
                .unwrap()
                .as_any()
                .downcast_ref::<Function>()
                .unwrap()
                .2
                == std::any::TypeId::of::<Vec<f32>>()
            {
                // Create copy node
                let copy_node = graph
                    .add_op(CudaCopyToDevice::<f16>::new(dev.clone()))
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
                    .add_op(CudaCopyFromDevice::<f16>::new(dev.clone()))
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
                    .is::<Function>()
            })
            .map(|n| {
                (
                    *n,
                    graph
                        .graph
                        .edges_directed(*n, petgraph::Direction::Incoming)
                        .flat_map(|i| i.weight().as_data().map(|i| i.2))
                        .max_by_key(|s| s.n_physical_elements())
                        .unwrap(),
                )
            })
            .collect::<Vec<_>>()
        {
            // Create copy node
            let copy_node = graph
                .add_op(CudaCopyFromDevice::<f16>::new(dev.clone()))
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
                        .find(|i| !i.weight().is_schedule())
                        .unwrap()
                        .id(),
                )
            })
            .collect::<Vec<_>>()
        {
            // Create copy node
            let (source, shape) = (
                graph.graph.edge_endpoints(edge).unwrap().0,
                graph.graph.edge_weight(edge).unwrap().as_data().unwrap().2,
            );
            let copy_node = graph
                .add_op(CudaCopyFromDevice::<f16>::new(dev.clone()))
                .input(source, 0, shape)
                .finish();
            graph.graph.add_edge(
                copy_node,
                output_node,
                Dependency::Data {
                    input_order: 0,
                    output_order: 0,
                    shape,
                },
            );
            graph.graph.remove_edge(edge);
        }

        // Swap primitive ops
        for id in graph.graph.node_indices().collect::<Vec<_>>() {
            let shapes = graph
                .graph
                .edges_directed(id, petgraph::Direction::Incoming)
                .filter_map(|e| e.weight().as_data())
                .sorted_by_key(|e| e.0)
                .map(|e| e.2)
                .collect::<Vec<_>>();
            let op = graph.graph.node_weight(id).unwrap().as_any().type_id();
            let op_ref = graph.graph.node_weight_mut(id).unwrap();
            if is::<Log2>(op) {
                *op_ref = Box::new(CudaLog2::<f16>::new(dev.clone()));
            } else if is::<Exp2>(op) {
                *op_ref = Box::new(CudaExp2::<f16>::new(dev.clone()));
            } else if is::<Sin>(op) {
                *op_ref = Box::new(CudaSin::<f16>::new(dev.clone()));
            } else if is::<Sqrt>(op) {
                *op_ref = Box::new(CudaSqrt::<f16>::new(dev.clone()));
            } else if let Some(c) = op_ref.as_any().downcast_ref::<Constant>() {
                *op_ref = Box::new(CudaConstant::<f16>::new(dev.clone(), f16::from_f32(c.0)));
            } else if is::<Recip>(op) {
                *op_ref = Box::new(CudaRecip::<f16>::new(dev.clone()));
            } else if is::<Add>(op) {
                *op_ref = Box::new(CudaAdd::<f16>::new(shapes[0], shapes[1], dev.clone()));
            } else if is::<Mul>(op) {
                *op_ref = Box::new(CudaMul::<f16>::new(shapes[0], shapes[1], dev.clone()));
            } else if is::<Mod>(op) {
                *op_ref = Box::new(CudaMod::<f16>::new(shapes[0], shapes[1], dev.clone()));
            } else if is::<LessThan>(op) {
                *op_ref = Box::new(CudaLessThan::<f16>::new(shapes[0], shapes[1], dev.clone()));
            } else if is::<Contiguous>(op) {
                *op_ref = Box::new(CudaContiguous::<f16>::new(shapes[0], dev.clone()));
            } else if let Some(SumReduce(dim)) = op_ref.as_any().downcast_ref() {
                *op_ref = Box::new(CudaSumReduce::<f16>::new(*dim, shapes[0], dev.clone()));
            } else if let Some(MaxReduce(dim)) = op_ref.as_any().downcast_ref() {
                *op_ref = Box::new(CudaMaxReduce::<f16>::new(*dim, shapes[0], dev.clone()));
            }
        }
    }
}

/// In 16 bit, summing above 2048 doesn't work. This precludes the .expand(Dim).sum_reduce() pattern to get a dim size in a tensor, so we need to replace these fake reductions with an elementwise mul
#[derive(LuminalPrint, Default)]
pub struct FakeReductionCompiler;

impl Compiler for FakeReductionCompiler {
    fn compile(&self, graph: &mut Graph) {
        let mut sum_reduce = NodeIndex::default();
        let s = SelectEdge::new(
            SelectOp::new().check(|o, _| {
                if let Some(c) = o.as_any().downcast_ref::<CudaConstant<f16>>() {
                    c.1 == f16::ONE
                } else {
                    false
                }
            }),
            SelectOp::new()
                .ty::<CudaSumReduce<f16>>()
                .check(|o, shapes| {
                    if let Some(o) = o.as_any().downcast_ref::<CudaSumReduce<f16>>() {
                        shapes[0].fake[shapes[0].indexes[o.2]] // Ensure dimension we are reducing is fake
                    } else {
                        false
                    }
                })
                .ptr(&mut sum_reduce),
        );

        for _ in s.search(graph) {
            let op_ref = graph.graph.node_weight_mut(sum_reduce).unwrap();
            let dim = op_ref
                .as_any()
                .downcast_ref::<CudaSumReduce<f16>>()
                .unwrap()
                .2;
            let dev = op_ref
                .as_any()
                .downcast_ref::<CudaSumReduce<f16>>()
                .unwrap()
                .1
                .clone();
            *op_ref = Box::new(FakeSumReduce::new(dev, dim));
        }
    }
}

#[derive(Debug, Clone)]
pub struct FakeSumReduce(CudaFunction, Arc<CudaDevice>, pub usize);
impl PartialEq for FakeSumReduce {
    fn eq(&self, _: &Self) -> bool {
        false
    }
}

impl FakeSumReduce {
    pub fn new(dev: Arc<CudaDevice>, dim: usize) -> Self {
        let mut code = "#include \"cuda_fp16.h\"
extern \"C\" __global__ void kernel(__half *out, const __half *inp, int numel, __half mul_factor) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numel) {
        out[i] = inp[i] * mul_factor;
    }
}"
        .to_string();
        let name = format!("kernel_{}", hash(&code));
        code = code.replace("kernel", &name);
        if !dev.has_func(&name, &name) {
            dev.load_ptx(
                compile_ptx_with_opts(
                    code,
                    CompileOptions {
                        arch: Some("sm_75"),
                        include_paths: vec!["/usr/local/cuda/include".to_string()],
                        ..Default::default()
                    },
                )
                .unwrap(),
                &name,
                &[&name],
            )
            .unwrap();
        }
        Self(dev.get_func(&name, &name).unwrap(), dev, dim)
    }
}

impl Operator for FakeSumReduce {
    fn process(&self, tensors: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        let dim_size = f16::from_usize(tensors[0].1.shape()[self.2].to_usize().unwrap()).unwrap();
        let inp = tensors[0]
            .0
            .borrowed()
            .data
            .as_any()
            .downcast_ref::<CudaSlice<f16>>()
            .unwrap();
        let inp_size = tensors[0].1.n_physical_elements();
        let mut out = self.1.alloc_zeros::<f16>(inp_size).unwrap();
        unsafe {
            self.0
                .clone()
                .launch(
                    LaunchConfig::for_num_elems(inp_size as u32),
                    (&mut out, inp, inp_size, dim_size),
                )
                .unwrap();
        }

        vec![Tensor {
            data: Box::new(out),
        }]
    }
}

// Sometimes CopyTo -> CopyFrom and CopyFrom -> CopyTo patterns remain, so let's clean them up
#[derive(Debug, Default)]
pub struct CopyCompiler;

impl Compiler for CopyCompiler {
    fn compile(&self, graph: &mut Graph) {
        for (first, second) in graph
            .graph
            .edge_indices()
            .filter_map(|e| graph.graph.edge_endpoints(e))
            .filter(|(a, b)| {
                (graph
                    .graph
                    .node_weight(*a)
                    .unwrap()
                    .as_any()
                    .is::<CudaCopyToDevice<f16>>()
                    && graph
                        .graph
                        .node_weight(*b)
                        .unwrap()
                        .as_any()
                        .is::<CudaCopyFromDevice<f16>>())
                    || (graph
                        .graph
                        .node_weight(*a)
                        .unwrap()
                        .as_any()
                        .is::<CudaCopyFromDevice<f16>>()
                        && graph
                            .graph
                            .node_weight(*b)
                            .unwrap()
                            .as_any()
                            .is::<CudaCopyToDevice<f16>>())
            })
            .unique_by(|n| n.0)
            .unique_by(|n| n.1)
            .collect::<Vec<_>>()
        {
            if graph
                .graph
                .edges_directed(first, petgraph::Direction::Outgoing)
                .filter(|e| graph.graph.contains_node(e.target()))
                .filter(|e| {
                    !graph
                        .graph
                        .node_weight(e.target())
                        .unwrap()
                        .as_any()
                        .is::<CudaCopyFromDevice<f16>>()
                        && !graph
                            .graph
                            .node_weight(e.target())
                            .unwrap()
                            .as_any()
                            .is::<CudaCopyToDevice<f16>>()
                })
                .count()
                > 0
                || graph.no_delete.contains(&first)
            {
                continue;
            }
            let source = graph.get_sources(first)[0];
            move_outgoing_edge(second, source.0, &mut graph.graph);
            move_references(
                &mut graph.id_remap,
                &mut graph.no_delete,
                &mut graph.to_retrieve,
                second,
                source.0,
            );
            graph.graph.remove_node(second);
            for dest in graph
                .get_dests(first)
                .iter()
                .map(|(i, _)| *i)
                .collect::<Vec<_>>()
            {
                move_outgoing_edge(dest, source.0, &mut graph.graph);
                move_references(
                    &mut graph.id_remap,
                    &mut graph.no_delete,
                    &mut graph.to_retrieve,
                    dest,
                    source.0,
                );
                graph.graph.remove_node(dest);
            }
            graph.graph.remove_node(first);
        }
    }
}

fn is<T: Any>(type_id: TypeId) -> bool {
    type_id == TypeId::of::<T>()
}

fn hash<T: Hash>(obj: T) -> u64 {
    let mut hasher = DefaultHasher::new();
    obj.hash(&mut hasher);
    hasher.finish()
}
