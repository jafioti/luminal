use std::any::{Any, TypeId};

use cudarc::driver::CudaDevice;
use itertools::Itertools;
use petgraph::visit::EdgeRef;

use crate::{
    compilers::cuda::prim::*,
    op::{Function as LFunction, *},
    prelude::*,
};

// Unary Op (A -> A)

// Binary Ops

/// Convert all primitive ops to cuda primitive ops, and insert copy to and from device ops
#[derive(LuminalPrint, Default)]
pub struct CudaPrimitiveCompiler;

impl Compiler for CudaPrimitiveCompiler {
    fn compile<T: ToIdsMut>(&self, graph: &mut Graph, mut remap: T) {
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
            // Create copy node
            let copy_node = graph
                .add_op(CudaCopyToDevice::<f32>::new(dev.clone()))
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

            // Insert copy from device for function inputs
            for (source, edge, edge_weight) in graph
                .graph
                .edges_directed(function_node, petgraph::Direction::Incoming)
                .map(|e| (e.source(), e.id(), *e.weight()))
                .collect::<Vec<_>>()
            {
                let copy_from_node = graph
                    .add_op(CudaCopyFromDevice::<f32>::new(dev.clone()))
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
            // Filter to non-functions
            .filter(|n| {
                !graph
                    .graph
                    .node_weight(**n)
                    .unwrap()
                    .as_any()
                    .is::<LFunction>()
            })
            .map(|n| {
                (
                    *n,
                    graph
                        .graph
                        .edges_directed(*n, petgraph::Direction::Incoming)
                        .filter_map(|e| e.weight().as_data())
                        .map(|i| i.2)
                        .max_by_key(|s| s.n_physical_elements().to_usize().unwrap_or_default())
                        .unwrap(),
                )
            })
            .collect::<Vec<_>>()
        {
            if graph
                .graph
                .node_weight(output_node)
                .unwrap()
                .as_any()
                .is::<CudaCopyToDevice<f32>>()
            {
                // This output is already a copy to, instead of adding a copy from, let's remap back to the source
                let src = graph
                    .graph
                    .neighbors_directed(output_node, petgraph::Direction::Incoming)
                    .next()
                    .unwrap();
                graph.no_delete.remove(&output_node);
                graph.to_retrieve.remove(&output_node);
                graph.no_delete.insert(src);
                graph.to_retrieve.insert(src);
            } else {
                // Create copy node
                let copy_node = graph
                    .add_op(CudaCopyFromDevice::<f32>::new(dev.clone()))
                    .input(output_node, 0, output_shape)
                    .finish();

                move_references(
                    &mut remap,
                    &mut graph.no_delete,
                    &mut graph.to_retrieve,
                    output_node,
                    copy_node,
                );
            }
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
                        .find(|e| !e.weight().is_schedule())
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
                .add_op(CudaCopyFromDevice::<f32>::new(dev.clone()))
                .input(source, 0, shape)
                .finish();
            graph.graph.add_edge(
                copy_node,
                output_node,
                Dependency::Data {
                    shape,
                    input_order: 0,
                    output_order: 0,
                },
            );
            graph.graph.remove_edge(edge);
        }

        fn is<T: Any>(type_id: TypeId) -> bool {
            type_id == TypeId::of::<T>()
        }

        // Swap primitive ops
        for id in graph.graph.node_indices().collect::<Vec<_>>() {
            let shapes = graph
                .graph
                .edges_directed(id, petgraph::Direction::Incoming)
                .filter_map(|i| i.weight().as_data())
                .sorted_by_key(|e| e.0)
                .map(|e| e.2)
                .collect::<Vec<_>>();
            let op = graph.graph.node_weight(id).unwrap().as_any().type_id();
            let op_ref = graph.graph.node_weight_mut(id).unwrap();
            if is::<Log2>(op) {
                *op_ref = Box::new(CudaLog2::<f32>::new(dev.clone()));
            } else if is::<Exp2>(op) {
                *op_ref = Box::new(CudaExp2::<f32>::new(dev.clone()));
            } else if is::<Sin>(op) {
                *op_ref = Box::new(CudaSin::<f32>::new(dev.clone()));
            } else if let Some(c) = op_ref.as_any().downcast_ref::<Constant>() {
                *op_ref = Box::new(CudaConstant::<f32>::new(
                    dev.clone(),
                    c.0.clone(),
                    &graph.dyn_map,
                ));
            } else if is::<Sqrt>(op) {
                *op_ref = Box::new(CudaSqrt::<f32>::new(dev.clone()));
            } else if is::<Recip>(op) {
                *op_ref = Box::new(CudaRecip::<f32>::new(dev.clone()));
            } else if is::<Add>(op) {
                *op_ref = Box::new(CudaAdd::<f32>::new(
                    shapes[0],
                    shapes[1],
                    dev.clone(),
                    &graph.dyn_map,
                ));
            } else if is::<Mul>(op) {
                *op_ref = Box::new(CudaMul::<f32>::new(
                    shapes[0],
                    shapes[1],
                    dev.clone(),
                    &graph.dyn_map,
                ));
            } else if is::<Mod>(op) {
                *op_ref = Box::new(CudaMod::<f32>::new(
                    shapes[0],
                    shapes[1],
                    dev.clone(),
                    &graph.dyn_map,
                ));
            } else if is::<LessThan>(op) {
                *op_ref = Box::new(CudaLessThan::<f32>::new(
                    shapes[0],
                    shapes[1],
                    dev.clone(),
                    &graph.dyn_map,
                ));
            } else if is::<Contiguous>(op) {
                *op_ref = Box::new(CudaContiguous::<f32>::new(
                    shapes[0],
                    dev.clone(),
                    &graph.dyn_map,
                ));
            } else if let Some(SumReduce(dim)) = op_ref.as_any().downcast_ref() {
                *op_ref = Box::new(CudaSumReduce::<f32>::new(
                    *dim,
                    shapes[0],
                    dev.clone(),
                    &graph.dyn_map,
                ));
            } else if let Some(MaxReduce(dim)) = op_ref.as_any().downcast_ref() {
                *op_ref = Box::new(CudaMaxReduce::<f32>::new(
                    *dim,
                    shapes[0],
                    dev.clone(),
                    &graph.dyn_map,
                ));
            }
        }
    }
}
