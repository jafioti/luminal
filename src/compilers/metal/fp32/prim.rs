use std::collections::HashMap;

use crate::{
    compilers::metal::*,
    op::{
        Add, Constant, Contiguous, Exp2, Function as LFunction, LessThan, Log2, MaxReduce, Mod,
        Mul, Print, Recip, Sin, Sqrt, SumReduce,
    },
    prelude::*,
};
use itertools::Itertools;
use metal_rs::*;
use petgraph::visit::EdgeRef;

#[derive(Default)]
pub struct PrimitiveCompiler;

impl Compiler for PrimitiveCompiler {
    fn compile(&self, graph: &mut Graph) {
        let dev = Device::system_default().unwrap();
        let queue = dev.new_command_queue();
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
                    .is::<LFunction>()
            })
            .collect::<Vec<_>>()
        {
            if graph
                .graph
                .node_weight(function_node)
                .unwrap()
                .as_any()
                .downcast_ref::<LFunction>()
                .unwrap()
                .2
                == std::any::TypeId::of::<Vec<f32>>()
            {
                // Create copy node
                let copy_node = graph
                    .add_op(MetalCopyToDevice::<f32>::new(dev.clone()))
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

                // If there are inputs to this function remap the function to the copy node
                if graph
                    .graph
                    .edges_directed(function_node, petgraph::Direction::Incoming)
                    .count()
                    != 0
                {
                    move_references(
                        &mut graph.id_remap,
                        &mut graph.no_delete,
                        &mut graph.to_retrieve,
                        function_node,
                        copy_node,
                    );
                }
            }

            // Insert copy from device for function inputs
            for (source, edge, edge_weight) in graph
                .graph
                .edges_directed(function_node, petgraph::Direction::Incoming)
                .map(|e| (e.source(), e.id(), *e.weight()))
                .collect::<Vec<_>>()
            {
                let copy_from_node = graph
                    .add_op(MetalCopyFromDevice::<f32>::new(dev.clone()))
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
                        .filter_map(|i| i.weight().as_data().map(|i| i.2))
                        .max_by_key(|s| s.n_physical_elements())
                        .unwrap(),
                )
            })
            .collect::<Vec<_>>()
        {
            // Create copy node
            let copy_node = graph
                .add_op(MetalCopyFromDevice::<f32>::new(dev.clone()))
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
                .add_op(MetalCopyFromDevice::<f32>::new(dev.clone()))
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
        let mut kernels = HashMap::new();
        for id in graph.graph.node_indices().collect::<Vec<_>>() {
            let src_shapes = graph
                .graph
                .edges_directed(id, petgraph::Direction::Incoming)
                .filter_map(|e| e.weight().as_data())
                .sorted_by_key(|e| e.0)
                .map(|e| e.2)
                .collect::<Vec<_>>();
            let op = graph.graph.node_weight(id).unwrap().as_any().type_id();
            let op_ref = graph.graph.node_weight_mut(id).unwrap();
            if is::<Log2>(op) {
                *op_ref = Box::new(MetalLog2::<f32>::new(
                    dev.clone(),
                    queue.clone(),
                    &mut kernels,
                ));
            } else if let Some(c) = op_ref.as_any().downcast_ref::<Constant>() {
                *op_ref = Box::new(MetalConstant(c.0, dev.clone()));
            } else if is::<Exp2>(op) {
                *op_ref = Box::new(MetalExp2::<f32>::new(
                    dev.clone(),
                    queue.clone(),
                    &mut kernels,
                ));
            } else if is::<Sin>(op) {
                *op_ref = Box::new(MetalSin::<f32>::new(
                    dev.clone(),
                    queue.clone(),
                    &mut kernels,
                ));
            } else if is::<Sqrt>(op) {
                *op_ref = Box::new(MetalSqrt::<f32>::new(
                    dev.clone(),
                    queue.clone(),
                    &mut kernels,
                ));
            } else if is::<Recip>(op) {
                *op_ref = Box::new(MetalRecip::<f32>::new(
                    dev.clone(),
                    queue.clone(),
                    &mut kernels,
                ));
            } else if is::<Add>(op) {
                *op_ref = Box::new(MetalAdd::<f32>::new(
                    src_shapes[0],
                    src_shapes[1],
                    dev.clone(),
                    queue.clone(),
                    &mut kernels,
                ));
            } else if is::<Mul>(op) {
                *op_ref = Box::new(MetalMul::<f32>::new(
                    src_shapes[0],
                    src_shapes[1],
                    dev.clone(),
                    queue.clone(),
                    &mut kernels,
                ));
            } else if is::<LessThan>(op) {
                *op_ref = Box::new(MetalLessThan::<f32>::new(
                    src_shapes[0],
                    src_shapes[1],
                    dev.clone(),
                    queue.clone(),
                    &mut kernels,
                ));
            } else if is::<Mod>(op) {
                *op_ref = Box::new(MetalMod::<f32>::new(
                    src_shapes[0],
                    src_shapes[1],
                    dev.clone(),
                    queue.clone(),
                    &mut kernels,
                ));
            } else if let Some(SumReduce(dim)) = op_ref.as_any().downcast_ref() {
                *op_ref = Box::new(MetalSumReduce::<f32>::new(
                    src_shapes[0],
                    *dim,
                    dev.clone(),
                    queue.clone(),
                    &mut kernels,
                ));
            } else if let Some(MaxReduce(dim)) = op_ref.as_any().downcast_ref() {
                *op_ref = Box::new(MetalMaxReduce::<f32>::new(
                    src_shapes[0],
                    *dim,
                    dev.clone(),
                    queue.clone(),
                    &mut kernels,
                ));
            } else if is::<Contiguous>(op) {
                *op_ref = Box::new(MetalContiguous::<f32>::new(
                    src_shapes[0],
                    dev.clone(),
                    &mut kernels,
                ));
            }
        }
    }
}
