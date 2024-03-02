use rustc_hash::{FxHashMap, FxHashSet};
use std::{any::Any, marker::PhantomData, ops::Deref, sync::Arc};

use itertools::Itertools;
use metal_rs::{
    objc::rc::autoreleasepool, Buffer, CommandBufferRef, CommandQueue, ComputePassDescriptor,
    ComputePipelineState, Device, MTLResourceOptions,
};

use luminal::{
    op::{InputTensor, Operator},
    prelude::{
        petgraph::{visit::EdgeRef, Direction},
        *,
    },
};

use crate::{get_buffer_from_tensor, MetalBuffer, MetalFloat, MetalKernel, MetalKernelWrapper};

use self::symbolic::BigExpression;

use super::{
    compile_function, get_idx_valid_exps, input_dyn_dims, prim::MetalConstant,
    render_dyn_dim_inputs, DispatchNElements, SetInt,
};

#[derive(Default, Debug)]
pub struct ElementwiseFusionCompiler<T>(PhantomData<T>);

impl<T: MetalFloat> Compiler for ElementwiseFusionCompiler<T> {
    fn compile<To: ToIdsMut>(&self, graph: &mut Graph, mut remap: To) {
        let device = Device::system_default().unwrap();
        let queue = device.new_command_queue();
        // Find two elementwise ops that have a contiguous edge
        let (mut a, mut b) = (NodeIndex::default(), NodeIndex::default());
        let mut selector = SelectOp::new()
            .check(|o, _| o.custom("elementwise", Box::<()>::default()).is_some())
            .ptr(&mut a)
            .edge(
                SelectOp::new()
                    .check(|o, _| o.custom("elementwise", Box::<()>::default()).is_some())
                    .ptr(&mut b),
            )
            .search(graph);
        let mut fused_ops = FxHashSet::default();

        while selector.next_match() {
            // More than one connecting edge
            if graph.no_delete.contains(&a)
                || (graph
                    .graph
                    .edges_directed(a, Direction::Outgoing)
                    .filter(|e| !e.weight().is_schedule())
                    .count()
                    > 1
                    && !graph
                        .graph
                        .node_weight(a)
                        .unwrap()
                        .as_any()
                        .is::<MetalConstant<T>>())
            {
                continue;
            }
            // Connecting shape isn't contiguous
            let (edge_id, (to_input, _, connecting_shape)) = graph
                .graph
                .edges_connecting(a, b)
                .find_map(|e| e.weight().as_data().map(|i| (e.id(), i)))
                .unwrap();
            if !connecting_shape.is_contiguous()
                || connecting_shape.is_sliced()
                || connecting_shape.is_padded()
            {
                continue;
            }

            // Fuse into a FusedElementwiseOp
            let new_op;
            let mut a_equation = graph
                .node_custom::<String, _>(a, "elementwise", ())
                .unwrap();
            let mut curr_input = to_input;
            // Keep track of original edges to a and b
            let a_orig_edges = graph
                .graph
                .edges_directed(a, Direction::Incoming)
                .filter_map(|e| e.weight().as_data().map(|(i, ind, _)| (e.source(), i, ind)))
                .sorted_by_key(|i| i.1)
                .collect::<Vec<_>>();
            let b_orig_edges = graph
                .graph
                .edges_directed(b, Direction::Incoming)
                .filter_map(|e| e.weight().as_data().map(|(i, ind, _)| (e.source(), i, ind)))
                .sorted_by_key(|i| i.1)
                .collect::<Vec<_>>();
            // Remove edge a -> b, and decrement indexes of all edges higher than it
            graph.graph.remove_edge(edge_id);
            for edge in graph
                .graph
                .edges_directed(b, Direction::Incoming)
                .map(|e| e.id())
                .collect_vec()
            {
                if let Some(Dependency::Data { input_order, .. }) =
                    graph.graph.edge_weight_mut(edge)
                {
                    if *input_order > curr_input {
                        *input_order -= 1;
                    }
                }
            }
            // Add edges if they don't exist
            for input_edge in graph
                .graph
                .edges_directed(a, Direction::Incoming)
                .filter_map(|e| e.weight().as_data().map(|(a, b, c)| (e.source(), a, b, c)))
                .sorted_by_key(|i| i.1)
                .collect_vec()
            {
                // Find edge or add it
                if !graph
                    .graph
                    .edges_directed(b, Direction::Incoming)
                    .filter_map(|e| e.weight().as_data().map(|(a, b, c)| (e.source(), a, b, c)))
                    .any(|(src, _, out_ind, _)| src == input_edge.0 && out_ind == input_edge.2)
                {
                    // Move all edges >= curr_input up by one
                    for edge in graph
                        .graph
                        .edges_directed(b, Direction::Incoming)
                        .map(|e| e.id())
                        .collect_vec()
                    {
                        if let Some(Dependency::Data { input_order, .. }) =
                            graph.graph.edge_weight_mut(edge)
                        {
                            if *input_order >= curr_input {
                                *input_order += 1;
                            }
                        }
                    }
                    // Add edge
                    graph.graph.add_edge(
                        input_edge.0,
                        b,
                        Dependency::Data {
                            input_order: curr_input,
                            output_order: input_edge.2,
                            shape: input_edge.3,
                        },
                    );
                    curr_input += 1;
                }
            }
            // Alter a_equation to reflect the correct input indexes
            let mut replacements = vec![];
            for (src, inp_ind, out_ind) in a_orig_edges {
                let n = graph
                    .graph
                    .edges_directed(b, Direction::Incoming)
                    .filter_map(|e| e.weight().as_data().map(|(a, b, c)| (e.source(), a, b, c)))
                    .find(|(c_src, _, c_out_ind, _)| *c_src == src && *c_out_ind == out_ind)
                    .unwrap();
                replacements.push((format!("input{inp_ind}"), format!("input{}", n.1)));
            }
            a_equation = multi_replace(&a_equation, &replacements);
            // Alter b_equation to reflect the correct input indexes
            replacements.clear();
            for (src, inp_ind, out_ind) in b_orig_edges {
                if inp_ind > to_input {
                    let n = graph
                        .graph
                        .edges_directed(b, Direction::Incoming)
                        .filter_map(|e| e.weight().as_data().map(|(a, b, c)| (e.source(), a, b, c)))
                        .find(|(c_src, _, c_out_ind, _)| *c_src == src && *c_out_ind == out_ind)
                        .unwrap();
                    replacements.push((format!("input{inp_ind}"), format!("input{}", n.1)));
                }
            }

            if let Some(fused_op) = graph
                .graph
                .node_weight_mut(b)
                .unwrap()
                .as_any_mut()
                .downcast_mut::<FusedElementwiseOp<T>>()
            {
                // B is already fused, just combine with b
                new_op = b;
                // Render a into b as input to_input
                fused_op.equation = multi_replace(&fused_op.equation, &replacements)
                    .replace(&format!("input{to_input}"), &format!("({a_equation})"));
            } else {
                let mut b_equation = graph
                    .node_custom::<String, _>(b, "elementwise", ())
                    .unwrap();
                b_equation = multi_replace(&b_equation, &replacements)
                    .replace(&format!("input{to_input}"), &format!("({a_equation})"));
                // B is not a fused op, let's create a new one
                new_op = graph
                    .add_op(FusedElementwiseOp::<T> {
                        kernel: None,
                        dyn_map: &graph.dyn_map,
                        dyn_chars: vec![],
                        equation: b_equation,
                        queue: queue.clone(),
                        device: device.clone(),
                        _phantom: Default::default(),
                    })
                    .finish();
                move_incoming_edge(b, new_op, &mut graph.graph);
                move_outgoing_edge(b, new_op, &mut graph.graph);
                move_references(
                    &mut remap,
                    &mut graph.no_delete,
                    &mut graph.to_retrieve,
                    b,
                    new_op,
                );
                graph.graph.remove_node(b);
                fused_ops.remove(&b);
            }
            // Remove a
            move_references(
                &mut remap,
                &mut graph.no_delete,
                &mut graph.to_retrieve,
                a,
                new_op,
            );
            if graph
                .graph
                .edges_directed(a, Direction::Outgoing)
                .filter(|e| !e.weight().is_schedule())
                .count()
                == 0
            {
                graph.graph.remove_node(a);
            }
            fused_ops.remove(&a);
            fused_ops.insert(new_op);
            selector.reset();
        }
        // Compile all the kernels we placed
        let type_name = T::type_name();
        for fused_op in fused_ops {
            let edges = graph
                .graph
                .edges_directed(fused_op, Direction::Incoming)
                .filter_map(|e| e.weight().as_data())
                .collect_vec();
            if let Some(op) = graph
                .graph
                .node_weight_mut(fused_op)
                .unwrap()
                .as_any_mut()
                .downcast_mut::<FusedElementwiseOp<T>>()
            {
                let (dyn_chars, rendered) = render_dyn_dim_inputs(
                    &edges.iter().map(|i| i.2).collect_vec(),
                    edges.len() + 2,
                );
                for (inp_ind, _, sh) in &edges {
                    let (ind, val) = get_idx_valid_exps(*sh);
                    if (sh.is_contiguous() && !sh.is_sliced() && !sh.is_padded())
                        || (!sh.is_sliced() && !sh.is_padded())
                    {
                        op.equation = op.equation.replace(
                            &format!("input{inp_ind}"),
                            &format!("(float)input{inp_ind}[{ind}]"),
                        );
                    } else {
                        op.equation = op.equation.replace(
                            &format!("input{inp_ind}"),
                            &format!("(({val} != 0) ? (float)input{inp_ind}[{ind}] : 0.0)"),
                        );
                    }
                }
                let kernel = format!(
                    "
#include <metal_stdlib>
using namespace metal;
kernel void mkernel({} device {type_name} *out [[buffer({})]], device uint& n_elements [[buffer({})]], uint idx [[thread_position_in_grid]]{rendered}) {{
    if (idx < n_elements) {{
        out[idx] = ({type_name})({});
    }}
}}",
                    edges
                        .iter()
                        .map(|(inp_ind, _, _)| format!(
                            "device {type_name}* input{inp_ind} [[buffer({inp_ind})]],"
                        ))
                        .collect_vec()
                        .join(" "),
                    edges.len(),
                    edges.len() + 1,
                    op.equation
                );
                op.kernel = Some(compile_function("mkernel", &kernel, &device));
                op.dyn_chars = dyn_chars;
            }
        }
    }
}

fn multi_replace(input: &str, replacements: &[(String, String)]) -> String {
    // Use Unicode Private Use Areas as unlikely placeholders
    // Starting at U+E000
    let mut placeholder_start = 0xE000;

    let mut output = input.to_string();

    // Generate placeholder characters for each replacement pair
    let mut placeholders: Vec<(String, char)> = Vec::new();
    for (from, _) in replacements {
        let placeholder = std::char::from_u32(placeholder_start).unwrap();
        placeholder_start += 1;
        placeholders.push((from.clone(), placeholder));
    }

    // First pass: Replace all target strings with placeholders
    for (from, placeholder) in &placeholders {
        output = output.replace(from, &placeholder.to_string());
    }

    // Second pass: Replace placeholders with final strings
    for ((_, placeholder), (_, to)) in placeholders.iter().zip(replacements) {
        output = output.replace(&placeholder.to_string(), to);
    }

    output
}

#[derive(LuminalPrint, LuminalEqFalse, Clone)]
pub struct FusedElementwiseOp<T> {
    kernel: Option<ComputePipelineState>,
    dyn_map: *const FxHashMap<char, usize>,
    dyn_chars: Vec<char>,
    equation: String,
    queue: CommandQueue,
    device: Device,
    _phantom: PhantomData<T>,
}
impl<T> MetalKernel for FusedElementwiseOp<T> {
    fn output_buffer_sizes(&self, input_shapes: &[ShapeTracker]) -> Vec<BigExpression> {
        if input_shapes.len() == 1 {
            // Assume since it's a unary op, we're outputting 1-1 elements from input
            vec![input_shapes[0].n_physical_elements() * std::mem::size_of::<T>()]
        } else {
            // If it isn't a unary op, output the contiguous buffer length
            vec![input_shapes[0].n_elements() * std::mem::size_of::<T>()]
        }
    }
    fn metal_forward(
        &self,
        inputs: &[(&Buffer, ShapeTracker)],
        command_buffer: &CommandBufferRef,
        _: &[&Buffer],
        output_buffers: &[&Buffer],
    ) {
        let encoder =
            command_buffer.compute_command_encoder_with_descriptor(ComputePassDescriptor::new());
        encoder.set_compute_pipeline_state(self.kernel.as_ref().unwrap());
        let out_size = inputs
            .iter()
            .map(|i| i.1.n_elements().to_usize().unwrap())
            .max()
            .unwrap();

        // Set function inputs
        for (i, (buf, _)) in inputs.iter().enumerate() {
            encoder.set_buffer(i as u64, Some(*buf), 0);
        }
        encoder.set_buffer(inputs.len() as u64, Some(output_buffers[0]), 0);
        encoder.set_u32(inputs.len() + 1, out_size as u32);
        input_dyn_dims(
            &self.dyn_chars,
            unsafe { self.dyn_map.as_ref().unwrap() },
            encoder,
            inputs.len() + 2,
        );

        // Execute
        println!("Out: {:?}", out_size);
        encoder.dispatch_1d(out_size);
        encoder.end_encoding();
    }
}

impl<T: MetalFloat> Operator for FusedElementwiseOp<T> {
    fn process(&mut self, tensors: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        autoreleasepool(|| {
            let command_buffer = self.queue.new_command_buffer();
            let out = self.device.new_buffer(
                self.output_buffer_sizes(&tensors.iter().map(|(_, s)| *s).collect_vec())[0]
                    .exec(unsafe { self.dyn_map.as_ref().unwrap() })
                    .unwrap() as u64,
                MTLResourceOptions::StorageModeShared,
            );

            self.metal_forward(
                &tensors
                    .iter()
                    .map(|(t, s)| (get_buffer_from_tensor(t).deref(), *s))
                    .collect_vec(),
                command_buffer,
                &[],
                &[&out],
            );

            command_buffer.commit();
            command_buffer.wait_until_completed();

            vec![Tensor::new(MetalBuffer(out))]
        })
    }

    fn custom(&mut self, key: &str, _: Box<dyn Any>) -> Option<Box<dyn Any>> {
        if key == "metal" {
            return Some(Box::new(MetalKernelWrapper(Arc::new(Box::new(
                self.clone(),
            )))));
        }
        // This op can accept non contiguous inputs
        if key == "non_contiguous" {
            return Some(Box::new(()));
        }
        if key == "elementwise" {
            return Some(Box::new(self.equation.clone()));
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use luminal::{
        prelude::*,
        tests::{assert_close, random_vec},
    };

    use crate::MetalCompiler;
    #[test]
    fn test_fusion() {
        let mut cx = Graph::new();
        let a = cx.named_tensor::<R1<10>>("a").set(random_vec(10)).keep();
        let b = cx.named_tensor::<R1<10>>("b").set(random_vec(10)).keep();
        let mut c = (a.exp2() - b.sin()).relu().retrieve();

        cx.execute();
        let unopt_c = c.data();
        c.drop();

        cx.compile(<(GenericCompiler, MetalCompiler<f16>)>::default(), &mut c);
        cx.execute();

        assert_close(&c.data(), &unopt_c);
    }
}
