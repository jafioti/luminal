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
    compile_function, get_idx_valid_exps, input_dyn_dims, render_dyn_dim_inputs, DispatchNElements,
    SetInt,
};

#[derive(Default, Debug)]
pub struct ElementwiseFusionCompiler<T>(PhantomData<T>);

impl<T: MetalFloat> Compiler for ElementwiseFusionCompiler<T> {
    fn compile<To: ToIdsMut>(&self, graph: &mut Graph, _: To) {
        graph.display();
        let device = Device::system_default().unwrap();
        let queue = device.new_command_queue();
        // Track fused ops to compile later
        let mut fused_ops = FxHashMap::default();

        let mut matched = true;
        while matched {
            matched = false;
            for edge in graph.edge_indices().collect::<Vec<_>>() {
                let Some((a, b)) = graph.edge_endpoints(edge) else {
                    continue;
                };
                if graph.edges_directed(a, Direction::Outgoing).count() > 1 {
                    continue; // More than one connecting edge. We'll handle this later
                }
                if graph.no_delete.contains(&a) {
                    continue;
                }
                let (Some(expression_a), Some(expression_b)) = (
                    graph.node_custom::<String, _>(a, "elementwise", Box::<()>::default()),
                    graph.node_custom::<String, _>(b, "elementwise", Box::<()>::default()),
                ) else {
                    continue;
                };
                // a and b are elementwise ops
                matched = true;
                // get views for each input in a and b
                #[allow(clippy::type_complexity)]
                let mut b_inputs: Vec<(
                    Vec<ShapeTracker>,
                    (NodeIndex, (u8, u8, ShapeTracker)),
                )> = graph
                    .try_get_op::<FusedElementwiseOp<T>>(b)
                    .map(|n| n.input_views.clone())
                    .unwrap_or_else(|| {
                        vec![vec![]; graph.edges_directed(b, Direction::Incoming).count()]
                    })
                    .into_iter()
                    .zip(
                        graph
                            .edges_directed(b, Direction::Incoming)
                            .filter_map(|e| e.weight().as_data().map(|i| (e.source(), i))),
                    )
                    .collect::<Vec<_>>();
                let a_to_b_index = graph
                    .edges_directed(b, Direction::Incoming)
                    .position(|e| e.source() == a)
                    .unwrap();
                let (connect_inp, (_, (_, _, sh))) = b_inputs.remove(a_to_b_index);
                let reshaped = !sh.is_contiguous() || sh.is_sliced() || sh.is_padded();
                let b_replacements = b_inputs
                    .iter()
                    .enumerate()
                    .skip(a_to_b_index + 1)
                    .map(|(i, _)| (i, i - 1))
                    .collect::<Vec<_>>();
                let mut a_replacements = vec![];
                for mut a_inp in graph
                    .try_get_op::<FusedElementwiseOp<T>>(a)
                    .map(|n| n.input_views.clone())
                    .unwrap_or_else(|| {
                        vec![vec![]; graph.edges_directed(a, Direction::Incoming).count()]
                    })
                    .into_iter()
                    .zip(
                        graph
                            .edges_directed(a, Direction::Incoming)
                            .filter_map(|e| e.weight().as_data().map(|i| (e.source(), i))),
                    )
                {
                    if !reshaped {
                        if let Some(pos) = b_inputs.iter().position(|(v, (n, w))| {
                            *v == a_inp.0
                                && *n == a_inp.1 .0
                                && w.1 == a_inp.1 .1 .1
                                && w.2 == a_inp.1 .1 .2
                        }) {
                            a_replacements.push((a_inp.1 .1 .0 as usize, pos));
                            continue;
                        }
                    }
                    a_inp.0.extend(connect_inp.iter().copied());
                    a_inp.1 .1 .0 = b_inputs.len() as u8;
                    b_inputs.push(a_inp);
                }
                // Combine the views into a final view array
                let new_views = b_inputs.iter().map(|(v, _)| v.clone()).collect::<Vec<_>>();
                // Get new input array
                let new_inputs = b_inputs
                    .into_iter()
                    .map(|(_, (n, (_, o, sh)))| (n, o, sh))
                    .collect::<Vec<_>>();
                // Combine expressions together to get final expression
                let a_replacements = a_replacements
                    .into_iter()
                    .map(|(from, to)| (format!("input{from}"), format!("input{to}")))
                    .collect::<Vec<_>>();
                println!("Exp A: {}", expression_a);
                let expression_a = multi_replace(&expression_a, &a_replacements);
                let mut b_replacements = b_replacements
                    .into_iter()
                    .map(|(from, to)| (format!("input{from}"), format!("input{to}")))
                    .collect::<Vec<_>>();
                b_replacements.push((format!("input{a_to_b_index}"), expression_a));
                println!("Exp B: {}", expression_b);
                let equation = multi_replace(&expression_b, &b_replacements);
                // Delete old ops
                let b_outgoing = graph
                    .edges_directed(b, Direction::Outgoing)
                    .map(|e| (e.target(), *e.weight()))
                    .collect::<Vec<_>>();
                let a_prev_ops = fused_ops.remove(&a).unwrap_or_else(|| {
                    vec![format!(
                        "{:?}({})",
                        graph.node_weight(a).unwrap(),
                        a.index()
                    )]
                });
                let b_prev_ops = fused_ops.remove(&b).unwrap_or_else(|| {
                    vec![format!(
                        "{:?}({})",
                        graph.node_weight(b).unwrap(),
                        b.index()
                    )]
                });
                graph.remove_node(a);
                graph.remove_node(b);
                println!(
                    "Fused {:?}",
                    [a_prev_ops.clone(), b_prev_ops.clone()].concat()
                );
                println!("Into {:?}", equation);
                // Create new fused op
                let new_op = graph
                    .add_op(FusedElementwiseOp::<T> {
                        kernel: None,
                        dyn_map: &graph.dyn_map,
                        dyn_chars: vec![],
                        equation,
                        queue: queue.clone(),
                        device: device.clone(),
                        input_views: new_views,
                        _phantom: Default::default(),
                    })
                    .finish();
                for (i, (node, output, shape)) in new_inputs.into_iter().enumerate() {
                    graph.add_edge(
                        node,
                        new_op,
                        Dependency::Data {
                            input_order: i as u8,
                            output_order: output,
                            shape,
                        },
                    );
                }
                for (node, weight) in b_outgoing {
                    graph.add_edge(new_op, node, weight);
                }
                // Keep track of the fused op so we can compile it later
                fused_ops.insert(new_op, [a_prev_ops, b_prev_ops].concat());
            }
        }
        // Compile all the kernels we placed
        let type_name = T::type_name();
        for (fused_op, prev_ops) in fused_ops {
            let edges = graph
                .edges_directed(fused_op, Direction::Incoming)
                .filter_map(|e| e.weight().as_data())
                .collect_vec();
            if let Some(op) = graph
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
                println!("Fused {:?}", prev_ops);
                println!("Into {:?}", op.equation);
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
        graph.display();
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
    input_views: Vec<Vec<ShapeTracker>>,
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
        let c = cx.constant(3.4);
        let d = cx.named_tensor::<R1<10>>("d").set(random_vec(10)).keep();
        let mut out = ((a.exp2() - b.sin()).relu() * c.expand())
            .less_than(d)
            .retrieve();

        cx.execute();
        let unopt_out = out.data();
        out.drop();

        cx.compile(<(GenericCompiler, MetalCompiler<f16>)>::default(), &mut out);
        cx.execute();

        assert_close(&out.data(), &unopt_out);
    }
}
