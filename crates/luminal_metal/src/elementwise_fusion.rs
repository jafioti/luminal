use regex::Regex;
use rustc_hash::{FxHashMap, FxHashSet};
use std::{any::Any, fmt::Debug, iter::once, marker::PhantomData, ops::Deref, sync::Arc};

use itertools::Itertools;
use metal_rs::{
    objc::rc::autoreleasepool, Buffer, CommandBufferRef, CommandQueue, ComputePassDescriptor,
    ComputePipelineState, Device, MTLResourceOptions,
};

use luminal::prelude::{
    petgraph::{visit::EdgeRef, Direction},
    *,
};

use crate::{
    expr_to_metal_string, get_buffer_from_tensor, prim::MetalConstant, MetalBuffer, MetalFloat,
    MetalKernel, MetalKernelWrapper,
};

use super::{compile_function, input_dyn_dims, render_dyn_dim_inputs, DispatchNElements, SetInt};

#[derive(Default, Debug)]
pub struct ElementwiseFusionCompiler<T>(PhantomData<T>);

fn get_inputs(node: NodeIndex, graph: &Graph) -> Vec<(NodeIndex, u8, ShapeTracker)> {
    graph
        .edges_directed(node, Direction::Incoming)
        .filter_map(|e| e.weight().as_data().map(|i| (e.source(), i)))
        .sorted_by_key(|(_, i)| i.0)
        .map(|(a, (_, b, c))| (a, b, c))
        .collect()
}

// Check if we stack the views, does more than one view exist for one of a set of given inputs
fn is_more_than_one_view(
    subexpressions: &[(String, ShapeTracker)],
    subexp_indexes: &[usize],
) -> bool {
    let intermediate_match = Regex::new(r"intermediate(\d+)").unwrap();
    let mut subexp_views = subexpressions
        .iter()
        .map(|(_, sh)| vec![*sh])
        .collect::<Vec<_>>();
    for i in (0..subexp_views.len()).rev() {
        for capture in intermediate_match.captures_iter(&subexpressions[i].0) {
            let index = capture.get(1).unwrap().as_str().parse::<usize>().unwrap();
            if subexp_views[index].len() == 1 {
                let v = subexp_views[i].clone();
                subexp_views[index].extend(v);
            }
        }
    }
    if !subexpressions
        .iter()
        .positions(|(s, _)| {
            subexp_indexes
                .iter()
                .any(|i| s.contains(&format!("input{i}")))
        })
        .map(|subexp_index| &subexp_views[subexp_index])
        .all_equal()
    {
        return true;
    }
    false
}

impl<T: MetalFloat> Compiler for ElementwiseFusionCompiler<T> {
    type Output = ();
    fn compile<To: ToIdsMut>(&self, graph: &mut Graph, mut ids: To) {
        let device = Device::system_default().unwrap();
        let queue = device.new_command_queue();
        // Track fused ops to compile later
        let mut fused_ops = FxHashSet::default();

        let mut matched = true;
        let mut elementwise_ops = FxHashMap::default();
        for op in graph.node_indices().collect::<Vec<_>>() {
            if let Some(exp) = graph.node_custom::<String, _>(op, "elementwise", ()) {
                elementwise_ops.insert(op, exp);
            }
        }
        let mut intermediate_regexes = FxHashMap::default();
        let mut input_regexes = FxHashMap::default();
        while matched {
            matched = false;
            for edge in graph.edge_indices().collect::<Vec<_>>() {
                let Some((a, b)) = graph.edge_endpoints(edge) else {
                    continue;
                };
                if graph.no_delete.contains(&a)
                    || graph.no_delete.contains(&b)
                    || (!graph.check_node_type::<MetalConstant<T>>(a)
                        && graph
                            .edges_directed(a, Direction::Outgoing)
                            .filter(|e| e.target() != b)
                            .count()
                            > 0)
                {
                    continue; // A is not a constant and is feeding into some other node
                }
                let (Some(expression_a), Some(expression_b)) =
                    (elementwise_ops.get(&a), elementwise_ops.get(&b))
                else {
                    continue;
                };
                // a and b are elementwise ops
                // Make sure all edges from a to b share the same shape
                if !graph
                    .edges_connecting(a, b)
                    .map(|e| e.weight().as_data().unwrap().2)
                    .all_equal()
                {
                    continue;
                }
                // Check if there are more than one view of this input. If so, we can't merge
                let mut subexpressions_b = graph
                    .try_get_op::<FusedElementwiseOp<T>>(b)
                    .map(|o| o.subexpressions.clone())
                    .unwrap_or_else(|| vec![(expression_b.clone(), ShapeTracker::new(&[]))]);
                let a_to_b_indexes = graph
                    .edges_connecting(a, b)
                    .map(|e| e.weight().as_data().unwrap().0 as usize)
                    .sorted()
                    .collect::<Vec<_>>();
                if is_more_than_one_view(&subexpressions_b, &a_to_b_indexes) {
                    continue;
                }
                matched = true;
                let a_inputs = get_inputs(a, graph);
                let mut b_inputs = get_inputs(b, graph);
                let (_, _, connecting_shape) = b_inputs.remove(*a_to_b_indexes.last().unwrap());
                for i in a_to_b_indexes.iter().take(a_to_b_indexes.len() - 1).rev() {
                    b_inputs.remove(*i);
                }
                // Get subexpressions
                let mut subexpressions_a = graph
                    .try_get_op::<FusedElementwiseOp<T>>(a)
                    .map(|o| o.subexpressions.clone())
                    .unwrap_or_else(|| vec![(expression_a.clone(), ShapeTracker::new(&[]))]);
                subexpressions_a.last_mut().unwrap().1 = connecting_shape;
                // Re-reference b intermediates
                for i in (0..subexpressions_b.len()).rev() {
                    let re = if let Some(r) = intermediate_regexes.get(&i) {
                        r
                    } else {
                        intermediate_regexes.insert(
                            i,
                            Regex::new(&format!(r"intermediate{i}([^0-9]|$)")).unwrap(),
                        );
                        intermediate_regexes.get(&i).unwrap()
                    };
                    for (exp, _) in subexpressions_b.iter_mut() {
                        *exp = re
                            .replace_all(
                                exp,
                                format!("intermediate{}$1", i + subexpressions_a.len()),
                            )
                            .to_string();
                    }
                }
                // Re-reference b inputs to a
                for index in &a_to_b_indexes {
                    let re = if let Some(r) = input_regexes.get(index) {
                        r
                    } else {
                        input_regexes.insert(
                            *index,
                            Regex::new(&format!(r"input{index}([^0-9]|$)")).unwrap(),
                        );
                        input_regexes.get(index).unwrap()
                    };
                    for (exp, _) in subexpressions_b.iter_mut() {
                        *exp = re
                            .replace_all(
                                exp,
                                format!("intermediate{}$1", subexpressions_a.len() - 1),
                            )
                            .to_string();
                    }
                }
                // Re-reference b inputs
                for (sub_factor, index) in a_to_b_indexes.iter().enumerate() {
                    for i in (*index - sub_factor + 1)..(b_inputs.len() + a_to_b_indexes.len()) {
                        let re = if let Some(r) = input_regexes.get(&i) {
                            r
                        } else {
                            input_regexes
                                .insert(i, Regex::new(&format!(r"input{i}([^0-9]|$)")).unwrap());
                            input_regexes.get(&i).unwrap()
                        };
                        for (exp, _) in subexpressions_b.iter_mut() {
                            *exp = re.replace_all(exp, format!("input{}$1", i - 1)).to_string();
                        }
                    }
                }
                // Combine inputs for a and b
                for i in (0..a_inputs.len()).rev() {
                    // Re-reference the a inputs
                    let re = if let Some(r) = input_regexes.get(&i) {
                        r
                    } else {
                        input_regexes
                            .insert(i, Regex::new(&format!(r"input{i}([^0-9]|$)")).unwrap());
                        input_regexes.get(&i).unwrap()
                    };
                    for (exp, _) in subexpressions_a.iter_mut() {
                        *exp = re
                            .replace_all(exp, format!("input{}$1", i + b_inputs.len()))
                            .to_string();
                    }
                }
                b_inputs.extend(a_inputs);
                // a intermediates should remain valid
                // Combine subexpressions
                for subexp in subexpressions_a.into_iter().rev() {
                    subexpressions_b.insert(0, subexp);
                }

                // Create new fused op
                let output_buffer_sizes = graph
                    .node_custom::<MetalKernelWrapper, _>(b, "metal", ())
                    .unwrap()
                    .output_buffer_sizes(
                        &graph
                            .edges_directed(b, Direction::Incoming)
                            .filter_map(|e| e.weight().as_data())
                            .sorted_by_key(|(i, _, _)| *i)
                            .map(|(_, _, s)| s)
                            .collect::<Vec<_>>(),
                    );
                let new_op = graph
                    .add_op(FusedElementwiseOp::<T> {
                        kernel_str: "".to_string(),
                        kernel: None,
                        dyn_map: &graph.dyn_map,
                        dyn_chars: vec![],
                        subexpressions: subexpressions_b.clone(),
                        queue: queue.clone(),
                        device: device.clone(),
                        output_buffer_sizes,
                        _phantom: Default::default(),
                    })
                    .finish();
                // Add edges to new op
                move_outgoing_edge(b, new_op, graph);
                for (i, (node, output, shape)) in b_inputs.into_iter().enumerate() {
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
                graph.remove_node(b);
                graph.safe_remove_node(a, 0);
                // Keep track of the fused op so we can compile it later
                fused_ops.remove(&a);
                fused_ops.remove(&b);
                fused_ops.insert(new_op);
                elementwise_ops.remove(&a);
                elementwise_ops.remove(&b);
                elementwise_ops.insert(new_op, String::new());
                if !graph.contains_node(a) {
                    remap(a, new_op, &mut ids, graph);
                }
                remap(b, new_op, &mut ids, graph);
            }
        }
        // Convert non-fused elementwise ops to fused elementwise ops
        for (op, op_string) in elementwise_ops {
            if !fused_ops.contains(&op) {
                let input_shapes = graph
                    .edges_directed(op, Direction::Incoming)
                    .filter_map(|e| e.weight().as_data())
                    .sorted_by_key(|(i, _, _)| *i)
                    .map(|(_, _, s)| s)
                    .collect::<Vec<_>>();
                let output_buffer_sizes = graph
                    .node_custom::<MetalKernelWrapper, _>(op, "metal", ())
                    .unwrap()
                    .output_buffer_sizes(&input_shapes);
                let new_op = graph
                    .add_op(FusedElementwiseOp::<T> {
                        kernel_str: "".to_string(),
                        kernel: None,
                        dyn_map: &graph.dyn_map,
                        dyn_chars: vec![],
                        subexpressions: vec![(op_string, ShapeTracker::new(&[]))],
                        queue: queue.clone(),
                        device: device.clone(),
                        output_buffer_sizes,
                        _phantom: Default::default(),
                    })
                    .finish();
                // Add edges to new op
                move_incoming_edge(op, new_op, graph);
                move_outgoing_edge(op, new_op, graph);
                graph.remove_node(op);
                remap(op, new_op, &mut ids, graph);
                fused_ops.insert(new_op);
            }
        }
        // Compile all the kernels we placed
        let intermediate_match = Regex::new(r"intermediate(\d+)([^0-9]|$)").unwrap();
        for fused_op in fused_ops {
            let inputs = graph
                .edges_directed(fused_op, Direction::Incoming)
                .flat_map(|e| e.weight().as_data())
                .sorted_by_key(|(i, _, _)| *i)
                .map(|(_, _, sh)| sh)
                .collect::<Vec<_>>();
            let op = graph.get_op_mut::<FusedElementwiseOp<T>>(fused_op);
            // Stack index expressions and replace them in the subexpressions
            // Track all shapes used, will pull dyn dims from these
            op.pre_compile(inputs, &mut input_regexes, &intermediate_match);
            op.compile(&device);
        }
    }
}

#[derive(Clone)]
pub struct FusedElementwiseOp<T> {
    pub kernel: Option<ComputePipelineState>,
    pub kernel_str: String,
    pub dyn_map: *const FxHashMap<char, usize>,
    pub dyn_chars: Vec<char>,
    pub subexpressions: Vec<(String, ShapeTracker)>,
    pub queue: CommandQueue,
    pub device: Device,
    pub output_buffer_sizes: Vec<BigExpression>,
    pub _phantom: PhantomData<T>,
}
crate::debug_type!(FusedElementwiseOp);

impl<T: MetalFloat> FusedElementwiseOp<T> {
    pub fn pre_compile(
        &mut self,
        input_shapes: Vec<ShapeTracker>,
        input_regexes: &mut FxHashMap<usize, Regex>,
        intermediate_match: &Regex,
    ) {
        let mut subexpressions = self.subexpressions.clone();
        let shapes_used = subexpressions
            .iter()
            .map(|(_, s)| *s)
            .chain(input_shapes.clone())
            .collect::<Vec<_>>();
        // Track the views of each subexpression by going in reverse order and appending the current subexpression's views to the referenced subexpression
        let mut subexp_views = subexpressions
            .iter()
            .map(|(_, sh)| vec![*sh]) // Start with the current view for this subexpression
            .collect::<Vec<_>>();
        for i in (0..subexp_views.len() - 1).rev() {
            for capture in intermediate_match.captures_iter(&subexpressions[i].0) {
                let index = capture.get(1).unwrap().as_str().parse::<usize>().unwrap();
                if subexp_views[index].len() == 1 {
                    let v = subexp_views[i].clone();
                    subexp_views[index].extend(v);
                } else {
                    assert_eq!(subexp_views[index][1..], subexp_views[i][..]);
                }
            }
        }
        // Stack views for each input by going to the first subexpression that uses it and combining it's stacked shape with the input's shape
        let stacked_shapes: Vec<Vec<ShapeTracker>> = input_shapes
            .iter()
            .enumerate()
            .map(|(i, s)| {
                // Find the first subexpression that uses this input
                let re = if let Some(r) = input_regexes.get(&i) {
                    r
                } else {
                    input_regexes.insert(i, Regex::new(&format!(r"input{i}([^0-9]|$)")).unwrap());
                    input_regexes.get(&i).unwrap()
                };
                let using_subexp = subexpressions
                    .iter()
                    .position(|(s, _)| re.is_match(s))
                    .unwrap();

                once(*s)
                    .chain(
                        subexp_views[using_subexp]
                            .iter()
                            .copied()
                            .filter(|s| !s.is_empty()),
                    )
                    .collect()
            })
            .collect();
        // Stack index expressions
        let stacked_index_expressions_partial = stacked_shapes
            .iter()
            .map(|s| {
                s.iter()
                    .rev()
                    .take(s.len() - 1)
                    .fold(BigExpression::from('z'), |acc, inp| {
                        inp.index_expression().substitute('z', acc)
                    })
            })
            .collect::<Vec<_>>();
        let stacked_index_expressions = stacked_index_expressions_partial
            .iter()
            .cloned()
            .zip(&stacked_shapes)
            .map(|(partial, sh)| sh[0].index_expression().substitute('z', partial))
            .collect::<Vec<_>>();
        let stacked_valid_expressions = stacked_index_expressions_partial
            .iter()
            .cloned()
            .zip(&stacked_shapes)
            .map(|(partial, sh)| sh[0].valid_expression().substitute('z', partial))
            .collect::<Vec<_>>();

        // Replace in subexpressions
        let n_subexpressions = subexpressions.len();
        for (i, ((subexp, _), stacked_shapes)) in
            subexpressions.iter_mut().zip(subexp_views).enumerate()
        {
            // Index
            for (i, (ind_exp, val_exp)) in stacked_index_expressions
                .iter()
                .zip(&stacked_valid_expressions)
                .enumerate()
            {
                let re = if let Some(r) = input_regexes.get(&i) {
                    r
                } else {
                    input_regexes.insert(i, Regex::new(&format!(r"input{i}([^0-9]|$)")).unwrap());
                    input_regexes.get(&i).unwrap()
                };
                let (ind, val) = (ind_exp.clone().simplify(), val_exp.clone().simplify());
                *subexp = re
                    .replace_all(
                        subexp,
                        &if val != true {
                            format!(
                                "({} != 0 ? (float)input{i}[{}] : 0.0)$1",
                                expr_to_metal_string(&val),
                                expr_to_metal_string(&ind)
                            )
                        } else {
                            format!("(float)input{i}[{}]$1", expr_to_metal_string(&ind))
                        },
                    )
                    .to_string();
            }
            // Valid (not on last subexpression)
            if i != n_subexpressions - 1 {
                let val_exp = stacked_shapes
                    .iter()
                    .rev()
                    .fold(
                        (BigExpression::from(true), BigExpression::from('z')),
                        |(_, ind_acc), inp| {
                            (
                                inp.valid_expression().substitute('z', ind_acc.clone()),
                                inp.index_expression().substitute('z', ind_acc),
                            )
                        },
                    )
                    .0
                    .simplify();
                if val_exp != true {
                    *subexp = format!(
                        "(({} != 0) ? {subexp} : 0.0)",
                        expr_to_metal_string(&val_exp)
                    );
                }
            }
        }

        let (dyn_chars, rendered) = render_dyn_dim_inputs(&shapes_used, input_shapes.len() + 2);
        let type_name = T::type_name();
        self.kernel_str = format!(
            "
#include <metal_stdlib>
using namespace metal;
kernel void mkernel({} device {type_name} *out [[buffer({})]], device uint& n_elements [[buffer({})]], uint idx [[thread_position_in_grid]]{rendered}) {{
if (idx < n_elements) {{
{}
out[idx] = ({type_name})({});
}}
}}",
            (0..input_shapes.len())
                .map(|inp_ind| format!(
                    "device {type_name}* input{inp_ind} [[buffer({inp_ind})]],"
                ))
                .collect::<Vec<_>>()
                .join(" "),
            input_shapes.len(),
            input_shapes.len() + 1,
            subexpressions.iter().take(subexpressions.len() - 1).enumerate().map(|(i, (subexp, _))| format!("float intermediate{i} = {subexp};")).join("\n        "),
            subexpressions.last().unwrap().0
        );
        self.dyn_chars = dyn_chars;
    }

    pub fn compile(&mut self, device: &Device) {
        self.kernel = Some(compile_function("mkernel", &self.kernel_str, device));
    }
}

impl<T> MetalKernel for FusedElementwiseOp<T> {
    fn output_buffer_sizes(&self, _: &[ShapeTracker]) -> Vec<BigExpression> {
        self.output_buffer_sizes.clone()
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
        // Use output buffer size to work out the dispatch size
        let dyn_map = unsafe { self.dyn_map.as_ref().unwrap() };
        let out_size =
            self.output_buffer_sizes[0].exec(dyn_map).unwrap() / std::mem::size_of::<T>();

        // Set function inputs
        for (i, (buf, _)) in inputs.iter().enumerate() {
            encoder.set_buffer(i as u64, Some(*buf), 0);
        }
        encoder.set_buffer(inputs.len() as u64, Some(output_buffers[0]), 0);
        encoder.set_u32(inputs.len() + 1, out_size as u32);
        input_dyn_dims(&self.dyn_chars, dyn_map, encoder, inputs.len() + 2);

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
                self.output_buffer_sizes[0]
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
        if key == "elementwise" {
            return Some(Box::new("".to_string()));
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use luminal::{
        prelude::{binary::F32Pow, *},
        tests::{assert_close, assert_close_precision, random_vec, random_vec_rng},
    };
    use luminal_nn::*;
    use rand::{rngs::StdRng, SeedableRng};
    use std::{marker::PhantomData, ops::Div};

    use crate::MetalCompiler;

    #[test]
    fn test_fusion_simple() {
        let mut cx = Graph::new();
        let mut rng = StdRng::seed_from_u64(0);
        let inp = cx.tensor::<R1<5>>().set(random_vec_rng(10, &mut rng));
        let mut out = inp.exp2().cos().sqrt().retrieve();

        cx.execute();
        let unopt_out = out.data();
        out.drop();

        cx.compile(<(GenericCompiler, MetalCompiler<f16>)>::default(), &mut out);
        cx.execute();

        assert_close(&out.data(), &unopt_out);
    }
    #[test]
    fn test_fusion_binary() {
        let mut cx = Graph::new();
        let mut rng = StdRng::seed_from_u64(0);
        let a = cx.tensor::<R1<5>>().set(random_vec_rng(10, &mut rng));
        let b = cx.tensor::<R1<5>>().set(random_vec_rng(10, &mut rng));
        let mut out = (a.exp2() + b.cos()).retrieve();

        cx.execute();
        let unopt_out = out.data();
        out.drop();

        cx.compile(<(GenericCompiler, MetalCompiler<f16>)>::default(), &mut out);
        cx.execute();

        assert_close(&out.data(), &unopt_out);
    }

    #[test]
    fn test_fusion_subexpression_complex() {
        let mut cx = Graph::new();
        let a = cx.named_tensor::<R1<10>>("a").set(random_vec(10)).keep();
        let b = cx.named_tensor::<R1<10>>("b").set(random_vec(10)).keep();
        let d = cx.named_tensor::<R1<10>>("d").set(random_vec(10)).keep();
        let mut out = ((a.exp2() - b.sin()).sin() * 3.4).less_than(d).retrieve();

        cx.execute();
        let unopt_out = out.data();
        out.drop();

        cx.compile(<(GenericCompiler, MetalCompiler<f16>)>::default(), &mut out);
        cx.execute();

        assert_close(&out.data(), &unopt_out);
    }

    #[test]
    fn test_fusion_slicing_padding() {
        let mut cx = Graph::new();
        let mut rng = StdRng::seed_from_u64(0);
        let inp = random_vec_rng(10, &mut rng);
        let a = cx.named_tensor::<R2<2, 5>>("a").set(inp);
        let mut padded = a
            .slice((..Expression::from(1), ..))
            .realize::<R2<1, 5>>()
            .cos()
            .pad::<R2<2, 5>>(((0, 1), (0, 0)))
            .exp2()
            .retrieve();
        cx.execute();
        let unopt_out = padded.data();
        padded.drop();

        cx.compile(
            <(GenericCompiler, MetalCompiler<f16>)>::default(),
            &mut padded,
        );
        cx.execute();

        assert_close(&padded.data(), &unopt_out);
    }

    #[test]
    fn test_fusion_subexpression() {
        let mut cx = Graph::new();
        let mut rng = StdRng::seed_from_u64(0);
        let data = random_vec_rng(10, &mut rng);
        let a = cx.tensor::<R2<2, 5>>().set(data);
        let mut out = (a.sqrt().exp() + a.sqrt().sin()).retrieve();
        cx.execute();
        let unopt_out = out.data();
        out.drop();

        cx.compile(<(GenericCompiler, MetalCompiler<f32>)>::default(), &mut out);
        cx.execute();

        assert_close(&out.data(), &unopt_out);
    }

    #[test]
    fn test_fusion_rope_emb() {
        let mut cx = Graph::new();
        const SEQ: usize = 2;
        const HEAD_DIM: usize = 4;
        const HEAD_DIM_OVER_2: usize = HEAD_DIM / 2;
        let freqs = (cx.arange::<Const<HEAD_DIM_OVER_2>>() * 2.0) / (HEAD_DIM as f32);
        let freqs = 1000000_f32.pow(freqs);
        let pos = cx.arange::<Const<SEQ>>() + BigExpression::from(0);
        let mut emb = pos
            .expand::<(_, Const<1>), _>()
            .matmul(freqs.expand())
            .retrieve();

        cx.execute();
        let unopt_out = emb.data();
        emb.drop();

        cx.compile(<(GenericCompiler, MetalCompiler<f16>)>::default(), &mut emb);
        cx.execute();
        assert_close(&emb.data(), &unopt_out);
    }

    #[test]
    fn test_fusion_rotate() {
        let mut cx = Graph::new();
        let mut rng = StdRng::seed_from_u64(0);
        const SEQ: usize = 2;
        const HEAD_DIM: usize = 4;
        const HEAD_DIM_OVER_2: usize = HEAD_DIM / 2;
        let a = cx
            .named_tensor::<R2<SEQ, HEAD_DIM>>("a")
            .set(random_vec_rng(SEQ * HEAD_DIM, &mut rng))
            .keep();
        let b = cx
            .tensor::<R3<SEQ, HEAD_DIM_OVER_2, 1>>()
            .set(random_vec_rng(SEQ * HEAD_DIM_OVER_2, &mut rng))
            .keep();
        // Split input into evens and odds
        let split = a.reshape::<R3<SEQ, HEAD_DIM_OVER_2, 2>>();
        let x0: GraphTensor<R3<SEQ, HEAD_DIM_OVER_2, 1>> =
            split.slice((.., .., ..Expression::from(1))).realize();
        let x1: GraphTensor<R3<SEQ, HEAD_DIM_OVER_2, 1>> =
            split.slice((.., .., Expression::from(1)..)).realize();

        let x0_out = x0 * b - x1 * b.cos();
        let x1_out = x0 + x1;

        // Combine back into output
        let mut out: GraphTensor<R2<SEQ, HEAD_DIM>> = x0_out
            .concat_along::<R3<SEQ, HEAD_DIM_OVER_2, 2>, Axis<2>, _>(x1_out)
            .reshape()
            .retrieve();
        cx.execute();

        let unopt_out = out.data();
        out.drop();

        cx.compile(<(GenericCompiler, MetalCompiler<f16>)>::default(), &mut out);
        cx.execute();
        assert_close(&out.data(), &unopt_out);
    }

    #[test]
    fn test_fusion_rope_full() {
        let mut cx = Graph::new();
        let mut rng = StdRng::seed_from_u64(0);
        const BATCH: usize = 1;
        const N_HEADS: usize = 8;
        const SEQ: usize = 2;
        const HEAD_DIM: usize = 4;
        const HEAD_DIM_OVER_2: usize = HEAD_DIM / 2;
        let a = cx
            .named_tensor::<R4<BATCH, N_HEADS, SEQ, HEAD_DIM>>("a")
            .set(random_vec_rng(BATCH * N_HEADS * SEQ * HEAD_DIM, &mut rng))
            .keep();
        let freqs = (cx.arange::<Const<HEAD_DIM_OVER_2>>() * 2.0) / (HEAD_DIM as f32);
        let freqs = 1000000_f32.pow(freqs);
        let pos = cx.arange::<Const<SEQ>>() + BigExpression::from(0);
        let emb = pos.expand::<(_, Const<1>), _>().matmul(freqs.expand());
        // Split input into evens and odds
        let split = a.reshape::<R5<BATCH, N_HEADS, SEQ, HEAD_DIM_OVER_2, 2>>();
        let x0: GraphTensor<R5<BATCH, N_HEADS, SEQ, HEAD_DIM_OVER_2, 1>> = split
            .slice((.., .., .., .., ..Expression::from(1)))
            .contiguous()
            .realize();
        let x1: GraphTensor<R5<BATCH, N_HEADS, SEQ, HEAD_DIM_OVER_2, 1>> = split
            .slice((.., .., .., .., Expression::from(1)..))
            .contiguous()
            .realize();

        // Apply sin and cos embeddings
        let x0_out = x0 * emb.cos().expand() - x1 * emb.sin().expand();
        let x1_out = x0 * emb.sin().expand() + x1 * emb.cos().expand();

        // Combine back into output
        let mut out: GraphTensor<R4<BATCH, N_HEADS, SEQ, HEAD_DIM>> = x0_out
            .concat_along::<R5<BATCH, N_HEADS, SEQ, HEAD_DIM_OVER_2, 2>, Axis<4>, _>(x1_out)
            .reshape()
            .retrieve();
        cx.execute();
        let unopt_out = out.data();
        out.drop();

        cx.compile(<(GenericCompiler, MetalCompiler<f16>)>::default(), &mut out);
        cx.execute();

        assert_close(&out.data(), &unopt_out);
    }

    #[test]
    fn test_fusion_transformer() {
        pub const HIDDEN_DIM: usize = 128;
        pub const N_HEADS: usize = 2;
        pub const N_KV_HEADS: usize = 2;
        pub const MLP_DIM: usize = 256;
        pub const NUM_LAYERS: usize = 2;
        pub const SEQ_LEN: usize = 65;
        pub const N_ATTENTION_GROUPS: usize = N_HEADS / N_KV_HEADS;
        pub const HEAD_DIM: usize = HIDDEN_DIM / N_HEADS;
        pub const HEAD_DIM_OVER_2: usize = HEAD_DIM / 2;
        pub const ATTN_PROJ_DIM: usize = HEAD_DIM * N_KV_HEADS;
        pub struct Mlp<const I: usize, const H: usize> {
            pub gate_proj: PermutedLinear<H, I>,
            pub down_proj: PermutedLinear<I, H>,
            pub up_proj: PermutedLinear<H, I>,
        }

        pub type KVCache<Batch, Seq> = (
            GraphTensor<(Batch, Const<N_KV_HEADS>, Seq, Const<HEAD_DIM>)>,
            GraphTensor<(Batch, Const<N_KV_HEADS>, Seq, Const<HEAD_DIM>)>,
        );

        impl<Sh: Shape, Im: Shape, const I: usize, const H: usize> Module<GraphTensor<Sh>> for Mlp<I, H>
        where
            GraphTensor<Sh>: Matmul<R2<H, I>, Output = GraphTensor<Im>>,
            GraphTensor<Im>: Matmul<R2<I, H>, Output = GraphTensor<Sh>>,
        {
            type Output = GraphTensor<Sh>;

            fn forward(&self, input: GraphTensor<Sh>) -> Self::Output {
                let gate = self.gate_proj.forward(input).swish();
                let up = self.up_proj.forward(input) * gate;
                self.down_proj.forward(up)
            }
        }
        impl<const I: usize, const H: usize> InitModule for Mlp<I, H> {
            fn initialize(cx: &mut Graph) -> Self {
                Self {
                    gate_proj: InitModule::initialize(cx),
                    up_proj: InitModule::initialize(cx),
                    down_proj: InitModule::initialize(cx),
                }
            }
        }
        fn apply_rotary_embeddings_ggml<const N_HEADS: usize, Batch: Dimension, Seq: Dimension>(
            input: GraphTensor<(Batch, Const<N_HEADS>, Seq, Const<HEAD_DIM>)>,
            prev_seq: BigExpression,
        ) -> GraphTensor<(Batch, Const<N_HEADS>, Seq, Const<HEAD_DIM>)> {
            // Get freqs
            let freqs =
                (input.graph().arange::<Const<HEAD_DIM_OVER_2>>() * 2.0) / (HEAD_DIM as f32);
            let freqs = 1000000_f32.pow(freqs);
            let pos = input.graph().arange::<Seq>() + prev_seq;
            let emb = pos.expand::<(_, Const<1>), _>().matmul(freqs.expand());

            // Split input into evens and odds
            let split =
                input.reshape::<(Batch, Const<N_HEADS>, Seq, Const<HEAD_DIM_OVER_2>, Const<2>)>();
            let x0: GraphTensor<(Batch, Const<N_HEADS>, Seq, Const<HEAD_DIM_OVER_2>, Const<1>)> =
                split
                    .slice((.., .., .., .., ..Expression::from(1)))
                    .contiguous()
                    .realize();
            let x1: GraphTensor<(Batch, Const<N_HEADS>, Seq, Const<HEAD_DIM_OVER_2>, Const<1>)> =
                split
                    .slice((.., .., .., .., Expression::from(1)..))
                    .contiguous()
                    .realize();

            // Apply sin and cos embeddings
            let x0_out = x0 * emb.cos().expand() - x1 * emb.sin().expand();
            let x1_out = x0 * emb.sin().expand() + x1 * emb.cos().expand();

            // Combine back into output
            x0_out
                .concat_along::<(Batch, Const<N_HEADS>, Seq, Const<HEAD_DIM_OVER_2>, Const<2>), Axis<4>, _>(
                    x1_out,
                )
                .reshape()
        }
        pub struct SelfAttention {
            pub q_proj: GraphTensor<R2<HIDDEN_DIM, HIDDEN_DIM>>,
            pub k_proj: GraphTensor<R2<ATTN_PROJ_DIM, HIDDEN_DIM>>,
            pub v_proj: GraphTensor<R2<ATTN_PROJ_DIM, HIDDEN_DIM>>,
            pub o_proj: GraphTensor<R2<HIDDEN_DIM, HIDDEN_DIM>>,
        }

        impl<Batch: Dimension, CurSeq: Dimension, PrevSeq: Dimension, TotSeq: Dimension>
            Module<(
                GraphTensor<(Batch, CurSeq, Const<HIDDEN_DIM>)>,
                KVCache<Batch, PrevSeq>,
                PhantomData<TotSeq>,
            )> for SelfAttention
        {
            type Output = (
                GraphTensor<(Batch, CurSeq, Const<HIDDEN_DIM>)>,
                KVCache<Batch, TotSeq>,
            );
            fn forward(
                &self,
                (x, (k_cache, v_cache), _): (
                    GraphTensor<(Batch, CurSeq, Const<HIDDEN_DIM>)>,
                    KVCache<Batch, PrevSeq>,
                    PhantomData<TotSeq>,
                ),
            ) -> Self::Output {
                // Apply the Projections
                let queries = x
                    .matmul(self.q_proj.permute())
                    .reshape::<(Batch, CurSeq, Const<N_HEADS>, Const<HEAD_DIM>)>()
                    .permute::<_, Axes4<0, 2, 1, 3>>();

                let keys = x
                    .matmul(self.k_proj.permute())
                    .reshape::<(Batch, CurSeq, Const<N_KV_HEADS>, Const<HEAD_DIM>)>()
                    .permute::<_, Axes4<0, 2, 1, 3>>();

                let values = x
                    .matmul(self.v_proj.permute())
                    .reshape::<(Batch, CurSeq, Const<N_KV_HEADS>, Const<HEAD_DIM>)>()
                    .permute::<_, Axes4<0, 2, 1, 3>>();

                // Rotary embed queries and keys
                let queries = apply_rotary_embeddings_ggml(queries, PrevSeq::size().big());
                let keys = apply_rotary_embeddings_ggml(keys, PrevSeq::size().big());

                // Add KV cache
                let (keys, values) = (
                    k_cache.concat_along::<_, Axis<2>, _>(keys),
                    v_cache.concat_along::<_, Axis<2>, _>(values),
                );

                // Repeat the KV States for Grouped-Query Attention
                let repeated_keys = keys.expand::<(_, _, Const<N_ATTENTION_GROUPS>, _, _), _>();
                let repeated_values = values.expand::<(_, _, Const<N_ATTENTION_GROUPS>, _, _), _>();

                // Calculate attention weights
                let mut attention_weights = queries
                    .reshape::<(_, Const<N_KV_HEADS>, Const<N_ATTENTION_GROUPS>, _, _)>() // Split query heads into groups
                    .matmul(repeated_keys.permute())
                    .div((HEAD_DIM as f32).sqrt());

                let attention_mask = self.k_proj.graph().triu::<CurSeq>(1) * f16::MIN.to_f32();
                attention_weights += attention_mask
                    .pad::<(CurSeq, TotSeq)>(((0, 0), (TotSeq::size() - CurSeq::size(), 0)))
                    .expand();

                // Calculate final outputs
                let output = attention_weights
                    .softmax::<Axis<4>>()
                    // Apply distribution to values
                    .matmul(repeated_values)
                    // Merge heads
                    .permute::<_, Axes5<0, 3, 1, 2, 4>>()
                    .reshape::<(Batch, CurSeq, Const<HIDDEN_DIM>)>();
                let output = output
                    // Apply output projection
                    .matmul(self.o_proj.permute());
                (output, (keys.contiguous(), values.contiguous())) // Cache needs to be contiguous for transferring to another graph
            }
        }

        impl InitModule for SelfAttention {
            fn initialize(cx: &mut Graph) -> Self {
                Self {
                    q_proj: cx
                        .named_tensor("Q Proj")
                        .set(random_vec(HIDDEN_DIM * HIDDEN_DIM)),
                    k_proj: cx
                        .named_tensor("K Proj")
                        .set(random_vec(ATTN_PROJ_DIM * HIDDEN_DIM)),
                    v_proj: cx
                        .named_tensor("V Proj")
                        .set(random_vec(ATTN_PROJ_DIM * HIDDEN_DIM)),
                    o_proj: cx
                        .named_tensor("O Proj")
                        .set(random_vec(HIDDEN_DIM * HIDDEN_DIM)),
                }
            }
        }

        impl SerializeModule for SelfAttention {
            fn serialize(&self, s: &mut Serializer) {
                s.tensor("attn_q/weight", self.q_proj);
                s.tensor("attn_v/weight", self.v_proj);
                s.tensor("attn_k/weight", self.k_proj);
                s.tensor("attn_output/weight", self.o_proj);
            }
        }

        pub struct TransformerBlock {
            pub attention: SelfAttention,
            pub attention_norm: LayerNorm<HIDDEN_DIM>,
            pub feed_forward: Mlp<MLP_DIM, HIDDEN_DIM>,
            pub feed_forward_norm: LayerNorm<HIDDEN_DIM>,
        }

        impl<Batch: Dimension, CurSeq: Dimension, PrevSeq: Dimension, TotSeq: Dimension>
            Module<(
                GraphTensor<(Batch, CurSeq, Const<HIDDEN_DIM>)>,
                KVCache<Batch, PrevSeq>,
                PhantomData<TotSeq>,
            )> for TransformerBlock
        {
            type Output = (
                GraphTensor<(Batch, CurSeq, Const<HIDDEN_DIM>)>,
                KVCache<Batch, TotSeq>,
            );
            fn forward(
                &self,
                (mut x, cache, _): (
                    GraphTensor<(Batch, CurSeq, Const<HIDDEN_DIM>)>,
                    KVCache<Batch, PrevSeq>,
                    PhantomData<TotSeq>,
                ),
            ) -> Self::Output {
                // Attention
                let normed = self.attention_norm.forward(x);
                let (y, cache) = self
                    .attention
                    .forward((normed, cache, PhantomData::<TotSeq>));

                // Residual Addition
                x += y;

                // Feed Forward
                let y = self.feed_forward.forward(self.feed_forward_norm.forward(x));

                // Residual Addition
                (x + y, cache)
            }
        }

        impl InitModule for TransformerBlock {
            fn initialize(cx: &mut Graph) -> Self {
                Self {
                    attention: InitModule::initialize(cx),
                    attention_norm: LayerNorm::new(false, false, false, 1e-5, cx),
                    feed_forward: InitModule::initialize(cx),
                    feed_forward_norm: LayerNorm::new(false, false, false, 1e-5, cx),
                }
            }
        }

        pub struct MistralLM {
            // Transformer layers
            pub layers: Vec<TransformerBlock>,
            // Final Norm layer
            pub norm: LayerNorm<HIDDEN_DIM>,
        }

        impl<Batch: Dimension, CurSeq: Dimension, PrevSeq: Dimension, TotSeq: Dimension>
            Module<(
                GraphTensor<(Batch, CurSeq, Const<HIDDEN_DIM>)>,
                Vec<KVCache<Batch, PrevSeq>>,
                PhantomData<TotSeq>,
            )> for MistralLM
        {
            type Output = (
                GraphTensor<(Batch, CurSeq, Const<HIDDEN_DIM>)>,
                Vec<KVCache<Batch, TotSeq>>,
            );
            fn forward(
                &self,
                (input, cache, _): (
                    GraphTensor<(Batch, CurSeq, Const<HIDDEN_DIM>)>,
                    Vec<KVCache<Batch, PrevSeq>>,
                    PhantomData<TotSeq>,
                ),
            ) -> Self::Output {
                let mut x = input;

                // Run through layers and collect new caches
                let mut new_caches = vec![];
                let mut new_cache;
                for (i, layer) in self.layers.iter().enumerate() {
                    (x, new_cache) = layer.forward((x, cache[i], PhantomData::<TotSeq>));
                    new_caches.push(new_cache);
                }
                // Run through last norm and output projection
                let normed = self.norm.forward(x);
                (normed, new_caches)
            }
        }

        impl InitModule for MistralLM {
            fn initialize(cx: &mut Graph) -> Self {
                Self {
                    norm: LayerNorm::new(false, false, false, 1e-5, cx),
                    layers: (0..NUM_LAYERS)
                        .map(|_| InitModule::initialize(cx))
                        .collect(),
                }
            }
        }

        let mut cx = Graph::new();
        let model = MistralLM::initialize(&mut cx);
        let caches = (0..NUM_LAYERS)
            .map(|_| {
                (
                    cx.tensor::<(Const<1>, Const<N_KV_HEADS>, Dyn<'p'>, Const<HEAD_DIM>)>()
                        .set_dyn(
                            random_vec(SEQ_LEN * N_KV_HEADS * HEAD_DIM),
                            &[1, N_KV_HEADS, SEQ_LEN, HEAD_DIM],
                        ),
                    cx.tensor::<(Const<1>, Const<N_KV_HEADS>, Dyn<'p'>, Const<HEAD_DIM>)>()
                        .set_dyn(
                            random_vec(SEQ_LEN * N_KV_HEADS * HEAD_DIM),
                            &[1, N_KV_HEADS, SEQ_LEN, HEAD_DIM],
                        ),
                )
            })
            .collect();
        let input = cx
            .tensor::<(Const<1>, Dyn<'s'>, luminal::shape::Const<HIDDEN_DIM>)>()
            .set_dyn(random_vec(2 * HIDDEN_DIM), &[1, 2, HIDDEN_DIM]);
        let (mut out, _) = model.forward((input, caches, PhantomData::<Dyn<'t'>>));
        out.retrieve();

        cx.set_dyn_dim('t', SEQ_LEN + 2);
        cx.execute();

        let unopt_out = out.data();
        out.drop();

        cx.compile(<(GenericCompiler, MetalCompiler<f16>)>::default(), &mut out);
        cx.execute();

        assert_close_precision(&out.data(), &unopt_out, 1e-2);
    }
}
