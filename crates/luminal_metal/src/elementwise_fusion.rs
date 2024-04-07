use regex::Regex;
use rustc_hash::{FxHashMap, FxHashSet};
use std::{any::Any, iter::once, marker::PhantomData, ops::Deref, sync::Arc};

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

use crate::{
    expr_to_metal_string, get_buffer_from_tensor, prim::MetalConstant, MetalBuffer, MetalFloat,
    MetalKernel, MetalKernelWrapper,
};

use self::symbolic::BigExpression;

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
        // graph.display();
        let device = Device::system_default().unwrap();
        let queue = device.new_command_queue();
        // Track fused ops to compile later
        let mut fused_ops = FxHashSet::default();

        let mut matched = true;
        while matched {
            matched = false;
            for edge in graph.edge_indices().collect::<Vec<_>>() {
                let Some((a, b)) = graph.edge_endpoints(edge) else {
                    continue;
                };
                if graph.no_delete.contains(&a)
                    || (!graph.check_node_type::<MetalConstant<T>>(a)
                        && graph
                            .edges_directed(a, Direction::Outgoing)
                            .filter(|e| e.target() != b)
                            .count()
                            > 0)
                {
                    continue; // A is not a constant and is feeding into some other node
                }
                let (Some(expression_a), Some(expression_b)) = (
                    graph.node_custom::<String, _>(a, "elementwise", ()),
                    graph.node_custom::<String, _>(b, "elementwise", ()),
                ) else {
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
                    .unwrap_or_else(|| vec![(expression_b, ShapeTracker::new(&[]))]);
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
                    .unwrap_or_else(|| vec![(expression_a, ShapeTracker::new(&[]))]);
                subexpressions_a.last_mut().unwrap().1 = connecting_shape;
                // Re-reference b intermediates
                for i in (0..subexpressions_b.len()).rev() {
                    for (exp, _) in subexpressions_b.iter_mut() {
                        *exp = exp.replace(
                            &format!("intermediate{i}"),
                            &format!("intermediate{}", i + subexpressions_a.len()),
                        );
                    }
                }
                // Re-reference b inputs to a
                for index in &a_to_b_indexes {
                    for (exp, _) in subexpressions_b.iter_mut() {
                        *exp = exp.replace(
                            &format!("input{index}"),
                            &format!("intermediate{}", subexpressions_a.len() - 1),
                        );
                    }
                }
                // Re-reference b inputs
                for (sub_factor, index) in a_to_b_indexes.iter().enumerate() {
                    for i in (*index - sub_factor + 1)..(b_inputs.len() + a_to_b_indexes.len()) {
                        for (exp, _) in subexpressions_b.iter_mut() {
                            *exp = exp.replace(&format!("input{i}"), &format!("input{}", i - 1));
                        }
                    }
                }
                // Combine inputs for a and b
                for i in (0..a_inputs.len()).rev() {
                    // Re-reference the a inputs
                    for (exp, _) in subexpressions_a.iter_mut() {
                        *exp = exp.replace(
                            &format!("input{i}"),
                            &format!("input{}", i + b_inputs.len()),
                        );
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
                        kernel: None,
                        dyn_map: &graph.dyn_map,
                        dyn_chars: vec![],
                        subexpressions: subexpressions_b,
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
                if !graph.contains_node(a) {
                    remap(a, new_op, &mut ids, graph);
                }
                remap(b, new_op, &mut ids, graph);
            }
        }
        // Compile all the kernels we placed
        let type_name = T::type_name();
        for fused_op in fused_ops {
            let inputs = graph
                .edges_directed(fused_op, Direction::Incoming)
                .flat_map(|e| e.weight().as_data())
                .sorted_by_key(|(i, _, _)| *i)
                .map(|(_, _, sh)| sh)
                .collect::<Vec<_>>();
            let op = graph.get_op_mut::<FusedElementwiseOp<T>>(fused_op);
            // Stack index expressions and replace them in the subexpressions
            let intermediate_match = Regex::new(r"intermediate(\d+)").unwrap();
            // Track all shapes used, will pull dyn dims from these
            let shapes_used = op
                .subexpressions
                .iter()
                .map(|(_, s)| *s)
                .chain(inputs.clone())
                .collect::<Vec<_>>();
            // Track the views of each subexpression by going in reverse order and appending the current subexpression's views to the referenced subexpression
            let mut subexp_views = op
                .subexpressions
                .iter()
                .map(|(_, sh)| vec![*sh]) // Start with the current view for this subexpression
                .collect::<Vec<_>>();
            for i in (0..subexp_views.len() - 1).rev() {
                for capture in intermediate_match.captures_iter(&op.subexpressions[i].0) {
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
            let stacked_shapes: Vec<Vec<ShapeTracker>> = inputs
                .iter()
                .enumerate()
                .map(|(i, s)| {
                    // Find the first subexpression that uses this input
                    let using_subexp = op
                        .subexpressions
                        .iter()
                        .position(|(s, _)| s.contains(&format!("input{i}")))
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
            let stacked_index_expressions = stacked_shapes
                .iter()
                .map(|s| {
                    s.iter().rev().fold(BigExpression::from('z'), |acc, inp| {
                        inp.index_expression().substitute('z', acc)
                    })
                })
                .collect::<Vec<_>>();

            // Replace in subexpressions
            let n_subexpressions = op.subexpressions.len();
            for (i, ((subexp, _), stacked_shapes)) in
                op.subexpressions.iter_mut().zip(subexp_views).enumerate()
            {
                // Index
                for (i, ind_exp) in stacked_index_expressions.iter().enumerate() {
                    *subexp = subexp.replace(
                        &format!("input{i}"),
                        &format!("(float)input{i}[{}]", expr_to_metal_string(ind_exp.clone())),
                    );
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
                        .0;
                    if val_exp != true.into() {
                        *subexp = format!(
                            "(({} != 0) ? {subexp} : 0.0)",
                            expr_to_metal_string(val_exp)
                        );
                    }
                }
            }

            let (dyn_chars, rendered) = render_dyn_dim_inputs(&shapes_used, inputs.len() + 2);
            let kernel = format!(
                    "
#include <metal_stdlib>
using namespace metal;
kernel void mkernel({} device {type_name} *out [[buffer({})]], device uint& n_elements [[buffer({})]], uint idx [[thread_position_in_grid]]{rendered}) {{
    if (idx < n_elements) {{
        {}
        out[idx] = ({type_name})({});
    }}
}}",
                    (0..inputs.len())
                        .map(|inp_ind| format!(
                            "device {type_name}* input{inp_ind} [[buffer({inp_ind})]],"
                        ))
                        .collect::<Vec<_>>()
                        .join(" "),
                    inputs.len(),
                    inputs.len() + 1,
                    op.subexpressions.iter().take(op.subexpressions.len() - 1).enumerate().map(|(i, (subexp, _))| format!("float intermediate{i} = {subexp};")).join("\n        "),
                    op.subexpressions.last().unwrap().0
                );
            op.kernel = Some(compile_function("mkernel", &kernel, &device));
            op.dyn_chars = dyn_chars;
        }
    }
}

#[derive(LuminalPrint, LuminalEqFalse, Clone)]
pub struct FusedElementwiseOp<T> {
    kernel: Option<ComputePipelineState>,
    dyn_map: *const FxHashMap<char, usize>,
    dyn_chars: Vec<char>,
    subexpressions: Vec<(String, ShapeTracker)>,
    queue: CommandQueue,
    device: Device,
    output_buffer_sizes: Vec<BigExpression>,
    _phantom: PhantomData<T>,
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
        shape::symbolic::{BigExpression, Expression},
        tests::{assert_close, random_vec, random_vec_rng},
    };
    use rand::{rngs::StdRng, SeedableRng};

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
            .pad::<R2<2, 5>, _, _>(&[(0, 1), (0, 0)])
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
}
