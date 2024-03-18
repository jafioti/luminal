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

use crate::{
    expr_to_metal_string, get_buffer_from_tensor, prim::MetalConstant, MetalBuffer, MetalFloat,
    MetalKernel, MetalKernelWrapper,
};

use self::symbolic::BigExpression;

use super::{compile_function, input_dyn_dims, render_dyn_dim_inputs, DispatchNElements, SetInt};

#[derive(Default, Debug)]
pub struct ElementwiseFusionCompiler<T>(PhantomData<T>);

fn get_inputs<T: MetalFloat>(
    node: NodeIndex,
    graph: &Graph,
) -> Vec<(Vec<ShapeTracker>, NodeIndex, u8, u8, ShapeTracker)> {
    graph
        .try_get_op::<FusedElementwiseOp<T>>(node)
        .map(|n| n.input_views.clone())
        .unwrap_or_else(|| vec![vec![]; graph.edges_directed(node, Direction::Incoming).count()])
        .into_iter()
        .zip(
            graph
                .edges_directed(node, Direction::Incoming)
                .filter_map(|e| e.weight().as_data().map(|i| (e.source(), i)))
                .sorted_by_key(|(_, i)| i.0),
        )
        .map(|(a, (b, (c, d, e)))| (a, b, c, d, e))
        .collect()
}

impl<T: MetalFloat> Compiler for ElementwiseFusionCompiler<T> {
    type Output = ();
    fn compile<To: ToIdsMut>(&self, graph: &mut Graph, mut ids: To) {
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
                        && graph.edges_directed(a, Direction::Outgoing).count() > 1)
                {
                    continue; // A is not a constant and is feeding into some other node
                }
                let (Some(expression_a), Some(expression_b)) = (
                    graph.node_custom::<String, _>(a, "elementwise", Box::<()>::default()),
                    graph.node_custom::<String, _>(b, "elementwise", Box::<()>::default()),
                ) else {
                    continue;
                };
                // a and b are elementwise ops
                matched = true;
                let a_to_b_index = graph
                    .edges_connecting(a, b)
                    .next()
                    .map(|e| e.weight().as_data().unwrap().0 as usize)
                    .unwrap();
                // Combine inputs for a and b
                let a_inputs = get_inputs::<T>(a, graph);
                let mut b_inputs = get_inputs::<T>(b, graph);
                let (connect_inp, _, _, _, sh) = b_inputs.remove(a_to_b_index);
                let reshaped = sh.is_reshaped();
                let mut add_index = a_to_b_index;
                let mut a_replacements = vec![];
                let orig_b_inputs = b_inputs.clone();
                for (mut views, src, inp, out, shape) in a_inputs {
                    if reshaped {
                        views.push(sh);
                    }
                    views.extend(connect_inp.iter().copied());
                    a_replacements.push((inp, add_index));
                    b_inputs.insert(add_index, (views, src, add_index as u8, out, shape));
                    add_index += 1;
                }
                // Combine expressions together to get final expression
                let a_replacements = a_replacements
                    .into_iter()
                    .map(|(from, to)| (format!("input{from}"), format!("input{to}")))
                    .collect::<Vec<_>>();
                let expression_a = multi_replace(&expression_a, &a_replacements);
                let added_inputs = graph.edges_directed(a, Direction::Incoming).count();
                let mut b_replacements = orig_b_inputs
                    .iter()
                    .enumerate()
                    .skip(a_to_b_index)
                    .map(|(i, _)| {
                        (
                            format!("input{}", i + 1),
                            format!("input{}", i + added_inputs),
                        )
                    })
                    .collect::<Vec<_>>();
                b_replacements.push((format!("input{a_to_b_index}"), format!("({expression_a})")));
                let equation = multi_replace(&expression_b, &b_replacements);

                // Create new fused op
                let new_op = graph
                    .add_op(FusedElementwiseOp::<T> {
                        kernel: None,
                        dyn_map: &graph.dyn_map,
                        dyn_chars: vec![],
                        equation,
                        queue: queue.clone(),
                        device: device.clone(),
                        input_views: b_inputs.iter().map(|(v, _, _, _, _)| v.clone()).collect(),
                        output_buffer_sizes: graph
                            .edges_directed(b, Direction::Outgoing)
                            .filter_map(|e| e.weight().as_data())
                            .sorted_by_key(|(i, _, _)| *i)
                            .map(|(_, _, sh)| sh.n_physical_elements())
                            .collect(),
                        _phantom: Default::default(),
                    })
                    .finish();
                // Add edges to new op
                move_outgoing_edge(b, new_op, graph);
                for (i, (_, node, _, output, shape)) in b_inputs.into_iter().enumerate() {
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
            let edges = graph
                .edges_directed(fused_op, Direction::Incoming)
                .flat_map(|e| e.weight().as_data())
                .sorted_by_key(|(i, _, _)| *i)
                .map(|(_, _, sh)| sh)
                .collect::<Vec<_>>();
            let op = graph.get_op_mut::<FusedElementwiseOp<T>>(fused_op);
            for (inp_ind, (sh, input_views)) in edges.iter().zip(&op.input_views).enumerate() {
                // Stack views in reverse order
                let mut val_exp = BigExpression::from(true);
                let mut ind_exp = BigExpression::from('z');
                for v in input_views.iter().rev() {
                    val_exp = val_exp & v.valid_expression().substitute('z', ind_exp.clone());
                    ind_exp = v.index_expression().substitute('z', ind_exp);
                }
                // Stack the final view on
                val_exp = val_exp & sh.valid_expression().substitute('z', ind_exp.clone());
                ind_exp = sh.index_expression().substitute('z', ind_exp);
                if val_exp == true.into() {
                    op.equation = op.equation.replace(
                        &format!("input{inp_ind}"),
                        &format!("(float)input{inp_ind}[{}]", expr_to_metal_string(ind_exp)),
                    );
                } else {
                    op.equation = op.equation.replace(
                        &format!("input{inp_ind}"),
                        &format!(
                            "(({} != 0) ? (float)input{inp_ind}[{}] : 0.0)",
                            expr_to_metal_string(val_exp),
                            expr_to_metal_string(ind_exp)
                        ),
                    );
                }
            }
            let (dyn_chars, rendered) = render_dyn_dim_inputs(&edges, edges.len() + 2);
            let kernel = format!(
                    "
#include <metal_stdlib>
using namespace metal;
kernel void mkernel({} device {type_name} *out [[buffer({})]], device uint& n_elements [[buffer({})]], uint idx [[thread_position_in_grid]]{rendered}) {{
    if (idx < n_elements) {{
        out[idx] = ({type_name})({});
    }}
}}",
                    (0..edges.len())
                        .map(|inp_ind| format!(
                            "device {type_name}* input{inp_ind} [[buffer({inp_ind})]],"
                        ))
                        .collect::<Vec<_>>()
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
    output_buffer_sizes: Vec<BigExpression>,
    input_views: Vec<Vec<ShapeTracker>>,
    _phantom: PhantomData<T>,
}
impl<T> MetalKernel for FusedElementwiseOp<T> {
    fn output_buffer_sizes(&self, _: &[ShapeTracker]) -> Vec<BigExpression> {
        self.output_buffer_sizes
            .iter()
            .map(|e| e.clone() * std::mem::size_of::<T>())
            .collect()
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
        let out_size = self.output_buffer_sizes[0].exec(dyn_map).unwrap();

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
                (self.output_buffer_sizes[0]
                    .exec(unsafe { self.dyn_map.as_ref().unwrap() })
                    .unwrap()
                    * std::mem::size_of::<T>()) as u64,
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
            return Some(Box::new(self.equation.clone()));
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

    #[test]
    fn test_fusion_slicing_padding() {
        let mut cx = Graph::new();
        let a = cx.named_tensor::<R2<2, 5>>("a").set(random_vec(10)).keep();
        let sliced: GraphTensor<(Const<1>, Const<5>)> =
            a.slice((..Expression::from(1), ..)).realize();
        let exp = sliced.exp2();
        let exp = exp.contiguous().sin();
        let mut padded = exp
            .pad::<R2<2, 5>, _, _>(&[(0, 1), (0, 0)])
            .log2()
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
        let a = cx
            .named_tensor::<R2<2, 5>>("a")
            .set(random_vec_rng(10, &mut rng))
            .keep();
        let sqrt = a.sqrt();
        let mut out = (sqrt.exp() + sqrt.sin()).retrieve();
        cx.execute();
        let unopt_out = out.data();
        out.drop();

        cx.compile(<(GenericCompiler, MetalCompiler<f16>)>::default(), &mut out);
        cx.execute();

        assert_close(&out.data(), &unopt_out);
    }

    #[test]
    fn test_fusion_rope() {
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
