use std::{any::Any, marker::PhantomData, sync::Arc};

use luminal_cudarc::{
    cublas::{sys::cublasOperation_t::*, CudaBlas},
    driver::{CudaDevice, CudaFunction, DevicePtr, DevicePtrMut},
};
use rustc_hash::{FxHashMap, FxHashSet};

use crate::{
    compile_and_load_kernel, expr_to_cuda_string, get_idx_valid_exps,
    prim::{CudaMul, CudaSumReduce},
    render_dyn_dim_inputs, CudaData, CudaFloat,
};
use luminal::{
    op::{InputTensor, Operator},
    prelude::{
        petgraph::{
            visit::{EdgeRef, IntoEdgesDirected},
            Direction,
        },
        *,
    },
};

#[derive(Debug, Default)]
pub struct ElementwiseFusionCompiler<T>(PhantomData<T>);

impl<T: CudaFloat> Compiler for ElementwiseFusionCompiler<T> {
    fn compile<To: ToIdsMut>(&self, graph: &mut Graph, _: To) {
        let device = CudaDevice::new(0).unwrap();
        // Track fused ops to compile later
        let mut fused_ops = FxHashSet::default();

        let mut matched = true;
        while matched {
            matched = false;
            for edge in graph.edge_indices().collect::<Vec<_>>() {
                let Some((a, b)) = graph.edge_endpoints(edge) else {
                    continue;
                };
                if graph.edges_connecting(a, b).count() > 1 {
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
                    .edges_connecting(a, b)
                    .next()
                    .unwrap()
                    .weight()
                    .as_data()
                    .unwrap()
                    .0 as usize;
                let (connect_inp, (_, (_, _, sh))) = b_inputs.remove(a_to_b_index);
                let reshaped = !sh.is_contiguous() || sh.is_sliced() || sh.is_padded();
                let b_replacements = b_inputs
                    .iter()
                    .enumerate()
                    .skip(a_to_b_index)
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
                let expression_a = multi_replace(&expression_a, &a_replacements);
                let mut b_replacements = b_replacements
                    .into_iter()
                    .map(|(from, to)| (format!("input{from}"), format!("input{to}")))
                    .collect::<Vec<_>>();
                b_replacements.push((format!("input{a_to_b_index}"), expression_a));
                let equation = multi_replace(&expression_b, &b_replacements);
                // Delete old ops
                let b_outgoing = graph
                    .edges_directed(b, Direction::Outgoing)
                    .map(|e| (e.target(), *e.weight()))
                    .collect::<Vec<_>>();
                graph.remove_node(a);
                graph.remove_node(b);
                // Create new fused op
                let new_op = graph
                    .add_op(FusedElementwiseOp::<T> {
                        kernel: None,
                        dyn_map: &graph.dyn_map,
                        dyn_chars: vec![],
                        equation,
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
                fused_ops.insert(new_op);
            }
        }

        // Compile all fused ops
        let type_name = T::type_name();
        for fused_op in fused_ops {
            let edges = graph
                .edges_directed(fused_op, Direction::Incoming)
                .filter_map(|e| e.weight().as_data())
                .collect::<Vec<_>>();
            if let Some(op) = graph
                .node_weight_mut(fused_op)
                .unwrap()
                .as_any_mut()
                .downcast_mut::<FusedElementwiseOp<T>>()
            {
                let (dyn_chars, rendered) = render_dyn_dim_inputs(
                    &edges
                        .iter()
                        .map(|i| i.2)
                        .chain(op.input_views.iter().flatten().copied())
                        .collect::<Vec<_>>(),
                );
                for ((inp_ind, _, sh), views) in edges.iter().zip(op.input_views.iter()) {
                    let view = views
                        .iter()
                        .rev()
                        .skip(1)
                        .fold(views[0].index_expression(), |acc, i| {
                            acc.substitute('z', i.index_expression()).minimize()
                        });
                    let ind =
                        expr_to_cuda_string(sh.index_expression().substitute('z', view).minimize());
                    let (_, val) = get_idx_valid_exps(*sh);
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
#include \"cuda_fp16.h\"
extern \"C\" __global__ void kernel({} {type_name} *out, int n_elements{rendered}) {{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_elements) {{
        out[idx] = ({type_name})({});
    }}
}}",
                    edges
                        .iter()
                        .map(|(inp_ind, _, _)| format!("const {type_name}* input{inp_ind},"))
                        .collect::<Vec<_>>()
                        .join(" "),
                    op.equation
                );
                op.kernel = Some(compile_and_load_kernel(kernel, &device));
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

#[derive(LuminalPrint, Clone)]
pub struct FusedElementwiseOp<T> {
    kernel: Option<CudaFunction>,
    dyn_map: *const FxHashMap<char, usize>,
    dyn_chars: Vec<char>,
    equation: String,
    device: Arc<CudaDevice>,
    input_views: Vec<Vec<ShapeTracker>>,
    _phantom: PhantomData<T>,
}
impl<T> PartialEq for FusedElementwiseOp<T> {
    fn eq(&self, other: &Self) -> bool {
        self.equation == other.equation && self.input_views == other.input_views
    }
}

impl<T: CudaFloat> Operator for FusedElementwiseOp<T> {
    fn process(&mut self, inp: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        todo!()
    }

    fn custom(&mut self, key: &str, _: Box<dyn Any>) -> Option<Box<dyn Any>> {
        if key == "elementwise" {
            return Some(Box::new(self.equation.clone()));
        }
        None
    }
}
