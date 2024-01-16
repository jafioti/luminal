use std::{any::Any, collections::HashMap, marker::PhantomData, sync::Arc};

use itertools::Itertools;
use metal_rs::{
    objc::rc::autoreleasepool, Buffer, CommandBufferRef, CommandQueue, ComputePassDescriptor,
    ComputePipelineState, Device, MTLResourceOptions,
};
use petgraph::{visit::EdgeRef, Direction};

use crate::{
    op::{InputTensor, Operator},
    prelude::{metal::get_buffer_from_tensor, *},
};

use self::symbolic::BigExpression;

use super::{
    compile_function, get_idx_valid_exps, input_dyn_dims, render_dyn_dim_inputs, DispatchNElements,
    SetInt,
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
        let mut fused_ops = vec![];

        while selector.next_match() {
            // More than one connecting edge
            if graph.no_delete.contains(&a)
                || graph
                    .graph
                    .edges_connecting(a, b)
                    .filter(|e| !e.weight().is_schedule())
                    .count()
                    > 1
            {
                continue;
            }
            // Connecting shape isn't contiguous
            let (to_input, _, connecting_shape) = graph
                .graph
                .edges_connecting(a, b)
                .find_map(|e| e.weight().as_data())
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
            let mut n_edges = graph
                .graph
                .edges_directed(a, Direction::Incoming)
                .filter(|e| !e.weight().is_schedule())
                .count() as u8;
            // Adjust variables in a_equation to the new inputs
            for input_edge in graph
                .graph
                .edges_directed(a, Direction::Incoming)
                .filter_map(|e| e.weight().as_data().map(|(a, b, c)| (e.source(), a, b, c)))
                .collect_vec()
            {
                // Find edge or add it
                if let Some(n) = graph
                    .graph
                    .edges_directed(b, Direction::Incoming)
                    .filter_map(|e| e.weight().as_data().map(|(a, b, c)| (e.source(), a, b, c)))
                    .find(|(src, inp_ind, _, _)| *src == input_edge.0 && *inp_ind == input_edge.2)
                {
                    a_equation = a_equation
                        .replace(&format!("input{}", input_edge.1), &format!("input{}", n.1));
                } else {
                    graph.graph.add_edge(
                        input_edge.0,
                        b,
                        Dependency::Data {
                            input_order: n_edges,
                            output_order: input_edge.2,
                            shape: input_edge.3,
                        },
                    );
                    a_equation = a_equation.replace(
                        &format!("input{}", input_edge.1),
                        &format!("input{}", n_edges),
                    );
                    n_edges += 1;
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
                fused_op.equation = fused_op
                    .equation
                    .replace(&format!("input{to_input}"), &format!("({a_equation})"));
                // Since we are removing the input from a, we must decrement all inputs larger than that
                for i in to_input + 1..n_edges {
                    fused_op.equation = fused_op
                        .equation
                        .replace(&format!("input{i}"), &format!("input{}", i - 1));
                }
            } else {
                let mut b_equation = graph
                    .node_custom::<String, _>(b, "elementwise", ())
                    .unwrap();
                b_equation =
                    b_equation.replace(&format!("input{to_input}"), &format!("({a_equation})"));
                // Since we are removing the input from a, we must decrement all inputs larger than that
                for i in to_input + 1..n_edges {
                    b_equation =
                        b_equation.replace(&format!("input{i}"), &format!("input{}", i - 1));
                }
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
            }
            // Remove a
            move_references(
                &mut remap,
                &mut graph.no_delete,
                &mut graph.to_retrieve,
                a,
                new_op,
            );
            graph.graph.remove_node(a);
            // Bring input indexes back in line
            for (i, e) in graph
                .graph
                .edges_directed(new_op, Direction::Incoming)
                .filter(|e| !e.weight().is_schedule())
                .sorted_by_key(|e| e.weight().as_data().unwrap().0)
                .map(|e| e.id())
                .enumerate()
                .collect_vec()
            {
                if let Dependency::Data { input_order, .. } =
                    graph.graph.edge_weight_mut(e).unwrap()
                {
                    *input_order = i as u8;
                }
            }
            fused_ops.push(new_op);
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
                let (dyn_chars, rendered) =
                    render_dyn_dim_inputs(&edges.iter().map(|i| i.2).collect_vec(), 0);
                for (inp_ind, _, sh) in &edges {
                    let (ind, val) = get_idx_valid_exps(*sh);
                    op.equation = op.equation.replace(
                        &format!("input{inp_ind}"),
                        &format!("({val} != 0) ? input{inp_ind}[{ind}] : 0.0"),
                    );
                }
                let kernel = format!(
                    "
#include <metal_stdlib>
using namespace metal;
kernel void mkernel({} device {type_name} *out [[buffer({})]], device uint& n_elements [[buffer({})]], uint idx [[thread_position_in_grid]]{rendered}) {{
    if (idx < n_elements) {{
        out[idx] = {};
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

#[derive(LuminalPrint, LuminalEq, Clone)]
pub struct FusedElementwiseOp<T> {
    kernel: Option<ComputePipelineState>,
    dyn_map: *const HashMap<char, usize>,
    dyn_chars: Vec<char>,
    equation: String,
    queue: CommandQueue,
    device: Device,
    _phantom: PhantomData<T>,
}
impl<T> MetalKernel for FusedElementwiseOp<T> {
    fn output_buffer_sizes(&self, input_shapes: &[ShapeTracker]) -> Vec<BigExpression> {
        vec![input_shapes[0].n_physical_elements() * std::mem::size_of::<T>()]
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
        let out_size = inputs[0].1.n_physical_elements().to_usize().unwrap();

        // Set function inputs
        for (i, (buf, _)) in inputs.iter().enumerate() {
            encoder.set_buffer(i as u64, Some(*buf), 0);
        }
        encoder.set_buffer(inputs.len() as u64, Some(output_buffers[0]), 0);
        encoder.set_u32(inputs.len() + 1, out_size as u32);
        input_dyn_dims(
            &self.dyn_chars,
            unsafe { self.dyn_map.as_ref().unwrap() },
            &encoder,
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
                    .map(|(t, s)| (get_buffer_from_tensor(t), *s))
                    .collect_vec(),
                command_buffer,
                &[],
                &[&out],
            );

            command_buffer.commit();
            command_buffer.wait_until_completed();

            vec![Tensor::new(out)]
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
    crate::test_imports!();
    #[test]
    fn test_fusion() {
        let mut cx = Graph::new();
        let a = cx.tensor::<R1<10>>().set(random_vec(10)).keep();
        let mut b = a.exp2().sin().retrieve();

        cx.execute();
        let unopt_b = b.data();
        b.drop();

        cx.compile(GenericCompiler::<MetalFp16Compiler>::default(), &mut b);
        cx.execute();

        assert_close(&b.data(), &unopt_b);
    }
}
