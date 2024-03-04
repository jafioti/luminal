use crate::{
    op::{InputTensor, Mul, Operator, SumReduce},
    prelude::*,
};

pub type MatMulCompiler = (MatMul2DCompiler, BatchMatMul2DCompiler);

#[derive(Debug, Default)]
pub struct MatMul2DCompiler;

impl Compiler for MatMul2DCompiler {
    fn compile<T: ToIdsMut>(&self, graph: &mut Graph, mut remap: T) {
        // Look for the matmul pattern
        // Mul ([A, C(fake), B] | [A(fake), C, B]) -> SumReduce(2) -> [A, C]
        // Actually starts at [A,B] | [B, C]
        let mut mul = op::<Mul>();
        mul.shapes([['A', 'C', 'B'], ['A', 'C', 'B']]);
        mul.fakes([
            [Some(false), Some(true), Some(false)],
            [Some(true), Some(false), Some(false)],
        ]);
        let mut sum_reduce = unary::<SumReduce>(mul.clone());
        sum_reduce.check(|o, _| o.is_equal(&SumReduce(0)));
        let mut s = sum_reduce.clone().search(graph);
        while s.next_match() {
            if s.check_no_delete(&[sum_reduce.id]) {
                // The intermediate mul can't be deleted
                continue;
            }
            let (mul, sum_reduce) = (s.get(&mul), s.get(&sum_reduce));
            // Insert MatMul2D op
            let mut srcs = graph.get_sources(mul);
            // Undo expansions and permute
            srcs[0].2.remove_dim(1);
            srcs[1].2.remove_dim(0);
            srcs[1].2.permute(&[1, 0]);
            let new_op = graph
                .add_op(MatMul2D)
                .input(srcs[0].0, 0, srcs[0].2)
                .input(srcs[1].0, 0, srcs[1].2)
                .finish();

            // Create edges to dests
            move_outgoing_edge(sum_reduce, new_op, &mut graph.graph);
            move_references(
                &mut remap,
                &mut graph.no_delete,
                &mut graph.to_retrieve,
                sum_reduce,
                new_op,
            );
            move_references(
                &mut remap,
                &mut graph.no_delete,
                &mut graph.to_retrieve,
                mul,
                new_op,
            );

            // Remove the old ops
            graph.graph.remove_node(sum_reduce);
            graph.safe_remove_node(mul, 0);
        }
    }
}

#[derive(Debug, PartialEq)]
pub struct MatMul2D;

impl Operator for MatMul2D {
    fn process(&mut self, inp: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        let (a_shape, b_shape) = (inp[0].1.shape(), inp[1].1.shape());
        let (a_strides, b_strides) = (inp[0].1.strides(), inp[1].1.strides());
        let a_data = inp[0]
            .0
            .borrowed()
            .data
            .as_any()
            .downcast_ref::<Vec<f32>>()
            .unwrap();
        let b_data = inp[1]
            .0
            .borrowed()
            .data
            .as_any()
            .downcast_ref::<Vec<f32>>()
            .unwrap();
        let mut c = vec![0.; a_shape[0].to_usize().unwrap() * b_shape[1].to_usize().unwrap()];
        unsafe {
            matrixmultiply::sgemm(
                a_shape[0].to_usize().unwrap(),
                a_shape[1].to_usize().unwrap(),
                b_shape[1].to_usize().unwrap(),
                1.0,
                a_data.as_ptr(),
                a_strides[0].to_usize().unwrap() as isize,
                a_strides[1].to_usize().unwrap() as isize,
                b_data.as_ptr(),
                b_strides[0].to_usize().unwrap() as isize,
                b_strides[1].to_usize().unwrap() as isize,
                0.0,
                c.as_mut_ptr(),
                b_shape[1].to_usize().unwrap() as isize,
                1,
            );
        }

        vec![Tensor::new(c)]
    }
}

#[derive(Debug, Default)]
pub struct BatchMatMul2DCompiler;

impl Compiler for BatchMatMul2DCompiler {
    fn compile<T: ToIdsMut>(&self, graph: &mut Graph, mut remap: T) {
        // Look for the matmul pattern
        // Mul ([A, C(fake), B] | [A(fake), C, B]) -> SumReduce(2) -> [A, C]
        // Actually starts at [A,B] | [B, C]
        let mut mul = op::<Mul>();
        mul.shapes([['D', 'A', 'C', 'B'], ['D', 'A', 'C', 'B']]);
        mul.fakes([
            [Some(false), Some(false), Some(true), Some(false)],
            [Some(true), Some(true), Some(false), Some(false)],
        ]);
        let mut sum_reduce = unary::<SumReduce>(mul.clone());
        sum_reduce.check(|o, _| o.is_equal(&SumReduce(3)));
        let mut s = sum_reduce.clone().search(graph);
        while s.next_match() {
            if s.check_no_delete(&[sum_reduce.id]) {
                // The intermediate mul can't be deleted
                continue;
            }
            let (mul, sum_reduce) = (s.get(&mul), s.get(&sum_reduce));
            // Insert MatMul2D op
            let mut srcs = graph.get_sources(mul);
            // Undo expansions and permute
            srcs[0].2.remove_dim(2);
            srcs[1].2.remove_dim(1);
            srcs[1].2.remove_dim(0);
            srcs[1].2.permute(&[1, 0]);
            let new_op = graph
                .add_op(BatchedMatMul2D)
                .input(srcs[0].0, 0, srcs[0].2)
                .input(srcs[1].0, 0, srcs[1].2)
                .finish();

            // Create edges to dests
            move_outgoing_edge(sum_reduce, new_op, &mut graph.graph);
            move_references(
                &mut remap,
                &mut graph.no_delete,
                &mut graph.to_retrieve,
                sum_reduce,
                new_op,
            );
            move_references(
                &mut remap,
                &mut graph.no_delete,
                &mut graph.to_retrieve,
                mul,
                new_op,
            );

            // Remove the old ops
            graph.graph.remove_node(mul);
            graph.graph.remove_node(sum_reduce);
        }
    }
}

#[derive(Debug, PartialEq)]
pub struct BatchedMatMul2D;

// ABCxCD -> ABD
impl Operator for BatchedMatMul2D {
    fn process(&mut self, inp: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        let (a_shape, b_shape) = (inp[0].1.shape(), inp[1].1.shape());
        let (a_strides, b_strides) = (inp[0].1.strides(), inp[1].1.strides());
        let a_data = inp[0]
            .0
            .borrowed()
            .data
            .as_any()
            .downcast_ref::<Vec<f32>>()
            .unwrap();
        let b_data = inp[1]
            .0
            .borrowed()
            .data
            .as_any()
            .downcast_ref::<Vec<f32>>()
            .unwrap();
        let mut c = vec![
            0.;
            a_shape[0].to_usize().unwrap()
                * a_shape[1].to_usize().unwrap()
                * b_shape[1].to_usize().unwrap()
        ];

        let mat_size = a_shape[1].to_usize().unwrap() * b_shape[1].to_usize().unwrap();
        for i in 0..a_shape[0].to_usize().unwrap() {
            unsafe {
                matrixmultiply::sgemm(
                    a_shape[1].to_usize().unwrap(),
                    a_shape[2].to_usize().unwrap(),
                    b_shape[1].to_usize().unwrap(),
                    1.0,
                    a_data.as_ptr().add(i * a_strides[0].to_usize().unwrap()),
                    a_strides[1].to_usize().unwrap() as isize,
                    a_strides[2].to_usize().unwrap() as isize,
                    b_data.as_ptr(),
                    b_strides[0].to_usize().unwrap() as isize,
                    b_strides[1].to_usize().unwrap() as isize,
                    0.0,
                    c.as_mut_ptr().add(i * mat_size),
                    b_shape[1].to_usize().unwrap() as isize,
                    1,
                );
            }
        }

        vec![Tensor::new(c)]
    }
}
