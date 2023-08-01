use itertools::Itertools;

use crate::prelude::*;

use crate::op;
use std::ops::{Add, Div, Mul, Neg, Rem, Sub};

// The high level interface implemented on GraphTensor. All of these ops get translated to primitive ops.

// Unary ops
impl<S: Shape> GraphTensor<S> {
    /// Base 2 log
    pub fn log_2(self) -> GraphTensor<S> {
        let graph = unsafe { self.graph_ref.as_mut().unwrap() };
        let shape = self.shape_tracker();
        let new_id = graph
            .add_op(op::Log2, shape.clone())
            .input(self.id)
            .finish();
        GraphTensor::from_id(new_id, self.graph_ref)
    }

    /// Base 2 exp
    pub fn exp_2(self) -> GraphTensor<S> {
        let graph = unsafe { self.graph_ref.as_mut().unwrap() };
        let shape = self.shape_tracker();
        let new_id = graph
            .add_op(op::Exp2, shape.clone())
            .input(self.id)
            .finish();
        GraphTensor::from_id(new_id, self.graph_ref)
    }

    /// Natural exp
    pub fn exp(self) -> GraphTensor<S> {
        (self * (1.0 / f32::ln(2.))).exp_2()
    }

    /// Natural log
    pub fn log(self) -> GraphTensor<S> {
        self.log_2() * f32::ln(2.)
    }

    pub fn recip(self) -> GraphTensor<S> {
        let graph = unsafe { self.graph_ref.as_mut().unwrap() };
        let shape = self.shape_tracker();
        let new_id = graph
            .add_op(op::Recip, shape.clone())
            .input(self.id)
            .finish();
        GraphTensor::from_id(new_id, self.graph_ref)
    }

    pub fn sin(self) -> GraphTensor<S> {
        let graph = unsafe { self.graph_ref.as_mut().unwrap() };
        let shape = self.shape_tracker();
        let new_id = graph.add_op(op::Sin, shape.clone()).input(self.id).finish();
        GraphTensor::from_id(new_id, self.graph_ref)
    }

    pub fn sqrt(self) -> GraphTensor<S> {
        let graph = unsafe { self.graph_ref.as_mut().unwrap() };
        let shape = self.shape_tracker();
        let new_id = graph
            .add_op(op::Sqrt, shape.clone())
            .input(self.id)
            .finish();
        GraphTensor::from_id(new_id, self.graph_ref)
    }

    pub fn layer_norm<const DIM: isize>(self) -> GraphTensor<S>
    where
        <S as ReduceShape<Axis<DIM>>>::Reduced: Shape,
        S: ReduceShape<Axis<DIM>>,
    {
        let mean = self
            .mean_reduce::<<S as ReduceShape<Axis<DIM>>>::Reduced, _>()
            .expand();
        let centered = self - mean;
        let std = centered
            .mul(centered)
            .mean_reduce::<<S as ReduceShape<Axis<DIM>>>::Reduced, _>()
            .expand()
            .add(1e-5)
            .sqrt();
        centered / std
    }

    pub fn softmax<const DIM: isize>(self) -> GraphTensor<S>
    where
        <S as ReduceShape<Axis<DIM>>>::Reduced: Shape,
        S: ReduceShape<Axis<DIM>>,
    {
        let m = self
            - self
                .max_reduce::<<S as ReduceShape<Axis<DIM>>>::Reduced, _>()
                .expand();
        let exp = m.exp();
        exp / exp
            .sum_reduce::<<S as ReduceShape<Axis<DIM>>>::Reduced, _>()
            .expand()
    }

    pub fn abs(self) -> GraphTensor<S> {
        self.relu() + (-self).relu()
    }

    pub fn sign(self) -> GraphTensor<S> {
        self / (self.abs() + 1e-10)
    }

    // Approxamate, see full impl here: https://github.com/tinygrad/tinygrad/blob/a32c67760140dd26b60d7932268f2e62e96a66e0/tinygrad/tensor.py#L568
    pub fn pow(self, e: f32) -> GraphTensor<S> {
        self.abs().log().mul(e).exp()
    }

    /// Take a slice of the original tensor. Any dimension with bounds becomes a dynamic dimension
    pub fn slice<Slice: SliceOfShape<S>>(self, slice: Slice) -> GraphTensor<Slice::OutputShape> {
        let slice = slice.to_range_vec();
        let graph = unsafe { self.graph_ref.as_mut().unwrap() };
        let new_id = graph
            .add_op(op::Slice(slice), Slice::OutputShape::realized_shape())
            .input(self.id)
            .finish();
        GraphTensor::from_id(new_id, self.graph_ref)
    }

    // pub fn pad<Slice: SliceOfShape<S>>(self, slice: Slice) -> GraphTensor<Slice::OutputShape> {
    //     let slice = slice.to_range_vec();
    //     let graph = unsafe { self.graph_ref.as_mut().unwrap() };
    //     let new_id = graph
    //         .add_op(op::Slice(slice), Slice::OutputShape::realized_shape())
    //         .input(self.id)
    //         .finish();
    //     GraphTensor::from_id(new_id, self.graph_ref)
    // }
}

// Reduction ops
impl<S: Shape> GraphTensor<S> {
    pub fn sum_reduce<Dst: Shape, Ax: Axes>(self) -> GraphTensor<Dst>
    where
        S: HasAxes<Ax> + ReduceShapeTo<Dst, Ax>,
    {
        let graph = unsafe { self.graph_ref.as_mut().unwrap() };
        let mut shape = self.shape_tracker().clone();

        let mut new_id = self.id;
        for dim in Ax::as_array().into_iter().collect_vec().into_iter().rev() {
            shape.remove(dim as usize);
            new_id = graph
                .add_op(op::SumReduce(dim as usize), shape.clone())
                .input(new_id)
                .finish();
        }
        GraphTensor::from_id(new_id, self.graph_ref)
    }

    pub fn max_reduce<Dst: Shape, Ax: Axes>(self) -> GraphTensor<Dst>
    where
        S: HasAxes<Ax> + ReduceShapeTo<Dst, Ax>,
    {
        let graph = unsafe { self.graph_ref.as_mut().unwrap() };
        let mut shape = self.shape_tracker().clone();

        let mut new_id = self.id;
        for dim in Ax::as_array().into_iter().collect_vec().into_iter().rev() {
            shape.remove(dim as usize);
            new_id = graph
                .add_op(op::MaxReduce(dim as usize), shape.clone())
                .input(new_id)
                .finish();
        }
        GraphTensor::from_id(new_id, self.graph_ref)
    }

    pub fn mean_reduce<Dst: Shape, Ax: Axes>(self) -> GraphTensor<Dst>
    where
        S: HasAxes<Ax> + ReduceShapeTo<Dst, Ax>,
    {
        let graph = unsafe { self.graph_ref.as_mut().unwrap() };
        let mut shape = <S as Shape>::realized_shape();

        let mut node_id = self.id;
        for dim in Ax::as_array().into_iter().collect_vec().into_iter().rev() {
            // Reduce shape
            shape.remove(dim as usize);
            // Sum reduce
            node_id = graph
                .add_op(op::SumReduce(dim as usize), shape.clone())
                .input(node_id)
                .finish();
            // Create div tensor
            let size_t = graph
                .add_op(
                    op::Function(Box::new(move |inp, i| {
                        let s = inp[0].1.shape.shape()[dim as usize];
                        (
                            Some(Tensor {
                                data: Box::new(vec![s as f32]),
                            }),
                            TensorView {
                                tensor_id: i,
                                shape: ShapeTracker::new(vec![]),
                            },
                        )
                    })),
                    vec![],
                )
                .input(self.id)
                .finish();
            let size_t: GraphTensor<R0> = GraphTensor::from_id(size_t, graph);
            let mut size_t = size_t.id;
            let mut size_t_shape = vec![];
            // Expand div tensor
            for (dim, size) in shape.iter().enumerate() {
                size_t_shape.insert(dim, *size);
                size_t = graph
                    .add_op(op::Expand(dim, *size), size_t_shape.clone())
                    .input(size_t)
                    .finish();
            }
            // Divide by div tensor
            node_id = graph
                .add_op(op::Div, shape.clone())
                .input(node_id)
                .input(size_t)
                .finish();
        }
        GraphTensor::from_id(node_id, self.graph_ref)
    }
}

// Activation functions
impl<S: Shape> GraphTensor<S> {
    /// The Rectified Linear Unit activation function
    pub fn relu(self) -> GraphTensor<S> {
        self.max_f32(0.)
    }

    /// The sigmoid activation function
    pub fn sigmoid(self) -> GraphTensor<S> {
        // Based on https://github.com/tinygrad/tinygrad/blob/9d142430cbe61121c864c0015f1de83c94a7d2c0/tinygrad/mlops.py#L70
        let graph = unsafe { self.graph_ref.as_mut().unwrap() };
        let one = graph.new_tensor::<R0>();
        one.set(vec![1.]);
        one.expand() / (one.expand() + (self * (-1. / 2_f32.ln())).exp_2())
    }

    /// The swish activation function
    pub fn swish(self) -> GraphTensor<S> {
        self * self.sigmoid()
    }

    /// The tanh activation function
    pub fn tanh(self) -> GraphTensor<S> {
        (self * 2.0).sigmoid() * 2.0 - 1.0
    }

    /// The leaky relu activation function
    pub fn leaky_relu(self, neg_slope: f32) -> GraphTensor<S> {
        self.relu() - (self * -neg_slope).relu()
    }
}

// Clipping ops (min, max, clip)
impl<S: Shape> GraphTensor<S> {
    /// Take the elementwise maximum of two tensors
    pub fn max(self, rhs: GraphTensor<S>) -> GraphTensor<S> {
        let graph = unsafe { self.graph_ref.as_mut().unwrap() };
        let new_id = graph
            .add_op(op::Max, self.shape_tracker().clone())
            .input(self.id)
            .input(rhs.id)
            .finish();
        GraphTensor::from_id(new_id, self.graph_ref)
    }

    /// Take the elementwise maximum of a tensor and a float
    pub fn max_f32(self, rhs: f32) -> GraphTensor<S> {
        let graph = unsafe { self.graph_ref.as_mut().unwrap() };
        let rhs_t = graph.new_tensor::<R0>();
        rhs_t.set(vec![rhs]);
        self.max(rhs_t.expand())
    }

    /// Take the elementwise minimum of two tensors
    pub fn min(self, rhs: GraphTensor<S>) -> GraphTensor<S> {
        -(-self).max(-rhs)
    }

    /// Take the elementwise minimum of a tensor and a float
    pub fn min_f32(self, rhs: f32) -> GraphTensor<S> {
        -(-self).max_f32(-rhs)
    }

    /// Clip a tensor in a range
    pub fn clip(self, min: f32, max: f32) -> GraphTensor<S> {
        self.min_f32(min).max_f32(max)
    }
}

// Movement ops
impl<S: Shape> GraphTensor<S> {
    pub fn permute<Dst: Shape, Ax: Axes>(self) -> GraphTensor<Dst>
    where
        S: PermuteShapeTo<Dst, Ax>,
    {
        let graph = unsafe { self.graph_ref.as_mut().unwrap() };
        let new_id = graph
            .add_op(
                op::Permute(Ax::as_array().into_iter().map(|i| i as usize).collect_vec()),
                <Dst as Shape>::realized_shape(),
            )
            .input(self.id)
            .finish();
        GraphTensor::from_id(new_id, self.graph_ref)
    }

    pub fn expand<Dst: Shape, Ax: Axes>(self) -> GraphTensor<Dst>
    where
        S: BroadcastShapeTo<Dst, Ax>,
    {
        let graph = unsafe { self.graph_ref.as_mut().unwrap() };
        let new_shape = <Dst as Shape>::realized_shape();
        let mut shape = self.shape_tracker().clone();

        let mut new_id = self.id;
        for (dim, size) in Ax::as_array()
            .into_iter()
            .map(|i| (i as usize, new_shape[i as usize]))
        {
            shape.insert(dim, size);
            new_id = graph
                .add_op(op::Expand(dim, size), shape.clone())
                .input(new_id)
                .finish();
        }
        GraphTensor::from_id(new_id, self.graph_ref)
    }

    pub fn reshape<N: Shape>(self) -> GraphTensor<N> {
        // <S as AssertSameNumel<N>>::assert_same_numel();
        let graph = unsafe { self.graph_ref.as_mut().unwrap() };
        let new_id = graph
            .add_op(
                op::Reshape(<N as Shape>::realized_shape()),
                <N as Shape>::realized_shape(),
            )
            .input(self.id)
            .finish();
        GraphTensor::from_id(new_id, self.graph_ref)
    }

    /// Dynamically reshape. Panics if destination shape doesn't match given shape.
    pub fn dyn_reshape<N: Shape>(self, shape: Vec<RealDim>) -> GraphTensor<N> {
        for (a, b) in N::realized_shape().iter().zip(shape.iter()) {
            match a {
                RealDim::Const(n) => assert_eq!(RealDim::Const(*n), *b),
                RealDim::Dyn => {}
            }
        }
        let graph = unsafe { self.graph_ref.as_mut().unwrap() };
        let new_id = graph
            .add_op(op::Reshape(shape.clone()), shape)
            .input(self.id)
            .finish();
        GraphTensor::from_id(new_id, self.graph_ref)
    }
}

// Matmuls

// ABxBC -> AC
impl<A: Dim, B: Dim> GraphTensor<(A, B)> {
    pub fn matmul<C: Dim>(self, rhs: GraphTensor<(B, C)>) -> GraphTensor<(A, C)> {
        // Reshape
        let w: GraphTensor<(C, B)> = rhs.permute::<_, Axes2<1, 0>>();

        // Broadcasted Multiply
        let mul = self.expand::<(A, C, B), _>() * w.expand::<(A, C, B), _>();

        // Sum Reduce
        mul.sum_reduce::<_, Axis<2>>()
    }
}

// AxAB -> B
impl<A: Dim> GraphTensor<(A,)> {
    pub fn matmul<B: Dim>(self, rhs: GraphTensor<(A, B)>) -> GraphTensor<(B,)> {
        let s: GraphTensor<(Const<1>, A)> = self.expand();

        // Run normal matmul
        let r = s.matmul(rhs);

        // Sum Reduce
        r.sum_reduce::<_, Axis<0>>()
    }
}

// ABCxCD -> ABD
impl<A: Dim, B: Dim, C: Dim> GraphTensor<(A, B, C)> {
    pub fn matmul<D: Dim>(self, rhs: GraphTensor<(C, D)>) -> GraphTensor<(A, B, D)> {
        // Reshape
        let w: GraphTensor<(D, C)> = rhs.permute::<_, Axes2<1, 0>>();

        // Broadcasted Multiply
        let mul = self.expand::<(A, B, D, C), _>() * w.expand::<(A, B, D, C), _>();

        // Sum Reduce
        mul.sum_reduce::<_, Axis<3>>()
    }
}

// ABCxACD -> ABD
impl<A: Dim, B: Dim, C: Dim> GraphTensor<(A, B, C)> {
    pub fn batch_matmul<D: Dim>(self, rhs: GraphTensor<(A, C, D)>) -> GraphTensor<(A, B, D)> {
        // Reshape
        let w: GraphTensor<(A, D, C)> = rhs.permute::<_, Axes3<0, 2, 1>>();

        // Broadcasted Multiply
        let mul = self.expand::<(A, B, D, C), _>() * w.expand::<(A, B, D, C), _>();

        // Sum Reduce
        mul.sum_reduce::<_, Axis<3>>()
    }
}

// ABCDxABDE -> ABCE
impl<A: Dim, B: Dim, C: Dim, D: Dim> GraphTensor<(A, B, C, D)> {
    pub fn batch_matmul<E: Dim>(self, rhs: GraphTensor<(A, B, D, E)>) -> GraphTensor<(A, B, C, E)> {
        // Reshape
        let w: GraphTensor<(A, B, E, D)> = rhs.permute::<_, Axes4<0, 1, 3, 2>>();

        // Broadcasted Multiply
        let mul = self.expand::<(A, B, C, E, D), _>() * w.expand::<(A, B, C, E, D), _>();

        // Sum Reduce
        mul.sum_reduce::<_, Axis<4>>()
    }
}

impl<A: Dim> GraphTensor<(A,)> {
    /// Simple dot product of two vectors
    pub fn dot(self, rhs: GraphTensor<(A,)>) -> GraphTensor<R0> {
        (self * rhs).sum_reduce()
    }
}

impl<S: Shape> Add<GraphTensor<S>> for GraphTensor<S> {
    type Output = GraphTensor<S>;

    fn add(self, rhs: GraphTensor<S>) -> Self::Output {
        let graph = unsafe { self.graph_ref.as_mut().unwrap() };
        let new_id = graph
            .add_op(op::Add, self.shape_tracker().clone())
            .input(self.id)
            .input(rhs.id)
            .finish();
        GraphTensor::from_id(new_id, self.graph_ref)
    }
}

impl<S: Shape> Sub<GraphTensor<S>> for GraphTensor<S> {
    type Output = GraphTensor<S>;

    fn sub(self, rhs: GraphTensor<S>) -> Self::Output {
        let graph = unsafe { self.graph_ref.as_mut().unwrap() };
        let new_id = graph
            .add_op(op::Sub, self.shape_tracker().clone())
            .input(self.id)
            .input(rhs.id)
            .finish();
        GraphTensor::from_id(new_id, self.graph_ref)
    }
}

impl<S: Shape> Mul<GraphTensor<S>> for GraphTensor<S> {
    type Output = GraphTensor<S>;

    fn mul(self, rhs: GraphTensor<S>) -> Self::Output {
        let graph = unsafe { self.graph_ref.as_mut().unwrap() };
        let new_id = graph
            .add_op(op::Mul, self.shape_tracker().clone())
            .input(self.id)
            .input(rhs.id)
            .finish();
        GraphTensor::from_id(new_id, self.graph_ref)
    }
}

impl<S: Shape> Div<GraphTensor<S>> for GraphTensor<S> {
    type Output = GraphTensor<S>;

    fn div(self, rhs: GraphTensor<S>) -> Self::Output {
        let graph = unsafe { self.graph_ref.as_mut().unwrap() };
        let new_id = graph
            .add_op(op::Div, self.shape_tracker().clone())
            .input(self.id)
            .input(rhs.id)
            .finish();
        GraphTensor::from_id(new_id, self.graph_ref)
    }
}

impl<S: Shape> Rem<GraphTensor<S>> for GraphTensor<S> {
    type Output = GraphTensor<S>;

    fn rem(self, rhs: GraphTensor<S>) -> Self::Output {
        let graph = unsafe { self.graph_ref.as_mut().unwrap() };
        let new_id = graph
            .add_op(op::Mod, self.shape_tracker().clone())
            .input(self.id)
            .input(rhs.id)
            .finish();
        GraphTensor::from_id(new_id, self.graph_ref)
    }
}

impl<S: Shape> Neg for GraphTensor<S> {
    type Output = GraphTensor<S>;

    fn neg(self) -> Self::Output {
        self * -1.0
    }
}

impl<S: Shape> Add<f32> for GraphTensor<S> {
    type Output = GraphTensor<S>;

    fn add(self, rhs: f32) -> Self::Output {
        let graph = unsafe { self.graph_ref.as_mut().unwrap() };
        let rhs_t = graph.new_tensor::<R0>();
        rhs_t.set(vec![rhs]);
        self + rhs_t.expand()
    }
}

impl<S: Shape> Sub<f32> for GraphTensor<S> {
    type Output = GraphTensor<S>;

    fn sub(self, rhs: f32) -> Self::Output {
        let graph = unsafe { self.graph_ref.as_mut().unwrap() };
        let rhs_t = graph.new_tensor::<R0>();
        rhs_t.set(vec![rhs]);
        self - rhs_t.expand()
    }
}

impl<S: Shape> Mul<f32> for GraphTensor<S> {
    type Output = GraphTensor<S>;

    fn mul(self, rhs: f32) -> Self::Output {
        let graph = unsafe { self.graph_ref.as_mut().unwrap() };
        let rhs_t = graph.new_tensor::<R0>();
        rhs_t.set(vec![rhs]);
        self * rhs_t.expand()
    }
}

impl<S: Shape> Div<f32> for GraphTensor<S> {
    type Output = GraphTensor<S>;

    fn div(self, rhs: f32) -> Self::Output {
        let graph = unsafe { self.graph_ref.as_mut().unwrap() };
        let rhs_t = graph.new_tensor::<R0>();
        rhs_t.set(vec![rhs]);
        self / rhs_t.expand()
    }
}

impl<S: Shape> Rem<f32> for GraphTensor<S> {
    type Output = GraphTensor<S>;

    fn rem(self, rhs: f32) -> Self::Output {
        let graph = unsafe { self.graph_ref.as_mut().unwrap() };
        let rhs_t = graph.new_tensor::<R0>();
        rhs_t.set(vec![rhs]);
        self % rhs_t.expand()
    }
}

#[cfg(test)]
mod tests {
    use crate::{prelude::*, tests::assert_close_data};
    use dfdx::prelude::*;

    #[test]
    fn test_layer_norm() {
        let mut cx = Graph::new();
        let a = cx.new_tensor::<R2<2, 3>>();
        a.set(vec![1., 2., 3., 3., 1., 3.]);
        let b = a.layer_norm::<0>();
        let c = a.layer_norm::<1>();
        b.mark();
        c.mark();
        cx.execute();

        let d_dev = Cpu::default();
        let d_a = d_dev.tensor([[1., 2., 3.], [3., 1., 3.]]);
        let d_b = d_a.clone().normalize::<dfdx::shapes::Axis<0>>(1e-5);
        let d_c = d_a.normalize::<dfdx::shapes::Axis<1>>(1e-5);

        assert_close_data(
            &b.retrieve().unwrap().real_data(b.view().unwrap()).unwrap(),
            &d_b.as_vec(),
        );
        assert_close_data(
            &c.retrieve().unwrap().real_data(c.view().unwrap()).unwrap(),
            &d_c.as_vec(),
        );
    }

    #[test]
    fn test_matrix_vector() {
        let mut cx = Graph::new();
        let a = cx.new_tensor::<R1<3>>();
        a.set(vec![1., 2., 3.]);
        let b = cx.new_tensor::<R2<3, 2>>();
        b.set(vec![1., 2., 3., 1., 2., 3.]);
        let c = a.matmul(b);
        cx.execute();

        let d_dev = Cpu::default();
        let d_a = d_dev.tensor([1., 2., 3.]);
        let d_b = d_dev.tensor([[1., 2.], [3., 1.], [2., 3.]]);
        let d_c = d_a.matmul(d_b);

        assert_close_data(
            &c.retrieve().unwrap().real_data(c.view().unwrap()).unwrap(),
            &d_c.as_vec(),
        );
    }

    #[test]
    fn test_matmul() {
        let mut cx = Graph::new();
        let a = cx.new_tensor::<R2<2, 3>>();
        a.set(vec![1., 2., 3., 1., 2., 3.]);
        let b = cx.new_tensor::<R2<3, 3>>();
        b.set(vec![1., 2., 3., 1., 2., 3., 1., 2., 3.]);
        let c = a.matmul(b);

        cx.execute();

        let d_dev = Cpu::default();
        let d_a = d_dev.tensor([[1., 2., 3.], [1., 2., 3.]]);
        let d_b = d_dev.tensor([[1., 2., 3.], [1., 2., 3.], [1., 2., 3.]]);
        let d_c = d_a.matmul(d_b);

        assert_close_data(
            &c.retrieve().unwrap().real_data(c.view().unwrap()).unwrap(),
            &d_c.as_vec(),
        );
    }

    #[test]
    fn test_batch_matmul() {
        let mut cx = Graph::new();
        let a = cx.new_tensor::<R3<2, 3, 2>>();
        a.set(vec![1., 2., 3., 1., 2., 3., 1., 2., 3., 1., 2., 3.]);
        let b = cx.new_tensor::<R2<2, 4>>();
        b.set(vec![1., 2., 3., 1., 1., 2., 3., 1.]);
        let c = a.matmul(b);

        cx.execute();

        let d_dev = Cpu::default();
        let d_a = d_dev.tensor([
            [[1., 2.], [3., 1.], [2., 3.]],
            [[1., 2.], [3., 1.], [2., 3.]],
        ]);
        let d_b = d_dev.tensor([[1., 2., 3., 1.], [1., 2., 3., 1.]]);
        let d_c = d_a.matmul(d_b);

        assert_close_data(
            &c.retrieve().unwrap().real_data(c.view().unwrap()).unwrap(),
            &d_c.as_vec(),
        );
    }

    #[test]
    fn test_batch_batch_matmul() {
        let mut cx = Graph::new();
        let a = cx.new_tensor::<R3<1, 2, 3>>();
        a.set(vec![8.6, 8.0, 12.0, 9.9, 10.0, 15.0]);
        let b = cx.new_tensor::<R3<1, 2, 3>>();
        b.set(vec![4.0, -12.0, 12.0, 5.0, 70.0, 15.0]);
        let c = a.batch_matmul(b.permute());

        cx.execute();

        let d_dev = Cpu::default();
        let d_a = d_dev.tensor_from_vec(
            vec![8.6, 8.0, 12.0, 9.9, 10.0, 15.0],
            (
                dfdx::shapes::Const::<1>,
                dfdx::shapes::Const::<2>,
                dfdx::shapes::Const::<3>,
            ),
        );
        let d_b = d_dev.tensor_from_vec(
            vec![4.0, -12.0, 12.0, 5.0, 70.0, 15.0],
            (
                dfdx::shapes::Const::<1>,
                dfdx::shapes::Const::<2>,
                dfdx::shapes::Const::<3>,
            ),
        );
        let d_c = d_a.matmul(d_b.permute::<Rank3<1, 3, 2>, dfdx::shapes::Axes3<0, 2, 1>>());

        assert_close_data(
            &c.retrieve().unwrap().real_data(c.view().unwrap()).unwrap(),
            &d_c.as_vec(),
        );
    }

    #[test]
    fn test_batch_batch_matmul2() {
        let mut cx = Graph::new();
        let a = cx.new_tensor::<(usize, usize)>();
        a.set_dyn(vec![0.0, 1.0, 0.0, 1.0], vec![2, 2]);
        let a = a.expand::<(crate::shape::Const<1>, usize, usize), _>();
        let b = cx.new_tensor::<(crate::shape::Const<1>, usize, crate::shape::Const<3>)>();
        b.set_dyn(vec![32.0, -2.0, 0.0, -17.0, 40.0, -3.0], vec![1, 2, 3]);
        let c = a.batch_matmul(b);

        cx.execute();

        let d_dev = Cpu::default();
        let d_a = d_dev.tensor_from_vec(
            vec![0.0, 1.0, 0.0, 1.0],
            (
                dfdx::shapes::Const::<1>,
                dfdx::shapes::Const::<2>,
                dfdx::shapes::Const::<2>,
            ),
        );
        let d_b = d_dev.tensor_from_vec(
            vec![32.0, -2.0, 0.0, -17.0, 40.0, -3.0],
            (
                dfdx::shapes::Const::<1>,
                dfdx::shapes::Const::<2>,
                dfdx::shapes::Const::<3>,
            ),
        );
        let d_c = d_a.matmul(d_b);

        assert_close_data(
            &c.retrieve().unwrap().real_data(c.view().unwrap()).unwrap(),
            &d_c.as_vec(),
        );
    }

    #[test]
    fn test_mean_reduce() {
        let mut cx = Graph::new();
        let a = cx.new_tensor::<R2<2, 3>>();
        a.set(vec![1., 2., 3., 1., 2., 3.]);
        let b = a.mean_reduce::<_, crate::prelude::Axis<1>>();
        b.mark();

        cx.execute();

        let d_dev = Cpu::default();
        let d_a = d_dev.tensor([[1., 2., 3.], [1., 2., 3.]]);
        let d_b = d_a.mean::<_, dfdx::shapes::Axis<1>>();

        assert_close_data(
            &b.retrieve().unwrap().real_data(b.view().unwrap()).unwrap(),
            &d_b.as_vec(),
        );
    }

    #[test]
    fn test_softmax() {
        let mut cx = Graph::new();
        let a = cx.new_tensor::<R2<2, 3>>();
        a.set(vec![
            5.51743, 6.896794, 5.51743, 5.528703, 6.9108624, 5.528703,
        ]);
        let b = a.softmax::<1>();
        b.mark();

        cx.execute();

        let d_dev = Cpu::default();
        let d_a = d_dev.tensor_from_vec(
            vec![5.51743, 6.896794, 5.51743, 5.528703, 6.9108624, 5.528703],
            (dfdx::shapes::Const::<2>, dfdx::shapes::Const::<3>),
        );
        let d_b = d_a.softmax::<dfdx::shapes::Axis<1>>();

        let r = b.retrieve().unwrap().real_data(b.view().unwrap()).unwrap();
        assert_close_data(&r, &d_b.as_vec());
    }
}
