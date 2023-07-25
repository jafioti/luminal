use itertools::Itertools;

use crate::prelude::*;

use crate::op;
use std::ops::{Add, Div, Mul, Neg, Rem, Sub};

// The high level interface implemented on GraphTensor. All of these ops get translated to primitive ops.

// Unary ops
impl<S: ConstShape> GraphTensor<S> {
    pub fn log_2(self) -> GraphTensor<S> {
        let graph = unsafe { self.graph_ref.as_mut().unwrap() };
        let shape = self.shape_tracker();
        let new_id = graph
            .add_op(op::Log2, shape.clone())
            .input(self.id)
            .finish();
        GraphTensor::from_id(new_id, self.graph_ref)
    }

    pub fn exp_2(self) -> GraphTensor<S> {
        let graph = unsafe { self.graph_ref.as_mut().unwrap() };
        let shape = self.shape_tracker();
        let new_id = graph
            .add_op(op::Exp2, shape.clone())
            .input(self.id)
            .finish();
        GraphTensor::from_id(new_id, self.graph_ref)
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
}

// Reduction ops
impl<S: ConstShape> GraphTensor<S> {
    pub fn sum_reduce<Dst: ConstShape, Ax: Axes>(self) -> GraphTensor<Dst>
    where
        S: HasAxes<Ax> + ReduceShapeTo<Dst, Ax>,
    {
        let graph = unsafe { self.graph_ref.as_mut().unwrap() };
        let mut shape = self.shape_tracker().clone();

        let mut new_id = self.id;
        for dim in Ax::as_array().into_iter().collect_vec().into_iter().rev() {
            let mut s = shape.shape().clone();
            s.remove(dim as usize);
            shape.reshape(s);
            new_id = graph
                .add_op(op::SumReduce(dim as usize), shape.clone())
                .input(new_id)
                .finish();
        }
        GraphTensor::from_id(new_id, self.graph_ref)
    }

    pub fn max_reduce<Dst: ConstShape, Ax: Axes>(self) -> GraphTensor<Dst>
    where
        S: HasAxes<Ax> + ReduceShapeTo<Dst, Ax>,
    {
        let graph = unsafe { self.graph_ref.as_mut().unwrap() };
        let mut shape = self.shape_tracker().clone();

        let mut new_id = self.id;
        for dim in Ax::as_array().into_iter().collect_vec().into_iter().rev() {
            let mut s = shape.shape().clone();
            s.remove(dim as usize);
            shape.reshape(s);
            new_id = graph
                .add_op(op::MaxReduce(dim as usize), shape.clone())
                .input(new_id)
                .finish();
        }
        GraphTensor::from_id(new_id, self.graph_ref)
    }

    pub fn mean_reduce<Dst: ConstShape, Ax: Axes>(self) -> GraphTensor<Dst>
    where
        S: HasAxes<Ax> + ReduceShapeTo<Dst, Ax>,
    {
        let graph = unsafe { self.graph_ref.as_mut().unwrap() };
        let mut shape = self.shape_tracker().clone();

        let mut node_id = self.id;
        for dim in Ax::as_array().into_iter().collect_vec().into_iter().rev() {
            // Reduce shape
            let mut sh = shape.shape().clone();
            let size = sh.remove(dim as usize);
            shape.reshape(sh);
            // Sum reduce
            node_id = graph
                .add_op(op::SumReduce(dim as usize), shape.clone())
                .input(node_id)
                .finish();
            // Create div tensor
            let size_t = graph.new_tensor::<R0>();
            size_t.set(vec![size as f32]);
            let mut size_t = size_t.id;
            let mut size_t_shape = ShapeTracker::new(vec![]);
            // Expand div tensor
            for (dim, size) in shape.shape().iter().enumerate() {
                size_t_shape.expand(dim, *size);
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
impl<S: ConstShape> GraphTensor<S> {
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
impl<S: ConstShape> GraphTensor<S> {
    /// Take the elementwise maximum of two tensors
    pub fn max(self, rhs: GraphTensor<S>) -> GraphTensor<S> {
        let graph = unsafe { self.graph_ref.as_mut().unwrap() };
        let new_id = graph
            .add_op(
                op::Max,
                ShapeTracker::new(self.shape_tracker().shape().clone()),
            )
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
impl<S: ConstShape> GraphTensor<S> {
    pub fn permute<N: ConstShape, Dst: ConstShape, Ax: Axes>(self) -> GraphTensor<N>
    where
        N: PermuteShapeTo<Dst, Ax>,
    {
        let graph = unsafe { self.graph_ref.as_mut().unwrap() };
        let orig_shape = self.shape_tracker().clone();
        let mut shape = self.shape_tracker().clone();
        shape.reshape(Dst::realized_shape());
        let new_id = graph
            .add_op(
                op::Permute(Ax::as_array().into_iter().map(|i| i as usize).collect_vec()),
                orig_shape,
            )
            .input(self.id)
            .finish();
        GraphTensor::from_id(new_id, self.graph_ref)
    }

    pub fn expand<Dst: ConstShape, Ax: Axes>(self) -> GraphTensor<Dst>
    where
        S: BroadcastShapeTo<Dst, Ax>,
    {
        let graph = unsafe { self.graph_ref.as_mut().unwrap() };
        let new_shape = Dst::realized_shape();
        let mut shape = self.shape_tracker().clone();

        let mut new_id = self.id;
        for (dim, size) in Ax::as_array()
            .into_iter()
            .map(|i| (i as usize, new_shape[i as usize]))
        {
            shape.expand(dim, size);
            new_id = graph
                .add_op(op::Expand(dim, size), shape.clone())
                .input(new_id)
                .finish();
        }
        GraphTensor::from_id(new_id, self.graph_ref)
    }

    pub fn reshape<N: ConstShape>(self) -> GraphTensor<N> {
        <S as AssertSameNumel<N>>::assert_same_numel();
        let graph = unsafe { self.graph_ref.as_mut().unwrap() };
        let mut shape = self.shape_tracker().clone();
        shape.reshape(N::realized_shape());
        let new_id = graph
            .add_op(op::Reshape(N::realized_shape()), shape)
            .input(self.id)
            .finish();
        GraphTensor::from_id(new_id, self.graph_ref)
    }
}

// Matmul 2x2, 2x3 (broadcast 2 across batch), 2x4 (broadcast 2 across 2 batch dims), 3x3 (make sure shape matches up, multiply each consituent matrix)

// ABxBC -> AC
impl<const A: usize, const B: usize> GraphTensor<R2<A, B>> {
    pub fn matmul<const C: usize>(self, rhs: GraphTensor<R2<B, C>>) -> GraphTensor<R2<A, C>> {
        // Reshape
        let w: GraphTensor<R2<C, B>> = rhs.permute::<_, _, Axes2<1, 0>>();

        // Broadcasted Multiply
        let mul = self.expand::<R3<A, C, B>, _>() * w.expand::<R3<A, C, B>, _>();

        // Sum Reduce
        mul.sum_reduce::<_, Axis<2>>()
    }
}

// AxAB -> B (Don't know if this is correct)
impl<const A: usize> GraphTensor<R1<A>> {
    pub fn matmul<const B: usize>(self, rhs: GraphTensor<R2<A, B>>) -> GraphTensor<R1<B>> {
        let s: GraphTensor<R2<1, A>> = self.expand();

        // Run normal matmul
        let r = s.matmul(rhs);

        // Sum Reduce
        r.sum_reduce::<_, Axis<0>>()
    }
}

// ABCxCD -> ABD
impl<const A: usize, const B: usize, const C: usize> GraphTensor<R3<A, B, C>> {
    pub fn matmul<const D: usize>(self, rhs: GraphTensor<R2<C, D>>) -> GraphTensor<R3<A, B, D>> {
        // Reshape
        let w: GraphTensor<R2<D, C>> = rhs.permute::<_, _, Axes2<1, 0>>();

        // Broadcasted Multiply
        let mul = self.expand::<R4<A, B, D, C>, _>() * w.expand::<R4<A, B, D, C>, _>();

        // Sum Reduce
        mul.sum_reduce::<_, Axis<3>>()
    }
}

impl<const A: usize> GraphTensor<R1<A>> {
    /// Simple dot product of two vectors
    pub fn dot(self, rhs: GraphTensor<R1<A>>) -> GraphTensor<R0> {
        (self * rhs).sum_reduce()
    }
}

impl<S: ConstShape> Add<GraphTensor<S>> for GraphTensor<S> {
    type Output = GraphTensor<S>;

    fn add(self, rhs: GraphTensor<S>) -> Self::Output {
        let graph = unsafe { self.graph_ref.as_mut().unwrap() };
        let new_id = graph
            .add_op(
                op::Add,
                ShapeTracker::new(self.shape_tracker().shape().clone()),
            )
            .input(self.id)
            .input(rhs.id)
            .finish();
        GraphTensor::from_id(new_id, self.graph_ref)
    }
}

impl<S: ConstShape> Sub<GraphTensor<S>> for GraphTensor<S> {
    type Output = GraphTensor<S>;

    fn sub(self, rhs: GraphTensor<S>) -> Self::Output {
        let graph = unsafe { self.graph_ref.as_mut().unwrap() };
        let new_id = graph
            .add_op(
                op::Sub,
                ShapeTracker::new(self.shape_tracker().shape().clone()),
            )
            .input(self.id)
            .input(rhs.id)
            .finish();
        GraphTensor::from_id(new_id, self.graph_ref)
    }
}

impl<S: ConstShape> Mul<GraphTensor<S>> for GraphTensor<S> {
    type Output = GraphTensor<S>;

    fn mul(self, rhs: GraphTensor<S>) -> Self::Output {
        let graph = unsafe { self.graph_ref.as_mut().unwrap() };
        let new_id = graph
            .add_op(
                op::Mul,
                ShapeTracker::new(self.shape_tracker().shape().clone()),
            )
            .input(self.id)
            .input(rhs.id)
            .finish();
        GraphTensor::from_id(new_id, self.graph_ref)
    }
}

impl<S: ConstShape> Div<GraphTensor<S>> for GraphTensor<S> {
    type Output = GraphTensor<S>;

    fn div(self, rhs: GraphTensor<S>) -> Self::Output {
        let graph = unsafe { self.graph_ref.as_mut().unwrap() };
        let new_id = graph
            .add_op(
                op::Div,
                ShapeTracker::new(self.shape_tracker().shape().clone()),
            )
            .input(self.id)
            .input(rhs.id)
            .finish();
        GraphTensor::from_id(new_id, self.graph_ref)
    }
}

impl<S: ConstShape> Rem<GraphTensor<S>> for GraphTensor<S> {
    type Output = GraphTensor<S>;

    fn rem(self, rhs: GraphTensor<S>) -> Self::Output {
        let graph = unsafe { self.graph_ref.as_mut().unwrap() };
        let new_id = graph
            .add_op(
                op::Mod,
                ShapeTracker::new(self.shape_tracker().shape().clone()),
            )
            .input(self.id)
            .input(rhs.id)
            .finish();
        GraphTensor::from_id(new_id, self.graph_ref)
    }
}

impl<S: ConstShape> Neg for GraphTensor<S> {
    type Output = GraphTensor<S>;

    fn neg(self) -> Self::Output {
        let graph = unsafe { self.graph_ref.as_mut().unwrap() };
        let neg_one = graph.new_tensor::<R0>();
        neg_one.set(vec![-1.0]);
        self * neg_one.expand()
    }
}

impl<S: ConstShape> Add<f32> for GraphTensor<S> {
    type Output = GraphTensor<S>;

    fn add(self, rhs: f32) -> Self::Output {
        let graph = unsafe { self.graph_ref.as_mut().unwrap() };
        let rhs_t = graph.new_tensor::<R0>();
        rhs_t.set(vec![rhs]);
        self + rhs_t.expand()
    }
}

impl<S: ConstShape> Sub<f32> for GraphTensor<S> {
    type Output = GraphTensor<S>;

    fn sub(self, rhs: f32) -> Self::Output {
        let graph = unsafe { self.graph_ref.as_mut().unwrap() };
        let rhs_t = graph.new_tensor::<R0>();
        rhs_t.set(vec![rhs]);
        self - rhs_t.expand()
    }
}

impl<S: ConstShape> Mul<f32> for GraphTensor<S> {
    type Output = GraphTensor<S>;

    fn mul(self, rhs: f32) -> Self::Output {
        let graph = unsafe { self.graph_ref.as_mut().unwrap() };
        let rhs_t = graph.new_tensor::<R0>();
        rhs_t.set(vec![rhs]);
        self * rhs_t.expand()
    }
}

impl<S: ConstShape> Div<f32> for GraphTensor<S> {
    type Output = GraphTensor<S>;

    fn div(self, rhs: f32) -> Self::Output {
        let graph = unsafe { self.graph_ref.as_mut().unwrap() };
        let rhs_t = graph.new_tensor::<R0>();
        rhs_t.set(vec![rhs]);
        self / rhs_t.expand()
    }
}

impl<S: ConstShape> Rem<f32> for GraphTensor<S> {
    type Output = GraphTensor<S>;

    fn rem(self, rhs: f32) -> Self::Output {
        let graph = unsafe { self.graph_ref.as_mut().unwrap() };
        let rhs_t = graph.new_tensor::<R0>();
        rhs_t.set(vec![rhs]);
        self % rhs_t.expand()
    }
}
