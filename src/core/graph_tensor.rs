use crate::{
    graph::Graph,
    op::{self},
    shape::*,
    tensor::Tensor,
};
use std::{
    marker::PhantomData,
    ops::{Add, Div, Mul, Rem, Sub},
};

use itertools::Itertools;
use petgraph::graph::NodeIndex;

#[derive(Clone, Copy)]
pub struct GraphTensor<S: ConstShape> {
    pub id: NodeIndex,
    pub graph_ref: *mut Graph,
    pub(crate) _phantom: PhantomData<S>,
}

impl<S: ConstShape> GraphTensor<S> {
    fn from_id(id: NodeIndex, graph_ref: *mut Graph) -> Self {
        Self {
            id,
            graph_ref,
            _phantom: Default::default(),
        }
    }

    pub fn shape_tracker(&self) -> &ShapeTracker {
        let graph = unsafe { self.graph_ref.as_mut().unwrap() };
        &graph.graph.node_weight(self.id).unwrap().2
    }

    /// Mark this tensor to be retrieved later
    pub fn mark(&self) {
        unsafe {
            self.graph_ref.as_mut().unwrap().no_delete.insert(self.id);
            self.graph_ref.as_mut().unwrap().to_retrieve.insert(self.id);
        }
    }

    /// Get the value of the tensor (if the graph was executed)
    pub fn retrieve(self) -> Option<Tensor> {
        unsafe { self.graph_ref.as_mut().unwrap().get_tensor(self.id) }
    }

    /// Set the value of the tensor
    pub fn set(&self, data: Vec<f32>) {
        unsafe { self.graph_ref.as_mut().unwrap().set_tensor(*self, data) }
    }

    pub fn log_2(self) -> GraphTensor<S> {
        let graph = unsafe { self.graph_ref.as_mut().unwrap() };
        let shape = self.shape_tracker();
        let new_id = graph
            .add_op(op::Log2, shape.clone())
            .input(self.id, shape.clone())
            .finish();
        GraphTensor::from_id(new_id, self.graph_ref)
    }

    pub fn exp_2(self) -> GraphTensor<S> {
        let graph = unsafe { self.graph_ref.as_mut().unwrap() };
        let shape = self.shape_tracker();
        let new_id = graph
            .add_op(op::Exp2, shape.clone())
            .input(self.id, shape.clone())
            .finish();
        GraphTensor::from_id(new_id, self.graph_ref)
    }

    pub fn recip(self) -> GraphTensor<S> {
        let graph = unsafe { self.graph_ref.as_mut().unwrap() };
        let shape = self.shape_tracker();
        let new_id = graph
            .add_op(op::Recip, shape.clone())
            .input(self.id, shape.clone())
            .finish();
        GraphTensor::from_id(new_id, self.graph_ref)
    }

    pub fn sin(self) -> GraphTensor<S> {
        let graph = unsafe { self.graph_ref.as_mut().unwrap() };
        let shape = self.shape_tracker();
        let new_id = graph
            .add_op(op::Sin, shape.clone())
            .input(self.id, shape.clone())
            .finish();
        GraphTensor::from_id(new_id, self.graph_ref)
    }

    pub fn sqrt(self) -> GraphTensor<S> {
        let graph = unsafe { self.graph_ref.as_mut().unwrap() };
        let shape = self.shape_tracker();
        let new_id = graph
            .add_op(op::Sqrt, shape.clone())
            .input(self.id, shape.clone())
            .finish();
        GraphTensor::from_id(new_id, self.graph_ref)
    }

    pub fn max(self, rhs: GraphTensor<S>) -> GraphTensor<S> {
        let graph = unsafe { self.graph_ref.as_mut().unwrap() };
        let new_id = graph
            .add_op(
                op::Max,
                ShapeTracker::new(self.shape_tracker().shape().clone()),
            )
            .input(self.id, self.shape_tracker().clone())
            .input(rhs.id, rhs.shape_tracker().clone())
            .finish();
        GraphTensor::from_id(new_id, self.graph_ref)
    }

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
            .input(self.id, shape)
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
            let pre_shape = shape.clone();
            shape.expand(dim, size);
            new_id = graph
                .add_op(op::Expand(dim, size), shape.clone())
                .input(new_id, pre_shape)
                .finish();
        }
        GraphTensor::from_id(new_id, self.graph_ref)
    }

    pub fn reshape<N: ConstShape>(self) -> GraphTensor<N> {
        <S as AssertSameNumel<N>>::assert_same_numel();
        let graph = unsafe { self.graph_ref.as_mut().unwrap() };
        let pre_shape = self.shape_tracker().clone();
        let mut shape = pre_shape.clone();
        shape.reshape(N::realized_shape());
        let new_id = graph
            .add_op(op::Reshape(N::realized_shape()), shape)
            .input(self.id, pre_shape)
            .finish();
        GraphTensor::from_id(new_id, self.graph_ref)
    }

    pub fn sum_reduce<Dst: ConstShape, Ax: Axes>(self) -> GraphTensor<Dst>
    where
        S: HasAxes<Ax> + ReduceShapeTo<Dst, Ax>,
    {
        let graph = unsafe { self.graph_ref.as_mut().unwrap() };
        let dim = Ax::as_array().into_iter().next().unwrap() as usize;
        let mut tracker = self.shape_tracker().clone();
        let mut shape = tracker.shape().clone();
        shape.remove(dim);
        tracker.reshape(shape);
        let new_id = graph
            .add_op(op::SumReduce(dim), tracker)
            .input(self.id, self.shape_tracker().clone())
            .finish();
        GraphTensor::from_id(new_id, self.graph_ref)
    }

    pub fn max_reduce<Dst: ConstShape, Ax: Axes>(self) -> GraphTensor<Dst>
    where
        S: HasAxes<Ax> + ReduceShapeTo<Dst, Ax>,
    {
        let graph = unsafe { self.graph_ref.as_mut().unwrap() };
        let dim = Ax::as_array().into_iter().next().unwrap() as usize;
        let mut tracker = self.shape_tracker().clone();
        let mut shape = tracker.shape().clone();
        shape.remove(dim);
        tracker.reshape(shape);
        let new_id = graph
            .add_op(op::MaxReduce(dim), tracker)
            .input(self.id, self.shape_tracker().clone())
            .finish();
        GraphTensor::from_id(new_id, self.graph_ref)
    }
}

impl<const A: usize> GraphTensor<R1<A>> {
    /// Simple dot product of two vectors
    pub fn dot(self, rhs: GraphTensor<R1<A>>) -> GraphTensor<R0> {
        (self * rhs).sum_reduce()
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

impl<S: ConstShape> Add<GraphTensor<S>> for GraphTensor<S> {
    type Output = GraphTensor<S>;

    fn add(self, rhs: GraphTensor<S>) -> Self::Output {
        let graph = unsafe { self.graph_ref.as_mut().unwrap() };
        let new_id = graph
            .add_op(
                op::Add,
                ShapeTracker::new(self.shape_tracker().shape().clone()),
            )
            .input(self.id, self.shape_tracker().clone())
            .input(rhs.id, rhs.shape_tracker().clone())
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
            .input(self.id, self.shape_tracker().clone())
            .input(rhs.id, rhs.shape_tracker().clone())
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
            .input(self.id, self.shape_tracker().clone())
            .input(rhs.id, rhs.shape_tracker().clone())
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
            .input(self.id, self.shape_tracker().clone())
            .input(rhs.id, rhs.shape_tracker().clone())
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
            .input(self.id, self.shape_tracker().clone())
            .input(rhs.id, rhs.shape_tracker().clone())
            .finish();
        GraphTensor::from_id(new_id, self.graph_ref)
    }
}
