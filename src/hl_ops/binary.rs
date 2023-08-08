use crate::op;
use crate::prelude::*;
use std::ops::{Add, Div, Mul, Neg, Rem, Sub};

impl<S: Shape> Add<GraphTensor<S>> for GraphTensor<S> {
    type Output = GraphTensor<S>;

    fn add(self, rhs: GraphTensor<S>) -> Self::Output {
        let graph = unsafe { self.graph_ref.as_mut().unwrap() };
        let new_id = graph
            .add_op(op::Add, self.shape().clone())
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
            .add_op(op::Sub, self.shape().clone())
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
            .add_op(op::Mul, self.shape().clone())
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
            .add_op(op::Div, self.shape().clone())
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
            .add_op(op::Mod, self.shape().clone())
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
        let rhs_t = graph.new_tensor::<R0>("Const");
        rhs_t.set(vec![rhs]);
        self + rhs_t.expand()
    }
}

impl<S: Shape> Sub<f32> for GraphTensor<S> {
    type Output = GraphTensor<S>;

    fn sub(self, rhs: f32) -> Self::Output {
        let graph = unsafe { self.graph_ref.as_mut().unwrap() };
        let rhs_t = graph.new_tensor::<R0>("Const");
        rhs_t.set(vec![rhs]);
        self - rhs_t.expand()
    }
}

impl<S: Shape> Mul<f32> for GraphTensor<S> {
    type Output = GraphTensor<S>;

    fn mul(self, rhs: f32) -> Self::Output {
        let graph = unsafe { self.graph_ref.as_mut().unwrap() };
        let rhs_t = graph.new_tensor::<R0>("Const");
        rhs_t.set(vec![rhs]);
        self * rhs_t.expand()
    }
}

impl<S: Shape> Div<f32> for GraphTensor<S> {
    type Output = GraphTensor<S>;

    fn div(self, rhs: f32) -> Self::Output {
        let graph = unsafe { self.graph_ref.as_mut().unwrap() };
        let rhs_t = graph.new_tensor::<R0>("Const");
        rhs_t.set(vec![rhs]);
        self / rhs_t.expand()
    }
}

impl<S: Shape> Rem<f32> for GraphTensor<S> {
    type Output = GraphTensor<S>;

    fn rem(self, rhs: f32) -> Self::Output {
        let graph = unsafe { self.graph_ref.as_mut().unwrap() };
        let rhs_t = graph.new_tensor::<R0>("Const");
        rhs_t.set(vec![rhs]);
        self % rhs_t.expand()
    }
}

// Clipping ops (min, max, clip)
impl<S: Shape> GraphTensor<S> {
    /// Take the elementwise maximum of two tensors
    pub fn max(self, rhs: GraphTensor<S>) -> GraphTensor<S> {
        let graph = unsafe { self.graph_ref.as_mut().unwrap() };
        let new_id = graph
            .add_op(op::Max, self.shape().clone())
            .input(self.id)
            .input(rhs.id)
            .finish();
        GraphTensor::from_id(new_id, self.graph_ref)
    }

    /// Take the elementwise maximum of a tensor and a float
    pub fn max_f32(self, rhs: f32) -> GraphTensor<S> {
        let graph = unsafe { self.graph_ref.as_mut().unwrap() };
        let rhs_t = graph.new_tensor::<R0>("Const");
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
