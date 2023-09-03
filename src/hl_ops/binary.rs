use crate::op;
use crate::prelude::*;
use std::ops::{Add, Div, Mul, Neg, Rem, Sub};

impl<S: Shape> Add<GraphTensor<S>> for GraphTensor<S> {
    type Output = GraphTensor<S>;

    fn add(mut self, mut rhs: GraphTensor<S>) -> Self::Output {
        let new_id = self
            .graph()
            .add_op(op::Add)
            .input(self.id, self.shape)
            .input(rhs.id, rhs.shape)
            .finish();
        resolve_shapes(&mut self.shape, &mut rhs.shape, false);
        GraphTensor::from_id(new_id, self.shape.contiguous(), self.graph_ref)
    }
}

impl<S: Shape> Sub<GraphTensor<S>> for GraphTensor<S> {
    type Output = GraphTensor<S>;

    fn sub(self, rhs: GraphTensor<S>) -> Self::Output {
        self + -rhs
    }
}

impl<S: Shape> Mul<GraphTensor<S>> for GraphTensor<S> {
    type Output = GraphTensor<S>;

    fn mul(mut self, mut rhs: GraphTensor<S>) -> Self::Output {
        let new_id = self
            .graph()
            .add_op(op::Mul)
            .input(self.id, self.shape)
            .input(rhs.id, rhs.shape)
            .finish();
        resolve_shapes(&mut self.shape, &mut rhs.shape, false);
        GraphTensor::from_id(new_id, self.shape.contiguous(), self.graph_ref)
    }
}

#[allow(clippy::suspicious_arithmetic_impl)]
impl<S: Shape> Div<GraphTensor<S>> for GraphTensor<S> {
    type Output = GraphTensor<S>;

    fn div(self, rhs: GraphTensor<S>) -> Self::Output {
        self * rhs.recip()
    }
}

impl<S: Shape> Rem<GraphTensor<S>> for GraphTensor<S> {
    type Output = GraphTensor<S>;

    fn rem(mut self, mut rhs: GraphTensor<S>) -> Self::Output {
        let new_id = self
            .graph()
            .add_op(op::Mod)
            .input(self.id, self.shape)
            .input(rhs.id, rhs.shape)
            .finish();
        resolve_shapes(&mut self.shape, &mut rhs.shape, false);
        GraphTensor::from_id(new_id, self.shape.contiguous(), self.graph_ref)
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
        self + self.graph().constant(rhs).expand()
    }
}

impl<S: Shape> Sub<f32> for GraphTensor<S> {
    type Output = GraphTensor<S>;

    fn sub(self, rhs: f32) -> Self::Output {
        self - self.graph().constant(rhs).expand()
    }
}

impl<S: Shape> Mul<f32> for GraphTensor<S> {
    type Output = GraphTensor<S>;

    fn mul(self, rhs: f32) -> Self::Output {
        self * self.graph().constant(rhs).expand()
    }
}

impl<S: Shape> Div<f32> for GraphTensor<S> {
    type Output = GraphTensor<S>;

    fn div(self, rhs: f32) -> Self::Output {
        self / self.graph().constant(rhs).expand()
    }
}

impl<S: Shape> Rem<f32> for GraphTensor<S> {
    type Output = GraphTensor<S>;

    fn rem(self, rhs: f32) -> Self::Output {
        self % self.graph().constant(rhs).expand()
    }
}

// Comparisons (based on https://github.com/tinygrad/tinygrad/blob/3e0c2d256fe9f4f5f85cd3e4d8733a51d7b4a984/tinygrad/tensor.py#L653)
impl<S: Shape> GraphTensor<S> {
    pub fn less_than(mut self, mut rhs: GraphTensor<S>) -> GraphTensor<S> {
        let new_id = self
            .graph()
            .add_op(op::LessThan)
            .input(self.id, self.shape)
            .input(rhs.id, rhs.shape)
            .finish();
        resolve_shapes(&mut self.shape, &mut rhs.shape, false);
        GraphTensor::from_id(new_id, self.shape.contiguous(), self.graph_ref)
    }

    pub fn greater_than(self, rhs: GraphTensor<S>) -> GraphTensor<S> {
        rhs.less_than(self)
    }

    pub fn less_than_equal(self, rhs: GraphTensor<S>) -> GraphTensor<S> {
        -self.greater_than(rhs) + 1.0
    }

    pub fn greater_than_equal(self, rhs: GraphTensor<S>) -> GraphTensor<S> {
        -self.less_than(rhs) + 1.0
    }

    pub fn not_equals(self, rhs: GraphTensor<S>) -> GraphTensor<S> {
        self.less_than(rhs) + self.greater_than(rhs)
    }

    pub fn equals(self, rhs: GraphTensor<S>) -> GraphTensor<S> {
        -self.not_equals(rhs) + 1.0
    }
}

// Clipping ops (min, max, clip)
impl<S: Shape> GraphTensor<S> {
    /// Take the elementwise maximum of two tensors
    pub fn max(self, rhs: GraphTensor<S>) -> GraphTensor<S> {
        (self.less_than(rhs) * rhs) + (rhs.less_than_equal(self) * self)
    }

    /// Take the elementwise maximum of a tensor and a float
    pub fn max_f32(self, rhs: f32) -> GraphTensor<S> {
        self.max(self.graph().constant(rhs).expand())
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
