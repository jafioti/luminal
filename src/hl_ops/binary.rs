use crate::op;
use crate::prelude::*;
use std::ops::AddAssign;
use std::ops::DivAssign;
use std::ops::MulAssign;
use std::ops::RemAssign;
use std::ops::SubAssign;
use std::ops::{Add, Div, Mul, Rem, Sub};

use self::symbolic::ExpressionStorage;
use self::symbolic::GenericExpression;
use self::symbolic::Term;

impl<S: Shape> Add for GraphTensor<S> {
    type Output = GraphTensor<S>;

    fn add(mut self, mut rhs: GraphTensor<S>) -> Self::Output {
        resolve_local_dyn_dims(&mut self.shape, &mut rhs.shape, false);
        let new_id = self
            .graph()
            .add_op(op::Add)
            .input(self.id, 0, self.shape)
            .input(rhs.id, 0, rhs.shape)
            .finish();
        GraphTensor::from_id(new_id, self.shape.contiguous(), self.graph_ref)
    }
}

impl<S: Shape> Add<GraphTensor<S>> for f32 {
    type Output = GraphTensor<S>;

    fn add(self, rhs: GraphTensor<S>) -> Self::Output {
        rhs + self
    }
}

impl<S: Shape> AddAssign for GraphTensor<S> {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl<S: Shape> Sub for GraphTensor<S> {
    type Output = GraphTensor<S>;

    fn sub(self, rhs: GraphTensor<S>) -> Self::Output {
        self + -rhs
    }
}

impl<S: Shape> Sub<GraphTensor<S>> for f32 {
    type Output = GraphTensor<S>;

    fn sub(self, rhs: GraphTensor<S>) -> Self::Output {
        self + -rhs
    }
}

impl<S: Shape> SubAssign for GraphTensor<S> {
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl<S: Shape> Mul for GraphTensor<S> {
    type Output = GraphTensor<S>;

    fn mul(mut self, mut rhs: GraphTensor<S>) -> Self::Output {
        resolve_local_dyn_dims(&mut self.shape, &mut rhs.shape, false);
        let new_id = self
            .graph()
            .add_op(op::Mul)
            .input(self.id, 0, self.shape)
            .input(rhs.id, 0, rhs.shape)
            .finish();
        GraphTensor::from_id(new_id, self.shape.contiguous(), self.graph_ref)
    }
}

impl<S: Shape> Mul<GraphTensor<S>> for f32 {
    type Output = GraphTensor<S>;

    fn mul(self, rhs: GraphTensor<S>) -> Self::Output {
        rhs * self
    }
}

impl<S: Shape> MulAssign for GraphTensor<S> {
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

#[allow(clippy::suspicious_arithmetic_impl)]
impl<S: Shape> Div<GraphTensor<S>> for GraphTensor<S> {
    type Output = GraphTensor<S>;

    fn div(self, rhs: GraphTensor<S>) -> Self::Output {
        self * rhs.recip()
    }
}

#[allow(clippy::suspicious_arithmetic_impl)]
impl<S: Shape> Div<GraphTensor<S>> for f32 {
    type Output = GraphTensor<S>;

    fn div(self, rhs: GraphTensor<S>) -> Self::Output {
        self * rhs.recip()
    }
}

impl<S: Shape> DivAssign for GraphTensor<S> {
    fn div_assign(&mut self, rhs: Self) {
        *self = *self / rhs;
    }
}

impl<S: Shape> Rem<GraphTensor<S>> for GraphTensor<S> {
    type Output = GraphTensor<S>;

    fn rem(mut self, mut rhs: GraphTensor<S>) -> Self::Output {
        resolve_local_dyn_dims(&mut self.shape, &mut rhs.shape, false);
        let new_id = self
            .graph()
            .add_op(op::Mod)
            .input(self.id, 0, self.shape)
            .input(rhs.id, 0, rhs.shape)
            .finish();
        GraphTensor::from_id(new_id, self.shape.contiguous(), self.graph_ref)
    }
}

impl<S: Shape> RemAssign for GraphTensor<S> {
    fn rem_assign(&mut self, rhs: Self) {
        *self = *self % rhs;
    }
}

impl<S: Shape> Add<f32> for GraphTensor<S> {
    type Output = GraphTensor<S>;

    fn add(self, rhs: f32) -> Self::Output {
        self + self.graph().constant(rhs).expand()
    }
}

impl<S: Shape, St: ExpressionStorage> Add<GenericExpression<St>> for GraphTensor<S>
where
    GenericExpression<Vec<Term>>: From<GenericExpression<St>>,
{
    type Output = GraphTensor<S>;

    fn add(self, rhs: GenericExpression<St>) -> Self::Output {
        self + self.graph().constant_expr(rhs).expand()
    }
}

impl<S: Shape> Sub<f32> for GraphTensor<S> {
    type Output = GraphTensor<S>;

    fn sub(self, rhs: f32) -> Self::Output {
        self - self.graph().constant(rhs).expand()
    }
}

impl<S: Shape, St: ExpressionStorage> Sub<GenericExpression<St>> for GraphTensor<S>
where
    GenericExpression<Vec<Term>>: From<GenericExpression<St>>,
{
    type Output = GraphTensor<S>;

    fn sub(self, rhs: GenericExpression<St>) -> Self::Output {
        self - self.graph().constant_expr(rhs).expand()
    }
}

impl<S: Shape> Mul<f32> for GraphTensor<S> {
    type Output = GraphTensor<S>;

    fn mul(self, rhs: f32) -> Self::Output {
        self * self.graph().constant(rhs).expand()
    }
}

impl<S: Shape, St: ExpressionStorage> Mul<GenericExpression<St>> for GraphTensor<S>
where
    GenericExpression<Vec<Term>>: From<GenericExpression<St>>,
{
    type Output = GraphTensor<S>;

    fn mul(self, rhs: GenericExpression<St>) -> Self::Output {
        self * self.graph().constant_expr(rhs).expand()
    }
}

#[allow(clippy::suspicious_arithmetic_impl)]
impl<S: Shape> Div<f32> for GraphTensor<S> {
    type Output = GraphTensor<S>;

    fn div(self, rhs: f32) -> Self::Output {
        self * self.graph().constant(rhs.recip()).expand()
    }
}

impl<S: Shape, St: ExpressionStorage> Div<GenericExpression<St>> for GraphTensor<S>
where
    GenericExpression<Vec<Term>>: From<GenericExpression<St>>,
{
    type Output = GraphTensor<S>;

    fn div(self, rhs: GenericExpression<St>) -> Self::Output {
        self / self.graph().constant_expr(rhs).expand()
    }
}

impl<S: Shape> Rem<f32> for GraphTensor<S> {
    type Output = GraphTensor<S>;

    fn rem(self, rhs: f32) -> Self::Output {
        self % self.graph().constant(rhs).expand()
    }
}

impl<S: Shape, St: ExpressionStorage> Rem<GenericExpression<St>> for GraphTensor<S>
where
    GenericExpression<Vec<Term>>: From<GenericExpression<St>>,
{
    type Output = GraphTensor<S>;

    fn rem(self, rhs: GenericExpression<St>) -> Self::Output {
        self % self.graph().constant_expr(rhs).expand()
    }
}

// Comparisons (based on https://github.com/tinygrad/tinygrad/blob/3e0c2d256fe9f4f5f85cd3e4d8733a51d7b4a984/tinygrad/tensor.py#L653)
impl<S: Shape> GraphTensor<S> {
    pub fn less_than(mut self, mut rhs: GraphTensor<S>) -> GraphTensor<S> {
        resolve_local_dyn_dims(&mut self.shape, &mut rhs.shape, false);
        let new_id = self
            .graph()
            .add_op(op::LessThan)
            .input(self.id, 0, self.shape)
            .input(rhs.id, 0, rhs.shape)
            .finish();
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

    /// Raise the tensor to a power
    pub fn pow<T>(self, e: T) -> GraphTensor<S>
    where
        Self: Mul<T, Output = Self>,
    {
        // Approximate, see full impl here: https://github.com/tinygrad/tinygrad/blob/a32c67760140dd26b60d7932268f2e62e96a66e0/tinygrad/tensor.py#L568
        self.abs().ln().mul(e).exp()
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

pub trait F32Pow {
    fn pow<S: Shape>(self, e: GraphTensor<S>) -> GraphTensor<S>;
}

impl F32Pow for f32 {
    fn pow<S: Shape>(self, e: GraphTensor<S>) -> GraphTensor<S> {
        e.mul(self.abs().ln()).exp().recip()
    }
}
