use crate::op;
use crate::prelude::*;
use std::ops::AddAssign;
use std::ops::DivAssign;
use std::ops::MulAssign;
use std::ops::RemAssign;
use std::ops::SubAssign;
use std::ops::{Add, Div, Mul, Rem, Sub};

impl Add for GraphTensor {
    type Output = GraphTensor;

    fn add(self, rhs: GraphTensor) -> Self::Output {
        // assert_eq!(
        //     self.dims()
        //         .into_iter()
        //         .map(|i| i.simplify())
        //         .collect::<Vec<_>>(),
        //     rhs.dims()
        //         .into_iter()
        //         .map(|i| i.simplify())
        //         .collect::<Vec<_>>(),
        //     "Dims must match to add tensors."
        // );
        let new_id = self
            .graph()
            .add_op(op::Add)
            .input(self.id, 0, self.shape)
            .input(rhs.id, 0, rhs.shape)
            .finish();
        GraphTensor::from_id(new_id, self.shape.contiguous(), self.graph_ref)
    }
}

impl Add<GraphTensor> for f32 {
    type Output = GraphTensor;

    fn add(self, rhs: GraphTensor) -> Self::Output {
        rhs + self
    }
}

impl AddAssign for GraphTensor {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl Sub for GraphTensor {
    type Output = GraphTensor;

    fn sub(self, rhs: GraphTensor) -> Self::Output {
        self + -rhs
    }
}

impl Sub<GraphTensor> for f32 {
    type Output = GraphTensor;

    fn sub(self, rhs: GraphTensor) -> Self::Output {
        self + -rhs
    }
}

impl SubAssign for GraphTensor {
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl Mul for GraphTensor {
    type Output = GraphTensor;

    fn mul(self, rhs: GraphTensor) -> Self::Output {
        // assert_eq!(
        //     self.dims(),
        //     rhs.dims(),
        //     "Dims must match to multiply tensors."
        // );
        let new_id = self
            .graph()
            .add_op(op::Mul)
            .input(self.id, 0, self.shape)
            .input(rhs.id, 0, rhs.shape)
            .finish();
        GraphTensor::from_id(new_id, self.shape.contiguous(), self.graph_ref)
    }
}

impl Mul<GraphTensor> for f32 {
    type Output = GraphTensor;

    fn mul(self, rhs: GraphTensor) -> Self::Output {
        rhs * self
    }
}

impl MulAssign for GraphTensor {
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

#[allow(clippy::suspicious_arithmetic_impl)]
impl Div<GraphTensor> for GraphTensor {
    type Output = GraphTensor;

    fn div(self, rhs: GraphTensor) -> Self::Output {
        self * rhs.recip()
    }
}

#[allow(clippy::suspicious_arithmetic_impl)]
impl Div<GraphTensor> for f32 {
    type Output = GraphTensor;

    fn div(self, rhs: GraphTensor) -> Self::Output {
        self * rhs.recip()
    }
}

impl DivAssign for GraphTensor {
    fn div_assign(&mut self, rhs: Self) {
        *self = *self / rhs;
    }
}

impl Rem<GraphTensor> for GraphTensor {
    type Output = GraphTensor;

    fn rem(self, rhs: GraphTensor) -> Self::Output {
        assert_eq!(self.dims(), rhs.dims(), "Dims must match to mod tensors.");
        let new_id = self
            .graph()
            .add_op(op::Mod)
            .input(self.id, 0, self.shape)
            .input(rhs.id, 0, rhs.shape)
            .finish();
        GraphTensor::from_id(new_id, self.shape.contiguous(), self.graph_ref)
    }
}

impl RemAssign for GraphTensor {
    fn rem_assign(&mut self, rhs: Self) {
        *self = *self % rhs;
    }
}

impl Add<f32> for GraphTensor {
    type Output = GraphTensor;

    fn add(self, rhs: f32) -> Self::Output {
        self + self.graph().constant(rhs).expand_to(self.shape)
    }
}

impl<S: Into<Expression>> Add<S> for GraphTensor {
    type Output = GraphTensor;

    fn add(self, rhs: S) -> Self::Output {
        self + self.graph().constant(rhs).expand_to(self.shape)
    }
}

impl Sub<f32> for GraphTensor {
    type Output = GraphTensor;

    fn sub(self, rhs: f32) -> Self::Output {
        self - self.graph().constant(rhs).expand_to(self.shape)
    }
}

impl<S: Into<Expression>> Sub<S> for GraphTensor {
    type Output = GraphTensor;

    fn sub(self, rhs: S) -> Self::Output {
        self - self.graph().constant(rhs).expand_to(self.shape)
    }
}

impl Mul<f32> for GraphTensor {
    type Output = GraphTensor;

    fn mul(self, rhs: f32) -> Self::Output {
        self * self.graph().constant(rhs).expand_to(self.shape)
    }
}

impl<S: Into<Expression>> Mul<S> for GraphTensor {
    type Output = GraphTensor;

    fn mul(self, rhs: S) -> Self::Output {
        self * self.graph().constant(rhs).expand_to(self.shape)
    }
}

#[allow(clippy::suspicious_arithmetic_impl)]
impl Div<f32> for GraphTensor {
    type Output = GraphTensor;

    fn div(self, rhs: f32) -> Self::Output {
        self * self.graph().constant(rhs.recip()).expand_to(self.shape)
    }
}

impl<S: Into<Expression>> Div<S> for GraphTensor {
    type Output = GraphTensor;

    fn div(self, rhs: S) -> Self::Output {
        self / self.graph().constant(rhs).expand_to(self.shape)
    }
}

impl Rem<f32> for GraphTensor {
    type Output = GraphTensor;

    fn rem(self, rhs: f32) -> Self::Output {
        self % self.graph().constant(rhs).expand_to(self.shape)
    }
}

impl<S: Into<Expression>> Rem<S> for GraphTensor {
    type Output = GraphTensor;

    fn rem(self, rhs: S) -> Self::Output {
        self % self.graph().constant(rhs).expand_to(self.shape)
    }
}

// Comparisons (based on https://github.com/tinygrad/tinygrad/blob/3e0c2d256fe9f4f5f85cd3e4d8733a51d7b4a984/tinygrad/tensor.py#L653)
impl GraphTensor {
    pub fn less_than(self, rhs: GraphTensor) -> GraphTensor {
        assert_eq!(self.dims(), rhs.dims(), "Dims must match to lt tensors.");
        let new_id = self
            .graph()
            .add_op(op::LessThan)
            .input(self.id, 0, self.shape)
            .input(rhs.id, 0, rhs.shape)
            .finish();
        GraphTensor::from_id(new_id, self.shape.contiguous(), self.graph_ref)
    }

    pub fn greater_than(self, rhs: GraphTensor) -> GraphTensor {
        rhs.less_than(self)
    }

    pub fn less_than_equal(self, rhs: GraphTensor) -> GraphTensor {
        -self.greater_than(rhs) + 1.0
    }

    pub fn greater_than_equal(self, rhs: GraphTensor) -> GraphTensor {
        -self.less_than(rhs) + 1.0
    }

    pub fn not_equals(self, rhs: GraphTensor) -> GraphTensor {
        self.less_than(rhs) + self.greater_than(rhs)
    }

    pub fn equals(self, rhs: GraphTensor) -> GraphTensor {
        -self.not_equals(rhs) + 1.0
    }

    /// Raise the tensor to a power
    pub fn pow<T>(self, e: T) -> GraphTensor
    where
        Self: Mul<T, Output = Self>,
    {
        // Approximate, see full impl here: https://github.com/tinygrad/tinygrad/blob/a32c67760140dd26b60d7932268f2e62e96a66e0/tinygrad/tensor.py#L568
        self.abs().ln().mul(e).exp()
    }
}

// Clipping ops (min, max, clip)
impl GraphTensor {
    /// Take the elementwise maximum of two tensors
    pub fn max(self, rhs: GraphTensor) -> GraphTensor {
        (self.less_than(rhs) * rhs) + (rhs.less_than_equal(self) * self)
    }

    /// Take the elementwise maximum of a tensor and a float
    pub fn max_f32(self, rhs: f32) -> GraphTensor {
        self.max(self.graph().constant(rhs).expand_to(self.shape))
    }

    /// Take the elementwise minimum of two tensors
    pub fn min(self, rhs: GraphTensor) -> GraphTensor {
        -(-self).max(-rhs)
    }

    /// Take the elementwise minimum of a tensor and a float
    pub fn min_f32(self, rhs: f32) -> GraphTensor {
        -(-self).max_f32(-rhs)
    }

    /// Clip (clamp) a tensor into the range [`min`, `max`]
    pub fn clip(self, min: f32, max: f32) -> GraphTensor {
        self.max_f32(min).min_f32(max)
    }
}

pub trait F32Pow {
    fn pow(self, e: GraphTensor) -> GraphTensor;
}

impl F32Pow for f32 {
    fn pow(self, e: GraphTensor) -> GraphTensor {
        e.mul(self.abs().ln()).exp().recip()
    }
}

#[cfg(test)]
mod tests {
    crate::test_imports!();

    #[test]
    fn test_clip() {
        let mut cx = Graph::new();
        let a = cx
            .tensor((3, 2))
            .set([[[-1.0], [-2.0], [3.0]], [[-1.5], [0.0], [5.0]]]);
        let result = a.clip(-1.5, 3.4).retrieve();
        let expected_result = cx
            .tensor((3, 2))
            .set([[[-1.0], [-1.5], [3.0]], [[-1.5], [0.0], [3.4]]])
            .retrieve();
        cx.execute();

        assert_close(&result.data(), &expected_result.data());
    }
}
