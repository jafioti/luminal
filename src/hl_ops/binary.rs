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
        // Use the Sub primitive directly to avoid circular dependency with negation
        let new_id = self
            .graph()
            .add_op(op::Sub)
            .input(self.id, 0, self.shape)
            .input(rhs.id, 0, rhs.shape)
            .finish();
        GraphTensor::from_id(new_id, self.shape.contiguous(), self.graph_ref)
    }
}

impl Sub<GraphTensor> for f32 {
    type Output = GraphTensor;

    fn sub(self, rhs: GraphTensor) -> Self::Output {
        rhs.graph().constant(self).expand(rhs.shape) - rhs
    }
}

impl SubAssign for GraphTensor {
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl Mul for GraphTensor {
    type Output = GraphTensor;

    /// Element-wise multiplication implemented without the `Mul` primitive.
    /// Uses the identity `a * b = 2^{log2(a) + log2(b)}`.
    fn mul(self, rhs: GraphTensor) -> Self::Output {
        assert_eq!(
            self.dims(),
            rhs.dims(),
            "Dims must match to multiply tensors."
        );

        // Use the Mul primitive directly to avoid circular dependency
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
        assert_eq!(
            self.dims(),
            rhs.dims(),
            "Dims must match to divide tensors."
        );

        // Use the Div primitive directly to avoid circular dependency
        let new_id = self
            .graph()
            .add_op(op::Div)
            .input(self.id, 0, self.shape)
            .input(rhs.id, 0, rhs.shape)
            .finish();
        GraphTensor::from_id(new_id, self.shape.contiguous(), self.graph_ref)
    }
}

#[allow(clippy::suspicious_arithmetic_impl)]
impl Div<GraphTensor> for f32 {
    type Output = GraphTensor;

    fn div(self, rhs: GraphTensor) -> Self::Output {
        rhs.graph().constant(self).expand(rhs.shape) / rhs
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
        self + self.graph().constant(rhs).expand(self.shape)
    }
}

impl<S: Into<Expression>> Add<S> for GraphTensor {
    type Output = GraphTensor;

    fn add(self, rhs: S) -> Self::Output {
        self + self.graph().constant(rhs).expand(self.shape)
    }
}

impl Sub<f32> for GraphTensor {
    type Output = GraphTensor;

    fn sub(self, rhs: f32) -> Self::Output {
        self - self.graph().constant(rhs).expand(self.shape)
    }
}

impl<S: Into<Expression>> Sub<S> for GraphTensor {
    type Output = GraphTensor;

    fn sub(self, rhs: S) -> Self::Output {
        self - self.graph().constant(rhs).expand(self.shape)
    }
}

impl Mul<f32> for GraphTensor {
    type Output = GraphTensor;

    fn mul(self, rhs: f32) -> Self::Output {
        if rhs == 0.0 {
            self.graph().constant(0.0).expand(self.shape)
        } else {
            self * self.graph().constant(rhs).expand(self.shape)
        }
    }
}

impl<S: Into<Expression>> Mul<S> for GraphTensor {
    type Output = GraphTensor;

    fn mul(self, rhs: S) -> Self::Output {
        self * self.graph().constant(rhs).expand(self.shape)
    }
}

#[allow(clippy::suspicious_arithmetic_impl)]
impl Div<f32> for GraphTensor {
    type Output = GraphTensor;

    fn div(self, rhs: f32) -> Self::Output {
        if rhs == 0.0 {
            panic!("Division by zero");
        }
        self / self.graph().constant(rhs).expand(self.shape)
    }
}

#[allow(clippy::suspicious_arithmetic_impl)]
impl<S: Into<Expression>> Div<S> for GraphTensor {
    type Output = GraphTensor;

    fn div(self, rhs: S) -> Self::Output {
        self / self.graph().constant(rhs).expand(self.shape)
    }
}

impl Rem<f32> for GraphTensor {
    type Output = GraphTensor;

    fn rem(self, rhs: f32) -> Self::Output {
        self % self.graph().constant(rhs).expand(self.shape)
    }
}

impl<S: Into<Expression>> Rem<S> for GraphTensor {
    type Output = GraphTensor;

    fn rem(self, rhs: S) -> Self::Output {
        self % self.graph().constant(rhs).expand(self.shape)
    }
}

// Comparisons (based on https://github.com/tinygrad/tinygrad/blob/3e0c2d256fe9f4f5f85cd3e4d8733a51d7b4a984/tinygrad/tensor.py#L653)
impl GraphTensor {
    pub fn lt(self, rhs: GraphTensor) -> GraphTensor {
        assert_eq!(self.dims(), rhs.dims(), "Dims must match to lt tensors.");
        let new_id = self
            .graph()
            .add_op(op::LessThan)
            .input(self.id, 0, self.shape)
            .input(rhs.id, 0, rhs.shape)
            .finish();
        GraphTensor::from_id(new_id, self.shape.contiguous(), self.graph_ref)
    }

    pub fn gt(self, rhs: GraphTensor) -> GraphTensor {
        assert_eq!(self.dims(), rhs.dims(), "Dims must match to gt tensors.");
        let new_id = self
            .graph()
            .add_op(op::GreaterThan)
            .input(self.id, 0, self.shape)
            .input(rhs.id, 0, rhs.shape)
            .finish();
        GraphTensor::from_id(new_id, self.shape.contiguous(), self.graph_ref)
    }

    pub fn le(self, rhs: GraphTensor) -> GraphTensor {
        assert_eq!(self.dims(), rhs.dims(), "Dims must match to le tensors.");
        let new_id = self
            .graph()
            .add_op(op::LessEqual)
            .input(self.id, 0, self.shape)
            .input(rhs.id, 0, rhs.shape)
            .finish();
        GraphTensor::from_id(new_id, self.shape.contiguous(), self.graph_ref)
    }

    pub fn ge(self, rhs: GraphTensor) -> GraphTensor {
        assert_eq!(self.dims(), rhs.dims(), "Dims must match to ge tensors.");
        let new_id = self
            .graph()
            .add_op(op::GreaterEqual)
            .input(self.id, 0, self.shape)
            .input(rhs.id, 0, rhs.shape)
            .finish();
        GraphTensor::from_id(new_id, self.shape.contiguous(), self.graph_ref)
    }

    pub fn ne(self, rhs: GraphTensor) -> GraphTensor {
        assert_eq!(self.dims(), rhs.dims(), "Dims must match to ne tensors.");
        let new_id = self
            .graph()
            .add_op(op::NotEqual)
            .input(self.id, 0, self.shape)
            .input(rhs.id, 0, rhs.shape)
            .finish();
        GraphTensor::from_id(new_id, self.shape.contiguous(), self.graph_ref)
    }

    pub fn eq(self, rhs: GraphTensor) -> GraphTensor {
        assert_eq!(self.dims(), rhs.dims(), "Dims must match to eq tensors.");
        let new_id = self
            .graph()
            .add_op(op::Equal)
            .input(self.id, 0, self.shape)
            .input(rhs.id, 0, rhs.shape)
            .finish();
        GraphTensor::from_id(new_id, self.shape.contiguous(), self.graph_ref)
    }

    /// Raise the tensor to a power
    pub fn pow<T>(self, e: T) -> GraphTensor
    where
        Self: Mul<T, Output = Self>,
    {
        // Approximate, see full impl here: https://github.com/tinygrad/tinygrad/blob/a32c67760140dd26b60d7932268f2e62e96a66e0/tinygrad/tensor.py#L568
        self.abs().log().mul(e).exp()
    }
}

// Clipping ops (minimum, maximum, clip)
impl GraphTensor {
    /// Take the elementwise maximum of two tensors
    pub fn maximum(self, rhs: GraphTensor) -> GraphTensor {
        (self.lt(rhs) * rhs) + (rhs.le(self) * self)
    }

    /// Take the elementwise maximum of a tensor and a float
    pub fn maximum_f32(self, rhs: f32) -> GraphTensor {
        self.maximum(self.graph().constant(rhs).expand(self.shape))
    }

    /// Take the elementwise minimum of two tensors
    pub fn minimum(self, rhs: GraphTensor) -> GraphTensor {
        (self.gt(rhs) * rhs) + (rhs.ge(self) * self)
    }

    /// Take the elementwise minimum of a tensor and a float
    pub fn minimum_f32(self, rhs: f32) -> GraphTensor {
        self.minimum(self.graph().constant(rhs).expand(self.shape))
    }

    /// Clip (clamp) a tensor into the range [`min`, `max`]
    pub fn clip(self, min: f32, max: f32) -> GraphTensor {
        self.maximum_f32(min).minimum_f32(max)
    }
}

pub trait F32Pow {
    fn pow(self, e: GraphTensor) -> GraphTensor;
}

impl F32Pow for f32 {
    fn pow(self, e: GraphTensor) -> GraphTensor {
        e.mul(self.abs().ln()).exp()
    }
}

#[cfg(test)]
mod tests {
    use super::F32Pow;

    crate::test_imports!();

    #[test]
    fn test_clip() {
        let mut cx = Graph::new();
        let a = cx
            .tensor((3, 2))
            .set([[-1.0, -2.0], [3.0, -1.5], [0.0, 5.0]]);
        let result = a.clip(-1.5, 3.4).retrieve();
        let expected_result = cx
            .tensor((3, 2))
            .set([[-1.0, -1.5], [3.0, -1.5], [0.0, 3.4]])
            .retrieve();
        cx.execute();

        assert_close(&result.data(), &expected_result.data());
    }

    #[test]
    fn test_pow() {
        let base = 2_f32;
        let mut cx = Graph::new();
        let a = cx
            .tensor((3, 2))
            .set([[[-1.0], [-2.0], [3.0]], [[1.0], [0.0], [5.0]]]);

        let expected_result = cx
            .tensor((3, 2))
            .set([[[0.5], [0.25], [8.0]], [[2.0], [1.0], [32.0]]])
            .retrieve();

        let result = base.pow(a).retrieve();
        cx.execute();

        assert_close(&result.data(), &expected_result.data());
    }
}
