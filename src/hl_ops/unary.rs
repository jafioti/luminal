use crate::{op, prelude::*};
use std::ops::{Add, Mul, Neg};

impl Neg for GraphTensor {
    type Output = GraphTensor;

    fn neg(self) -> Self::Output {
        self * -1.0
    }
}

impl GraphTensor {
    /// Base 2 log
    pub fn log2(self) -> GraphTensor {
        let new_id = self
            .graph()
            .add_op(op::Log2)
            .input(self.id, 0, self.shape)
            .finish();
        GraphTensor::from_id(new_id, self.shape.contiguous(), self.graph_ref)
    }

    /// Base 2 exp
    pub fn exp2(self) -> GraphTensor {
        let new_id = self
            .graph()
            .add_op(op::Exp2)
            .input(self.id, 0, self.shape)
            .finish();
        GraphTensor::from_id(new_id, self.shape.contiguous(), self.graph_ref)
    }

    /// Natural exp
    pub fn exp(self) -> GraphTensor {
        (self * (1.0 / f32::ln(2.))).exp2()
    }

    /// Natural log
    pub fn log(self) -> GraphTensor {
        self.log2() * f32::ln(2.)
    }

    /// Take the reciprocal of each element
    pub fn reciprocal(self) -> GraphTensor {
        let new_id = self
            .graph()
            .add_op(op::Recip)
            .input(self.id, 0, self.shape)
            .finish();
        GraphTensor::from_id(new_id, self.shape.contiguous(), self.graph_ref)
    }

    /// The sin(x) function
    pub fn sin(self) -> GraphTensor {
        let new_id = self
            .graph()
            .add_op(op::Sin)
            .input(self.id, 0, self.shape)
            .finish();
        GraphTensor::from_id(new_id, self.shape.contiguous(), self.graph_ref)
    }

    /// The cos(x) function
    pub fn cos(self) -> GraphTensor {
        ((std::f32::consts::PI / 2.) - self).sin()
    }

    /// Square every element in the tensor
    pub fn square(self) -> GraphTensor {
        self * self
    }

    /// The square root function
    pub fn sqrt(self) -> GraphTensor {
        let new_id = self
            .graph()
            .add_op(op::Sqrt)
            .input(self.id, 0, self.shape)
            .finish();
        GraphTensor::from_id(new_id, self.shape.contiguous(), self.graph_ref)
    }

    /// Scale so std is 1.0
    pub fn std_norm<T>(self, axes: impl ToAxes, epsilon: T) -> GraphTensor
    where
        GraphTensor: Add<T, Output = GraphTensor>,
    {
        (self * self)
            .mean(axes)
            .add(epsilon)
            .sqrt()
            .reciprocal()
            .expand_to(self.shape)
            .mul(self)
    }

    /// Center so mean is 0.0
    pub fn mean_norm(self, axes: impl ToAxes) -> GraphTensor {
        self - self.mean(axes).expand_to(self.shape)
    }

    /// Applies a layer norm along an axis
    pub fn layer_norm<T>(self, axes: impl ToAxes, epsilon: T) -> GraphTensor
    where
        GraphTensor: Add<T, Output = GraphTensor>,
    {
        self.mean_norm(axes.to_axes()).std_norm(axes, epsilon)
    }

    /// Applies a softmax function along an axis
    pub fn softmax(self, axes: impl ToAxes) -> GraphTensor {
        let m = self - self.max(axes.to_axes()).expand_to(self.shape);
        let exp = m.exp();
        exp / exp.sum(axes).expand_to(exp.shape)
    }

    /// Applies a log softmax function along an axis
    pub fn log_softmax(self, axes: impl ToAxes) -> GraphTensor {
        let m = self - self.max(axes.to_axes()).expand_to(self.shape);
        m - m.exp().sum(axes.to_axes()).log().expand_to(m.shape)
    }

    /// Get the indicies of the max elements along the specified axis
    pub fn argmax(self, axis: usize) -> GraphTensor {
        assert!(axis < self.shape.len(), "axis out of bounds");
        // Get one-hot along the axis
        let x_equal = self.eq(self.max(axis).expand_to(self.shape));

        // Create index arange for the axis dimension and expand to tensor shape
        let dims = self.dims();
        let mut r = self.graph().arange(dims[axis]);
        for i in (0..axis).rev() {
            r = r.expand(0, dims[i]);
        }
        for i in axis + 1..dims.len() {
            r = r.expand(r.shape.len(), dims[i]);
        }

        // Multiply one-hot by expanded index arange
        (x_equal * r).max(axis)
    }

    /// Take the absolute value
    pub fn abs(self) -> GraphTensor {
        self.relu() + (-self).relu()
    }

    /// Get the sign of each element, '1' for positive and '-1' for negative
    pub fn sign(self) -> GraphTensor {
        self / (self.abs() + 1e-10)
    }

    /// The Rectified Linear Unit activation function
    pub fn relu(self) -> GraphTensor {
        self.maximum_f32(0.)
    }

    /// The sigmoid activation function
    pub fn sigmoid(self) -> GraphTensor {
        // Based on https://github.com/tinygrad/tinygrad/blob/9d142430cbe61121c864c0015f1de83c94a7d2c0/tinygrad/mlops.py#L70
        1. / (1. + (-self).exp())
    }

    /// The swish (aka silu) activation function
    pub fn swish(self) -> GraphTensor {
        self * self.sigmoid()
    }

    /// The silu (aka swish) activation function
    pub fn silu(self) -> GraphTensor {
        self.swish()
    }

    /// The tanh activation function
    pub fn tanh(self) -> GraphTensor {
        (self * 2.0).sigmoid() * 2.0 - 1.0
    }

    /// The leaky relu activation function
    pub fn leaky_relu(self, neg_slope: f32) -> GraphTensor {
        self.relu() - (self * -neg_slope).relu()
    }

    /// The Gaussian Error Linear Unit activation function
    #[allow(clippy::excessive_precision)]
    pub fn gelu(self) -> GraphTensor {
        // Based on https://github.com/tinygrad/tinygrad/blob/9fc4465557831b614b56dd645eebc940ca0fa1bb/tinygrad/tensor.py#L1162C26-L1162C104
        0.5 * self * (1. + (0.7978845608 * self * (1. + 0.044715 * self * self)).tanh())
    }
}

#[cfg(test)]
mod tests {
    crate::test_imports!();

    #[test]
    fn test_exp() {
        let mut cx = Graph::new();
        let a_data = random_vec(6);
        let a = cx.tensor((2, 3)).set(a_data.clone());
        let b = a.exp().retrieve();
        cx.execute();

        let d_dev = Cpu::default();
        let d_a = d_dev.tensor_from_vec(a_data, (DConst::<2>, DConst::<3>));
        let d_b = d_a.exp();

        assert_close(&b.data(), &d_b.as_vec());
    }

    #[test]
    fn test_layer_norm() {
        let mut cx = Graph::new();
        let a_data = random_vec(6);
        let a = cx.tensor((2, 3)).set(a_data.clone());
        let b = a.layer_norm(0, 1e-5).retrieve();
        let c = a.layer_norm(1, 1e-5).retrieve();
        cx.execute();

        let d_dev = Cpu::default();
        let d_a = d_dev.tensor_from_vec(a_data, (DConst::<2>, DConst::<3>));
        let d_b = d_a.clone().normalize::<DAxis<0>>(1e-5);
        let d_c = d_a.normalize::<DAxis<1>>(1e-5);

        assert_close(&b.data(), &d_b.as_vec());
        assert_close(&c.data(), &d_c.as_vec());
    }

    #[test]
    fn test_softmax() {
        let mut cx = Graph::new();
        let a_data = random_vec(6);
        let a = cx.tensor((2, 3)).set(a_data.clone());
        let b = a.softmax(1).retrieve();

        cx.execute();

        let d_dev = Cpu::default();
        let d_a = d_dev.tensor_from_vec(a_data, (DConst::<2>, DConst::<3>));
        let d_b = d_a.softmax::<DAxis<1>>();

        let r = b.data();
        assert_close(&r, &d_b.as_vec());
    }

    #[test]
    fn test_sin() {
        let mut cx = Graph::new();
        let a_data = random_vec(6);
        let a = cx.tensor((2, 3)).set(a_data.clone());
        let b = a.sin().retrieve();

        cx.execute();

        let d_dev = Cpu::default();
        let d_a = d_dev.tensor_from_vec(a_data, (DConst::<2>, DConst::<3>));
        let d_b = d_a.sin();

        let r = b.data();
        assert_close(&r, &d_b.as_vec());
    }

    #[test]
    fn test_cos() {
        let mut cx = Graph::new();
        let a_data = random_vec(6);
        let a = cx.tensor((2, 3)).set(a_data.clone());
        let b = a.cos().retrieve();

        cx.execute();

        let d_dev = Cpu::default();
        let d_a = d_dev.tensor_from_vec(a_data, (DConst::<2>, DConst::<3>));
        let d_b = d_a.cos();
        assert_close(&b.data(), &d_b.as_vec());
    }

    #[test]
    fn test_relu() {
        let mut cx = Graph::new();
        let a_data = random_vec(4);
        let a = cx.tensor(('a', 'b')).set_dyn(a_data.clone(), (2, 2));
        let b = a.relu().retrieve();

        cx.execute();

        let d_dev = Cpu::default();
        let d_a = d_dev.tensor_from_vec(a_data, (DConst::<2>, DConst::<2>));
        let d_b = d_a.relu();
        assert_close(&b.data(), &d_b.as_vec());
    }

    #[test]
    fn test_gelu() {
        let mut cx = Graph::new();
        let a_data = random_vec(4);
        let a = cx.tensor(('a', 'b')).set_dyn(a_data.clone(), (2, 2));
        let b = a.gelu().retrieve();

        cx.execute();

        let d_dev = Cpu::default();
        let d_a = d_dev.tensor_from_vec(a_data, (DConst::<2>, DConst::<2>));
        let d_b = d_a.fast_gelu();
        assert_close(&b.data(), &d_b.as_vec());
    }

    #[test]
    fn test_sigmoid() {
        let mut cx = Graph::new();
        let a_data = random_vec(4);
        let a = cx.tensor(('a', 'b')).set_dyn(a_data.clone(), (2, 2));
        let b = a.sigmoid().retrieve();

        cx.execute();

        let d_dev = Cpu::default();
        let d_a = d_dev.tensor_from_vec(a_data, (DConst::<2>, DConst::<2>));
        let d_b = d_a.sigmoid();
        assert_close(&b.data(), &d_b.as_vec());
    }

    #[test]
    fn test_swish() {
        let mut cx = Graph::new();
        let a_data = random_vec(4);
        let a = cx.tensor(('a', 'b')).set_dyn(a_data.clone(), (2, 2));
        let b = a.swish().retrieve();
        cx.execute();

        let d_dev = Cpu::default();
        let d_a = d_dev.tensor_from_vec(a_data, (DConst::<2>, DConst::<2>));
        let d_b = d_a.clone() * d_a.sigmoid();
        assert_close(&b.data(), &d_b.as_vec());
    }

    #[test]
    fn test_argmax_dim1() {
        let mut cx = Graph::new();
        let a_data = random_vec(6);
        let a = cx.tensor((2, 3)).set(a_data.clone());
        let b = a.argmax(1).retrieve();

        cx.execute();

        let mut expected = Vec::new();
        for row in 0..2 {
            let mut max_val = f32::MIN;
            let mut idx = 0usize;
            for col in 0..3 {
                let v = a_data[row * 3 + col];
                if v > max_val {
                    max_val = v;
                    idx = col;
                }
            }
            expected.push(idx as f32);
        }

        assert_exact(&b.data(), &expected);
    }

    #[test]
    fn test_argmax_dim0() {
        let mut cx = Graph::new();
        let a_data = random_vec(6);
        let a = cx.tensor((2, 3)).set(a_data.clone());
        let b = a.argmax(0).retrieve();

        cx.execute();

        let mut expected = Vec::new();
        for col in 0..3 {
            let mut max_val = f32::MIN;
            let mut idx = 0usize;
            for row in 0..2 {
                let v = a_data[row * 3 + col];
                if v > max_val {
                    max_val = v;
                    idx = row;
                }
            }
            expected.push(idx as f32);
        }

        assert_exact(&b.data(), &expected);
    }

    #[test]
    fn test_tanh() {
        let mut cx = Graph::new();
        let a_data = random_vec(4);
        let a = cx.tensor(('a', 'b')).set_dyn(a_data.clone(), (2, 2));
        let b = a.tanh().retrieve();

        cx.execute();

        let d_dev = Cpu::default();
        let d_a = d_dev.tensor_from_vec(a_data, (DConst::<2>, DConst::<2>));
        let d_b = d_a.tanh();
        assert_close(&b.data(), &d_b.as_vec());
    }
}
