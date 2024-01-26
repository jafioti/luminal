use crate::{op, prelude::*};
use std::ops::{Add, Mul, Neg};

impl<S: Shape> Neg for GraphTensor<S> {
    type Output = GraphTensor<S>;

    fn neg(self) -> Self::Output {
        self * -1.0
    }
}

impl<S: Shape> GraphTensor<S> {
    /// Base 2 log
    pub fn log2(self) -> GraphTensor<S> {
        let new_id = self
            .graph()
            .add_op(op::Log2)
            .input(self.id, 0, self.shape)
            .finish();
        GraphTensor::from_id(new_id, self.shape, self.graph_ref)
    }

    /// Base 2 exp
    pub fn exp2(self) -> GraphTensor<S> {
        let new_id = self
            .graph()
            .add_op(op::Exp2)
            .input(self.id, 0, self.shape)
            .finish();
        GraphTensor::from_id(new_id, self.shape, self.graph_ref)
    }

    /// Natural exp
    pub fn exp(self) -> GraphTensor<S> {
        (self * (1.0 / f32::ln(2.))).exp2()
    }

    /// Natural log
    pub fn ln(self) -> GraphTensor<S> {
        self.log2() * f32::ln(2.)
    }

    /// Take the reciprocal of each element
    pub fn recip(self) -> GraphTensor<S> {
        let new_id = self
            .graph()
            .add_op(op::Recip)
            .input(self.id, 0, self.shape)
            .finish();
        GraphTensor::from_id(new_id, self.shape, self.graph_ref)
    }

    /// The sin(x) function
    pub fn sin(self) -> GraphTensor<S> {
        let new_id = self
            .graph()
            .add_op(op::Sin)
            .input(self.id, 0, self.shape)
            .finish();
        GraphTensor::from_id(new_id, self.shape, self.graph_ref)
    }

    /// The cos(x) function
    pub fn cos(self) -> GraphTensor<S> {
        (-self + (std::f32::consts::PI / 2.)).sin()
    }

    /// The square root function
    pub fn sqrt(self) -> GraphTensor<S> {
        let new_id = self
            .graph()
            .add_op(op::Sqrt)
            .input(self.id, 0, self.shape)
            .finish();
        GraphTensor::from_id(new_id, self.shape, self.graph_ref)
    }

    /// Scale so std is 1.0
    pub fn std_norm<const DIM: usize>(self, epsilon: f32) -> GraphTensor<S>
    where
        <S as ReduceShape<Axis<DIM>>>::Reduced: Shape,
        S: ReduceShape<Axis<DIM>>,
    {
        (self * self)
            .mean_reduce::<<S as ReduceShape<Axis<DIM>>>::Reduced, _>()
            .add(epsilon)
            .sqrt()
            .recip()
            .expand()
            .mul(self)
    }

    /// Center so mean is 0.0
    pub fn mean_norm<const DIM: usize>(self) -> GraphTensor<S>
    where
        <S as ReduceShape<Axis<DIM>>>::Reduced: Shape,
        S: ReduceShape<Axis<DIM>>,
    {
        self - self
            .mean_reduce::<<S as ReduceShape<Axis<DIM>>>::Reduced, _>()
            .expand()
    }

    /// Applies a layer norm along an axis
    pub fn layer_norm<const DIM: usize>(self, epsilon: f32) -> GraphTensor<S>
    where
        <S as ReduceShape<Axis<DIM>>>::Reduced: Shape,
        S: ReduceShape<Axis<DIM>>,
    {
        self.mean_norm().std_norm(epsilon)
    }

    /// Applies a softmax function along an axis
    pub fn softmax<const DIM: usize>(self) -> GraphTensor<S>
    where
        <S as ReduceShape<Axis<DIM>>>::Reduced: Shape,
        S: ReduceShape<Axis<DIM>>,
    {
        let m = self
            - self
                .max_reduce::<<S as ReduceShape<Axis<DIM>>>::Reduced, _>()
                .expand();
        let exp = m.exp();
        let exp_sum = exp.sum_reduce::<<S as ReduceShape<Axis<DIM>>>::Reduced, _>();
        exp / exp_sum.expand()
    }

    /// Get the indicies of the max elements along the last axis
    pub fn argmax(self) -> GraphTensor<<S as ReduceShape<<S as Shape>::LastAxis>>::Reduced> {
        let x_equal = self.equals(self.max_reduce::<_, S::LastAxis>().expand());
        // ARange to shape
        let r = self.graph().constant(1.).expand().cumsum_last_dim() - 1.;
        (x_equal * r).max_reduce::<_, S::LastAxis>()
    }

    /// Take the absolute value
    pub fn abs(self) -> GraphTensor<S> {
        self.relu() + (-self).relu()
    }

    /// Get the sign of each element, '1' for positive and '-1' for negative
    pub fn sign(self) -> GraphTensor<S> {
        self / (self.abs() + 1e-10)
    }

    /// Raise the tensor to a power
    /// Approximate, see full impl here: https://github.com/tinygrad/tinygrad/blob/a32c67760140dd26b60d7932268f2e62e96a66e0/tinygrad/tensor.py#L568
    pub fn pow(self, e: f32) -> GraphTensor<S> {
        self.abs().ln().mul(e).exp()
    }

    /// 1 / (base ^ x)
    pub fn inv_pow(self, base: f32) -> GraphTensor<S> {
        self.mul(base.abs().ln()).exp()
    }

    /// The Rectified Linear Unit activation function
    pub fn relu(self) -> GraphTensor<S> {
        self.max_f32(0.)
    }

    /// The sigmoid activation function
    pub fn sigmoid(self) -> GraphTensor<S> {
        // Based on https://github.com/tinygrad/tinygrad/blob/9d142430cbe61121c864c0015f1de83c94a7d2c0/tinygrad/mlops.py#L70
        let one = self.graph().constant(1.0);
        one.expand() / (one.expand() + (-self).exp())
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

#[cfg(test)]
mod tests {
    crate::test_imports!();

    #[test]
    fn test_exp() {
        let mut cx = Graph::new();
        let a_data = random_vec(6);
        let a = cx.tensor::<R2<2, 3>>().set(a_data.clone());
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
        let a = cx.tensor::<R2<2, 3>>().set(a_data.clone());
        let b = a.layer_norm::<0>(1e-5).retrieve();
        let c = a.layer_norm::<1>(1e-5).retrieve();
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
        let a = cx.tensor::<R2<2, 3>>().set(a_data.clone());
        let b = a.softmax::<1>().retrieve();

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
        let a = cx.tensor::<R2<2, 3>>().set(a_data.clone());
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
        let a = cx.tensor::<R2<2, 3>>().set(a_data.clone());
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
        let a = cx
            .tensor::<(Dyn<'a'>, Dyn<'b'>)>()
            .set_dyn(a_data.clone(), vec![2, 2]);
        let b = a.relu().retrieve();

        cx.execute();

        let d_dev = Cpu::default();
        let d_a = d_dev.tensor_from_vec(a_data, (DConst::<2>, DConst::<2>));
        let d_b = d_a.relu();
        assert_close(&b.data(), &d_b.as_vec());
    }

    #[test]
    fn test_sigmoid() {
        let mut cx = Graph::new();
        let a_data = random_vec(4);
        let a = cx
            .tensor::<(Dyn<'a'>, Dyn<'b'>)>()
            .set_dyn(a_data.clone(), vec![2, 2]);
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
        let a = cx
            .tensor::<(Dyn<'a'>, Dyn<'b'>)>()
            .set_dyn(a_data.clone(), vec![2, 2]);
        let b = a.swish().retrieve();
        cx.execute();

        let d_dev = Cpu::default();
        let d_a = d_dev.tensor_from_vec(a_data, (DConst::<2>, DConst::<2>));
        let d_b = d_a.clone() * d_a.sigmoid();
        assert_close(&b.data(), &d_b.as_vec());
    }

    #[test]
    fn test_tanh() {
        let mut cx = Graph::new();
        let a_data = random_vec(4);
        let a = cx
            .tensor::<(Dyn<'a'>, Dyn<'b'>)>()
            .set_dyn(a_data.clone(), vec![2, 2]);
        let b = a.tanh().retrieve();

        cx.execute();

        let d_dev = Cpu::default();
        let d_a = d_dev.tensor_from_vec(a_data, (DConst::<2>, DConst::<2>));
        let d_b = d_a.tanh();
        assert_close(&b.data(), &d_b.as_vec());
    }
}
