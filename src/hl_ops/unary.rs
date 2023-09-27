use std::ops::{Add, Mul};

use crate::{op, prelude::*};

impl<S: Shape> GraphTensor<S> {
    /// Base 2 log
    pub fn log_2(self) -> GraphTensor<S> {
        let new_id = self
            .graph()
            .add_op(op::Log2)
            .input(self.id, 0, self.shape)
            .finish();
        GraphTensor::from_id(new_id, self.shape, self.graph_ref)
    }

    /// Base 2 exp
    pub fn exp_2(self) -> GraphTensor<S> {
        let new_id = self
            .graph()
            .add_op(op::Exp2)
            .input(self.id, 0, self.shape)
            .finish();
        GraphTensor::from_id(new_id, self.shape, self.graph_ref)
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
        let new_id = self
            .graph()
            .add_op(op::Recip)
            .input(self.id, 0, self.shape)
            .finish();
        GraphTensor::from_id(new_id, self.shape, self.graph_ref)
    }

    pub fn sin(self) -> GraphTensor<S> {
        let new_id = self
            .graph()
            .add_op(op::Sin)
            .input(self.id, 0, self.shape)
            .finish();
        GraphTensor::from_id(new_id, self.shape, self.graph_ref)
    }

    pub fn cos(self) -> GraphTensor<S> {
        (-self + (std::f32::consts::PI / 2.)).sin()
    }

    pub fn sqrt(self) -> GraphTensor<S> {
        let new_id = self
            .graph()
            .add_op(op::Sqrt)
            .input(self.id, 0, self.shape)
            .finish();
        GraphTensor::from_id(new_id, self.shape, self.graph_ref)
    }

    pub fn layer_norm<const DIM: isize>(self) -> GraphTensor<S>
    where
        <S as ReduceShape<Axis<DIM>>>::Reduced: Shape,
        S: ReduceShape<Axis<DIM>>,
    {
        let centered = self
            - self
                .mean_reduce::<<S as ReduceShape<Axis<DIM>>>::Reduced, _>()
                .expand();
        let std = centered
            .mul(centered)
            .mean_reduce::<<S as ReduceShape<Axis<DIM>>>::Reduced, _>()
            .add(1e-5)
            .sqrt();
        centered / std.expand()
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

    /// The Rectified Linear Unit activation function
    pub fn relu(self) -> GraphTensor<S> {
        self.max_f32(0.)
    }

    /// The sigmoid activation function
    pub fn sigmoid(self) -> GraphTensor<S> {
        // Based on https://github.com/tinygrad/tinygrad/blob/9d142430cbe61121c864c0015f1de83c94a7d2c0/tinygrad/mlops.py#L70
        let one = self.graph().constant(1.0);
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

#[cfg(test)]
mod tests {
    crate::test_imports!();

    #[test]
    fn test_exp() {
        let mut cx = Graph::new();
        let a_data = random_vec(6);
        let a = cx.new_tensor::<R2<2, 3>>("Input");
        a.set(a_data.clone());
        let b = a.exp();
        b.mark();
        cx.execute();

        let d_dev = Cpu::default();
        let d_a = d_dev.tensor_from_vec(a_data, (DConst::<2>, DConst::<3>));
        let d_b = d_a.exp();

        assert_close_data(&b.data(), &d_b.as_vec());
    }

    #[test]
    fn test_layer_norm() {
        let mut cx = Graph::new();
        let a_data = random_vec(6);
        let a = cx.new_tensor::<R2<2, 3>>("Input");
        a.set(a_data.clone());
        let b = a.layer_norm::<0>();
        let c = a.layer_norm::<1>();
        b.mark();
        c.mark();
        cx.execute();

        let d_dev = Cpu::default();
        let d_a = d_dev.tensor_from_vec(a_data, (DConst::<2>, DConst::<3>));
        let d_b = d_a.clone().normalize::<DAxis<0>>(1e-5);
        let d_c = d_a.normalize::<DAxis<1>>(1e-5);

        assert_close_data(&b.data(), &d_b.as_vec());
        assert_close_data(&c.data(), &d_c.as_vec());
    }

    #[test]
    fn test_softmax() {
        let mut cx = Graph::new();
        let a_data = random_vec(6);
        let a = cx.new_tensor::<R2<2, 3>>("Input");
        a.set(a_data.clone());
        let b = a.softmax::<1>();
        b.mark();

        cx.execute();

        let d_dev = Cpu::default();
        let d_a = d_dev.tensor_from_vec(a_data, (DConst::<2>, DConst::<3>));
        let d_b = d_a.softmax::<DAxis<1>>();

        let r = b.data();
        assert_close_data(&r, &d_b.as_vec());
    }

    #[test]
    fn test_sin() {
        let mut cx = Graph::new();
        let a_data = random_vec(6);
        let a = cx.new_tensor::<R2<2, 3>>("Input");
        a.set(a_data.clone());
        let b = a.sin();
        b.mark();

        cx.execute();

        let d_dev = Cpu::default();
        let d_a = d_dev.tensor_from_vec(a_data, (DConst::<2>, DConst::<3>));
        let d_b = d_a.sin();

        let r = b.data();
        assert_close_data(&r, &d_b.as_vec());
    }

    #[test]
    fn test_cos() {
        let mut cx = Graph::new();
        let a_data = random_vec(6);
        let a = cx.new_tensor::<R2<2, 3>>("Input");
        a.set(a_data.clone());
        let b = a.cos();
        b.mark();

        cx.execute();

        let d_dev = Cpu::default();
        let d_a = d_dev.tensor_from_vec(a_data, (DConst::<2>, DConst::<3>));
        let d_b = d_a.cos();
        assert_close_data(&b.data(), &d_b.as_vec());
    }

    #[test]
    fn test_relu() {
        let mut cx = Graph::new();
        let a_data = random_vec(4);
        let a = cx.new_tensor::<(Dyn<'a'>, Dyn<'b'>)>("Input");
        a.set_dyn(a_data.clone(), vec![2, 2]);
        let b = a.relu();
        b.mark();

        cx.execute();

        let d_dev = Cpu::default();
        let d_a = d_dev.tensor_from_vec(a_data, (DConst::<2>, DConst::<2>));
        let d_b = d_a.relu();
        assert_close_data(&b.dyn_data(&cx.dyn_map), &d_b.as_vec());
    }

    #[test]
    fn test_sigmoid() {
        let mut cx = Graph::new();
        let a_data = random_vec(4);
        let a = cx.new_tensor::<(Dyn<'a'>, Dyn<'b'>)>("Input");
        a.set_dyn(a_data.clone(), vec![2, 2]);
        let b = a.sigmoid();
        b.mark();

        cx.execute();

        let d_dev = Cpu::default();
        let d_a = d_dev.tensor_from_vec(a_data, (DConst::<2>, DConst::<2>));
        let d_b = d_a.sigmoid();
        assert_close_data(&b.dyn_data(&cx.dyn_map), &d_b.as_vec());
    }

    #[test]
    fn test_tanh() {
        let mut cx = Graph::new();
        let a_data = random_vec(4);
        let a = cx.new_tensor::<(Dyn<'a'>, Dyn<'b'>)>("Input");
        a.set_dyn(a_data.clone(), vec![2, 2]);
        let b = a.tanh();
        b.mark();

        cx.execute();

        let d_dev = Cpu::default();
        let d_a = d_dev.tensor_from_vec(a_data, (DConst::<2>, DConst::<2>));
        let d_b = d_a.tanh();
        assert_close_data(&b.dyn_data(&cx.dyn_map), &d_b.as_vec());
    }
}
