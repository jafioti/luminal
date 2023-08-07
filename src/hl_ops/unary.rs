use std::ops::{Add, Mul};

use crate::{op, prelude::*};

impl<S: Shape> GraphTensor<S> {
    /// Base 2 log
    pub fn log_2(self) -> GraphTensor<S> {
        let graph = unsafe { self.graph_ref.as_mut().unwrap() };
        let shape = self.shape_tracker();
        let new_id = graph
            .add_op(op::Log2, shape.clone())
            .input(self.id)
            .finish();
        GraphTensor::from_id(new_id, self.graph_ref)
    }

    /// Base 2 exp
    pub fn exp_2(self) -> GraphTensor<S> {
        let graph = unsafe { self.graph_ref.as_mut().unwrap() };
        let shape = self.shape_tracker();
        let new_id = graph
            .add_op(op::Exp2, shape.clone())
            .input(self.id)
            .finish();
        GraphTensor::from_id(new_id, self.graph_ref)
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

    pub fn cos(self) -> GraphTensor<S> {
        (-self + (std::f32::consts::PI / 2.)).sin()
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

    pub fn layer_norm<const DIM: isize>(self) -> GraphTensor<S>
    where
        <S as ReduceShape<Axis<DIM>>>::Reduced: Shape,
        S: ReduceShape<Axis<DIM>>,
    {
        let mean = self
            .mean_reduce::<<S as ReduceShape<Axis<DIM>>>::Reduced, _>()
            .expand();
        let centered = self - mean;
        let std = centered
            .mul(centered)
            .mean_reduce::<<S as ReduceShape<Axis<DIM>>>::Reduced, _>()
            .expand()
            .add(1e-5)
            .sqrt();
        centered / std
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
}

#[cfg(test)]
mod tests {
    use crate::{prelude::*, tests::assert_close_data};
    use dfdx::prelude::*;

    #[test]
    fn test_layer_norm() {
        let mut cx = Graph::new();
        let a = cx.new_tensor::<R2<2, 3>>("Input");
        a.set(vec![1., 2., 3., 3., 1., 3.]);
        let b = a.layer_norm::<0>();
        let c = a.layer_norm::<1>();
        b.mark();
        c.mark();
        cx.execute();

        let d_dev = Cpu::default();
        let d_a = d_dev.tensor([[1., 2., 3.], [3., 1., 3.]]);
        let d_b = d_a.clone().normalize::<dfdx::shapes::Axis<0>>(1e-5);
        let d_c = d_a.normalize::<dfdx::shapes::Axis<1>>(1e-5);

        assert_close_data(
            &b.retrieve().unwrap().real_data(b.view().unwrap()).unwrap(),
            &d_b.as_vec(),
        );
        assert_close_data(
            &c.retrieve().unwrap().real_data(c.view().unwrap()).unwrap(),
            &d_c.as_vec(),
        );
    }

    #[test]
    fn test_softmax() {
        let mut cx = Graph::new();
        let a = cx.new_tensor::<R2<2, 3>>("Input");
        a.set(vec![
            5.51743, 6.896794, 5.51743, 5.528703, 6.9108624, 5.528703,
        ]);
        let b = a.softmax::<1>();
        b.mark();

        cx.execute();

        let d_dev = Cpu::default();
        let d_a = d_dev.tensor_from_vec(
            vec![5.51743, 6.896794, 5.51743, 5.528703, 6.9108624, 5.528703],
            (dfdx::shapes::Const::<2>, dfdx::shapes::Const::<3>),
        );
        let d_b = d_a.softmax::<dfdx::shapes::Axis<1>>();

        let r = b.retrieve().unwrap().real_data(b.view().unwrap()).unwrap();
        assert_close_data(&r, &d_b.as_vec());
    }

    #[test]
    fn test_sin() {
        let mut cx = Graph::new();
        let a = cx.new_tensor::<R2<2, 3>>("Input");
        a.set(vec![
            5.51743, 6.896794, 5.51743, 5.528703, 6.9108624, 5.528703,
        ]);
        let b = a.sin();
        b.mark();

        cx.execute();

        let d_dev = Cpu::default();
        let d_a = d_dev.tensor_from_vec(
            vec![5.51743, 6.896794, 5.51743, 5.528703, 6.9108624, 5.528703],
            (dfdx::shapes::Const::<2>, dfdx::shapes::Const::<3>),
        );
        let d_b = d_a.sin();

        let r = b.retrieve().unwrap().real_data(b.view().unwrap()).unwrap();
        assert_close_data(&r, &d_b.as_vec());
    }

    #[test]
    fn test_cos() {
        let mut cx = Graph::new();
        let a = cx.new_tensor::<R2<2, 3>>("Input");
        a.set(vec![
            5.51743, 6.896794, 5.51743, 5.528703, 6.9108624, 5.528703,
        ]);
        let b = a.cos();
        b.mark();

        cx.execute();

        let d_dev = Cpu::default();
        let d_a = d_dev.tensor_from_vec(
            vec![5.51743, 6.896794, 5.51743, 5.528703, 6.9108624, 5.528703],
            (dfdx::shapes::Const::<2>, dfdx::shapes::Const::<3>),
        );
        let d_b = d_a.cos();

        let r = b.retrieve().unwrap().real_data(b.view().unwrap()).unwrap();
        assert_close_data(&r, &d_b.as_vec());
    }
}
