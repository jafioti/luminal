use crate::prelude::*;

impl<S: Shape> GraphTensor<S> {
    /// The Rectified Linear Unit activation function
    pub fn relu(self) -> GraphTensor<S> {
        self.max_f32(0.)
    }

    /// The sigmoid activation function
    pub fn sigmoid(self) -> GraphTensor<S> {
        // Based on https://github.com/tinygrad/tinygrad/blob/9d142430cbe61121c864c0015f1de83c94a7d2c0/tinygrad/mlops.py#L70
        let graph = unsafe { self.graph_ref.as_mut().unwrap() };
        let one = graph.new_tensor::<R0>("Const");
        one.set(vec![1.]);
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
    use dfdx::tensor::{Cpu, TensorFromVec};

    use crate::{prelude::*, tests::assert_close_data};

    #[test]
    fn test_relu() {
        let mut cx = Graph::new();
        let a = cx.new_tensor::<(usize, usize)>("Input");
        a.set_dyn(vec![0.0, 1.0, 0.0, 1.0], vec![2, 2]);
        let b = a.relu();
        b.mark();

        cx.execute();

        let d_dev = Cpu::default();
        let d_a = d_dev.tensor_from_vec(
            vec![0.0, 1.0, 0.0, 1.0],
            (dfdx::shapes::Const::<2>, dfdx::shapes::Const::<2>),
        );
        let d_b = d_a.relu();
        assert_close_data(
            &b.retrieve().unwrap().real_data(b.view().unwrap()).unwrap(),
            &d_b.as_vec(),
        );
    }

    #[test]
    fn test_sigmoid() {
        let mut cx = Graph::new();
        let a = cx.new_tensor::<(usize, usize)>("Input");
        a.set_dyn(vec![0.0, 1.0, 0.0, 1.0], vec![2, 2]);
        let b = a.sigmoid();
        b.mark();

        cx.execute();

        let d_dev = Cpu::default();
        let d_a = d_dev.tensor_from_vec(
            vec![0.0, 1.0, 0.0, 1.0],
            (dfdx::shapes::Const::<2>, dfdx::shapes::Const::<2>),
        );
        let d_b = d_a.sigmoid();
        assert_close_data(
            &b.retrieve().unwrap().real_data(b.view().unwrap()).unwrap(),
            &d_b.as_vec(),
        );
    }

    #[test]
    fn test_tanh() {
        let mut cx = Graph::new();
        let a = cx.new_tensor::<(usize, usize)>("Input");
        a.set_dyn(vec![0.0, 1.0, 0.0, 1.0], vec![2, 2]);
        let b = a.tanh();
        b.mark();

        cx.execute();

        let d_dev = Cpu::default();
        let d_a = d_dev.tensor_from_vec(
            vec![0.0, 1.0, 0.0, 1.0],
            (dfdx::shapes::Const::<2>, dfdx::shapes::Const::<2>),
        );
        let d_b = d_a.tanh();
        assert_close_data(
            &b.retrieve().unwrap().real_data(b.view().unwrap()).unwrap(),
            &d_b.as_vec(),
        );
    }
}
