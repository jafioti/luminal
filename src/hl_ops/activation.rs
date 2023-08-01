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
        let one = graph.new_tensor::<R0>();
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
