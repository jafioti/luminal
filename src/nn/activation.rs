use crate::prelude::*;

pub struct ReLU {
    zeros: GraphTensor<()>,
}

impl InitModule for ReLU {
    fn initialize(cx: &mut Graph) -> Self {
        let s = Self {
            zeros: cx.new_tensor(),
        };
        s.zeros.set(vec![0.]);
        s
    }
}

impl<S: ConstShape> Module<GraphTensor<S>> for ReLU {
    type Output = GraphTensor<S>;

    fn forward(&self, input: GraphTensor<S>) -> Self::Output {
        input.max(self.zeros.expand())
    }
}
