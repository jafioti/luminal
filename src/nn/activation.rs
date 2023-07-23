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

#[cfg(test)]
mod tests {
    use super::ReLU;
    use crate::{nn::linear::Linear, prelude::*, tests::assert_close};

    #[test]
    fn test_relu_and_linear() {
        let mut cx = Graph::new();
        let batch = cx.new_tensor::<R2<2, 3>>();
        let a = cx.new_tensor::<R1<3>>();

        let model: (Linear<3, 4>, ReLU, Linear<4, 2>) = InitModule::initialize(&mut cx);
        let b = model.forward(a);
        let batch_out = model.forward(batch);

        b.mark();
        a.mark();
        batch_out.mark();
        a.set(vec![1.0, 2.0, 3.0]);
        batch.set(vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);
        cx.execute();

        let unoptimized_b = b.retrieve().unwrap();
        let unoptimized_batch_out = batch_out.retrieve().unwrap();

        cx.optimize((CPUOptimizer, GeneralOpt::default()));
        cx.execute();

        assert_close(&unoptimized_b, &b.retrieve().unwrap());
        assert_close(&unoptimized_batch_out, &batch_out.retrieve().unwrap());
    }
}
