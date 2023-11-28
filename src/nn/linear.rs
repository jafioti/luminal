use rand::{thread_rng, Rng};

use crate::prelude::*;

/// A simple linear layer
pub struct Linear<const A: usize, const B: usize> {
    pub weight: GraphTensor<R2<A, B>>,
}

impl<const A: usize, const B: usize> InitModule for Linear<A, B> {
    fn initialize(cx: &mut Graph) -> Self {
        let s = Self {
            weight: cx.named_tensor("Weight"),
        };
        // Init weight as uniform(-1, 1)
        let mut rng = thread_rng();
        s.weight.set(
            (0..(A * B))
                .map(|_| rng.gen_range(-1_f32..1_f32))
                .collect::<Vec<_>>(),
        );
        s
    }
}

impl<const A: usize, const B: usize> SerializeModule for Linear<A, B> {
    fn serialize(&self, s: &mut crate::serialization::Serializer) {
        s.tensor("weight", self.weight);
    }
}

// Single
impl<const A: usize, const B: usize> Module<GraphTensor<R1<A>>> for Linear<A, B> {
    type Output = GraphTensor<R1<B>>;

    fn forward(&self, input: GraphTensor<R1<A>>) -> Self::Output {
        input.matmul(self.weight)
    }
}

// Batched
impl<const A: usize, const B: usize, C: Dimension> Module<GraphTensor<(C, Const<A>)>>
    for Linear<A, B>
{
    type Output = GraphTensor<(C, Const<B>)>;

    fn forward(&self, input: GraphTensor<(C, Const<A>)>) -> Self::Output {
        input.matmul(self.weight)
    }
}

// 2x Batched
impl<const A: usize, const B: usize, C: Dimension, D: Dimension>
    Module<GraphTensor<(C, D, Const<A>)>> for Linear<A, B>
{
    type Output = GraphTensor<(C, D, Const<B>)>;

    fn forward(&self, input: GraphTensor<(C, D, Const<A>)>) -> Self::Output {
        input.matmul(self.weight)
    }
}

// 3x Batched
impl<const A: usize, const B: usize, C: Dimension, D: Dimension, E: Dimension>
    Module<GraphTensor<(C, D, E, Const<A>)>> for Linear<A, B>
{
    type Output = GraphTensor<(C, D, E, Const<B>)>;

    fn forward(&self, input: GraphTensor<(C, D, E, Const<A>)>) -> Self::Output {
        input.matmul(self.weight.expand())
    }
}

#[cfg(test)]
mod tests {
    use super::Linear;
    use crate::{prelude::*, tests::assert_close};
    #[test]
    fn test_linear() {
        let mut cx = Graph::new();
        let batch = cx.tensor::<R2<2, 3>>();
        let a = cx.tensor::<R1<3>>();

        let model: Linear<3, 4> = Linear::initialize(&mut cx);
        let b = model.forward(a).retrieve();
        let batch_out = model.forward(batch).retrieve();

        a.set(vec![1.0, 2.0, 3.0]);
        batch.set(vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);

        cx.execute();

        let unoptimized_b = b.data();
        let unoptimized_batch_out = batch_out.data();

        cx.compile(GenericCompiler::default());
        cx.execute();

        assert_close(&unoptimized_b, &b.data());
        assert_close(&unoptimized_batch_out, &batch_out.data());
    }
}
