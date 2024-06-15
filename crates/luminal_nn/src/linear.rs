use rand::{thread_rng, Rng};

use luminal::prelude::*;

/// A simple unbiased linear layer
pub struct Linear<const A: usize, const B: usize> {
    pub weight: GraphTensor<R2<A, B>>,
}

impl<const A: usize, const B: usize> InitModule for Linear<A, B> {
    fn initialize(cx: &mut Graph) -> Self {
        // Init weight as uniform(-1, 1)
        let mut rng = thread_rng();
        Self {
            weight: cx.named_tensor("Weight").set(
                (0..(A * B))
                    .map(|_| rng.gen_range(-1_f32..1_f32))
                    .collect::<Vec<_>>(),
            ),
        }
    }
}

impl<const A: usize, const B: usize> SerializeModule for Linear<A, B> {
    fn serialize(&self, s: &mut luminal::module::Serializer) {
        s.tensor("weight", self.weight);
    }
}

impl<const A: usize, const B: usize, S: Shape> Module<GraphTensor<S>> for Linear<A, B>
where
    GraphTensor<S>: Matmul<R2<A, B>>,
{
    type Output = <GraphTensor<S> as Matmul<R2<A, B>>>::Output;

    fn forward(&self, input: GraphTensor<S>) -> Self::Output {
        input.matmul(self.weight)
    }
}

/// A simple unbiased linear layer with a permuted weight matrix
pub struct PermutedLinear<const A: usize, const B: usize> {
    pub weight: GraphTensor<R2<B, A>>,
    pub bias: Option<GraphTensor<R1<B>>>,
}

impl<const A: usize, const B: usize> InitModule for PermutedLinear<A, B> {
    fn initialize(cx: &mut Graph) -> Self {
        // Init weight as uniform(-1, 1)
        let mut rng = thread_rng();
        Self {
            weight: cx.named_tensor("Weight").set(
                (0..(A * B))
                    .map(|_| rng.gen_range(-1_f32..1_f32))
                    .collect::<Vec<_>>(),
            ),
            bias: None,
        }
    }
}

impl<const A: usize, const B: usize> PermutedLinear<A, B> {
    pub fn new(bias: bool, cx: &mut Graph) -> Self {
        Self::named("", bias, cx)
    }

    pub fn named(name: &str, bias: bool, cx: &mut Graph) -> Self {
        Self {
            weight: cx.named_tensor(&format!("{name} Weight")),
            bias: if bias {
                Some(cx.named_tensor(&format!("{name} Bias")))
            } else {
                None
            },
        }
    }
}

impl<const A: usize, const B: usize> SerializeModule for PermutedLinear<A, B> {
    fn serialize(&self, s: &mut luminal::module::Serializer) {
        s.tensor("weight", self.weight);
        if let Some(b) = self.bias {
            s.tensor("bias", b);
        }
    }
}

impl<const A: usize, const B: usize, Batch: Dimension> Module<GraphTensor<(Batch, Const<A>)>>
    for PermutedLinear<A, B>
{
    type Output = GraphTensor<(Batch, Const<B>)>;

    fn forward(&self, input: GraphTensor<(Batch, Const<A>)>) -> Self::Output {
        let o = input.matmul(self.weight.permute());
        if let Some(b) = self.bias {
            o + b.expand()
        } else {
            o
        }
    }
}

impl<const A: usize, const B: usize, Batch: Dimension, Batch1: Dimension>
    Module<GraphTensor<(Batch, Batch1, Const<A>)>> for PermutedLinear<A, B>
{
    type Output = GraphTensor<(Batch, Batch1, Const<B>)>;

    fn forward(&self, input: GraphTensor<(Batch, Batch1, Const<A>)>) -> Self::Output {
        let o = input.matmul(self.weight.permute());
        if let Some(b) = self.bias {
            o + b.expand()
        } else {
            o
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Linear;
    use luminal::{prelude::*, tests::assert_close};
    #[test]
    fn test_linear() {
        let mut cx = Graph::new();
        let batch = cx
            .tensor::<R2<2, 3>>()
            .set(vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);
        let a = cx.tensor::<R1<3>>().set(vec![1.0, 2.0, 3.0]);

        let model: Linear<3, 4> = Linear::initialize(&mut cx);
        let mut b = model.forward(a).retrieve();
        let mut batch_out = model.forward(batch).retrieve();

        cx.execute();

        let unoptimized_b = b.data();
        let unoptimized_batch_out = batch_out.data();

        cx.compile(GenericCompiler::default(), (&mut b, &mut batch_out));
        cx.execute();

        assert_close(&unoptimized_b, &b.data());
        assert_close(&unoptimized_batch_out, &batch_out.data());
    }
}
