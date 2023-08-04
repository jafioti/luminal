use rand::{thread_rng, Rng};

use crate::{prelude::*, serialization::SerializeModule};

/// A simple linear layer
pub struct Linear<const A: usize, const B: usize> {
    pub(crate) weight: GraphTensor<R2<A, B>>,
}

impl<const A: usize, const B: usize> InitModule for Linear<A, B> {
    fn initialize(cx: &mut Graph) -> Self {
        let s = Self {
            weight: cx.new_tensor("Weight"),
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
impl<const A: usize, const B: usize, C: Dim> Module<GraphTensor<(C, Const<A>)>> for Linear<A, B> {
    type Output = GraphTensor<(C, Const<B>)>;

    fn forward(&self, input: GraphTensor<(C, Const<A>)>) -> Self::Output {
        input.matmul(self.weight)
    }
}

// 2x Batched
impl<const A: usize, const B: usize, C: Dim, D: Dim> Module<GraphTensor<(C, D, Const<A>)>>
    for Linear<A, B>
{
    type Output = GraphTensor<(C, D, Const<B>)>;

    fn forward(&self, input: GraphTensor<(C, D, Const<A>)>) -> Self::Output {
        input.matmul(self.weight)
    }
}

// 3x Batched
impl<const A: usize, const B: usize, C: Dim, D: Dim, E: Dim>
    Module<GraphTensor<(C, D, E, Const<A>)>> for Linear<A, B>
{
    type Output = GraphTensor<(C, D, E, Const<B>)>;

    fn forward(&self, input: GraphTensor<(C, D, E, Const<A>)>) -> Self::Output {
        input.batch_matmul(self.weight.expand())
    }
}

#[cfg(test)]
mod tests {
    use super::Linear;
    use crate::{prelude::*, tests::assert_close};
    #[test]
    fn test_linear() {
        let mut cx = Graph::new();
        let batch = cx.new_tensor::<R2<2, 3>>("Input");
        let a = cx.new_tensor::<R1<3>>("Input");

        let model: Linear<3, 4> = Linear::initialize(&mut cx);
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

        cx.reset();
        cx.optimize(GenericOptimizer::default());
        cx.execute();

        assert_close(&unoptimized_b, &b.retrieve().unwrap());
        assert_close(&unoptimized_batch_out, &batch_out.retrieve().unwrap());
    }
}
