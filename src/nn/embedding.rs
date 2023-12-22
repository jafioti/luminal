use rand::{thread_rng, Rng};

use crate::prelude::*;

pub struct Embedding<const N: usize, const DIM: usize> {
    pub weight: GraphTensor<R2<N, DIM>>,
}

impl<const A: usize, const B: usize> InitModule for Embedding<A, B> {
    fn initialize(cx: &mut Graph) -> Self {
        let s = Self {
            weight: cx.named_tensor("Embedding Weight"),
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

impl<const A: usize, const B: usize> SerializeModule for Embedding<A, B> {
    fn serialize(&self, s: &mut crate::serialization::Serializer) {
        s.tensor("weight", self.weight);
    }
}

// Single
impl<S: Dimension, const N: usize, const DIM: usize> Module<GraphTensor<(S,)>>
    for Embedding<N, DIM>
{
    type Output = GraphTensor<(S, Const<DIM>)>;

    fn forward(&self, input: GraphTensor<(S,)>) -> Self::Output {
        <Self as Module<GraphTensor<(Const<1>, S)>>>::forward(self, input.expand()).max_reduce()
    }
}

// Batch
impl<B: Dimension, S: Dimension, const N: usize, const DIM: usize> Module<GraphTensor<(B, S)>>
    for Embedding<N, DIM>
{
    type Output = GraphTensor<(B, S, Const<DIM>)>;

    fn forward(&self, input: GraphTensor<(B, S)>) -> Self::Output {
        self.weight.gather(input)
    }
}

#[cfg(test)]
mod tests {
    use dfdx::{
        prelude::Module as DfdxModule,
        tensor::{Cpu, TensorFromVec},
    };

    use crate::prelude::Module;

    use super::Embedding;
    use dfdx::nn::BuildOnDevice;
    crate::test_imports!();

    #[test]
    fn test_embedding() {
        let mut cx = Graph::new();
        let batch = cx
            .tensor::<R2<2, 3>>()
            .set(vec![1.0, 0.0, 2.0, 1.0, 0.0, 1.0]);
        // let a = cx.tensor::<R1<3>>().set(vec![1.0, 0.0, 1.0]);

        let model: Embedding<3, 4> = InitModule::initialize(&mut cx);
        model
            .weight
            .set(vec![1.1, 2., 3., 1., 2., 3., 14., 2., 33., 1., 2., 3.]);
        // let b = model.forward(a).retrieve();
        let batch_out = model.forward(batch).retrieve();

        cx.compile(MetalFp16Compiler::default());
        cx.display();
        cx.execute();

        let d_dev = Cpu::default();
        let mut d_model = <dfdx::nn::modules::builders::Embedding<3, 4>>::build_on_device(&d_dev);
        d_model.weight = d_dev.tensor_from_vec(
            vec![1.1, 2., 3., 1., 2., 3., 14., 2., 33., 1., 2., 3.],
            (DConst::<3>, DConst::<4>),
        );
        // let d_a = d_dev.tensor_from_vec(vec![1, 0, 1], (DConst::<3>,));
        let d_batch = d_dev.tensor_from_vec(vec![1, 0, 2, 1, 0, 1], (DConst::<2>, DConst::<3>));

        // let d_b = d_model.forward(d_a);
        let d_batch_out = d_model.forward(d_batch);

        // assert_close(&b.data(), &d_b.as_vec());
        assert_close(&batch_out.data(), &d_batch_out.as_vec());
    }
}
