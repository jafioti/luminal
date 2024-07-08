use luminal::{prelude::*, tests::random_vec};

pub struct Embedding {
    permute: bool,
    pub weight: GraphTensor, // n embeddings x embedding dim
}

impl Embedding {
    pub fn new(n_embeddings: usize, embedding_dim: usize, cx: &mut Graph) -> Self {
        Self {
            weight: cx.named_tensor("Embedding Weight", (n_embeddings, embedding_dim)),
            permute: false,
        }
    }

    pub fn new_permuted(n_embeddings: usize, embedding_dim: usize, cx: &mut Graph) -> Self {
        Self {
            weight: cx.named_tensor("Embedding Weight", (n_embeddings, embedding_dim)),
            permute: true,
        }
    }

    pub fn initialize(self) -> Self {
        self.weight.set(random_vec(
            self.weight.shape.n_elements().to_usize().unwrap(),
        ));
        self
    }
}

impl SerializeModule for Embedding {
    fn serialize(&self, s: &mut luminal::module::Serializer) {
        s.tensor("weight", self.weight);
    }
}

impl Module<GraphTensor> for Embedding {
    type Output = GraphTensor;

    fn forward(&self, input: GraphTensor) -> Self::Output {
        // Flatten batches
        let batch_size = input.shape.n_elements();
        let inp = input.reshape(batch_size);
        let out = if self.permute {
            self.weight.permute((1, 0))
        } else {
            self.weight
        }
        .gather(inp);
        // Unflatten
        let mut new_shape = input.shape();
        new_shape.push(self.weight.shape()[1].clone());
        out.reshape(new_shape)
    }
}

#[cfg(test)]
mod tests {
    use dfdx::{
        prelude::Module as DfdxModule,
        tensor::{Cpu, TensorFromVec},
    };

    use luminal::prelude::Module;

    use super::Embedding;
    use dfdx::nn::BuildOnDevice;
    luminal::test_imports!();

    #[test]
    fn test_embedding() {
        let mut cx = Graph::new();
        let batch = cx.tensor((2, 3)).set(vec![1.0, 0.0, 2.0, 1.0, 0.0, 1.0]);
        let a = cx.tensor(3).set(vec![1.0, 0.0, 1.0]).retrieve();

        let model = Embedding::new(3, 4, &mut cx).initialize();
        model
            .weight
            .set(vec![1.1, 2., 3., 1., 2., 3., 14., 2., 33., 1., 2., 3.]);
        let mut b = model.forward(a).retrieve();
        let mut batch_out = model.forward(batch).retrieve();

        cx.compile(GenericCompiler::default(), (&mut b, &mut batch_out));

        cx.execute();

        let d_dev = Cpu::default();
        let mut d_model = <dfdx::nn::modules::builders::Embedding<3, 4>>::build_on_device(&d_dev);
        d_model.weight = d_dev.tensor_from_vec(
            vec![1.1, 2., 3., 1., 2., 3., 14., 2., 33., 1., 2., 3.],
            (DConst::<3>, DConst::<4>),
        );
        let d_a = d_dev.tensor_from_vec(vec![1, 0, 1], (DConst::<3>,));
        let d_batch = d_dev.tensor_from_vec(vec![1, 0, 2, 1, 0, 1], (DConst::<2>, DConst::<3>));

        let d_b = d_model.forward(d_a);
        let d_batch_out = d_model.forward(d_batch);

        assert_close(&b.data(), &d_b.as_vec());
        assert_close(&batch_out.data(), &d_batch_out.as_vec());
    }
}
