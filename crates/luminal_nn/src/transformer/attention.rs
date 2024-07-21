use std::ops::Mul;

use crate::Linear;
use luminal::prelude::*;

/// Multi-head self attention as layed out in [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762).
pub struct MultiHeadSelfAttention {
    pub w_q: Linear, // dim x k_dim
    pub w_k: Linear, // dim x k_dim
    pub w_v: Linear, // dim x v_dim
    pub w_o: Linear, // v_dim x dim
    k_dim: usize,
    v_dim: usize,
    heads: usize,
}

impl MultiHeadSelfAttention {
    pub fn new(dim: usize, k_dim: usize, v_dim: usize, heads: usize, cx: &mut Graph) -> Self {
        Self {
            w_q: Linear::new(dim, k_dim, false, cx),
            w_k: Linear::new(dim, k_dim, false, cx),
            w_v: Linear::new(dim, v_dim, false, cx),
            w_o: Linear::new(v_dim, dim, false, cx),
            k_dim,
            v_dim,
            heads,
        }
    }
}

impl SerializeModule for MultiHeadSelfAttention {
    fn serialize(&self, s: &mut Serializer) {
        s.module("w_q", &self.w_q);
        s.module("w_k", &self.w_k);
        s.module("w_v", &self.w_v);
        s.module("w_o", &self.w_o);
    }
}

// Batched
impl Module<GraphTensor> for MultiHeadSelfAttention {
    type Output = GraphTensor;

    fn forward(&self, input: GraphTensor) -> Self::Output {
        // Input: batch_dims, sequence, dim
        <Self as Module<(GraphTensor, GraphTensor, GraphTensor)>>::forward(
            self,
            (input, input, input),
        )
    }
}

// Batched different key-query-value
impl Module<(GraphTensor, GraphTensor, GraphTensor)> for MultiHeadSelfAttention {
    type Output = GraphTensor;

    fn forward(
        &self,
        (keys, queries, values): (
            GraphTensor, // batch, s1, dim
            GraphTensor, // batch, s2, dim
            GraphTensor, // batch, s1, dim
        ),
    ) -> Self::Output {
        let orig_query_shape = queries.dims();
        let s1 = keys.dims()[keys.shape.len() - 2];
        let s2 = queries.dims()[queries.shape.len() - 2];
        let n_batches = queries
            .dims()
            .into_iter()
            .take(queries.shape.len() - 2)
            .product::<Expression>()
            .max(1);
        let dim = *queries.dims().last().unwrap();
        let keys = keys.reshape((n_batches, s1, dim));
        let values = values.reshape((n_batches, s1, dim));
        let queries = queries.reshape((n_batches, s2, dim));
        let values = self
            .w_v
            .forward(values)
            .reshape((n_batches, s1, self.heads, self.k_dim / self.heads))
            .permute((0, 2, 1, 3));
        let keys = self
            .w_k
            .forward(keys)
            .reshape((n_batches, s1, self.heads, self.k_dim / self.heads))
            .permute((0, 2, 3, 1));
        let queries = self
            .w_q
            .forward(queries)
            .reshape((n_batches, s2, self.heads, self.k_dim / self.heads))
            .permute((0, 2, 1, 3));

        let weights = queries
            .matmul(keys)
            .mul((1.0 / ((self.k_dim / self.heads) as f64).sqrt()) as f32)
            .softmax(3);

        let tokens = weights
            .matmul(values)
            .permute((0, 2, 1, 3))
            .reshape((n_batches, s2, self.v_dim));
        self.w_o.forward(tokens).reshape(orig_query_shape) // batch_dims, s2, dim
    }
}

#[cfg(test)]
mod tests {
    use dfdx::prelude::{Module as DfdxModule, *};
    use luminal::{
        prelude::{Module, *},
        tests::assert_close,
    };

    use super::MultiHeadSelfAttention;
    #[test]
    fn test_self_attention() {
        let mut cx = Graph::new();
        let model = MultiHeadSelfAttention::new(3, 3, 3, 1, &mut cx);
        model
            .w_k
            .weight
            .set(vec![1., 22., 3., 1., 2., 3., 1., 2., 3.]);
        model
            .w_q
            .weight
            .set(vec![3., 2., 3., 1.3, 2., 3., 3., 2., 3.]);
        model
            .w_v
            .weight
            .set(vec![-1., 12., 3., -1., 2., -3., 11., 2., 3.]);
        model
            .w_o
            .weight
            .set(vec![1., 22., 3., 1., 2., 3., 1., 2., 3.]);

        let a = cx.tensor(('d', 3));
        let e = cx.tensor(('e', 3));
        let b = model.forward((e, a, e));

        a.set_dyn(
            vec![
                0.56587636, -1.4053632, 0.8394869, 0.5916256, -1.4082357, 0.8166099,
            ],
            (2, 3),
        );
        e.set_dyn(vec![-1.0, 2.0, 3.0, 3.0, 3.0, -1.0, -1.0, 2.0, 3.0], (3, 3));
        b.retrieve();

        cx.execute();

        let d_dev = Cpu::default();
        let mut d_model: dfdx::nn::modules::MultiHeadAttention<3, 1, 3, 3, f32, Cpu> =
            d_dev.build_module::<MultiHeadAttention<3, 1, 3, 3>, f32>();
        d_model.w_k.bias.copy_from(&[0.0, 0.0, 0.0]);
        d_model.w_v.bias.copy_from(&[0.0, 0.0, 0.0]);
        d_model.w_q.bias.copy_from(&[0.0, 0.0, 0.0]);
        d_model.w_o.bias.copy_from(&[0.0, 0.0, 0.0]);
        d_model.w_o.weight = d_dev
            .tensor_from_vec(
                vec![1., 22., 3., 1., 2., 3., 1., 2., 3.],
                (dfdx::shapes::Const::<3>, dfdx::shapes::Const::<3>),
            )
            .permute();
        d_model.w_k.weight = d_dev
            .tensor_from_vec(
                vec![1., 22., 3., 1., 2., 3., 1., 2., 3.],
                (dfdx::shapes::Const::<3>, dfdx::shapes::Const::<3>),
            )
            .permute();
        d_model.w_q.weight = d_dev
            .tensor_from_vec(
                vec![3., 2., 3., 1.3, 2., 3., 3., 2., 3.],
                (dfdx::shapes::Const::<3>, dfdx::shapes::Const::<3>),
            )
            .permute();
        d_model.w_v.weight = d_dev
            .tensor_from_vec(
                vec![-1., 12., 3., -1., 2., -3., 11., 2., 3.],
                (dfdx::shapes::Const::<3>, dfdx::shapes::Const::<3>),
            )
            .permute();
        let d_a = d_dev.tensor_from_vec(
            vec![
                0.56587636, -1.4053632, 0.8394869, 0.5916256, -1.4082357, 0.8166099,
            ],
            (dfdx::shapes::Const::<2>, dfdx::shapes::Const::<3>),
        );
        let d_e = d_dev.tensor_from_vec(
            vec![-1.0, 2.0, 3.0, 3.0, 3.0, -1.0, -1.0, 2.0, 3.0],
            (dfdx::shapes::Const::<3>, dfdx::shapes::Const::<3>),
        );
        let d_b = d_model.forward((d_a, d_e.clone(), d_e));

        assert_close(&b.data(), &d_b.as_vec());
    }
}
