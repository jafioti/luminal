use std::ops::Mul;

use crate::{
    nn::linear::Linear,
    prelude::{tracker::Dim, *},
};

// This is still single head attention because I need a runtime reshape, like the try_reshape in dfdx
pub struct MultiHeadSelfAttention<
    const DIM: usize,
    const K_DIM: usize,
    const V_DIM: usize,
    const HEADS: usize,
> {
    pub(crate) w_q: Linear<DIM, K_DIM>,
    pub(crate) w_k: Linear<DIM, K_DIM>,
    pub(crate) w_v: Linear<DIM, V_DIM>,
    pub(crate) w_o: Linear<V_DIM, DIM>,
}

impl<const DIM: usize, const K_DIM: usize, const V_DIM: usize, const HEADS: usize> InitModule
    for MultiHeadSelfAttention<DIM, K_DIM, V_DIM, HEADS>
{
    fn initialize(cx: &mut Graph) -> Self {
        Self {
            w_q: InitModule::initialize(cx),
            w_k: InitModule::initialize(cx),
            w_v: InitModule::initialize(cx),
            w_o: InitModule::initialize(cx),
        }
    }
}

impl<const DIM: usize, const K_DIM: usize, const V_DIM: usize, const HEADS: usize> SerializeModule
    for MultiHeadSelfAttention<DIM, K_DIM, V_DIM, HEADS>
{
    fn serialize(&self, s: &mut Serializer) {
        s.module("w_q", &self.w_q);
        s.module("w_k", &self.w_k);
        s.module("w_v", &self.w_v);
        s.module("w_o", &self.w_o);
    }
}

// Single
impl<
        const DIM: usize,
        const K_DIM: usize,
        const V_DIM: usize,
        const HEADS: usize,
        S: Dimension,
    > Module<GraphTensor<(S, Const<DIM>)>> for MultiHeadSelfAttention<DIM, K_DIM, V_DIM, HEADS>
{
    type Output = GraphTensor<(S, Const<DIM>)>;

    fn forward(&self, input: GraphTensor<(S, Const<DIM>)>) -> Self::Output {
        // Pass to batched forward
        <Self as Module<GraphTensor<(Const<1>, S, Const<DIM>)>>>::forward(self, input.expand())
            .max_reduce()
    }
}

impl<
        const DIM: usize,
        const K_DIM: usize,
        const V_DIM: usize,
        const HEADS: usize,
        S: Dimension,
        S1: Dimension,
    >
    Module<(
        GraphTensor<(S, Const<DIM>)>,
        GraphTensor<(S1, Const<DIM>)>,
        GraphTensor<(S, Const<DIM>)>,
    )> for MultiHeadSelfAttention<DIM, K_DIM, V_DIM, HEADS>
{
    type Output = GraphTensor<(S1, Const<DIM>)>;

    fn forward(
        &self,
        (k, q, v): (
            GraphTensor<(S, Const<DIM>)>,
            GraphTensor<(S1, Const<DIM>)>,
            GraphTensor<(S, Const<DIM>)>,
        ),
    ) -> Self::Output {
        // Pass to batched forward
        <Self as Module<(
            GraphTensor<(Const<1>, S, Const<DIM>)>,
            GraphTensor<(Const<1>, S1, Const<DIM>)>,
            GraphTensor<(Const<1>, S, Const<DIM>)>,
        )>>::forward(self, (k.expand(), q.expand(), v.expand()))
        .max_reduce()
    }
}

// Batched
impl<
        const DIM: usize,
        const K_DIM: usize,
        const V_DIM: usize,
        const HEADS: usize,
        S: Dimension,
        B: Dimension,
    > Module<GraphTensor<(B, S, Const<DIM>)>> for MultiHeadSelfAttention<DIM, K_DIM, V_DIM, HEADS>
{
    type Output = GraphTensor<(B, S, Const<DIM>)>;

    fn forward(&self, input: GraphTensor<(B, S, Const<DIM>)>) -> Self::Output {
        <Self as Module<(
            GraphTensor<(B, S, Const<DIM>)>,
            GraphTensor<(B, S, Const<DIM>)>,
            GraphTensor<(B, S, Const<DIM>)>,
        )>>::forward(self, (input, input, input))
    }
}

// Batched different key-query-value
impl<
        const DIM: usize,
        const K_DIM: usize,
        const V_DIM: usize,
        const HEADS: usize,
        S1: Dimension,
        S2: Dimension,
        B: Dimension,
    >
    Module<(
        GraphTensor<(B, S1, Const<DIM>)>,
        GraphTensor<(B, S2, Const<DIM>)>,
        GraphTensor<(B, S1, Const<DIM>)>,
    )> for MultiHeadSelfAttention<DIM, K_DIM, V_DIM, HEADS>
{
    type Output = GraphTensor<(B, S2, Const<DIM>)>;

    fn forward(
        &self,
        (keys, queries, values): (
            GraphTensor<(B, S1, Const<DIM>)>,
            GraphTensor<(B, S2, Const<DIM>)>,
            GraphTensor<(B, S1, Const<DIM>)>,
        ),
    ) -> Self::Output {
        let values = self
            .w_v
            .forward(values)
            .dyn_reshape::<(B, S1, Dyn<'-'>, Dyn<'-'>)>(vec![
                B::const_size(),
                S1::const_size(),
                Dim::Known(HEADS),
                Dim::Known(K_DIM / HEADS),
            ])
            .permute::<_, Axes4<0, 2, 1, 3>>();
        let keys = self
            .w_k
            .forward(keys)
            .dyn_reshape::<(B, S1, Dyn<'-'>, Dyn<'-'>)>(vec![
                B::const_size(),
                S1::const_size(),
                Dim::Known(HEADS),
                Dim::Known(K_DIM / HEADS),
            ])
            .permute::<_, Axes4<0, 2, 3, 1>>();
        let queries = self
            .w_q
            .forward(queries)
            .dyn_reshape::<(B, S2, Dyn<'-'>, Dyn<'-'>)>(vec![
                B::const_size(),
                S2::const_size(),
                Dim::Known(HEADS),
                Dim::Known(K_DIM / HEADS),
            ])
            .permute::<_, Axes4<0, 2, 1, 3>>();

        let weights = queries
            .batch_matmul(keys)
            .mul((1.0 / ((K_DIM / HEADS) as f64).sqrt()) as f32)
            .softmax::<3>();

        let tokens: GraphTensor<(B, S2, Const<V_DIM>)> = weights
            .batch_matmul(values)
            .permute::<_, Axes4<0, 2, 1, 3>>()
            .dyn_reshape(vec![B::const_size(), S2::const_size(), Dim::Known(V_DIM)]);
        self.w_o.forward(tokens)
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        prelude::{Module, *},
        tests::assert_close_data,
    };
    use dfdx::prelude::{Module as DfdxModule, *};

    use super::MultiHeadSelfAttention;
    #[test]
    fn test_self_attention() {
        let mut cx = Graph::new();
        let model: MultiHeadSelfAttention<3, 3, 3, 1> = InitModule::initialize(&mut cx);
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

        let a = cx.new_tensor::<(Dyn<'d'>, crate::shape::Const<3>)>("Input");
        let e = cx.new_tensor::<(Dyn<'e'>, crate::shape::Const<3>)>("Input");
        let b = model.forward((e, a, e));

        a.set_dyn(
            vec![
                0.56587636, -1.4053632, 0.8394869, 0.5916256, -1.4082357, 0.8166099,
            ],
            vec![2, 3],
        );
        e.set_dyn(
            vec![-1.0, 2.0, 3.0, 3.0, 3.0, -1.0, -1.0, 2.0, 3.0],
            vec![3, 3],
        );
        b.mark();

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

        assert_close_data(&b.dyn_data(&cx.dyn_map), &d_b.as_vec());
    }
}
