use std::ops::Mul;

use crate::{nn::linear::Linear, prelude::*};

// This is still single head attention because I need a runtime reshape, like the try_reshape in dfdx
pub struct MultiHeadSelfAttention<const DIM: usize> {
    pub(crate) w_q: Linear<DIM, DIM>,
    pub(crate) w_k: Linear<DIM, DIM>,
    pub(crate) w_v: Linear<DIM, DIM>,
    pub(crate) w_o: Linear<DIM, DIM>,
}

impl<const DIM: usize> InitModule for MultiHeadSelfAttention<DIM> {
    fn initialize(cx: &mut Graph) -> Self {
        Self {
            w_q: InitModule::initialize(cx),
            w_k: InitModule::initialize(cx),
            w_v: InitModule::initialize(cx),
            w_o: InitModule::initialize(cx),
        }
    }
}

// Single
impl<const DIM: usize, S: Dim> Module<GraphTensor<(S, Const<DIM>)>>
    for MultiHeadSelfAttention<DIM>
{
    type Output = GraphTensor<(S, Const<DIM>)>;

    fn forward(&self, input: GraphTensor<(S, Const<DIM>)>) -> Self::Output {
        // Pass to batched forward
        <Self as Module<GraphTensor<(Const<1>, S, Const<DIM>)>>>::forward(self, input.expand())
            .max_reduce::<_, Axis<0>>()
    }
}

// Batched
impl<const DIM: usize, S: Dim, B: Dim> Module<GraphTensor<(B, S, Const<DIM>)>>
    for MultiHeadSelfAttention<DIM>
{
    type Output = GraphTensor<(B, S, Const<DIM>)>;

    fn forward(&self, input: GraphTensor<(B, S, Const<DIM>)>) -> Self::Output {
        let keys = self.w_k.forward(input);
        let values = self.w_v.forward(input);
        let queries = self.w_q.forward(input);

        let weights = queries
            .batch_matmul(keys.permute())
            .mul(1.0f32 / (DIM as f32).sqrt())
            .softmax::<2>();

        self.w_o.forward(weights.batch_matmul(values))
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
        let model: MultiHeadSelfAttention<3> = InitModule::initialize(&mut cx);
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

        let a = cx.new_tensor::<(usize, crate::shape::Const<3>)>();
        let b = model.forward(a);

        a.set_dyn(vec![-1., 2., 3., 3., 3., -1.], vec![2, 3]);
        b.mark();

        cx.execute();

        let d_dev = Cpu::default();
        let mut d_model: dfdx::nn::modules::MultiHeadAttention<3, 1, 3, 3, f32, Cpu> =
            d_dev.build_module::<MultiHeadAttention<3, 1, 3, 3>, f32>();
        d_model.w_k.bias.copy_from(&[0.0, 0.0, 0.0]);
        d_model.w_v.bias.copy_from(&[0.0, 0.0, 0.0]);
        d_model.w_q.bias.copy_from(&[0.0, 0.0, 0.0]);
        d_model.w_o.bias.copy_from(&[0., 0., 0.]);
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
            vec![-1., 2., 3., 3., 3., -1.],
            (dfdx::shapes::Const::<2>, dfdx::shapes::Const::<3>),
        );
        let d_b = d_model.forward(d_a);

        assert_close_data(&b.retrieve().unwrap().real_data().unwrap(), &d_b.as_vec());
    }
}
