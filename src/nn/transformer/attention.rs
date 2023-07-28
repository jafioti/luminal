use crate::{nn::linear::Linear, prelude::*};

// This is still single head attention because I need a runtime reshape, like the try_reshape in dfdx
pub struct MultiHeadSelfAttention<const DIM: usize> {
    pub(crate) w_q: Linear<DIM, DIM>,
    pub(crate) w_k: Linear<DIM, DIM>,
    pub(crate) w_v: Linear<DIM, DIM>,
}

impl<const DIM: usize> InitModule for MultiHeadSelfAttention<DIM> {
    fn initialize(cx: &mut Graph) -> Self {
        Self {
            w_q: InitModule::initialize(cx),
            w_k: InitModule::initialize(cx),
            w_v: InitModule::initialize(cx),
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

        let weights = queries.batch_matmul(keys.permute()).softmax::<1>();

        weights.batch_matmul(values)
    }
}

#[cfg(test)]
mod tests {
    use crate::prelude::*;

    use super::MultiHeadSelfAttention;
    #[test]
    fn test_self_attention() {
        let mut cx = Graph::new();
        let model: MultiHeadSelfAttention<3> = InitModule::initialize(&mut cx);

        let a = cx.new_tensor::<(usize, Const<3>)>();
        let b = model.forward(a);

        a.set_dyn(vec![1., 2., 3., 1., 2., 3.], vec![2, 3]);
        b.mark();

        cx.execute();
    }
}
