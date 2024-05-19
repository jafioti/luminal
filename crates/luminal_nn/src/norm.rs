use luminal::prelude::*;

/// A simple layer norm with an optional weight and bias
#[derive(Default)]
pub struct LayerNorm<const DIM: usize> {
    pub weight: Option<GraphTensor<R1<DIM>>>,
    pub bias: Option<GraphTensor<R1<DIM>>>,
    mean_norm: bool,
    epsilon: f32,
}

impl<const DIM: usize> LayerNorm<DIM> {
    pub fn new(weight: bool, bias: bool, mean_norm: bool, epsilon: f32, cx: &mut Graph) -> Self {
        Self {
            weight: if weight {
                Some(cx.named_tensor("LayerNorm Weight"))
            } else {
                None
            },
            bias: if bias {
                Some(cx.named_tensor("LayerNorm Bias"))
            } else {
                None
            },
            mean_norm,
            epsilon,
        }
    }
}

impl<const DIM: usize, S: Shape> Module<GraphTensor<S>> for LayerNorm<DIM>
where
    (Const<DIM>,): BroadcastShapeTo<S, S::AllButLast>,
{
    type Output = GraphTensor<S>;
    fn forward(&self, mut input: GraphTensor<S>) -> Self::Output {
        if self.mean_norm {
            input = input.mean_norm::<S::LastAxis>();
        }
        input = input.std_norm(self.epsilon);
        if let Some(w) = self.weight {
            input *= w.expand();
        }
        if let Some(b) = self.bias {
            input += b.expand();
        }
        input
    }
}

impl<const DIM: usize> SerializeModule for LayerNorm<DIM> {
    fn serialize(&self, s: &mut Serializer) {
        if let Some(w) = self.weight {
            s.tensor("weight", w);
        }
        if let Some(b) = self.bias {
            s.tensor("bias", b);
        }
    }
}
