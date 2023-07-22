use rand::{thread_rng, Rng};

use crate::prelude::*;

/// A simple linear layer
pub struct Linear<const A: usize, const B: usize> {
    weight: GraphTensor<R2<A, B>>,
}

impl<const A: usize, const B: usize> InitModule for Linear<A, B> {
    fn initialize(cx: &mut Graph) -> Self {
        let s = Self {
            weight: cx.new_tensor(),
        };
        // Init weight has uniforn(-1, 1)
        let mut rng = thread_rng();
        s.weight
            .set((0..(A * B)).map(|_| rng.gen_range(-1_f32..1_f32)).collect());
        s
    }
}

impl<const A: usize, const B: usize, const C: usize> Module<GraphTensor<R2<C, A>>>
    for Linear<A, B>
{
    type Output = GraphTensor<R2<C, B>>;

    fn forward(&self, input: GraphTensor<R2<C, A>>) -> Self::Output {
        input.matmul(self.weight)
    }
}

impl<const A: usize, const B: usize> LoadModule for Linear<A, B> {
    fn load(&mut self, state_dict: &mut StateDict) {
        self.weight.set(state_dict.data.remove("weight").unwrap().0)
    }
}
