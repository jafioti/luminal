use luminal::prelude::*;
use luminal_nn::Linear;
use luminal_training::Autograd;

const I: usize = 5;
const O: usize = 2;

pub struct LinearBiased {
    pub linear: Linear<I, O>,
    pub bias: GraphTensor<R1<O>>,
}

impl SerializeModule for LinearBiased {
    fn serialize(&self, s: &mut luminal::module::Serializer) {
        s.module("linear", &self.linear);
        s.tensor("bias", self.bias);
    }
}

impl InitModule for LinearBiased {
    fn initialize(cx: &mut Graph) -> Self {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        Self {
            linear: Linear::initialize(cx),
            bias: cx.named_tensor("Bias").set(
                (0..O)
                    .map(|_| rng.gen_range(-1_f32..1_f32))
                    .collect::<Vec<_>>(),
            ),
        }
    }
}

impl Module<GraphTensor<R1<I>>> for LinearBiased {
    type Output = GraphTensor<R1<O>>;
    fn forward(&self, x: GraphTensor<R1<I>>) -> Self::Output {
        let x: GraphTensor<R1<O>> = self.linear.forward(x);
        let bias: GraphTensor<R1<O>> = self.bias.expand::<R1<O>, Axis<0>>();
        // let bias: GraphTensor<R1<O>> = self.bias.clone();
        x + bias
    }
}

fn main() {
    let mut cx = Graph::new();
    let model = LinearBiased::initialize(&mut cx);
    let input: GraphTensor<R1<I>> = cx.tensor();
    let output: GraphTensor<R1<O>> = model.forward(input).retrieve();
    let loss = output.sum_reduce().retrieve();

    let weights = params(&model);
    let _grads = cx.compile(Autograd::new(&weights, loss), ());
}
