use crate::prelude::*;
use rand::{thread_rng, Rng};

// DIM_OUT = ((DIM_IN - (DILATION + 1) * (KERNEL - 1) - 1) / STRIDE) + 1

// TODO: I think the math is more complex here with dilations
// POOL_OUT = DIM_IN / STRIDE

// Conv1D
pub struct Conv1D<
    const CHANNELS_IN: usize,
    const CHANNELS_OUT: usize,
    const KERNEL: usize,
    const STRIDE: usize,
    const DILATION: usize,
    const DIM_OUT: usize,
    const POOL_OUT: usize,
> {
    pub weight: GraphTensor<R3<CHANNELS_IN, CHANNELS_OUT, KERNEL>>,
}

impl<
        const CHANNELS_IN: usize,
        const CHANNELS_OUT: usize,
        const KERNEL: usize,
        const STRIDE: usize,
        const DILATION: usize,
        const DIM_OUT: usize,
        const POOL_OUT: usize,
    > InitModule
    for Conv1D<CHANNELS_IN, CHANNELS_OUT, KERNEL, STRIDE, DILATION, DIM_OUT, POOL_OUT>
{
    fn initialize(cx: &mut Graph) -> Self {
        let conv = Self {
            weight: cx.named_tensor("Weight"),
        };

        // Init weight as uniform(-1, 1)
        let mut rng = thread_rng();
        conv.weight.set(
            (0..(CHANNELS_IN * CHANNELS_OUT * KERNEL))
                .map(|_| rng.gen_range(-1_f32..1_f32))
                .collect::<Vec<_>>(),
        );
        conv
    }
}

impl<
        const CHANNELS_IN: usize,
        const CHANNELS_OUT: usize,
        const KERNEL: usize,
        const STRIDE: usize,
        const DILATION: usize,
        const DIM_OUT: usize,
        const POOL_OUT: usize,
    > SerializeModule
    for Conv1D<CHANNELS_IN, CHANNELS_OUT, KERNEL, STRIDE, DILATION, DIM_OUT, POOL_OUT>
{
    fn serialize(&self, s: &mut crate::serialization::Serializer) {
        s.tensor("weight", self.weight);
    }
}

// Single
impl<
        const CHANNELS_IN: usize,
        const CHANNELS_OUT: usize,
        const KERNEL: usize,
        const STRIDE: usize,
        const DILATION: usize,
        const DIM_IN: usize,
        const DIM_OUT: usize,
        const POOL_OUT: usize,
    > Module<GraphTensor<R2<CHANNELS_IN, DIM_IN>>>
    for Conv1D<CHANNELS_IN, CHANNELS_OUT, KERNEL, STRIDE, DILATION, DIM_OUT, POOL_OUT>
{
    type Output = GraphTensor<R2<CHANNELS_OUT, DIM_OUT>>;

    fn forward(&self, input: GraphTensor<R2<CHANNELS_IN, DIM_IN>>) -> Self::Output {
        input
            .pool_last_dim::<R3<CHANNELS_IN, DIM_IN, KERNEL>>(KERNEL, STRIDE, DILATION)
            .permute::<_, Axes3<1, 0, 2>>()
            .reshape::<R2<DIM_OUT, KERNEL>>()
            .matmul(
                self.weight
                    .permute::<_, Axes3<2, 1, 0>>()
                    .reshape::<R2<KERNEL, CHANNELS_OUT>>(),
            )
            .permute::<_, Axes2<1, 0>>()
    }
}

#[cfg(test)]
mod tests {
    use super::Conv1D;
    use crate::{prelude::*, tests::assert_close};

    #[test]
    fn test_conv_simple() {
        let mut cx = Graph::new();

        const CHANNELS_IN: usize = 1;
        const CHANNELS_OUT: usize = 1;
        const KERNEL: usize = 2;
        const STRIDE: usize = KERNEL;
        const DILATION: usize = 0;
        const DIM_IN: usize = 6;
        const DIM_OUT: usize = 3;
        const POOL_OUT: usize = DIM_IN / STRIDE;

        let model: Conv1D<CHANNELS_IN, CHANNELS_OUT, KERNEL, KERNEL, DILATION, DIM_OUT, POOL_OUT> =
            Conv1D::initialize(&mut cx);
        model.weight.set(vec![0.0316, -0.2057]);

        let inp1 = cx.tensor::<R2<1, 6>>();
        inp1.set(vec![3., 0., 9., 6., 0., 6.]);

        let out1 = model.forward(inp1).retrieve();
        cx.execute();

        assert_close(&out1.data(), &[0.0948, -0.9498, -1.2342]);
    }
}
