use crate::prelude::*;
use rand::{thread_rng, Rng};

// Conv1D
pub struct Conv1D<
    const CHANNELS_IN: usize,
    const CHANNELS_OUT: usize,
    const KERNEL: usize,
    const STRIDE: usize,
    const DILATION: usize,
    const CHANNELS_IN_TIMES_KERNEL: usize,
> {
    pub weight: GraphTensor<R3<CHANNELS_IN, CHANNELS_OUT, KERNEL>>,
}

impl<
        const CHANNELS_IN: usize,
        const CHANNELS_OUT: usize,
        const KERNEL: usize,
        const STRIDE: usize,
        const DILATION: usize,
        const CHANNELS_IN_TIMES_KERNEL: usize,
    > InitModule
    for Conv1D<CHANNELS_IN, CHANNELS_OUT, KERNEL, STRIDE, DILATION, CHANNELS_IN_TIMES_KERNEL>
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
        const CHANNELS_IN_TIMES_KERNEL: usize,
    > SerializeModule
    for Conv1D<CHANNELS_IN, CHANNELS_OUT, KERNEL, STRIDE, DILATION, CHANNELS_IN_TIMES_KERNEL>
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
        const CHANNELS_IN_TIMES_KERNEL: usize,
    > Conv1D<CHANNELS_IN, CHANNELS_OUT, KERNEL, STRIDE, DILATION, CHANNELS_IN_TIMES_KERNEL>
{
    fn forward<const DIM_IN: usize, const DIM_OUT: usize>(
        &self,
        input: GraphTensor<R2<CHANNELS_IN, DIM_IN>>,
    ) -> GraphTensor<R2<CHANNELS_OUT, DIM_OUT>> {
        input
            .pool_last_dim::<R3<CHANNELS_IN, DIM_OUT, KERNEL>>(
                KERNEL.into(),
                STRIDE.into(),
                DILATION,
            )
            .permute::<_, Axes3<1, 0, 2>>()
            .reshape::<R2<DIM_OUT, CHANNELS_IN_TIMES_KERNEL>>()
            .matmul(
                self.weight
                    .permute::<_, Axes3<2, 1, 0>>()
                    .reshape::<R2<CHANNELS_IN_TIMES_KERNEL, CHANNELS_OUT>>(),
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
        const STRIDE: usize = 2;
        const DILATION: usize = 0;
        const DIM_IN: usize = 6;
        const DIM_OUT: usize = ((DIM_IN - (DILATION + 1) * (KERNEL - 1) - 1) / STRIDE) + 1;
        const CHANNELS_IN_TIMES_KERNEL: usize = CHANNELS_IN * KERNEL;

        let model: Conv1D<
            CHANNELS_IN,
            CHANNELS_OUT,
            KERNEL,
            KERNEL,
            DILATION,
            CHANNELS_IN_TIMES_KERNEL,
        > = Conv1D::initialize(&mut cx);
        model.weight.set(vec![0.0316, -0.2057]);

        let inp1 = cx.tensor::<R2<1, 6>>();
        inp1.set(vec![3., 0., 9., 6., 0., 6.]);

        let out1 = model.forward::<DIM_IN, DIM_OUT>(inp1).retrieve();
        cx.execute();

        assert_close(&out1.data(), &[0.0948, -0.9498, -1.2342]);
    }
}
