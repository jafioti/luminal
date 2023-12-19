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
    fn test_conv1d_simple() {
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

        let inp1 = cx.tensor::<R2<CHANNELS_IN, DIM_IN>>();
        inp1.set(vec![3., 0., 9., 6., 0., 6.]);

        let out1 = model.forward::<DIM_IN, DIM_OUT>(inp1).retrieve();
        cx.execute();

        assert_close(&out1.data(), &[0.0948, -0.9498, -1.2342]);
    }

    #[test]
    fn test_conv1d() {
        let mut cx = Graph::new();

        const CHANNELS_IN: usize = 8;
        const CHANNELS_OUT: usize = 4;
        const KERNEL: usize = 2;
        const STRIDE: usize = 2;
        const DILATION: usize = 0;
        const DIM_IN: usize = 12;
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
        model.weight.set(vec![
            -0.1700, -0.2000, 0.1000, -0.0200, 0.1000, 0.0200, -0.2100, -0.2300, -0.0600, 0.1500,
            0.1200, 0.1000, 0.1800, 0.0600, -0.1700, -0.0400, 0.1000, -0.0200, -0.1700, 0.1000,
            0.1100, 0.1600, 0.2000, 0.0100, -0.0500, 0.2100, -0.0200, 0.0300, -0.0900, -0.0500,
            0.1600, 0.0400, 0.0400, -0.1700, 0.1100, 0.0600, -0.1200, -0.2300, 0.2300, -0.2100,
            -0.2200, 0.1100, -0.0100, -0.1400, 0.1700, 0.0300, 0.1000, -0.1400, -0.2100, -0.1800,
            0.2000, -0.2300, -0.1600, 0.2200, 0.0900, 0.0700, -0.1000, -0.0400, -0.0500, 0.1400,
            0.0700, -0.1200, 0.1400, 0.2200,
        ]);
        let weight = model.weight.permute::<_, Axes3<1, 0, 2>>().retrieve();

        let inp1 = cx.tensor::<R2<CHANNELS_IN, DIM_IN>>();
        inp1.set(vec![
            1., 2., 6., 4., 8., 1., 6., 0., 1., 0., 6., 4., 3., 4., 9., 3., 8., 8., 5., 5., 0., 4.,
            2., 7., 6., 4., 2., 2., 8., 0., 7., 3., 0., 0., 7., 2., 3., 3., 1., 9., 5., 4., 5., 5.,
            8., 0., 0., 1., 2., 1., 8., 9., 4., 7., 7., 6., 8., 5., 0., 9., 1., 6., 0., 1., 4., 3.,
            3., 5., 8., 7., 9., 5., 6., 5., 6., 9., 7., 0., 9., 5., 6., 0., 6., 1., 2., 1., 0., 1.,
            3., 6., 8., 0., 6., 6., 3., 2.,
        ]);
        inp1.retrieve();

        let out1 = model.forward::<DIM_IN, DIM_OUT>(inp1).retrieve();
        let exp_out1 = cx.tensor::<R2<CHANNELS_OUT, DIM_OUT>>();
        exp_out1.set(vec![
            0.7600, -0.4700, 0.0100, -0.1600, -0.1800, 2.2300, 1.7200, 0.6900, 3.5100, 3.7700,
            3.4600, 3.8100, -1.2600, -1.3900, 0.9400, 0.5300, 0.6300, -0.0400, 0.3800, -1.4900,
            -0.8800, -0.3100, 1.7500, -2.7500,
        ]);
        exp_out1.retrieve();
        cx.execute();

        // println!("{:?}", weight);
        // println!("{:?}", inp1);
        println!("Expected: {:?}", exp_out1);
        println!("Actual: {:?}", out1);

        assert_close(
            &out1.data(),
            &[
                -0.2061, -1.3519, -2.4679, 0.6841, -1.4186, -2.2877, -0.1398, -0.8774, 2.7790,
                0.2744, 0.6171, 2.3570, -0.5874, 0.4322, 0.3313, -1.1620, 1.9789, 1.0397, 1.1491,
                -2.4564, 3.7550, 1.8571, 3.2417, 4.4339,
            ],
        );
    }
}
