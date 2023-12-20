use crate::prelude::*;
use rand::{thread_rng, Rng};

pub struct Conv1D<
    const CHANNELS_IN: usize,
    const CHANNELS_OUT: usize,
    const KERNEL: usize,
    const STRIDE: usize,
    const DILATION: usize,
    const CHANNELS_IN_TIMES_KERNEL: usize,
> {
    pub weight: GraphTensor<R2<CHANNELS_OUT, CHANNELS_IN_TIMES_KERNEL>>,
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
    pub fn forward<const DIM_IN: usize, const DIM_OUT: usize>(
        &self,
        input: GraphTensor<R2<CHANNELS_IN, DIM_IN>>,
    ) -> GraphTensor<R2<CHANNELS_OUT, DIM_OUT>> {
        self.weight.matmul(
            input
                .pool_last_dim::<R3<CHANNELS_IN, DIM_OUT, KERNEL>>(
                    KERNEL.into(),
                    STRIDE.into(),
                    DILATION,
                )
                .permute::<_, Axes3<0, 2, 1>>()
                .reshape::<R2<CHANNELS_IN_TIMES_KERNEL, DIM_OUT>>(),
        )
    }
}

pub struct Conv2D<
    const CHANNELS_IN: usize,
    const CHANNELS_OUT: usize,
    const KERNEL: usize,
    const STRIDE: usize,
    const DILATION: usize,
    const CHANNELS_IN_TIMES_KERNEL: usize,
> {
    pub weight: GraphTensor<R2<CHANNELS_OUT, CHANNELS_IN_TIMES_KERNEL>>,
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
        cx.execute();

        assert_close(
            &out1.data(),
            &[
                0.7600, -0.4700, 0.0100, -0.1600, -0.1800, 2.2300, 1.7200, 0.6900, 3.5100, 3.7700,
                3.4600, 3.8100, -1.2600, -1.3900, 0.9400, 0.5300, 0.6300, -0.0400, 0.3800, -1.4900,
                -0.8800, -0.3100, 1.7500, -2.7500,
            ],
        );
    }

    #[test]
    fn test_conv2d() {
        let mut cx = Graph::new();

        const CHANNELS_IN: usize = 5;
        const CHANNELS_OUT: usize = 2;
        const KERNELX: usize = 2;
        const KERNELY: usize = 2;
        const STRIDEX: usize = 2;
        const STRIDEY: usize = 2;
        const DILATIONX: usize = 0;
        const DILATIONY: usize = 0;
        const DIMX_IN: usize = 16;
        const DIMX_OUT: usize = ((DIMX_IN - (DILATIONX + 1) * (KERNELX - 1) - 1) / STRIDEX) + 1;
        const DIMY_IN: usize = 9;
        const DIMY_OUT: usize = ((DIMY_IN - (DILATIONY + 1) * (KERNELY - 1) - 1) / STRIDEY) + 1;
        const DIMX_TIMES_DIMY_OUT: usize = DIMX_OUT * DIMY_OUT;
        const CHANNELS_IN_TIMES_KERNELX_KERNELY: usize = CHANNELS_IN * KERNELX * KERNELY;

        let inp1 = cx.tensor::<R3<CHANNELS_IN, DIMX_IN, DIMY_IN>>();
        inp1.set(vec![
            8., 8., 5., 7., 0., 6., 5., 3., 0., 7., 0., 6., 6., 7., 7., 5., 0., 6., 9., 4., 0., 8.,
            8., 5., 7., 6., 2., 8., 9., 5., 0., 3., 1., 1., 8., 4., 1., 1., 5., 6., 9., 3., 2., 9.,
            4., 7., 1., 0., 7., 7., 4., 9., 5., 0., 4., 7., 4., 7., 8., 8., 4., 8., 4., 7., 9., 3.,
            7., 9., 5., 8., 5., 9., 0., 9., 5., 6., 8., 9., 5., 4., 1., 9., 7., 2., 2., 7., 9., 3.,
            1., 2., 8., 4., 0., 8., 0., 5., 6., 7., 7., 4., 3., 4., 6., 8., 3., 7., 8., 8., 7., 1.,
            5., 1., 8., 0., 1., 1., 7., 3., 2., 1., 0., 4., 5., 4., 3., 2., 5., 4., 2., 4., 1., 9.,
            4., 1., 9., 7., 7., 1., 2., 6., 3., 4., 1., 1., 6., 6., 8., 2., 7., 7., 9., 0., 9., 0.,
            1., 4., 2., 4., 9., 6., 8., 6., 1., 6., 3., 8., 3., 4., 5., 0., 2., 1., 8., 2., 2., 8.,
            7., 0., 7., 7., 3., 4., 5., 0., 7., 2., 1., 1., 4., 2., 9., 9., 6., 1., 5., 4., 6., 9.,
            5., 4., 1., 9., 1., 5., 5., 5., 8., 8., 0., 1., 3., 0., 8., 8., 5., 1., 6., 1., 5., 6.,
            4., 4., 4., 0., 1., 1., 5., 1., 7., 2., 3., 5., 5., 4., 9., 1., 3., 7., 6., 7., 1., 5.,
            3., 8., 6., 6., 6., 7., 3., 2., 2., 8., 1., 3., 0., 2., 7., 6., 5., 7., 5., 7., 8., 1.,
            2., 2., 5., 0., 2., 9., 1., 5., 3., 8., 7., 9., 7., 2., 8., 8., 8., 6., 3., 2., 7., 7.,
            0., 3., 7., 8., 3., 7., 2., 3., 2., 7., 5., 5., 6., 0., 9., 0., 9., 9., 1., 8., 7., 9.,
            6., 8., 7., 5., 4., 9., 5., 6., 3., 2., 8., 3., 0., 6., 3., 8., 3., 1., 8., 7., 2., 0.,
            7., 7., 7., 7., 8., 0., 4., 9., 8., 2., 0., 4., 4., 3., 5., 5., 3., 0., 3., 6., 3., 1.,
            2., 9., 9., 6., 8., 1., 2., 6., 8., 6., 0., 0., 2., 8., 8., 5., 0., 5., 9., 0., 8., 1.,
            1., 3., 5., 9., 3., 5., 8., 6., 3., 2., 9., 4., 8., 3., 9., 5., 2., 9., 0., 1., 6., 8.,
            0., 3., 0., 1., 2., 1., 0., 1., 4., 1., 1., 0., 6., 9., 2., 7., 2., 6., 0., 4., 8., 2.,
            6., 7., 2., 2., 7., 4., 5., 8., 1., 4., 7., 5., 9., 7., 2., 5., 9., 1., 6., 1., 7., 9.,
            5., 6., 9., 3., 5., 1., 6., 1., 3., 3., 9., 3., 9., 0., 1., 8., 1., 9., 8., 5., 3., 4.,
            4., 1., 5., 5., 4., 4., 5., 8., 7., 1., 1., 7., 3., 9., 0., 1., 3., 4., 8., 4., 0., 5.,
            6., 2., 0., 7., 8., 2., 6., 2., 9., 6., 2., 0., 3., 7., 5., 7., 1., 8., 5., 5., 9., 1.,
            0., 3., 5., 7., 5., 3., 2., 8., 6., 3., 0., 5., 8., 5., 7., 8., 8., 2., 9., 0., 1., 8.,
            6., 0., 3., 2., 5., 2., 9., 8., 9., 6., 2., 0., 3., 2., 5., 9., 1., 3., 6., 5., 2., 8.,
            2., 2., 1., 8., 6., 4., 1., 6., 0., 7., 3., 0., 9., 6., 5., 5., 5., 2., 4., 2., 8., 3.,
            0., 6., 3., 8., 8., 4., 9., 4., 7., 0., 3., 5., 1., 4., 6., 0., 0., 5., 9., 7., 8., 6.,
            7., 0., 6., 7., 0., 5., 8., 8., 6., 4., 6., 0., 2., 3., 2., 8., 7., 5., 9., 6., 6., 2.,
            0., 4., 4., 4., 4., 2., 7., 5., 3., 2., 6., 3., 7., 0., 7., 2., 5., 1., 4., 4., 5., 1.,
            6., 7., 5., 7., 0., 7., 8., 4., 7., 3., 9., 1., 7., 5., 6., 1., 0., 2., 0., 0., 5., 5.,
            8., 8., 7., 3., 7., 2., 9., 3., 8., 4., 5., 3., 8., 5., 2., 0., 2., 0., 5., 9., 0., 3.,
            8., 0., 4., 1., 8., 4., 8., 9., 1., 1., 4., 5., 0., 2., 0., 9., 4., 2., 3., 9., 0., 7.,
            3., 1., 5., 9., 1., 6., 5., 4., 2., 1., 2., 1., 1., 4., 7., 2.,
        ]);

        let exp_out1 = cx.tensor::<R3<CHANNELS_OUT, DIMX_OUT, DIMY_OUT>>();
        exp_out1.set(vec![
            3.9600, -0.3300, -1.7800, 4.0400, 1.5300, 0.2900, 2.8700, 3.0000, 0.9600, -1.8700,
            4.5900, 3.9700, 1.2800, 1.1800, 3.7800, 2.8500, 0.5500, 0.5600, 3.9800, 1.3200,
            -0.7100, -0.6500, 4.3900, 0.4000, 1.0300, 0.9800, 3.1200, 2.7400, 2.5100, 0.1200,
            1.8500, 2.0000, -0.7900, 1.0700, -0.3900, -0.8100, -2.5100, -2.9700, 0.2100, 1.8400,
            -0.7700, -0.3900, 1.2200, 0.1900, 4.1700, -4.3600, -1.8600, 0.4800, -2.4400, 2.6300,
            1.5000, -1.9700, 1.2800, -2.8200, -2.3200, 0.2200, -0.3800, 2.1800, -0.8200, -1.5700,
            1.2000, -3.4200, -1.6700, 0.9000,
        ]);

        let conv = cx.tensor::<R2<CHANNELS_OUT, CHANNELS_IN_TIMES_KERNELX_KERNELY>>();
        conv.set(vec![
            0.1600, 0.2000, 0.1900, -0.1100, 0.0100, -0.0300, -0.1200, -0.0800, -0.1300, -0.0300,
            0.1600, -0.1700, -0.0000, 0.1900, 0.1300, 0.0300, -0.1500, 0.0900, 0.0100, 0.0200,
            0.1500, 0.0700, -0.0800, 0.1700, 0.1000, -0.0700, 0.1600, -0.1600, -0.1900, -0.0500,
            -0.2100, 0.0100, -0.2000, 0.2100, -0.0400, -0.1400, 0.1500, 0.0500, -0.1700, 0.1400,
        ]);

        let inp1_pooled = inp1
            .pool_last_dim::<R4<CHANNELS_IN, DIMX_IN, DIMY_OUT, KERNELY>>(
                KERNELY.into(),
                STRIDEY.into(),
                DILATIONY,
            )
            .permute::<_, Axes4<0, 2, 3, 1>>()
            .pool_last_dim::<R5<CHANNELS_IN, DIMY_OUT, KERNELY, DIMX_OUT, KERNELX>>(
                KERNELX.into(),
                STRIDEX.into(),
                DILATIONX,
            )
            .permute::<_, Axes5<0, 4, 2, 3, 1>>()
            .reshape::<R2<CHANNELS_IN_TIMES_KERNELX_KERNELY, DIMX_TIMES_DIMY_OUT>>();

        let out1 = conv
            .matmul(inp1_pooled)
            .reshape::<R3<CHANNELS_OUT, DIMX_OUT, DIMY_OUT>>();

        let weight = conv.reshape::<R4<CHANNELS_OUT, CHANNELS_IN, KERNELX, KERNELY>>();

        weight.retrieve();
        exp_out1.retrieve();
        out1.retrieve();

        cx.execute();

        assert_close(&out1.data(), &exp_out1.data())
    }
}
