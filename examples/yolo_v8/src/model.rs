use luminal::prelude::*;
use luminal_nn::Conv2D;

struct Bottleneck<const CH_IN: usize, const CH_OUT: usize> {
    cv1: Conv2D<CH_IN, CH_OUT, 3, 3, 1, 1>,
    cv2: Conv2D<CH_OUT, CH_OUT, 3, 3, 1, 1>,
}

impl<const CH_IN: usize, const CH_OUT: usize, Width: Dimension, Height: Dimension>
    Module<GraphTensor<(Const<CH_IN>, Height, Width)>> for Bottleneck<CH_IN, CH_OUT>
{
    type Output = GraphTensor<(Const<CH_OUT>, Height, Width)>;
    fn forward(&self, input: GraphTensor<(Const<CH_IN>, Height, Width)>) -> Self::Output {
        self.cv2
            .forward(
                self.cv1
                    .forward::<Width, Height, Width, Height>(input.permute()),
            )
            .permute()
    }
}

struct C2F {
    cv1: ConvBlock,
    cv2: ConvBlock,
    bottleneck: Vec<Bottleneck>,
}

struct SPPF {
    cv1: ConvBlock,
    cv2: ConvBlock,
    k: usize,
}

struct DFL {
    conv: Conv2D,
    num_classes: usize,
}

struct DarkNet {
    b1_0: ConvBlock,
    b1_1: ConvBlock,
    b2_0: C2F,
    b2_1: ConvBlock,
    b2_2: C2F,
    b3_0: ConvBlock,
    b3_1: C2F,
    b4_0: ConvBlock,
    b4_1: C2F,
    b5: SPPF,
}

struct YoloNeck {
    n1: C2F,
    n2: C2F,
    n3: ConvBlock,
    n4: C2F,
    n5: ConvBlock,
    n6: C2F,
}

struct DetectionHead {
    dfl: DFL,
    cv2: [(ConvBlock, ConvBlock, Conv2D); 3],
    cv3: [(ConvBlock, ConvBlock, Conv2D); 3],
    ch: usize,
    no: usize,
}

pub struct Yolo {
    net: DarkNet,
    fpn: YoloNeck,
    head: DetectionHead,
}

impl<
        const CHANNELS: usize,
        const CLASSIFICATION: usize,
        Batch: Dimension,
        Height: Dimension,
        Width: Dimension,
    > Module<GraphTensor<(Batch, Const<CHANNELS>, Width, Height)>> for Yolo
{
    type Output = GraphTensor<(Batch,)>;

    fn forward(&self, input: GraphTensor<(Batch, Const<CHANNELS>, Width, Height)>) -> Self::Output {
    }
}
