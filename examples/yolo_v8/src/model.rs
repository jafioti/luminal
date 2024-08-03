use luminal::prelude::*;
use luminal_nn::Conv2D;

struct Upsample {
    scale_factor: usize,
}

impl Upsample {
    fn new(scale_factor: usize) -> Self {
        Upsample { scale_factor }
    }
}

impl Module<GraphTensor> for Upsample {
    type Output = GraphTensor;
    fn forward(&self, xs: GraphTensor) -> GraphTensor {
        let (batch, channels, h, w) = xs.dims4();
        xs.expand(3, self.scale_factor)
            .expand(5, self.scale_factor)
            .reshape((
                batch,
                channels,
                self.scale_factor * h,
                self.scale_factor * w,
            )) // Double height and width
    }
}

struct ConvBlock {
    conv: Conv2D,
    running_mean: GraphTensor,
    running_var: GraphTensor,
    weight: GraphTensor,
    bias: GraphTensor,
    original_weight: GraphTensor,
}

impl ConvBlock {
    fn new(
        ch_in: usize,
        ch_out: usize,
        kernel: (usize, usize),
        stride: (usize, usize),
        dilation: (usize, usize),
        cx: &mut Graph,
    ) -> Self {
        let eps = 1e-3;
        let mut conv = Conv2D::new(ch_in, ch_out, kernel, stride, dilation, false, cx);
        let original_weight = conv.weight;
        let running_mean = cx.constant(0.).expand(0, ch_out);
        let running_var = cx.constant(1.).expand(0, ch_out);
        let o_weight = cx.constant(1.).expand(0, ch_out);
        let o_bias = cx.constant(0.).expand(0, ch_out);
        let std_ = o_weight / ((running_var + eps).sqrt());
        let weight = conv.weight * std_.expand(1, conv.weight.dims2().1);
        let bias = o_bias - (std_ * running_mean);
        conv.weight = weight;
        conv.bias = Some(bias);
        Self {
            conv,
            running_var,
            running_mean,
            original_weight,
            weight: o_weight,
            bias: o_bias,
        }
    }
}

impl Module<GraphTensor> for ConvBlock {
    type Output = GraphTensor;
    fn forward(&self, input: GraphTensor) -> Self::Output {
        self.conv.forward(input)
    }
}

impl SerializeModule for ConvBlock {
    fn serialize(&self, s: &mut Serializer) {
        s.tensor("conv/weight", self.original_weight);
        s.tensor("bn/weight", self.weight);
        s.tensor("bn/bias", self.bias);
        s.tensor("bn/running_mean", self.running_mean);
        s.tensor("bn/running_var", self.running_var);
    }
}

struct Bottleneck {
    cv1: ConvBlock,
    cv2: ConvBlock,
    residual: bool,
}

impl Bottleneck {
    pub fn new(ch_in: usize, ch_out: usize, shortcut: bool, cx: &mut Graph) -> Self {
        Self {
            cv1: ConvBlock::new(ch_in, ch_out, (3, 3), (3, 3), (1, 1), cx),
            cv2: ConvBlock::new(ch_out, ch_out, (3, 3), (3, 3), (1, 1), cx),
            residual: ch_in == ch_out && shortcut,
        }
    }
}

impl SerializeModule for Bottleneck {
    fn serialize(&self, s: &mut Serializer) {
        s.module("cv1", &self.cv1);
        s.module("cv2", &self.cv2);
    }
}

impl Module<GraphTensor> for Bottleneck {
    type Output = GraphTensor;
    fn forward(&self, input: GraphTensor) -> Self::Output {
        let mut out = self
            .cv2
            .forward(self.cv1.forward(input.permute((0, 1, 3, 2))))
            .permute((0, 1, 3, 2));
        if self.residual {
            out += input
        }
        out
    }
}

struct C2f {
    cv1: ConvBlock,
    cv2: ConvBlock,
    bottleneck: Vec<Bottleneck>,
    c: usize,
}

impl C2f {
    pub fn new(c1: usize, c2: usize, n: usize, shortcut: bool, cx: &mut Graph) -> Self {
        let c = (c2 as f64 / 2.) as usize;
        Self {
            cv1: ConvBlock::new(c1, 2 * c, (1, 1), (1, 1), (1, 1), cx),
            cv2: ConvBlock::new((2 + n) * c, c2, (1, 1), (1, 1), (1, 1), cx),
            bottleneck: (0..n)
                .map(|_| Bottleneck::new(c, c, shortcut, cx))
                .collect(),
            c,
        }
    }
}

impl SerializeModule for C2f {
    fn serialize(&self, s: &mut Serializer) {
        s.module("cv1", &self.cv1);
        s.module("cv2", &self.cv2);
        for (i, l) in self.bottleneck.iter().enumerate() {
            s.module(&format!("bottleneck/{i}"), l);
        }
    }
}

impl Module<GraphTensor> for C2f {
    type Output = GraphTensor;
    fn forward(&self, input: GraphTensor) -> Self::Output {
        let ys = self.cv1.forward(input);
        let mut ys = chunk(ys, 2, 1);
        for m in self.bottleneck.iter() {
            ys.push(m.forward(*ys.last().unwrap()));
        }
        let mut fin = ys.remove(0);
        for t in ys {
            fin = fin.concat_along(t, 1);
        }
        self.cv2.forward(fin)
    }
}

fn chunk(tensor: GraphTensor, chunks: usize, dim: usize) -> Vec<GraphTensor> {
    let chunk_size = tensor.dims()[dim] / chunks;
    let mut t = vec![];
    for i in 0..chunks {
        t.push(tensor.slice_along(i * chunk_size..(i + 1) * chunk_size, dim));
    }
    t
}

#[allow(clippy::upper_case_acronyms)]
struct SPPF {
    cv1: ConvBlock,
    cv2: ConvBlock,
    k: usize,
}

impl SerializeModule for SPPF {
    fn serialize(&self, s: &mut Serializer) {
        s.module("cv1", &self.cv1);
        s.module("cv2", &self.cv2);
    }
}

impl SPPF {
    pub fn new(c1: usize, c2: usize, k: usize, cx: &mut Graph) -> Self {
        let c_ = c1 / 2;
        Self {
            cv1: ConvBlock::new(c1, c_, (1, 1), (1, 1), (1, 1), cx),
            cv2: ConvBlock::new(c_ * 4, c2, (1, 1), (1, 1), (1, 1), cx),
            k,
        }
    }
}

impl Module<GraphTensor> for SPPF {
    type Output = GraphTensor;
    fn forward(&self, xs: GraphTensor) -> Self::Output {
        let xs = self.cv1.forward(xs);
        let xs2 = xs
            .pad((
                (0, 0),
                (0, 0),
                (self.k / 2, self.k / 2),
                (self.k / 2, self.k / 2),
            ))
            .pool_last_dim(self.k, 1, 1)
            .max_reduce(4);
        let xs3 = xs2
            .pad((
                (0, 0),
                (0, 0),
                (self.k / 2, self.k / 2),
                (self.k / 2, self.k / 2),
            ))
            .pool_last_dim(self.k, 1, 1)
            .max_reduce(4);
        let xs4 = xs3
            .pad((
                (0, 0),
                (0, 0),
                (self.k / 2, self.k / 2),
                (self.k / 2, self.k / 2),
            ))
            .pool_last_dim(self.k, 1, 1)
            .max_reduce(4);
        self.cv2.forward(
            xs.concat_along(xs2, 1)
                .concat_along(xs3, 1)
                .concat_along(xs4, 1),
        )
    }
}

struct DFL {
    conv: Conv2D,
    num_classes: usize,
}

impl SerializeModule for DFL {
    fn serialize(&self, s: &mut Serializer) {
        s.module("conv", &self.conv);
    }
}

impl DFL {
    pub fn new(num_classes: usize, cx: &mut Graph) -> Self {
        Self {
            conv: Conv2D::new(num_classes, 1, (1, 1), (1, 1), (1, 1), false, cx),
            num_classes,
        }
    }
}

impl Module<GraphTensor> for DFL {
    type Output = GraphTensor;
    fn forward(&self, xs: GraphTensor) -> Self::Output {
        let (b_sz, _channels, anchors) = xs.dims3();
        let xs = xs
            .reshape((b_sz, 4, self.num_classes, anchors))
            .permute((0, 1, 3, 2))
            .softmax(1);
        self.conv.forward(xs).reshape((b_sz, 4, anchors))
    }
}

struct DarkNet {
    b1_0: ConvBlock,
    b1_1: ConvBlock,
    b2_0: C2f,
    b2_1: ConvBlock,
    b2_2: C2f,
    b3_0: ConvBlock,
    b3_1: C2f,
    b4_0: ConvBlock,
    b4_1: C2f,
    b5: SPPF,
}

impl SerializeModule for DarkNet {
    fn serialize(&self, s: &mut Serializer) {
        s.module("b1.0", &self.b1_0);
        s.module("b1.1", &self.b1_1);
        s.module("b2.0", &self.b2_0);
        s.module("b2.1", &self.b2_1);
        s.module("b2.2", &self.b2_2);
        s.module("b3.0", &self.b3_0);
        s.module("b3.1", &self.b3_1);
        s.module("b4.0", &self.b4_0);
        s.module("b4.1", &self.b4_1);
        s.module("b5.0", &self.b5);
    }
}

impl DarkNet {
    pub fn new(w: f64, r: f64, d: f64, cx: &mut Graph) -> Self {
        Self {
            b1_0: ConvBlock::new(3, (64. * w) as usize, (3, 3), (2, 2), (1, 1), cx),
            b1_1: ConvBlock::new(
                (64. * w) as usize,
                (128. * w) as usize,
                (3, 3),
                (2, 2),
                (1, 1),
                cx,
            ),
            b2_0: C2f::new(
                (128. * w) as usize,
                (128. * w) as usize,
                (3. * d).round() as usize,
                true,
                cx,
            ),
            b2_1: ConvBlock::new(
                (128. * w) as usize,
                (256. * w) as usize,
                (3, 3),
                (2, 2),
                (1, 1),
                cx,
            ),
            b2_2: C2f::new(
                (256. * w) as usize,
                (256. * w) as usize,
                (6. * d).round() as usize,
                true,
                cx,
            ),
            b3_0: ConvBlock::new(
                (256. * w) as usize,
                (512. * w) as usize,
                (3, 3),
                (2, 2),
                (1, 1),
                cx,
            ),
            b3_1: C2f::new(
                (512. * w) as usize,
                (512. * w) as usize,
                (6. * d).round() as usize,
                true,
                cx,
            ),
            b4_0: ConvBlock::new(
                (512. * w) as usize,
                (512. * w * r) as usize,
                (3, 3),
                (2, 2),
                (1, 1),
                cx,
            ),
            b4_1: C2f::new(
                (512. * w * r) as usize,
                (512. * w * r) as usize,
                (3. * d).round() as usize,
                true,
                cx,
            ),
            b5: SPPF::new((512. * w * r) as usize, (512. * w * r) as usize, 5, cx),
        }
    }
}

impl Module<GraphTensor> for DarkNet {
    type Output = (GraphTensor, GraphTensor, GraphTensor);
    fn forward(&self, xs: GraphTensor) -> Self::Output {
        let x1 = self.b1_1.forward(self.b1_0.forward(xs));
        let x2 = self.b2_2.forward(self.b2_1.forward(self.b2_0.forward(x1)));
        let x3 = self.b3_1.forward(self.b3_0.forward(x2));
        let x4 = self.b4_1.forward(self.b4_0.forward(x3));
        let x5 = self.b5.forward(x4);
        (x2, x3, x5)
    }
}

struct YoloNeck {
    up: Upsample,
    n1: C2f,
    n2: C2f,
    n3: ConvBlock,
    n4: C2f,
    n5: ConvBlock,
    n6: C2f,
}

impl SerializeModule for YoloNeck {
    fn serialize(&self, s: &mut Serializer) {
        s.module("n1", &self.n1);
        s.module("n2", &self.n2);
        s.module("n3", &self.n3);
        s.module("n4", &self.n4);
        s.module("n5", &self.n5);
        s.module("n6", &self.n6);
    }
}

impl YoloNeck {
    pub fn new(w: f64, r: f64, d: f64, cx: &mut Graph) -> Self {
        let n = (3. * d).round() as usize;
        Self {
            up: Upsample::new(2),
            n1: C2f::new(
                (512. * w * (1. + r)) as usize,
                (512. * w) as usize,
                n,
                false,
                cx,
            ),
            n2: C2f::new((768. * w) as usize, (256. * w) as usize, n, false, cx),
            n3: ConvBlock::new(
                (256. * w) as usize,
                (256. * w) as usize,
                (3, 3),
                (2, 2),
                (1, 1),
                cx,
            ),
            n4: C2f::new((768. * w) as usize, (512. * w) as usize, n, false, cx),
            n5: ConvBlock::new(
                (512. * w) as usize,
                (512. * w) as usize,
                (3, 3),
                (2, 2),
                (1, 1),
                cx,
            ),
            n6: C2f::new(
                (512. * w * (1. + r)) as usize,
                (512. * w * r) as usize,
                n,
                false,
                cx,
            ),
        }
    }
}

impl Module<(GraphTensor, GraphTensor, GraphTensor)> for YoloNeck {
    type Output = (GraphTensor, GraphTensor, GraphTensor);
    fn forward(&self, (p3, p4, p5): (GraphTensor, GraphTensor, GraphTensor)) -> Self::Output {
        let x = self.n1.forward(self.up.forward(p5).concat_along(p4, 1));
        let head_1 = self.n2.forward(self.up.forward(x).concat_along(p3, 1));
        let head_2 = self.n4.forward(self.n3.forward(head_1).concat_along(x, 1));
        let head_3 = self.n6.forward(self.n5.forward(head_2).concat_along(p5, 1));
        (head_1, head_2, head_3)
    }
}

struct DetectionHead {
    dfl: DFL,
    cv2: [(ConvBlock, ConvBlock, Conv2D); 3],
    cv3: [(ConvBlock, ConvBlock, Conv2D); 3],
    ch: usize,
    no: usize,
}

impl SerializeModule for DetectionHead {
    fn serialize(&self, s: &mut Serializer) {
        s.module("dfl", &self.dfl);
        for (i, m) in self.cv2.iter().enumerate() {
            s.module(&format!("cv2/{i}"), m);
        }
        for (i, m) in self.cv3.iter().enumerate() {
            s.module(&format!("cv3/{i}"), m);
        }
    }
}

impl DetectionHead {
    pub fn new(nc: usize, filters: (usize, usize, usize), cx: &mut Graph) -> Self {
        let ch = 16;
        let c1 = usize::max(filters.0, nc);
        let c2 = usize::max(filters.0 / 4, ch * 4);
        Self {
            dfl: DFL::new(ch, cx),
            cv2: [
                Self::new_cv2(c2, ch, filters.0, cx),
                Self::new_cv2(c2, ch, filters.1, cx),
                Self::new_cv2(c2, ch, filters.2, cx),
            ],
            cv3: [
                Self::new_cv3(c1, nc, filters.0, cx),
                Self::new_cv3(c1, nc, filters.1, cx),
                Self::new_cv3(c1, nc, filters.2, cx),
            ],
            ch,
            no: nc + ch * 4,
        }
    }

    fn new_cv3(
        c1: usize,
        nc: usize,
        filter: usize,
        cx: &mut Graph,
    ) -> (ConvBlock, ConvBlock, Conv2D) {
        (
            ConvBlock::new(filter, c1, (3, 3), (1, 1), (1, 1), cx),
            ConvBlock::new(c1, c1, (3, 3), (1, 1), (1, 1), cx),
            Conv2D::new(c1, nc, (1, 1), (1, 1), (1, 1), true, cx),
        )
    }

    fn new_cv2(
        c2: usize,
        ch: usize,
        filter: usize,
        cx: &mut Graph,
    ) -> (ConvBlock, ConvBlock, Conv2D) {
        (
            ConvBlock::new(filter, c2, (3, 3), (1, 1), (1, 1), cx),
            ConvBlock::new(c2, c2, (3, 3), (1, 1), (1, 1), cx),
            Conv2D::new(c2, 4 * ch, (1, 1), (1, 1), (1, 1), true, cx),
        )
    }
}

impl Module<(GraphTensor, GraphTensor, GraphTensor)> for DetectionHead {
    type Output = (GraphTensor, GraphTensor, GraphTensor);
    fn forward(&self, (xs0, xs1, xs2): (GraphTensor, GraphTensor, GraphTensor)) -> Self::Output {
        let forward_cv = |xs, i: usize| {
            let xs_2 = self.cv2[i].0.forward(xs);
            let xs_2 = self.cv2[i].1.forward(xs_2);
            let xs_2 = self.cv2[i].2.forward(xs_2);

            let xs_3 = self.cv3[i].0.forward(xs);
            let xs_3 = self.cv3[i].1.forward(xs_3);
            let xs_3 = self.cv3[i].2.forward(xs_3);
            xs_2.concat_along(xs_3, 1)
        };
        let xs0 = forward_cv(xs0, 0);
        let xs1 = forward_cv(xs1, 1);
        let xs2 = forward_cv(xs2, 2);

        let (anchors, strides) = make_anchors(xs0, xs1, xs2, (8, 16, 32), 0.5);
        let anchors = anchors.permute((1, 0)).expand(0, 1);
        let strides = strides.permute((1, 0));

        let reshape = |xs: GraphTensor| {
            let d = xs.dims()[0];
            let el = xs.shape.n_elements();
            xs.reshape((d, self.no, el / (d * self.no)))
        };
        let ys0 = reshape(xs0);
        let ys1 = reshape(xs1);
        let ys2 = reshape(xs2);

        let x_cat = ys0.concat_along(ys1, 2).concat_along(ys2, 2);
        let box_ = x_cat.slice((.., ..self.ch * 4));
        let cls = x_cat.slice((.., self.ch * 4..));

        let dbox = dist2bbox(self.dfl.forward(box_), anchors);
        let dbox = dbox * strides.expand_to(dbox.shape);
        let pred = dbox.concat_along(cls.sigmoid(), 1);
        (pred, anchors, strides)
    }
}

pub struct Yolo {
    net: DarkNet,        // Backbone
    fpn: YoloNeck,       // Neck
    head: DetectionHead, // Head
}

impl SerializeModule for Yolo {
    fn serialize(&self, s: &mut Serializer) {
        s.module("net", &self.net);
        s.module("fpn", &self.fpn);
        s.module("head", &self.head);
    }
}

impl Yolo {
    pub fn new(w: f64, r: f64, d: f64, num_classes: usize, cx: &mut Graph) -> Self {
        let f1 = (256. * w) as usize;
        let f2 = (512. * r) as usize;
        let f3 = (512. * w * r) as usize;
        Self {
            net: DarkNet::new(w, r, d, cx),
            fpn: YoloNeck::new(w, r, d, cx),
            head: DetectionHead::new(num_classes, (f1, f2, f3), cx),
        }
    }
}

fn make_anchors(
    xs0: GraphTensor,
    xs1: GraphTensor,
    xs2: GraphTensor,
    (s0, s1, s2): (usize, usize, usize),
    grid_cell_offset: f64,
) -> (GraphTensor, GraphTensor) {
    let cx = xs0.graph();
    let mut anchor_points = vec![];
    let mut stride_tensor = vec![];
    for (xs, stride) in [(xs0, s0), (xs1, s1), (xs2, s2)] {
        // xs is only used to extract the h and w dimensions.
        let (_, _, h, w) = xs.dims4();
        let sx = cx.arange(w) + grid_cell_offset as f32;
        let sy = cx.arange(h) + grid_cell_offset as f32;
        let sx = sx.reshape((1, w)).expand(0, h).reshape(h * w);
        let sy = sy.reshape((h, 1)).expand(1, w).reshape(h * w);
        anchor_points.push(sx.expand(0, 1).concat_along(sy.expand(0, 1), 0));
        stride_tensor.push(cx.constant(1.).expand(0, h * w) * stride as f32);
    }
    let anchor_points = anchor_points
        .into_iter()
        .reduce(|acc, t| acc.concat_along(t, 0))
        .unwrap();
    let stride_tensor = stride_tensor
        .into_iter()
        .reduce(|acc, t| acc.concat_along(t, 0))
        .unwrap()
        .expand(1, 1);
    (anchor_points, stride_tensor)
}

fn dist2bbox(distance: GraphTensor, anchor_points: GraphTensor) -> GraphTensor {
    let chunks = chunk(distance, 2, 1);
    let lt = chunks[0];
    let rb = chunks[1];
    let x1y1 = anchor_points - lt;
    let x2y2 = anchor_points + rb;
    let c_xy = (x1y1 + x2y2) * 0.5;
    let wh = x2y2 - x1y1;
    c_xy.concat_along(wh, 1)
}

impl Module<GraphTensor> for Yolo {
    type Output = GraphTensor;

    fn forward(&self, xs: GraphTensor) -> Self::Output {
        let (xs1, xs2, xs3) = self.net.forward(xs);
        let (xs1, xs2, xs3) = self.fpn.forward((xs1, xs2, xs3));
        let (pred, _, _) = self.head.forward((xs1, xs2, xs3));
        pred
    }
}
