use rand::{rngs::StdRng, SeedableRng};

use crate::{
    nn::{activation::ReLU, linear::Linear, transformer::Transformer},
    prelude::*,
};

use super::random_vec_rng;

pub fn matmul() -> (Graph, Vec<GraphTensor<()>>) {
    let mut rng = StdRng::seed_from_u64(0);
    let mut cx = Graph::new();
    let a = cx
        .tensor::<(Dyn<'a'>, Const<3>)>()
        .set_dyn(random_vec_rng(2 * 3, &mut rng), &[2, 3]);
    let b = cx.tensor::<R2<3, 3>>().set(random_vec_rng(3 * 3, &mut rng));
    let c = a.matmul(b).retrieve();
    (cx, vec![c.no_shape()])
}

pub fn batch_matmul() -> (Graph, Vec<GraphTensor<()>>) {
    let mut rng = StdRng::seed_from_u64(0);
    let mut cx = Graph::new();
    let a = cx
        .tensor::<(Dyn<'a'>, Dyn<'b'>, Const<2>)>()
        .set_dyn(random_vec_rng(2 * 3 * 2, &mut rng), &[2, 3, 2]);
    let b = cx.tensor::<R2<2, 4>>().set(random_vec_rng(2 * 4, &mut rng));
    let c = a.matmul(b).retrieve();
    (cx, vec![c.no_shape()])
}

pub fn feedforward() -> (Graph, Vec<GraphTensor<()>>) {
    let mut rng = StdRng::seed_from_u64(0);
    // Test single and batch, unoptimized and optimized
    let mut cx = Graph::new();
    let batch = cx
        .tensor::<(Dyn<'a'>, Const<3>)>()
        .set_dyn(random_vec_rng(2 * 3, &mut rng), &[2, 3]);
    let model: (Linear<3, 4>, ReLU, Linear<4, 2>) = InitModule::initialize(&mut cx);
    model.0.weight.set(random_vec_rng(3 * 4, &mut rng));
    model.2.weight.set(random_vec_rng(4 * 2, &mut rng));
    let batch_out = model.forward(batch).retrieve();

    (cx, vec![batch_out.no_shape()])
}

pub fn transformer() -> (Graph, Vec<GraphTensor<()>>) {
    let mut rng = StdRng::seed_from_u64(0);
    let mut cx = Graph::new();
    let model: Transformer<3, 4, 1, 1, 1, 1> = InitModule::initialize(&mut cx);
    model.decoder.layers[0]
        .self_attention
        .w_k
        .weight
        .set(random_vec_rng(3 * 3, &mut rng));
    model.decoder.layers[0]
        .self_attention
        .w_q
        .weight
        .set(random_vec_rng(3 * 3, &mut rng));
    model.decoder.layers[0]
        .self_attention
        .w_v
        .weight
        .set(random_vec_rng(3 * 3, &mut rng));
    model.decoder.layers[0]
        .self_attention
        .w_o
        .weight
        .set(random_vec_rng(3 * 3, &mut rng));
    model.decoder.layers[0]
        .cross_attention
        .w_k
        .weight
        .set(random_vec_rng(3 * 3, &mut rng));
    model.decoder.layers[0]
        .cross_attention
        .w_q
        .weight
        .set(random_vec_rng(3 * 3, &mut rng));
    model.decoder.layers[0]
        .cross_attention
        .w_v
        .weight
        .set(random_vec_rng(3 * 3, &mut rng));
    model.decoder.layers[0]
        .cross_attention
        .w_o
        .weight
        .set(random_vec_rng(3 * 3, &mut rng));
    model.decoder.layers[0]
        .ff
        .0
        .weight
        .set(random_vec_rng(3 * 4, &mut rng));
    model.decoder.layers[0]
        .ff
        .2
        .weight
        .set(random_vec_rng(3 * 4, &mut rng));
    model.encoder.modules[0]
        .attention
        .w_k
        .weight
        .set(random_vec_rng(3 * 3, &mut rng));
    model.encoder.modules[0]
        .attention
        .w_q
        .weight
        .set(random_vec_rng(3 * 3, &mut rng));
    model.encoder.modules[0]
        .attention
        .w_v
        .weight
        .set(random_vec_rng(3 * 3, &mut rng));
    model.encoder.modules[0]
        .attention
        .w_o
        .weight
        .set(random_vec_rng(3 * 3, &mut rng));
    model.encoder.modules[0]
        .ff
        .0
        .weight
        .set(random_vec_rng(3 * 4, &mut rng));
    model.encoder.modules[0]
        .ff
        .2
        .weight
        .set(random_vec_rng(3 * 4, &mut rng));

    let a = cx.tensor::<(Dyn<'d'>, crate::shape::Const<3>)>();
    let e = cx.tensor::<(Dyn<'e'>, crate::shape::Const<3>)>();
    let b = model.forward((a, e));

    a.set_dyn(random_vec_rng(2 * 3, &mut rng), &[2, 3]);
    e.set_dyn(random_vec_rng(3 * 3, &mut rng), &[3, 3]);
    b.retrieve();

    (cx, vec![b.no_shape()])
}
