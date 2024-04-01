use rand::{rngs::StdRng, SeedableRng};

use crate::prelude::*;

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
