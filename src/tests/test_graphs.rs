use rand::{rngs::StdRng, SeedableRng};

use crate::prelude::*;

use super::random_vec_rng;
#[cfg(test)]
use dfdx::prelude::*;

pub fn matmul() -> (Graph, Vec<GraphTensor>) {
    let mut rng = StdRng::seed_from_u64(0);
    let mut cx = Graph::new();
    let a = cx
        .tensor(('a', 3))
        .set_dyn(random_vec_rng(2 * 3, &mut rng), (2, 3));
    let b = cx.tensor((3, 3)).set(random_vec_rng(3 * 3, &mut rng));
    let c = a.matmul(b).retrieve();
    (cx, vec![c])
}

pub fn batch_matmul() -> (Graph, Vec<GraphTensor>) {
    let mut rng = StdRng::seed_from_u64(0);
    let mut cx = Graph::new();
    let a = cx
        .tensor(('a', 'b', 2))
        .set_dyn(random_vec_rng(2 * 3 * 2, &mut rng), (2, 3, 2));
    let b = cx.tensor((2, 4)).set(random_vec_rng(2 * 4, &mut rng));
    let c = a.matmul(b).retrieve();
    (cx, vec![c])
}

#[test]
fn execute_no_delete_keeps_tensors() {
    let mut cx = Graph::new();
    let a = cx.tensor(3).set([1., 2., 3.]);
    let b = cx.tensor(3).set([4., 5., 6.]);
    let c = (a + b).retrieve();

    // first run without deleting tensors
    cx.execute_no_delete();

    // ensure all tensors remain in the map
    assert!(cx.tensors.contains_key(&(a.id, 0)));
    assert!(cx.tensors.contains_key(&(b.id, 0)));
    assert!(cx.tensors.contains_key(&(c.id, 0)));

    // normal execute should still produce the correct result
    cx.execute();

    let d_dev = dfdx::tensor::Cpu::default();
    let d_a = d_dev.tensor([1., 2., 3.]);
    let d_b = d_dev.tensor([4., 5., 6.]);
    let d_c = d_a + d_b;

    super::assert_close(&c.data(), &d_c.as_vec());
}
