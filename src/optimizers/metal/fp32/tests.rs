use dfdx::prelude::{Module as DfdxModule, *};
use half::f16;
use itertools::Itertools;
use rand::Rng;

use super::MetalFp32Optimizer;
use crate::{
    nn::{activation::ReLU, linear::Linear},
    prelude::{Module, *},
};

crate::test_imports!();

#[test]
fn test_contiguous() {
    let mut cx = Graph::new();
    let data = random_vec(12);
    let a = cx.new_tensor::<R2<3, 4>>("Input");
    a.set(data.clone());
    let b = a.permute::<R2<4, 3>, _>().reshape::<R2<12, 1>>();
    b.mark();
    cx.optimize(MetalFp32Optimizer::default());
    cx.execute();

    let d_dev = Cpu::default();
    let d_a = d_dev.tensor_from_vec(data, (DConst::<3>, DConst::<4>));
    let d_b = d_a.permute::<Rank2<4, 3>, _>().reshape::<Rank2<12, 1>>();

    assert_close(&b.data(), &d_b.as_vec());
}
