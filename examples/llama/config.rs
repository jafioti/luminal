use luminal::shape::ConstDim;

pub const VOCAB: usize = 32_000;
pub const HEAD_DIM: usize = 128;
pub const HEAD_DIM_OVER_2: usize = 64;

pub trait LlamaConfig: Clone {
    type Hidden: ConstDim;
    type Intermediate: ConstDim;
    type NumHeads: ConstDim;
    const NUM_LAYERS: usize;
}

// Dev
// pub const HIDDEN: usize = 24;
// pub const INTERMEDIATE: usize = 48;
// pub const HEADS: usize = 1;
// pub const LAYERS: usize = 1;
// pub const VOCAB: usize = 4;
// pub const HEAD_DIM: usize = 24;
// pub const HEAD_DIM_OVER_2: usize = 12;

// 7B
pub const HIDDEN: usize = 4096;
pub const INTERMEDIATE: usize = 11008;
pub const HEADS: usize = 32;
// pub const LAYERS: usize = 32;
pub const LAYERS: usize = 24;

// 13B
// pub const HIDDEN: usize = 5120;
// pub const INTERMEDIATE: usize = 13824;
// pub const HEADS: usize = 40;
// pub const LAYERS: usize = 40;

// 65B
// pub const HIDDEN: usize = 8192;
// pub const INTERMEDIATE: usize = 22016;
// pub const HEADS: usize = 64;
// pub const LAYERS: usize = 80;
