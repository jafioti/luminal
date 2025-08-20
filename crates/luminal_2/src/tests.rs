use luminal::{
    prelude::{
        NodeIndex,
        petgraph::{Directed, prelude::StableGraph},
    },
    shape::{Expression, Term, expression_cleanup},
};
use rustc_hash::FxHashMap;

use crate::{GPUArch, codegen::codegen, run::run_graph};
use std::collections::HashMap;

use crate::{GraphTerm, utils::*};

// #[test]
// fn test_naive_matmul() {

// }
