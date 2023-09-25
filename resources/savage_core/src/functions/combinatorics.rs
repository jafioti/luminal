// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2021-2022  Philipp Emanuel Weidmann <pew@worldwidemann.com>

use num::range_inclusive;
use savage_macros::function;

use crate::{expression::Integer, functions::NonNegativeInteger};

#[function(
    name = "factorial",
    description = "factorial of a non-negative integer",
    examples = r#"[
        ("factorial(0)", "1"),
        ("factorial(1)", "1"),
        ("factorial(4)", "24"),
        ("factorial(10)", "3628800"),
    ]"#,
    categories = r#"[
        "combinatorics",
    ]"#
)]
fn factorial(n: NonNegativeInteger) -> Integer {
    range_inclusive::<Integer>(1.into(), n).product()
}
