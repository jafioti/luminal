// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2021-2022  Philipp Emanuel Weidmann <pew@worldwidemann.com>

use num::ToPrimitive;
use primal::StreamingSieve;
use savage_macros::function;

use crate::{
    expression::Expression,
    functions::{function_expression, NonNegativeInteger, PositiveInteger},
    helpers::*,
};

#[function(
    name = "is_prime",
    description = "whether the given non-negative integer is a prime number",
    examples = r#"[
        ("is_prime(0)", "false"),
        ("is_prime(1)", "false"),
        ("is_prime(2)", "true"),
        ("is_prime(29)", "true"),
        ("is_prime(2^31)", "false"),
        ("is_prime(2^31 - 1)", "true"),
    ]"#,
    categories = r#"[
        "number theory",
        "prime numbers",
    ]"#
)]
fn is_prime(n: NonNegativeInteger) -> Expression {
    if let Some(n) = n.to_u64() {
        Expression::Boolean(primal::is_prime(n))
    } else {
        fun(function_expression("is_prime").unwrap(), [int(n)])
    }
}

#[function(
    name = "nth_prime",
    description = "`n`th prime number, 1-indexed",
    examples = r#"[
        ("nth_prime(1)", "2"),
        ("nth_prime(10)", "29"),
        ("nth_prime(100)", "541"),
        ("nth_prime(1000)", "7919"),
    ]"#,
    categories = r#"[
        "number theory",
        "prime numbers",
    ]"#
)]
fn nth_prime(n: PositiveInteger) -> Expression {
    if let Some(n) = n.to_usize() {
        int(StreamingSieve::nth_prime(n))
    } else {
        fun(function_expression("nth_prime").unwrap(), [int(n)])
    }
}

#[function(
    name = "prime_pi",
    description = "number of prime numbers less than or equal to the given non-negative integer",
    examples = r#"[
        ("prime_pi(10)", "4"),
        ("prime_pi(100)", "25"),
        ("prime_pi(1000)", "168"),
        ("prime_pi(10000)", "1229"),
        ("prime_pi(100000)", "9592"),
    ]"#,
    categories = r#"[
        "number theory",
        "prime numbers",
    ]"#
)]
fn prime_pi(n: NonNegativeInteger) -> Expression {
    if let Some(n) = n.to_usize() {
        int(StreamingSieve::prime_pi(n))
    } else {
        fun(function_expression("prime_pi").unwrap(), [int(n)])
    }
}
