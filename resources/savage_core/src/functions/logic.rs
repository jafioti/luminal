// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2021-2022  Philipp Emanuel Weidmann <pew@worldwidemann.com>

use savage_macros::function;

#[function(
    name = "and",
    description = "logical conjunction",
    examples = r#"[
        ("and(true, true)", "true"),
        ("and(true, false)", "false"),
        ("and(false, true)", "false"),
        ("and(false, false)", "false"),
    ]"#,
    categories = r#"[
        "logic",
        "boolean operators",
    ]"#
)]
fn and(a: bool, b: bool) -> bool {
    a && b
}
