// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2021-2022  Philipp Emanuel Weidmann <pew@worldwidemann.com>

use crate::{expression::Expression, helpers::*};

impl Expression {
    /// Applies standard algebraic simplification rules to the expression,
    /// and returns the result.
    ///
    /// Note that this function does not itself recurse into sub-expressions;
    /// but since it is called from `evaluate_step`, which *does* recurse,
    /// simplifications are applied to the entire expression tree during evaluation.
    pub(crate) fn simplify(&self) -> Self {
        use crate::expression::Expression::*;

        match self {
            Negation(a) => {
                if let Negation(a) = &**a {
                    *a.clone()
                } else {
                    self.clone()
                }
            }
            Not(a) => {
                if let Not(a) = &**a {
                    *a.clone()
                } else {
                    self.clone()
                }
            }
            Sum(a, b) => {
                let a = *a.clone();
                let b = *b.clone();

                if a == int(0) {
                    b
                } else if b == int(0) {
                    a
                } else if a == b {
                    int(2) * a
                } else if a == -b.clone() || b == -a {
                    int(0)
                } else {
                    self.clone()
                }
            }
            Difference(a, b) => {
                let a = *a.clone();
                let b = *b.clone();

                if a == int(0) {
                    -b
                } else if b == int(0) {
                    a
                } else if a == b {
                    int(0)
                } else if a == -b.clone() || b == -a.clone() {
                    int(2) * a
                } else {
                    self.clone()
                }
            }
            Product(a, b) => {
                let a = *a.clone();
                let b = *b.clone();

                if a == int(1) {
                    b
                } else if b == int(1) {
                    a
                } else if a == int(0) || b == int(0) {
                    int(0)
                } else if a == b {
                    pow(a, int(2))
                } else if a == int(1) / b.clone() || b == int(1) / a {
                    int(1)
                } else {
                    self.clone()
                }
            }
            Quotient(a, b) => {
                let a = *a.clone();
                let b = *b.clone();

                if b == int(1) {
                    a
                } else if a == int(0) {
                    // FIXME: This is incorrect if `b` evaluates to zero!
                    int(0)
                } else if a == b {
                    // FIXME: This is incorrect if `b` evaluates to zero!
                    int(1)
                } else {
                    self.clone()
                }
            }
            Remainder(a, b) => {
                let a = *a.clone();
                let b = *b.clone();

                if a == int(0) || a == b {
                    // FIXME: This is incorrect if `b` evaluates to zero!
                    int(0)
                } else {
                    self.clone()
                }
            }
            Power(a, b) => {
                let a = *a.clone();
                let b = *b.clone();

                if a == int(1) {
                    int(1)
                } else if b == int(1) {
                    a
                } else if a == int(0) {
                    // FIXME: This is incorrect if `b` evaluates to zero!
                    int(0)
                } else if b == int(0) {
                    // FIXME: This is incorrect if `a` evaluates to zero!
                    int(1)
                } else {
                    self.clone()
                }
            }
            Equal(a, b) | LessThanOrEqual(a, b) | GreaterThanOrEqual(a, b) => {
                if a == b {
                    Boolean(true)
                } else {
                    self.clone()
                }
            }
            NotEqual(a, b) | LessThan(a, b) | GreaterThan(a, b) => {
                if a == b {
                    Boolean(false)
                } else {
                    self.clone()
                }
            }
            And(a, b) => {
                let a = *a.clone();
                let b = *b.clone();

                if a == Boolean(true) {
                    b
                } else if b == Boolean(true) {
                    a
                } else if a == Boolean(false) || b == Boolean(false) {
                    Boolean(false)
                } else if a == b {
                    a
                } else if a == !b.clone() || b == !a {
                    Boolean(false)
                } else {
                    self.clone()
                }
            }
            Or(a, b) => {
                let a = *a.clone();
                let b = *b.clone();

                if a == Boolean(false) {
                    b
                } else if b == Boolean(false) {
                    a
                } else if a == Boolean(true) || b == Boolean(true) {
                    Boolean(true)
                } else if a == b {
                    a
                } else if a == !b.clone() || b == !a {
                    Boolean(true)
                } else {
                    self.clone()
                }
            }
            _ => self.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::expression::Expression;

    #[track_caller]
    fn t(expression: &str, result: &str) {
        assert_eq!(
            expression
                .parse::<Expression>()
                .unwrap()
                .simplify()
                .to_string(),
            result,
        );
    }

    #[test]
    fn arithmetic() {
        t("-(-a)", "a");

        t("0 + a", "a");
        t("a + 0", "a");
        t("a + a", "2 * a");
        t("(-a) + a", "0");
        t("a + (-a)", "0");

        t("0 - a", "-a");
        t("a - 0", "a");
        t("a - a", "0");
        t("(-a) - a", "2 * -a");
        t("a - (-a)", "2 * a");

        t("1 * a", "a");
        t("a * 1", "a");
        t("0 * a", "0");
        t("a * 0", "0");
        t("a * a", "a ^ 2");
        t("(1 / a) * a", "1");
        t("a * (1 / a)", "1");

        t("a / 1", "a");
        t("0 / a", "0");
        t("a / a", "1");

        t("0 % a", "0");
        t("a % a", "0");

        t("1 ^ a", "1");
        t("a ^ 1", "a");
        t("0 ^ a", "0");
        t("a ^ 0", "1");
    }

    #[test]
    fn logic() {
        t("!(!a)", "a");

        t("true && a", "a");
        t("a && true", "a");
        t("false && a", "false");
        t("a && false", "false");
        t("a && a", "a");
        t("(!a) && a", "false");
        t("a && (!a)", "false");

        t("false || a", "a");
        t("a || false", "a");
        t("true || a", "true");
        t("a || true", "true");
        t("a || a", "a");
        t("(!a) || a", "true");
        t("a || (!a)", "true");
    }

    #[test]
    fn comparisons() {
        t("a == a", "true");
        t("a != a", "false");
        t("a < a", "false");
        t("a <= a", "true");
        t("a > a", "false");
        t("a >= a", "true");
    }
}
