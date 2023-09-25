// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2021-2022  Philipp Emanuel Weidmann <pew@worldwidemann.com>

use std::cmp::max;
use std::fmt::{Display, Formatter, Result};

use num::{One, Signed, Zero};

use crate::expression::{Expression, Integer, Rational};

/// Returns a pair of integers `(n, m)` such that `x = n / 10^m`,
/// or `None` if no such integers exist.
fn decimal_representation(x: &Rational) -> Option<(Integer, usize)> {
    // https://en.wikipedia.org/wiki/Decimal_representation#Finite_decimal_representations
    let mut denominator = x.denom().clone();

    let [power_of_2, power_of_5] = [2, 5].map(|n| {
        let mut power = 0;

        while (denominator.clone() % Integer::from(n)).is_zero() {
            denominator /= n;
            power += 1;
        }

        power
    });

    if denominator.is_one() {
        let multiplier = if power_of_2 < power_of_5 {
            Integer::from(2).pow(power_of_5 - power_of_2)
        } else {
            Integer::from(5).pow(power_of_2 - power_of_5)
        };

        Some((x.numer() * multiplier, max(power_of_2, power_of_5) as usize))
    } else {
        None
    }
}

impl Expression {
    /// Formats the expression as a unary prefix operator with the minimally necessary parentheses.
    fn fmt_prefix(&self, f: &mut Formatter<'_>, symbol: &str, a: &Self) -> Result {
        let a_needs_parentheses = a.precedence() <= self.precedence();

        write!(
            f,
            "{}{}{}{}",
            symbol,
            if a_needs_parentheses { "(" } else { "" },
            a,
            if a_needs_parentheses { ")" } else { "" },
        )
    }

    /// Formats the expression as a binary infix operator with the minimally necessary parentheses.
    fn fmt_infix(&self, f: &mut Formatter<'_>, symbol: &str, a: &Self, b: &Self) -> Result {
        use crate::expression::Associativity::*;

        let a_needs_parentheses = (a.precedence() < self.precedence())
            || ((a.precedence() == self.precedence())
                && (self.associativity() == RightAssociative));

        let b_needs_parentheses = (b.precedence() < self.precedence())
            || ((b.precedence() == self.precedence()) && (self.associativity() == LeftAssociative));

        write!(
            f,
            "{}{}{} {} {}{}{}",
            if a_needs_parentheses { "(" } else { "" },
            a,
            if a_needs_parentheses { ")" } else { "" },
            symbol,
            if b_needs_parentheses { "(" } else { "" },
            b,
            if b_needs_parentheses { ")" } else { "" },
        )
    }
}

impl Display for Expression {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        use crate::expression::{Expression::*, RationalRepresentation::*};

        match self {
            Variable(identifier) => write!(f, "{}", identifier),
            Function(identifier, _) => write!(f, "{}", identifier),
            FunctionValue(function, arguments) => {
                let function_needs_parentheses = function.precedence() < isize::MAX;

                write!(
                    f,
                    "{}{}{}({})",
                    if function_needs_parentheses { "(" } else { "" },
                    function,
                    if function_needs_parentheses { ")" } else { "" },
                    arguments
                        .iter()
                        .map(|a| a.to_string())
                        .collect::<Vec<_>>()
                        .join(", "),
                )
            }
            Integer(n) => write!(f, "{}", n),
            Rational(x, representation) => {
                match representation {
                    Fraction => write!(f, "{}", x),
                    Decimal => {
                        if let Some((mantissa, separator_position)) = decimal_representation(x) {
                            let mut string = mantissa.abs().to_string();

                            if separator_position > 0 {
                                if separator_position > string.len() - 1 {
                                    // Left-pad the string with enough zeros to be able
                                    // to insert the decimal separator at the indicated position.
                                    string = format!(
                                        "{}{}",
                                        "0".repeat(separator_position - (string.len() - 1)),
                                        string,
                                    );
                                }

                                string.insert(string.len() - separator_position, '.');
                            }

                            write!(f, "{}{}", if x.is_negative() { "-" } else { "" }, string)
                        } else {
                            // Fall back to fraction representation.
                            write!(f, "{}", x)
                        }
                    }
                }
            }
            Complex(z, representation) => {
                if z.im.is_zero() {
                    write!(f, "{}", Rational(z.re.clone(), *representation))
                } else if z.re.is_zero() {
                    if z.im.abs().is_one() {
                        write!(f, "{}i", if z.im.is_negative() { "-" } else { "" })
                    } else {
                        write!(f, "{}*i", Rational(z.im.clone(), *representation))
                    }
                } else if z.re.is_negative() && z.im.is_positive() {
                    if z.im.is_one() {
                        write!(f, "i - {}", Rational(z.re.abs(), *representation))
                    } else {
                        write!(
                            f,
                            "{}*i - {}",
                            Rational(z.im.clone(), *representation),
                            Rational(z.re.abs(), *representation),
                        )
                    }
                } else if z.im.abs().is_one() {
                    write!(
                        f,
                        "{} {} i",
                        Rational(z.re.clone(), *representation),
                        if z.im.is_negative() { "-" } else { "+" },
                    )
                } else {
                    write!(
                        f,
                        "{} {} {}*i",
                        Rational(z.re.clone(), *representation),
                        if z.im.is_negative() { "-" } else { "+" },
                        Rational(z.im.abs(), *representation),
                    )
                }
            }
            Vector(v) => write!(
                f,
                "[{}]",
                v.iter()
                    .map(|element| element.to_string())
                    .collect::<Vec<_>>()
                    .join(", "),
            ),
            VectorElement(vector, i) => {
                let vector_needs_parentheses = vector.precedence() < isize::MAX;

                write!(
                    f,
                    "{}{}{}[{}]",
                    if vector_needs_parentheses { "(" } else { "" },
                    vector,
                    if vector_needs_parentheses { ")" } else { "" },
                    i,
                )
            }
            Matrix(m) => write!(
                f,
                "[{}]",
                m.row_iter()
                    .map(|row| format!(
                        "[{}]",
                        row.iter()
                            .map(|element| element.to_string())
                            .collect::<Vec<_>>()
                            .join(", "),
                    ))
                    .collect::<Vec<_>>()
                    .join(", "),
            ),
            MatrixElement(matrix, i, j) => {
                let matrix_needs_parentheses = matrix.precedence() < isize::MAX;

                write!(
                    f,
                    "{}{}{}[{}, {}]",
                    if matrix_needs_parentheses { "(" } else { "" },
                    matrix,
                    if matrix_needs_parentheses { ")" } else { "" },
                    i,
                    j,
                )
            }
            Boolean(boolean) => write!(f, "{}", boolean),
            Negation(a) => self.fmt_prefix(f, "-", a),
            Not(a) => self.fmt_prefix(f, "!", a),
            Sum(a, b) => self.fmt_infix(f, "+", a, b),
            Min(a, b) => write!(f, "min({a}, {b})"),
            Difference(a, b) => self.fmt_infix(f, "-", a, b),
            Product(a, b) => self.fmt_infix(f, "*", a, b),
            Quotient(a, b) => self.fmt_infix(f, "/", a, b),
            Remainder(a, b) => self.fmt_infix(f, "%", a, b),
            Power(a, b) => self.fmt_infix(f, "^", a, b),
            Equal(a, b) => self.fmt_infix(f, "==", a, b),
            NotEqual(a, b) => self.fmt_infix(f, "!=", a, b),
            LessThan(a, b) => self.fmt_infix(f, "<", a, b),
            LessThanOrEqual(a, b) => self.fmt_infix(f, "<=", a, b),
            GreaterThan(a, b) => self.fmt_infix(f, ">", a, b),
            GreaterThanOrEqual(a, b) => self.fmt_infix(f, ">=", a, b),
            And(a, b) => self.fmt_infix(f, "&&", a, b),
            Or(a, b) => self.fmt_infix(f, "||", a, b),
        }
    }
}

#[cfg(test)]
mod tests {
    use nalgebra::{dmatrix, dvector};

    use crate::expression::{Expression, Expression::*};
    use crate::helpers::*;

    #[track_caller]
    fn t(expression: Expression, string: &str) {
        assert_eq!(expression.to_string(), string);
    }

    #[test]
    fn variables() {
        t(var("a"), "a");
        t(var("A"), "A");
        t(var("Named_Variable"), "Named_Variable");
    }

    #[test]
    fn functions() {
        t(fun(var("f"), []), "f()");
        t(fun(var("f"), [var("a")]), "f(a)");
        t(fun(var("f"), [var("a"), int(1)]), "f(a, 1)");
        t(
            fun(
                var("f"),
                [fun(var("g"), [var("a")]), fun(var("h"), [var("b")])],
            ),
            "f(g(a), h(b))",
        );
        t(fun(fun(var("f"), [var("a")]), [var("b")]), "(f(a))(b)");
        t(fun(var("f") + var("g"), [var("a")]), "(f + g)(a)");
    }

    #[test]
    fn integers() {
        t(int(0), "0");
        t(int(1), "1");
        t(int(-1), "-1");
        t(int(1234567890), "1234567890");
        t(int(-1234567890), "-1234567890");
        t(int(9876543210u64), "9876543210");
        t(int(-9876543210i64), "-9876543210");
    }

    #[test]
    fn rational_numbers() {
        t(rat(0, 1), "0");
        t(ratd(0, 1), "0");
        t(rat(0, -1), "0");
        t(ratd(0, -1), "0");
        t(rat(1, 1), "1");
        t(ratd(1, 1), "1");
        t(rat(-1, 1), "-1");
        t(ratd(-1, 1), "-1");
        t(rat(1, 2), "1/2");
        t(ratd(1, 2), "0.5");
        t(rat(3, 2), "3/2");
        t(ratd(3, 2), "1.5");
        t(rat(1, 3), "1/3");
        t(ratd(1, 3), "1/3");
        t(rat(123, 40), "123/40");
        t(ratd(123, 40), "3.075");
        t(rat(123, -40), "-123/40");
        t(ratd(123, -40), "-3.075");
        t(rat(-123, -40), "123/40");
        t(ratd(-123, -40), "3.075");
    }

    #[test]
    fn complex_numbers() {
        t(com(0, 1, 0, 1), "0");
        t(comd(0, 1, 0, 1), "0");
        t(com(1, 1, 0, 1), "1");
        t(comd(1, 1, 0, 1), "1");
        t(com(0, 1, 1, 1), "i");
        t(comd(0, 1, 1, 1), "i");
        t(com(-1, 1, 0, 1), "-1");
        t(comd(-1, 1, 0, 1), "-1");
        t(com(0, 1, -1, 1), "-i");
        t(comd(0, 1, -1, 1), "-i");
        t(com(1, 1, 1, 1), "1 + i");
        t(comd(1, 1, 1, 1), "1 + i");
        t(com(1, 1, -1, 1), "1 - i");
        t(comd(1, 1, -1, 1), "1 - i");
        t(com(-1, 1, 1, 1), "i - 1");
        t(comd(-1, 1, 1, 1), "i - 1");
        t(com(-1, 1, -1, 1), "-1 - i");
        t(comd(-1, 1, -1, 1), "-1 - i");
        t(com(123, -40, 1, 3), "1/3*i - 123/40");
        t(comd(123, -40, 1, 3), "1/3*i - 3.075");
        t(com(1, 3, 123, 40), "1/3 + 123/40*i");
        t(comd(1, 3, 123, 40), "1/3 + 3.075*i");
    }

    #[test]
    fn vectors() {
        t(Vector(dvector![]), "[]");
        t(Vector(dvector![int(1)]), "[1]");
        t(Vector(dvector![int(1), int(2), int(3)]), "[1, 2, 3]");
        t(
            Vector(dvector![
                int(1),
                fun(var("f"), [var("a"), int(1)]),
                Vector(dvector![int(1), int(2), int(3)])
            ]),
            "[1, f(a, 1), [1, 2, 3]]",
        );
    }

    #[test]
    fn matrices() {
        t(Matrix(dmatrix![]), "[]");
        t(Matrix(dmatrix![int(1)]), "[[1]]");
        t(Matrix(dmatrix![int(1), int(2), int(3)]), "[[1, 2, 3]]");
        t(
            Matrix(dmatrix![
                int(1), int(2), int(3);
                int(4), int(5), int(6)
            ]),
            "[[1, 2, 3], [4, 5, 6]]",
        );
        t(
            Matrix(dmatrix![
                fun(var("f"), []), int(2), Vector(dvector![int(1)]);
                Matrix(dmatrix![int(1)]), comd(123, -40, 1, 3), int(6)
            ]),
            "[[f(), 2, [1]], [[[1]], 1/3*i - 3.075, 6]]",
        );
    }

    #[test]
    fn booleans() {
        t(Boolean(true), "true");
        t(Boolean(false), "false");
    }

    #[test]
    fn operators() {
        t(-int(1), "-1");
        t(-(-int(1)), "-(-1)");
        t(!var("A"), "!A");
        t(!(!var("A")), "!(!A)");

        t((int(1) + int(2)) + int(3), "1 + 2 + 3");
        t((int(1) + int(2)) - int(3), "1 + 2 - 3");
        t((int(1) - int(2)) + int(3), "1 - 2 + 3");
        t((int(1) - int(2)) - int(3), "1 - 2 - 3");
        t(int(1) + (int(2) + int(3)), "1 + 2 + 3");
        t(int(1) + (int(2) - int(3)), "1 + 2 - 3");
        t(int(1) - (int(2) + int(3)), "1 - (2 + 3)");
        t(int(1) - (int(2) - int(3)), "1 - (2 - 3)");

        t((int(1) * int(2)) * int(3), "1 * 2 * 3");
        t((int(1) * int(2)) / int(3), "1 * 2 / 3");
        t((int(1) / int(2)) * int(3), "1 / 2 * 3");
        t((int(1) / int(2)) / int(3), "1 / 2 / 3");
        t(int(1) * (int(2) * int(3)), "1 * 2 * 3");
        t(int(1) * (int(2) / int(3)), "1 * 2 / 3");
        t(int(1) / (int(2) * int(3)), "1 / (2 * 3)");
        t(int(1) / (int(2) / int(3)), "1 / (2 / 3)");

        t((int(1) + int(2)) / int(3), "(1 + 2) / 3");
        t((int(1) / int(2)) + int(3), "1 / 2 + 3");
        t(int(1) + (int(2) / int(3)), "1 + 2 / 3");
        t(int(1) / (int(2) + int(3)), "1 / (2 + 3)");

        t(pow(int(1) * int(2), int(3)), "(1 * 2) ^ 3");
        t(pow(int(1), int(2)) * int(3), "1 ^ 2 * 3");
        t(int(1) * pow(int(2), int(3)), "1 * 2 ^ 3");
        t(pow(int(1), int(2) * int(3)), "1 ^ (2 * 3)");

        t(pow(pow(int(1), int(2)), int(3)), "(1 ^ 2) ^ 3");
        t(pow(int(1), pow(int(2), int(3))), "1 ^ 2 ^ 3");

        t(pow(int(1), int(2)), "1 ^ 2");
        t(pow(int(-1), int(2)), "(-1) ^ 2");
        t(pow(rat(1, 2), int(3)), "(1/2) ^ 3");
        t(pow(ratd(1, 2), int(3)), "0.5 ^ 3");
        t(pow(com(0, 1, -1, 1), int(2)), "(-i) ^ 2");
        t(com(1, 1, 1, 1) * int(2), "(1 + i) * 2");
        t(com(1, 1, -1, 1) - int(2), "1 - i - 2");
        t(int(2) - com(1, 1, -1, 1), "2 - (1 - i)");

        // TODO: Comparison operators!

        t(and(and(var("A"), var("B")), var("C")), "A && B && C");
        t(or(and(var("A"), var("B")), var("C")), "A && B || C");
        t(and(or(var("A"), var("B")), var("C")), "(A || B) && C");
        t(or(or(var("A"), var("B")), var("C")), "A || B || C");
        t(and(var("A"), and(var("B"), var("C"))), "A && B && C");
        t(and(var("A"), or(var("B"), var("C"))), "A && (B || C)");
        t(or(var("A"), and(var("B"), var("C"))), "A || B && C");
        t(or(var("A"), or(var("B"), var("C"))), "A || B || C");
    }
}
