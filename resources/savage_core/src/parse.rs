// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2021-2022  Philipp Emanuel Weidmann <pew@worldwidemann.com>

use std::{ops::Range, str::FromStr};

use chumsky::prelude::*;

use crate::{
    expression::{Expression, Integer, Matrix, Vector},
    helpers::*,
};

/// Error that occurred while trying to parse a character stream into an expression.
pub type Error = chumsky::error::Simple<char, Range<usize>>;

/// Reason why a parse error occurred.
pub type ErrorReason = chumsky::error::SimpleReason<char, Range<usize>>;

/// Returns a parser that produces expressions from character streams.
///
/// The purpose of this function is to be a building block for parsers that parse
/// expressions as parts of a more complex input language. If you simply want
/// to turn strings into expressions, use `"a + b".parse::<Expression>()`.
#[allow(clippy::let_and_return)]
pub fn parser() -> impl Parser<char, Expression, Error = Error> {
    recursive(|expression| {
        let identifier = text::ident()
            .map(|identifier: String| match identifier.as_str() {
                "true" => Expression::Boolean(true),
                "false" => Expression::Boolean(false),
                _ => var(identifier),
            })
            .labelled("identifier")
            .boxed();

        let number = text::int(10)
            .chain(just('.').ignore_then(text::digits(10)).or_not())
            .map(|parts: Vec<String>| match parts.as_slice() {
                [integer] => int(integer.parse::<Integer>().unwrap()),
                [integer_part, fractional_part] => {
                    let numerator = format!("{}{}", integer_part, fractional_part);
                    let denominator = format!("1{}", "0".repeat(fractional_part.len()));
                    ratd(
                        numerator.parse::<Integer>().unwrap(),
                        denominator.parse::<Integer>().unwrap(),
                    )
                }
                _ => unreachable!(),
            })
            .labelled("number")
            .boxed();

        let vector_or_matrix = expression
            .clone()
            .separated_by(just(','))
            .padded()
            .delimited_by(just('['), just(']'))
            .map(|elements| {
                if let Some(Expression::Vector(v)) = elements.first() {
                    // If all elements of the vector are themselves vectors, and have the same size,
                    // they are interpreted as the rows of a rectangular matrix.
                    let row_size = v.len();
                    let mut rows = Vec::new();

                    for element in &elements {
                        match element {
                            Expression::Vector(v) if v.len() == row_size => {
                                rows.push(v.transpose());
                            }
                            _ => return Expression::Vector(Vector::from_vec(elements)),
                        }
                    }

                    Expression::Matrix(Matrix::from_rows(&rows))
                } else {
                    Expression::Vector(Vector::from_vec(elements))
                }
            })
            .labelled("vector_or_matrix")
            .boxed();

        let atomic_expression = identifier
            .or(number)
            .or(vector_or_matrix)
            .or(expression.clone().delimited_by(just('('), just(')')))
            .padded()
            .boxed();

        let function_or_element = atomic_expression
            .then(
                expression
                    .clone()
                    .separated_by(just(','))
                    .padded()
                    .delimited_by(just('('), just(')'))
                    .map(|arguments| (Some(arguments), None))
                    .or(expression
                        .clone()
                        .separated_by(just(','))
                        .at_least(1)
                        .at_most(2)
                        .delimited_by(just('['), just(']'))
                        .map(|indices| (None, Some(indices))))
                    .or_not(),
            )
            .map(
                |(expression, arguments_or_indices)| match arguments_or_indices {
                    Some((Some(arguments), None)) => fun(expression, arguments),
                    Some((None, Some(indices))) => {
                        if indices.len() == 1 {
                            Expression::VectorElement(
                                Box::new(expression),
                                Box::new(indices[0].clone()),
                            )
                        } else {
                            Expression::MatrixElement(
                                Box::new(expression),
                                Box::new(indices[0].clone()),
                                Box::new(indices[1].clone()),
                            )
                        }
                    }
                    None => expression,
                    _ => unreachable!(),
                },
            )
            .padded()
            .boxed();

        let power = function_or_element
            .separated_by(just('^'))
            .at_least(1)
            .map(|expressions| {
                expressions
                    .into_iter()
                    .rev()
                    .reduce(|a, b| pow(b, a))
                    .unwrap()
            })
            .labelled("power")
            .boxed();

        let negation = just('-')
            .ignore_then(power.clone())
            .map(|a| -a)
            .or(just('!').ignore_then(power.clone()).map(|a| !a))
            .labelled("negation")
            .or(power)
            .padded()
            .boxed();

        let product_or_quotient_or_remainder = negation
            .clone()
            .then(
                just('*')
                    .or(just('/'))
                    .or(just('%'))
                    .then(negation)
                    .repeated(),
            )
            .foldl(|a, (operator, b)| match operator {
                '*' => a * b,
                '/' => a / b,
                '%' => a % b,
                _ => unreachable!(),
            })
            .labelled("product_or_quotient_or_remainder")
            .boxed();

        let sum_or_difference = product_or_quotient_or_remainder
            .clone()
            .then(
                just('+')
                    .or(just('-'))
                    .then(product_or_quotient_or_remainder)
                    .repeated(),
            )
            .foldl(|a, (operator, b)| match operator {
                '+' => a + b,
                '-' => a - b,
                _ => unreachable!(),
            })
            .labelled("sum_or_difference")
            .boxed();

        let comparison = sum_or_difference
            .clone()
            .then(
                just('=')
                    .chain(just('='))
                    .or(just('!').chain(just('=')))
                    .or(just('<').chain(just('=')))
                    .or(just('<').to(vec!['<']))
                    .or(just('>').chain(just('=')))
                    .or(just('>').to(vec!['>']))
                    .collect::<String>()
                    .then(sum_or_difference)
                    .repeated(),
            )
            .foldl(|a, (operator, b)| match operator.as_str() {
                "==" => eq(a, b),
                "!=" => ne(a, b),
                "<" => lt(a, b),
                "<=" => le(a, b),
                ">" => gt(a, b),
                ">=" => ge(a, b),
                _ => unreachable!(),
            })
            .labelled("comparison")
            .boxed();

        let conjunction = comparison
            .clone()
            .then(
                just('&')
                    .ignore_then(just('&'))
                    .ignore_then(comparison)
                    .repeated(),
            )
            .foldl(and)
            .labelled("conjunction")
            .boxed();

        let disjunction = conjunction
            .clone()
            .then(
                just('|')
                    .ignore_then(just('|'))
                    .ignore_then(conjunction)
                    .repeated(),
            )
            .foldl(or)
            .labelled("disjunction")
            .boxed();

        disjunction
    })
}

impl FromStr for Expression {
    type Err = Vec<Error>;

    fn from_str(string: &str) -> Result<Self, Self::Err> {
        parser().then_ignore(end()).parse(string)
    }
}

#[cfg(test)]
mod tests {
    use nalgebra::{dmatrix, dvector};

    use crate::expression::{Expression, Expression::*};
    use crate::helpers::*;

    #[track_caller]
    fn t(string: &str, expression: Expression) {
        assert_eq!(string.parse(), Ok(expression));
    }

    #[test]
    fn variables() {
        t("a   ", var("a"));
        t("     A", var("A"));
        t("  Named_Variable ", var("Named_Variable"));
    }

    #[test]
    fn functions() {
        t(" f(   )   ", fun(var("f"), []));
        t("f ( a) ", fun(var("f"), [var("a")]));
        t("f( a, 1  )", fun(var("f"), [var("a"), int(1)]));
        t(
            " f(  g(a ),h(   b )  ) ",
            fun(
                var("f"),
                [fun(var("g"), [var("a")]), fun(var("h"), [var("b")])],
            ),
        );
        t(
            " ( f ( a ) ) ( b ) ",
            fun(fun(var("f"), [var("a")]), [var("b")]),
        );
        t("(f +g)( a)", fun(var("f") + var("g"), [var("a")]));
    }

    #[test]
    fn integers() {
        t("0", int(0));
        t("  1", int(1));
        t("1234567890  ", int(1234567890));
        t("  9876543210  ", int(9876543210u64));
    }

    #[test]
    fn rational_numbers() {
        t("0.5", ratd(1, 2));
        t(" 1.5", ratd(3, 2));
        t("3.075 ", ratd(123, 40));
        t(" 100.000 ", ratd(100, 1));
    }

    #[test]
    fn vectors() {
        t(" [ ] ", Vector(dvector![]));
        t("[   1]", Vector(dvector![int(1)]));
        t(" [1,2,   3 ]", Vector(dvector![int(1), int(2), int(3)]));
        t(
            "[  1,f(a,1),[   1,2,3  ] ]  ",
            Vector(dvector![
                int(1),
                fun(var("f"), [var("a"), int(1)]),
                Vector(dvector![int(1), int(2), int(3)])
            ]),
        );
    }

    #[test]
    fn matrices() {
        t("[[1   ]   ]   ", Matrix(dmatrix![int(1)]));
        t("[ [ 1,2,3 ] ]", Matrix(dmatrix![int(1), int(2), int(3)]));
        t(
            " [[  1, 2,3 ],[ 4,5, 6]]",
            Matrix(dmatrix![
                int(1), int(2), int(3);
                int(4), int(5), int(6)
            ]),
        );
        t(
            "  [[f ( ) ,2, [1 ] ], [[ [ 1]],3.075,6] ] ",
            Matrix(dmatrix![
                fun(var("f"), []), int(2), Vector(dvector![int(1)]);
                Matrix(dmatrix![int(1)]), ratd(123, 40), int(6)
            ]),
        );
    }

    #[test]
    fn booleans() {
        t("   true", Boolean(true));
        t("false   ", Boolean(false));
    }

    #[test]
    fn operators() {
        t("  - 1 ", -int(1));
        t(" - (-    1 )", -(-int(1)));
        t("!A  ", !var("A"));
        t(" !( ! A)", !(!var("A")));

        t("1+2    +3   ", (int(1) + int(2)) + int(3));
        t(" 1 +2- 3 ", (int(1) + int(2)) - int(3));
        t("1  -2 +    3", (int(1) - int(2)) + int(3));
        t("1-2-3", (int(1) - int(2)) - int(3));
        t(" 1-(2 + 3)", int(1) - (int(2) + int(3)));
        t("1 - (2 - 3)", int(1) - (int(2) - int(3)));

        t(" 1 *2* 3 ", (int(1) * int(2)) * int(3));
        t("1*2 / 3", (int(1) * int(2)) / int(3));
        t("1 / 2*3  ", (int(1) / int(2)) * int(3));
        t("  1/2/3", (int(1) / int(2)) / int(3));
        t("1 /(  2 *3) ", int(1) / (int(2) * int(3)));
        t(" 1/ (2 /3)", int(1) / (int(2) / int(3)));

        t("(1+2)   /3 ", (int(1) + int(2)) / int(3));
        t("  1/ 2 +3", (int(1) / int(2)) + int(3));
        t("1 + 2 / 3", int(1) + (int(2) / int(3)));
        t("1/ (2+3)     ", int(1) / (int(2) + int(3)));

        t("(1 * 2)^3", pow(int(1) * int(2), int(3)));
        t("1^2 * 3 ", pow(int(1), int(2)) * int(3));
        t("1*  2  ^3", int(1) * pow(int(2), int(3)));
        t(" 1^( 2 * 3 )", pow(int(1), int(2) * int(3)));

        t(" (1^2)  ^  3", pow(pow(int(1), int(2)), int(3)));
        t("1 ^2 ^3 ", pow(int(1), pow(int(2), int(3))));

        // TODO: Comparison operators!

        t("A&&B&&C", and(and(var("A"), var("B")), var("C")));
        t("A  &&  B||C", or(and(var("A"), var("B")), var("C")));
        t(" ( A || B ) && C ", and(or(var("A"), var("B")), var("C")));
        t("A||B ||C", or(or(var("A"), var("B")), var("C")));
        t("A&& ( B|| C)", and(var("A"), or(var("B"), var("C"))));
        t("   A|| B &&C", or(var("A"), and(var("B"), var("C"))));
    }

    // TODO: Replace with a real benchmark once `#[bench]` is stable.
    #[test]
    fn benchmark() {
        for depth in 0..20 {
            t(
                &format!("{}a{}", "(".repeat(depth), ")".repeat(depth)),
                var("a"),
            );
        }
    }
}
