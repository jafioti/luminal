// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2021-2022  Philipp Emanuel Weidmann <pew@worldwidemann.com>

mod combinatorics;
mod linear_algebra;
mod logic;
mod number_theory;

use std::rc::Rc;

use num::Signed;
use savage_macros::functions;

use crate::expression::{Expression, Function as FunctionImplementation, Integer, Matrix};

/// Arbitrary-precision non-negative integer.
/// This type alias is intended for use in function signatures
/// to mark integer parameters that must be non-negative.
pub(crate) type NonNegativeInteger = Integer;

/// Arbitrary-precision positive integer.
/// This type alias is intended for use in function signatures
/// to mark integer parameters that must be positive.
pub(crate) type PositiveInteger = Integer;

/// Column-major square matrix with expressions as components.
/// This type alias is intended for use in function signatures
/// to mark matrix parameters that must be square matrices.
pub(crate) type SquareMatrix = Matrix;

/// Function parameter.
#[derive(PartialEq, Eq, Clone, Copy, Debug)]
pub enum Parameter {
    /// Any symbolic expression.
    Expression,
    /// Integer expression, or an expression that can be interpreted as an integer.
    Integer,
    /// Non-negative integer expression, or an expression that can be interpreted as a non-negative integer.
    NonNegativeInteger,
    /// Positive integer expression, or an expression that can be interpreted as a positive integer.
    PositiveInteger,
    /// Rational number expression, or an expression that can be interpreted as a rational number.
    Rational,
    /// Complex number expression, or an expression that can be interpreted as a complex number.
    Complex,
    /// Vector expression, or an expression that can be interpreted as a vector.
    Vector,
    /// Matrix expression, or an expression that can be interpreted as a matrix.
    Matrix,
    /// Matrix expression containing a square matrix, or an expression that can be interpreted as a square matrix.
    SquareMatrix,
    /// Boolean expression, or an expression that can be interpreted as a boolean value.
    Boolean,
}

/// Metadata associated with a function.
#[derive(PartialEq, Eq, Clone, Debug)]
pub struct Metadata {
    /// Name used to represent the function (also, default identifier for invoking the function).
    pub name: &'static str,
    /// Human-readable description of the function.
    pub description: &'static str,
    /// Parameters expected by the function, in the expected order.
    pub parameters: &'static [Parameter],
    /// Usage examples for the function, as pairs of REPL input and output.
    pub examples: &'static [(&'static str, &'static str)],
    /// Categories associated with the function.
    pub categories: &'static [&'static str],
}

/// Function definition.
pub struct Function {
    /// Metadata associated with the function.
    pub metadata: Metadata,
    /// Implementation of the function.
    pub implementation: Rc<FunctionImplementation>,
}

/// Returns a regular function implementation that type-checks its arguments
/// based on the given `parameters` and then invokes the given function `proxy`.
fn wrap_proxy(
    parameters: &'static [Parameter],
    proxy: impl Fn(&[Expression]) -> Result<Expression, Expression> + 'static,
) -> Rc<FunctionImplementation> {
    use crate::evaluate::Error::*;
    use crate::expression::Type::{Arithmetic, Boolean as Bool, Unknown};
    use Parameter::*;

    Rc::new(move |expression, arguments, _| {
        if arguments.len() != parameters.len() {
            return Err(InvalidNumberOfArguments {
                expression: expression.clone(),
                min_number: parameters.len(),
                max_number: parameters.len(),
                given_number: arguments.len(),
            });
        }

        for (argument, parameter) in arguments.iter().zip(parameters) {
            if let Bool(None) | Arithmetic | Unknown = argument.typ() {
                if *parameter != Expression {
                    return Ok(expression.clone());
                }
            }

            let mut argument_valid = true;

            match parameter {
                NonNegativeInteger => {
                    if let Ok(integer) = crate::expression::Integer::try_from(argument.clone()) {
                        argument_valid = !integer.is_negative();
                    }
                }
                PositiveInteger => {
                    if let Ok(integer) = crate::expression::Integer::try_from(argument.clone()) {
                        argument_valid = integer.is_positive();
                    }
                }
                SquareMatrix => {
                    if let Ok(matrix) = crate::expression::Matrix::try_from(argument.clone()) {
                        argument_valid = matrix.is_square() || matrix.is_empty();
                    }
                }
                _ => (),
            }

            if !argument_valid {
                return Err(InvalidArgument {
                    expression: expression.clone(),
                    argument: argument.clone(),
                });
            }
        }

        proxy(arguments).map_err(|argument| InvalidArgument {
            expression: expression.clone(),
            argument,
        })
    })
}

/// Returns all available functions.
pub fn functions() -> Vec<Function> {
    functions!(
        logic::and,
        combinatorics::factorial,
        linear_algebra::determinant,
        number_theory::is_prime,
        number_theory::nth_prime,
        number_theory::prime_pi,
    )
}

/// Returns an expression representing the function with the given name,
/// or `None` if the function library contains no function with that name.
pub fn function_expression(name: &str) -> Option<Expression> {
    for function in functions() {
        if function.metadata.name == name {
            return Some(Expression::Function(
                function.metadata.name.to_owned(),
                function.implementation,
            ));
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use crate::evaluate::default_context;
    use crate::expression::Expression;
    use crate::functions::functions;

    #[track_caller]
    fn t(expression: &str, result: &str) {
        assert_eq!(
            expression
                .parse::<Expression>()
                .unwrap()
                .evaluate(&default_context())
                .unwrap()
                .to_string(),
            result,
        );
    }

    #[test]
    fn examples() {
        for function in functions() {
            for (expression, result) in function.metadata.examples {
                t(expression, result);
            }
        }
    }
}
