// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2021-2022  Philipp Emanuel Weidmann <pew@worldwidemann.com>

use std::collections::HashMap;

use num::{One, ToPrimitive, Zero};

use crate::{
    expression::{Complex, Expression, RationalRepresentation},
    functions::functions,
};

/// Error that occurred while trying to evaluate an expression.
#[derive(PartialEq, Eq, Clone, Debug)]
pub enum Error {
    /// Operation on an expression that the operation is not defined for.
    InvalidOperand {
        expression: Expression,
        operand: Expression,
    },
    /// Operation on two expressions that cannot be combined using the operation.
    IncompatibleOperands {
        expression: Expression,
        operand_1: Expression,
        operand_2: Expression,
    },
    /// Division by an expression that evaluates to zero (undefined).
    DivisionByZero {
        expression: Expression,
        dividend: Expression,
        divisor: Expression,
    },
    /// An expression that evaluates to zero raised to the power of
    /// another expression that evaluates to zero (undefined).
    ZeroToThePowerOfZero {
        expression: Expression,
        base: Expression,
        exponent: Expression,
    },
    /// Vector or matrix expression indexed by an expression that evaluates to
    /// an integer outside the range of valid indices for that vector or matrix.
    IndexOutOfBounds {
        expression: Expression,
        vector_or_matrix: Expression,
        index: Expression,
    },
    /// Function expression evaluated with a number of arguments
    /// that is invalid for the function.
    InvalidNumberOfArguments {
        expression: Expression,
        min_number: usize,
        max_number: usize,
        given_number: usize,
    },
    /// Function expression evaluated with an argument
    /// that is invalid for the function in that position.
    InvalidArgument {
        expression: Expression,
        argument: Expression,
    },
}

/// Returns an evaluation context populated with standard variable and function definitions.
pub fn default_context() -> HashMap<String, Expression> {
    let mut default_context = HashMap::new();

    default_context.insert(
        "i".to_owned(),
        Expression::Complex(Complex::i(), RationalRepresentation::Fraction),
    );

    for function in functions() {
        default_context.insert(
            function.metadata.name.to_owned(),
            Expression::Function(function.metadata.name.to_owned(), function.implementation),
        );
    }

    default_context
}

impl Expression {
    /// Returns the result of performing a single evaluation step on
    /// the unary operator expression `self` with operand `a`, or an error
    /// if the expression cannot be evaluated. The `context` argument can be
    /// used to set the values of variables by their identifiers.
    fn evaluate_step_unary(
        &self,
        a: &Self,
        context: &HashMap<String, Self>,
    ) -> Result<Self, Error> {
        use crate::expression::Expression::*;
        use crate::expression::Type::{Arithmetic, Boolean as Bool, Matrix as Mat, Number as Num};
        use Error::*;

        let a_original = a;

        let a = a.evaluate_step(context)?;

        match (self, a.typ()) {
            (Negation(_), Bool(_)) | (Not(_), Num(_, _) | Mat(_) | Arithmetic) => {
                Err(InvalidOperand {
                    expression: self.clone(),
                    operand: a_original.clone(),
                })
            }

            (Negation(_), Num(a, representation)) => Ok(Complex(-a, representation)),
            (Negation(_), Mat(a)) => Ok(Matrix(-a)),
            (Negation(_), _) => Ok(Negation(Box::new(a))),

            (Not(_), Bool(Some(a))) => Ok(Boolean(!a)),
            (Not(_), _) => Ok(Not(Box::new(a))),

            (
                Variable(_)
                | Function(_, _)
                | FunctionValue(_, _)
                | Integer(_)
                | Rational(_, _)
                | Complex(_, _)
                | Vector(_)
                | VectorElement(_, _)
                | Matrix(_)
                | MatrixElement(_, _, _)
                | Boolean(_)
                | Sum(_, _)
                | Difference(_, _)
                | Product(_, _)
                | Quotient(_, _)
                | Remainder(_, _)
                | Power(_, _)
                | Equal(_, _)
                | NotEqual(_, _)
                | LessThan(_, _)
                | LessThanOrEqual(_, _)
                | GreaterThan(_, _)
                | GreaterThanOrEqual(_, _)
                | And(_, _)
                | Or(_, _)
                | Min(_, _),
                _,
            ) => unreachable!(),
        }
    }

    /// Returns the result of performing a single evaluation step on
    /// the binary operator expression `self` with operands `a` and `b`,
    /// or an error if the expression cannot be evaluated. The `context`
    /// argument can be used to set the values of variables by their
    /// identifiers.
    fn evaluate_step_binary(
        &self,
        a: &Self,
        b: &Self,
        context: &HashMap<String, Self>,
    ) -> Result<Self, Error> {
        use crate::expression::Expression::*;
        use crate::expression::Type::{Arithmetic, Boolean as Bool, Matrix as Mat, Number as Num};
        use Error::*;

        let a_original = a;
        let b_original = b;

        let a = a.evaluate_step(context)?;
        let b = b.evaluate_step(context)?;

        let a_evaluated = &a;
        let b_evaluated = &b;

        match (self, a.typ(), b.typ()) {
            (
                Sum(_, _)
                | Min(_, _)
                | Difference(_, _)
                | Product(_, _)
                | Quotient(_, _)
                | Remainder(_, _)
                | Power(_, _),
                Bool(_),
                _,
            )
            | (
                LessThan(_, _)
                | LessThanOrEqual(_, _)
                | GreaterThan(_, _)
                | GreaterThanOrEqual(_, _),
                Mat(_) | Bool(_),
                _,
            )
            | (And(_, _) | Or(_, _), Num(_, _) | Mat(_) | Arithmetic, _) => Err(InvalidOperand {
                expression: self.clone(),
                operand: a_original.clone(),
            }),

            (
                Sum(_, _)
                | Min(_, _)
                | Difference(_, _)
                | Product(_, _)
                | Quotient(_, _)
                | Remainder(_, _)
                | Power(_, _),
                _,
                Bool(_),
            )
            | (
                LessThan(_, _)
                | LessThanOrEqual(_, _)
                | GreaterThan(_, _)
                | GreaterThanOrEqual(_, _),
                _,
                Mat(_) | Bool(_),
            )
            | (And(_, _) | Or(_, _), _, Num(_, _) | Mat(_) | Arithmetic) => Err(InvalidOperand {
                expression: self.clone(),
                operand: b_original.clone(),
            }),

            (
                Sum(_, _) | Min(_, _) | Difference(_, _) | Equal(_, _) | NotEqual(_, _),
                Num(_, _),
                Mat(_),
            )
            | (
                Sum(_, _) | Min(_, _) | Difference(_, _) | Equal(_, _) | NotEqual(_, _),
                Mat(_),
                Num(_, _),
            )
            | (Equal(_, _) | NotEqual(_, _), Num(_, _) | Mat(_), Bool(_))
            | (Equal(_, _) | NotEqual(_, _), Bool(_), Num(_, _) | Mat(_)) => {
                Err(IncompatibleOperands {
                    expression: self.clone(),
                    operand_1: a_original.clone(),
                    operand_2: b_original.clone(),
                })
            }

            (
                Sum(_, _)
                | Min(_, _)
                | Difference(_, _)
                | Product(_, _)
                | Quotient(_, _)
                | Remainder(_, _)
                | Power(_, _)
                | Equal(_, _)
                | NotEqual(_, _)
                | LessThan(_, _)
                | LessThanOrEqual(_, _)
                | GreaterThan(_, _)
                | GreaterThanOrEqual(_, _),
                Num(a, a_representation),
                Num(b, b_representation),
            ) => {
                let representation = a_representation.merge(b_representation);

                match self {
                    Sum(_, _) => Ok(Complex(a + b, representation)),
                    Min(_, _) => unreachable!(),
                    Difference(_, _) => Ok(Complex(a - b, representation)),
                    Product(_, _) => Ok(Complex(a * b, representation)),
                    Quotient(_, _) | Remainder(_, _) => {
                        if b.is_zero() {
                            Err(DivisionByZero {
                                expression: self.clone(),
                                dividend: a_original.clone(),
                                divisor: b_original.clone(),
                            })
                        } else {
                            Ok(Complex(
                                match self {
                                    Quotient(_, _) => a / b,
                                    Remainder(_, _) => a % b,
                                    _ => unreachable!(),
                                },
                                representation,
                            ))
                        }
                    }
                    Power(_, _) => {
                        if a.is_zero() && b.is_zero() {
                            Err(ZeroToThePowerOfZero {
                                expression: self.clone(),
                                base: a_original.clone(),
                                exponent: b_original.clone(),
                            })
                        } else if let Some(b) = b.to_i32() {
                            Ok(Complex(a.powi(b), representation))
                        } else {
                            // TODO
                            Ok(Power(
                                Box::new(a_evaluated.clone()),
                                Box::new(b_evaluated.clone()),
                            ))
                        }
                    }
                    Equal(_, _) => Ok(Boolean(a == b)),
                    NotEqual(_, _) => Ok(Boolean(a != b)),
                    LessThan(_, _)
                    | LessThanOrEqual(_, _)
                    | GreaterThan(_, _)
                    | GreaterThanOrEqual(_, _) => {
                        if !a.im.is_zero() {
                            Err(InvalidOperand {
                                expression: self.clone(),
                                operand: a_original.clone(),
                            })
                        } else if !b.im.is_zero() {
                            Err(InvalidOperand {
                                expression: self.clone(),
                                operand: b_original.clone(),
                            })
                        } else {
                            let a = a.re;
                            let b = b.re;

                            Ok(Boolean(match self {
                                LessThan(_, _) => a < b,
                                LessThanOrEqual(_, _) => a <= b,
                                GreaterThan(_, _) => a > b,
                                GreaterThanOrEqual(_, _) => a >= b,
                                _ => unreachable!(),
                            }))
                        }
                    }
                    _ => unreachable!(),
                }
            }

            (Sum(_, _) | Min(_, _) | Difference(_, _), Mat(a), Mat(b)) => {
                if a.shape() == b.shape() {
                    Ok(Matrix(match self {
                        Sum(_, _) => a + b,
                        Difference(_, _) => a - b,
                        _ => unreachable!(),
                    }))
                } else {
                    Err(IncompatibleOperands {
                        expression: self.clone(),
                        operand_1: a_original.clone(),
                        operand_2: b_original.clone(),
                    })
                }
            }

            (Product(_, _), Mat(a), Mat(b)) => {
                if a.is_empty() && b.is_empty() {
                    Ok(Matrix(a))
                } else if !a.is_empty() && !b.is_empty() && a.ncols() == b.nrows() {
                    Ok(Matrix(crate::expression::Matrix::from_fn(
                        a.nrows(),
                        b.ncols(),
                        |i, j| {
                            (0..a.ncols())
                                .map(|k| a[(i, k)].clone() * b[(k, j)].clone())
                                .reduce(|a, b| a + b)
                                .unwrap()
                        },
                    )))
                } else {
                    Err(IncompatibleOperands {
                        expression: self.clone(),
                        operand_1: a_original.clone(),
                        operand_2: b_original.clone(),
                    })
                }
            }

            (Product(_, _), Mat(a), _) => Ok(Matrix(a.map(|element| element * b.clone()))),
            (Product(_, _), _, Mat(b)) => Ok(Matrix(b.map(|element| a.clone() * element))),

            (Equal(_, _), Bool(Some(a)), Bool(Some(b))) => Ok(Boolean(a == b)),
            (NotEqual(_, _), Bool(Some(a)), Bool(Some(b))) => Ok(Boolean(a != b)),
            (And(_, _), Bool(Some(a)), Bool(Some(b))) => Ok(Boolean(a && b)),
            (Or(_, _), Bool(Some(a)), Bool(Some(b))) => Ok(Boolean(a || b)),

            (Sum(_, _), _, _) => Ok(Sum(Box::new(a), Box::new(b))), // TODO
            (Min(_, _), _, _) => Ok(Min(Box::new(a), Box::new(b))), // TODO
            (Difference(_, _), _, _) => Ok(Difference(Box::new(a), Box::new(b))), // TODO
            (Product(_, _), _, _) => Ok(Product(Box::new(a), Box::new(b))), // TODO
            (Quotient(_, _), _, _) => Ok(Quotient(Box::new(a), Box::new(b))), // TODO
            (Remainder(_, _), _, _) => Ok(Remainder(Box::new(a), Box::new(b))), // TODO
            (Power(_, _), _, _) => Ok(Power(Box::new(a), Box::new(b))), // TODO
            (Equal(_, _), _, _) => Ok(Equal(Box::new(a), Box::new(b))), // TODO
            (NotEqual(_, _), _, _) => Ok(NotEqual(Box::new(a), Box::new(b))), // TODO
            (LessThan(_, _), _, _) => Ok(LessThan(Box::new(a), Box::new(b))), // TODO
            (LessThanOrEqual(_, _), _, _) => Ok(LessThanOrEqual(Box::new(a), Box::new(b))), // TODO
            (GreaterThan(_, _), _, _) => Ok(GreaterThan(Box::new(a), Box::new(b))), // TODO
            (GreaterThanOrEqual(_, _), _, _) => Ok(GreaterThanOrEqual(Box::new(a), Box::new(b))), // TODO
            (And(_, _), _, _) => Ok(And(Box::new(a), Box::new(b))), // TODO
            (Or(_, _), _, _) => Ok(Or(Box::new(a), Box::new(b))),   // TODO

            (
                Variable(_)
                | Function(_, _)
                | FunctionValue(_, _)
                | Integer(_)
                | Rational(_, _)
                | Complex(_, _)
                | Vector(_)
                | VectorElement(_, _)
                | Matrix(_)
                | MatrixElement(_, _, _)
                | Boolean(_)
                | Negation(_)
                | Not(_),
                _,
                _,
            ) => unreachable!(),
        }
    }

    /// Returns the result of performing a single evaluation step on the expression,
    /// or an error if the expression cannot be evaluated. The `context` argument
    /// can be used to set the values of variables by their identifiers.
    fn evaluate_step(&self, context: &HashMap<String, Self>) -> Result<Self, Error> {
        use crate::expression::Expression::*;
        use crate::expression::Type::{
            Boolean as Bool, Function as Fun, Matrix as Mat, Number as Num,
        };
        use Error::*;

        let expression = self.simplify();

        match &expression {
            Variable(identifier) => context
                .get(identifier)
                .map_or_else(|| Ok(expression), |x| x.evaluate_step(context)),
            Function(_, _) => Ok(expression),
            FunctionValue(function, arguments) => {
                let function_original = function;

                let function = function.evaluate_step(context)?;

                let mut arguments_evaluated = Vec::new();

                for argument in arguments {
                    arguments_evaluated.push(argument.evaluate_step(context)?);
                }

                match function.typ() {
                    Num(_, _) | Mat(_) | Bool(_) => Err(InvalidOperand {
                        expression: expression.clone(),
                        operand: *function_original.clone(),
                    }),

                    Fun(_, f) => f(&expression, &arguments_evaluated, context),

                    _ => Ok(FunctionValue(Box::new(function), arguments_evaluated)),
                }
            }
            Integer(_) => Ok(expression),
            Rational(x, _) => Ok(if x.denom().is_one() {
                Integer(x.numer().clone())
            } else {
                expression
            }),
            Complex(z, representation) => Ok(if z.im.is_zero() {
                Rational(z.re.clone(), *representation)
            } else {
                expression
            }),
            Vector(v) => {
                let mut elements = Vec::new();

                for element in v.iter() {
                    elements.push(element.evaluate_step(context)?);
                }

                Ok(Vector(crate::expression::Vector::from_vec(elements)))
            }
            VectorElement(vector, i) => {
                let vector_original = vector;
                let i_original = i;

                let vector = vector.evaluate_step(context)?;
                let i = i.evaluate_step(context)?;

                match (vector.typ(), i.typ()) {
                    (Num(_, _) | Bool(_), _) => Err(InvalidOperand {
                        expression: expression.clone(),
                        operand: *vector_original.clone(),
                    }),

                    (_, Mat(_) | Bool(_)) => Err(InvalidOperand {
                        expression: expression.clone(),
                        operand: *i_original.clone(),
                    }),

                    (Mat(vector), Num(i, _)) => {
                        if vector.ncols() != 1 {
                            Err(InvalidOperand {
                                expression: expression.clone(),
                                operand: *vector_original.clone(),
                            })
                        } else if let Some(i) = i.to_usize() {
                            if i >= vector.nrows() {
                                Err(IndexOutOfBounds {
                                    expression: expression.clone(),
                                    vector_or_matrix: *vector_original.clone(),
                                    index: *i_original.clone(),
                                })
                            } else {
                                Ok(vector[(i, 0)].clone())
                            }
                        } else {
                            Err(InvalidOperand {
                                expression: expression.clone(),
                                operand: *i_original.clone(),
                            })
                        }
                    }

                    _ => Ok(VectorElement(Box::new(vector), Box::new(i))),
                }
            }
            Matrix(m) => {
                let mut columns = Vec::new();

                for column in m.column_iter() {
                    let mut elements = Vec::new();

                    for element in column.iter() {
                        elements.push(element.evaluate_step(context)?);
                    }

                    columns.push(crate::expression::Vector::from_vec(elements));
                }

                Ok(if columns.is_empty() {
                    Vector(crate::expression::Vector::from_vec(Vec::new()))
                } else if columns.len() == 1 {
                    Vector(columns.remove(0))
                } else {
                    Matrix(crate::expression::Matrix::from_columns(&columns))
                })
            }
            MatrixElement(matrix, i, j) => {
                let matrix_original = matrix;
                let i_original = i;
                let j_original = j;

                let matrix = matrix.evaluate_step(context)?;
                let i = i.evaluate_step(context)?;
                let j = j.evaluate_step(context)?;

                match (matrix.typ(), i.typ(), j.typ()) {
                    (Num(_, _) | Bool(_), _, _) => Err(InvalidOperand {
                        expression: expression.clone(),
                        operand: *matrix_original.clone(),
                    }),

                    (_, Mat(_) | Bool(_), _) => Err(InvalidOperand {
                        expression: expression.clone(),
                        operand: *i_original.clone(),
                    }),

                    (_, _, Mat(_) | Bool(_)) => Err(InvalidOperand {
                        expression: expression.clone(),
                        operand: *j_original.clone(),
                    }),

                    (Mat(matrix), Num(i, _), Num(j, _)) => {
                        if let Some(i) = i.to_usize() {
                            if let Some(j) = j.to_usize() {
                                if i >= matrix.nrows() {
                                    Err(IndexOutOfBounds {
                                        expression: expression.clone(),
                                        vector_or_matrix: *matrix_original.clone(),
                                        index: *i_original.clone(),
                                    })
                                } else if j >= matrix.ncols() {
                                    Err(IndexOutOfBounds {
                                        expression: expression.clone(),
                                        vector_or_matrix: *matrix_original.clone(),
                                        index: *j_original.clone(),
                                    })
                                } else {
                                    Ok(matrix[(i, j)].clone())
                                }
                            } else {
                                Err(InvalidOperand {
                                    expression: expression.clone(),
                                    operand: *j_original.clone(),
                                })
                            }
                        } else {
                            Err(InvalidOperand {
                                expression: expression.clone(),
                                operand: *i_original.clone(),
                            })
                        }
                    }

                    _ => Ok(MatrixElement(Box::new(matrix), Box::new(i), Box::new(j))),
                }
            }
            Boolean(_) => Ok(expression),
            Negation(a) => expression.evaluate_step_unary(a, context),
            Not(a) => expression.evaluate_step_unary(a, context),
            Sum(a, b) => expression.evaluate_step_binary(a, b, context),
            Min(a, b) => expression.evaluate_step_binary(a, b, context),
            Difference(a, b) => expression.evaluate_step_binary(a, b, context),
            Product(a, b) => expression.evaluate_step_binary(a, b, context),
            Quotient(a, b) => expression.evaluate_step_binary(a, b, context),
            Remainder(a, b) => expression.evaluate_step_binary(a, b, context),
            Power(a, b) => expression.evaluate_step_binary(a, b, context),
            Equal(a, b) => expression.evaluate_step_binary(a, b, context),
            NotEqual(a, b) => expression.evaluate_step_binary(a, b, context),
            LessThan(a, b) => expression.evaluate_step_binary(a, b, context),
            LessThanOrEqual(a, b) => expression.evaluate_step_binary(a, b, context),
            GreaterThan(a, b) => expression.evaluate_step_binary(a, b, context),
            GreaterThanOrEqual(a, b) => expression.evaluate_step_binary(a, b, context),
            And(a, b) => expression.evaluate_step_binary(a, b, context),
            Or(a, b) => expression.evaluate_step_binary(a, b, context),
        }
    }

    /// Returns the result of evaluating the expression, or an error
    /// if the expression cannot be evaluated. The `context` argument
    /// can be used to set the values of variables by their identifiers.
    pub fn evaluate(&self, context: &HashMap<String, Self>) -> Result<Self, Error> {
        let mut old_expression: Self = self.clone();

        loop {
            let new_expression = old_expression.evaluate_step(context)?;

            if new_expression == old_expression {
                return Ok(new_expression);
            }

            old_expression = new_expression;
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::evaluate::default_context;
    use crate::expression::Expression;

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
    fn arithmetic() {
        t("-(-1)", "1");
        t("-0", "0");

        t("1 + 2", "3");
        t("1 + -1", "0");
        t("1/2 + 0.5", "1");
        t(
            "123456789987654321 + 987654321123456789",
            "1111111111111111110",
        );

        t("1 - 2", "-1");
        t("1 - -1", "2");
        t("1/2 - 0.5", "0");
        t(
            "123456789987654321 - 987654321123456789",
            "-864197531135802468",
        );

        t("1 * 2", "2");
        t("1 * -1", "-1");
        t("1/2 * 0.5", "0.25");
        t(
            "123456789987654321 * 987654321123456789",
            "121932632103337905662094193112635269",
        );

        t("1 / 2", "1/2");
        t("1 / -1", "-1");
        t("1/2 / 0.5", "1");
        t(
            "123456789987654321 / 987654321123456789",
            "101010101/808080809",
        );

        t("4 % 2", "0");
        t("0 % 2", "0");
        t("5 % 2", "1");
        t("-5 % 2", "-1");
        t("-5 % -2", "-1");
        t("0.75 % (1/4)", "0");
        t("0.75 % (1/3)", "1/12");
        t("987654321123456789 % 123456789987654321", "1222222221");

        t("i ^ 2", "-1");
        t("2 ^ 3", "8");
        t("2 ^ (-3)", "1/8");
        t("-2 ^ 4", "-16");
        t("(-2) ^ 4", "16");
        t("0.5 ^ 4", "0.0625");
        t(
            "987654321123456789 ^ 5",
            "939777062588963894467852986656442266299580252508947542802086985660852317355013741720482949",
        );
        t(
            "3 ^ 4 ^ 5",
            "373391848741020043532959754184866588225409776783734007750636931722079040617265251229993688938803977220468765065431475158108727054592160858581351336982809187314191748594262580938807019951956404285571818041046681288797402925517668012340617298396574731619152386723046235125934896058590588284654793540505936202376547807442730582144527058988756251452817793413352141920744623027518729185432862375737063985485319476416926263819972887006907013899256524297198527698749274196276811060702333710356481",
        );
    }

    #[test]
    fn linear_algebra() {
        t("[1] + [2]", "[3]");
        t("[1] - [2]", "[-1]");
        t("[1] * [2]", "[2]");
        t("[1] * 2", "[2]");
        t("1 * [2]", "[2]");

        t("[1, 2] + [3, 4]", "[4, 6]");
        t("[1, 2] - [3, 4]", "[-2, -2]");
        t("[1, 2] * [[3, 4]]", "[[3, 4], [6, 8]]");
        t("[[1, 2]] * [3, 4]", "[11]");
        t("2 * [3, 4]", "[6, 8]");
        t("[2, 3] * 4", "[8, 12]");

        t(
            "[[a, b], [c, d], [e, f]] * [[5, 6], [7, 8]]",
            "[[a * 5 + b * 7, a * 6 + b * 8], [c * 5 + d * 7, c * 6 + d * 8], [e * 5 + f * 7, e * 6 + f * 8]]",
        );
    }

    #[test]
    fn indices() {
        t("[a][0]", "a");
        t("[a, b, c][2]", "c");
        t("[1 + 2, 2 + 3, 3 + 4, 4 + 5][1 + 2]", "9");

        t("[[a]][0, 0]", "a");
        t("[[a, b, c], [d, e, f]][1, 2]", "f");
        t("[[1 + 2, 2 + 3], [3 + 4, 4 + 5]][0 + 0, 0 + 1]", "5");
    }

    #[test]
    fn logic() {
        t("!true", "false");
        t("!false", "true");

        t("true && true", "true");
        t("true && false", "false");
        t("false && true", "false");
        t("false && false", "false");

        t("true || true", "true");
        t("true || false", "true");
        t("false || true", "true");
        t("false || false", "false");
    }

    #[test]
    fn comparisons() {
        t("0 == 0", "true");
        t("0 == 0.0", "true");
        t("0.5 == 1/2", "true");
        t("1/2 == 2/4", "true");
        t("3 ^ 4 ^ 5 == 5 ^ 4 ^ 3", "false");

        t("0 != 0", "false");
        t("0 != 0.0", "false");
        t("0.5 != 1/2", "false");
        t("1/2 != 2/4", "false");
        t("3 ^ 4 ^ 5 != 5 ^ 4 ^ 3", "true");

        t("0 < 0", "false");
        t("0 < 0.0", "false");
        t("0.5 < 1/2", "false");
        t("1/2 < 2/4", "false");
        t("3 ^ 4 ^ 5 < 5 ^ 4 ^ 3", "false");

        t("0 <= 0", "true");
        t("0 <= 0.0", "true");
        t("0.5 <= 1/2", "true");
        t("1/2 <= 2/4", "true");
        t("3 ^ 4 ^ 5 <= 5 ^ 4 ^ 3", "false");

        t("0 > 0", "false");
        t("0 > 0.0", "false");
        t("0.5 > 1/2", "false");
        t("1/2 > 2/4", "false");
        t("3 ^ 4 ^ 5 > 5 ^ 4 ^ 3", "true");

        t("0 >= 0", "true");
        t("0 >= 0.0", "true");
        t("0.5 >= 1/2", "true");
        t("1/2 >= 2/4", "true");
        t("3 ^ 4 ^ 5 >= 5 ^ 4 ^ 3", "true");

        t("true == true", "true");
        t("true == false", "false");
        t("false == true", "false");
        t("false == false", "true");

        t("true != true", "false");
        t("true != false", "true");
        t("false != true", "true");
        t("false != false", "false");
    }
}
