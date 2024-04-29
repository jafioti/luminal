use itertools::Itertools;

use super::*;

pub fn reduce_triples<S: ExpressionStorage>(
    mut expr: GenericExpression<S>,
) -> GenericExpression<S> {
    fn get_triples<S: ExpressionStorage>(
        exp: &GenericExpression<S>,
    ) -> Vec<(Option<usize>, usize, Option<usize>)> {
        // Mark all terms with their index
        let terms = exp
            .terms
            .clone()
            .into_iter()
            .enumerate()
            .collect::<Vec<_>>();
        let mut stack = Vec::new();
        let mut triples = vec![];
        for (index, term) in terms {
            match term {
                Term::Num(_) | Term::Var(_) => stack.push((Some(index), term)),
                _ => {
                    let (a_ind, a_term) = stack.pop().unwrap();
                    let (b_ind, b_term) = stack.pop().unwrap();
                    triples.push((a_ind, index, b_ind));
                    if let (Term::Num(a), Term::Num(b)) = (a_term, b_term) {
                        if let Some(c) = term.as_op().unwrap()(a as i64, b as i64) {
                            stack.push((None, Term::Num(c as i32)));
                        } else {
                            break;
                        }
                    } else if let Term::Var(a) = a_term {
                        stack.push((None, Term::Var(a)));
                    } else if let Term::Var(b) = b_term {
                        stack.push((None, Term::Var(b)));
                    }
                }
            }
        }
        triples
    }
    fn remove_terms<S: ExpressionStorage>(terms: &mut S, inds: &[usize]) {
        for ind in inds.iter().sorted().rev() {
            terms.remove(*ind);
        }
    }

    #[macro_export]
    macro_rules! unwrap_cont {
        ($i: expr) => {
            if let Some(s) = $i {
                s
            } else {
                continue;
            }
        };
    }
    let mut changed = true;
    while changed {
        changed = false;
        let triples = get_triples(&expr);
        for (a_ind, op_ind, b_ind) in triples {
            let mut inner_changed = true;
            match (
                a_ind.map(|a| expr.terms[a]),
                expr.terms[op_ind],
                b_ind.map(|b| expr.terms[b]),
            ) {
                (Some(Term::Num(a)), term, Some(Term::Num(b))) if term.as_op().is_some() => {
                    if let Some(c) = term.as_op().unwrap()(a as i64, b as i64) {
                        expr.terms[unwrap_cont!(a_ind)] = Term::Num(c as i32);
                        remove_terms(&mut expr.terms, &[op_ind, unwrap_cont!(b_ind)]);
                    } else {
                        inner_changed = false;
                    }
                }
                // Remove min(i, inf) and min(inf, i)
                (Some(Term::Num(a)), Term::Min, _) if a == i32::MAX => {
                    remove_terms(&mut expr.terms, &[op_ind, unwrap_cont!(a_ind)]);
                }
                (_, Term::Min, Some(Term::Num(b))) if b == i32::MAX => {
                    remove_terms(&mut expr.terms, &[op_ind, unwrap_cont!(b_ind)]);
                }
                // Remove max(i, inf) and max(inf, i)
                (_, Term::Max, Some(Term::Num(i))) if i == i32::MAX => {
                    remove_terms(&mut expr.terms, &[op_ind, unwrap_cont!(a_ind)]);
                }
                (Some(Term::Num(i)), Term::Max, _) if i == i32::MAX => {
                    remove_terms(&mut expr.terms, &[op_ind, unwrap_cont!(b_ind)]);
                }
                _ => {
                    inner_changed = false;
                }
            }
            if inner_changed {
                changed = true;
                break;
            }
        }
    }
    expr
}
