use super::*;
#[test]
fn test_expressions() {
    let n = (Expression::from('x') + Term::Num(255)) / Term::Num(256) * Term::Num(256);
    assert_eq!(n.exec(&[('x', 767)].into_iter().collect()).unwrap(), 768);

    let n = (Expression::from('x') + Term::Num(255)) / Term::Num(256) * Term::Num(256);
    assert_eq!(n.exec(&[('x', 767)].into_iter().collect()).unwrap(), 768);
}

#[test]
fn test_minimizations() {
    let expr = ((BigExpression::from('a') * 1) + 0) / 1 + (1 - 1);
    let reduced_expr = expr.simplify();
    assert_eq!(reduced_expr, 'a');
}

#[test]
fn test_substitution() {
    let main = Expression::from('x') - 255;
    let sub = Expression::from('x') / 2;
    let new = main.substitute('x', sub);
    assert_eq!(new, (Expression::from('x') / 2) - 255);
}
