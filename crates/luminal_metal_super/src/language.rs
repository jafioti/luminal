use egg::*;

define_language! {
    enum Kernel {
        Num(i32),
        "+" = Add([Id; 2]),
        "-" = Sub([Id; 2]),
        "*" = Mul([Id; 2]),
        "/" = Div([Id; 2]),
        "%" = Mod([Id; 2]),
        "min" = Min([Id; 2]),
        "max" = Max([Id; 2]),
        "&&" = And([Id; 2]),
        "||" = Or([Id; 2]),
        "<" = LessThan([Id; 2]),
        ">=" = GreaterThanEqual([Id; 2]),
        Symbol(Symbol),
    }
}
