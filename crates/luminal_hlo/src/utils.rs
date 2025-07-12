pub fn is_binary_op(op: &str) -> bool {
    matches!(op,
        "stablehlo.add" |
        "stablehlo.subtract" |
        "stablehlo.multiply" |
        "stablehlo.divide" |
        "stablehlo.remainder"
        // "stablehlo.maximum" - not supported
        // "stablehlo.minimum" - not supported
        // "stablehlo.matmul" - not supported
        // "stablehlo.power" - not supported
        // "stablehlo.compare" - EQ, NE, LT, GE, GT, LE, LT - not supported
    )
}

pub fn is_unary_op(op: &str) -> bool {
    matches!(op,
        "stablehlo.abs" |
        "stablehlo.negate" |
        "stablehlo.sqrt" |
        "stablehlo.log" |
        "stablehlo.exponential" 
        // "stablehlo.logistic" - not supported
        // "stablehlo.reciprocal" - not supported
        // "stablehlo.sin" - not supported
        // "stablehlo.cos" - not supported
    )
}
