use std::collections::HashMap;

use luminal::prelude::*;

pub fn import_hlo(path: &str) -> (Box<Graph>, HashMap<String, GraphTensor>) {
    let contents = std::fs::read_to_string(path).expect("Failed to read file.");

    let mut cx = Box::new(Graph::new());
    let mut tensor_map = HashMap::new();

    for line in contents.lines().map(str::trim) {
        if line.starts_with("func.func") {
            parse_func_args(line, &mut cx, &mut tensor_map);
        } else if line.starts_with('%') && line.contains(" = stablehlo.") {
            parse_hlo_op(line, &mut cx, &mut tensor_map);
        } else if line.starts_with("return") {
            parse_return(line, &mut tensor_map);
        }
    }

    (cx, tensor_map)
}

fn parse_func_args(line: &str, cx: &mut Graph, tensor_map: &mut HashMap<String, GraphTensor>) {
    if let Some((start_idx, end_idx)) = line.find('(').zip(line.find(')')) {
        let args_str = &line[start_idx + 1..end_idx];

        for arg in args_str.split(',') {
            let arg_tokens: Vec<&str> = arg.trim().split(':').collect();

            if let [arg_name, tensor_shape_str] = arg_tokens.as_slice() {
                let arg_name = arg_name.trim();
                let tensor_shape_str = tensor_shape_str.trim();

                // Parse shape
                let tensor_shape: (usize, usize) =
                    if let Some(shape_start) = tensor_shape_str.find('<') {
                        if let Some(shape_end) = tensor_shape_str.find('>') {
                            let tensor_shape_str = &tensor_shape_str[shape_start + 1..shape_end];
                            let dims: Vec<usize> = tensor_shape_str
                                .split('x')
                                .filter_map(|s| s.parse::<usize>().ok())
                                .collect();
                            match dims.as_slice() {
                                [d1, d2] => (*d1, *d2),
                                [d1] => (*d1, 1),
                                [] => (1, 1),
                                _ => panic!(
                                    "Only supports up to 2D shapes for now: {}",
                                    tensor_shape_str
                                ),
                            }
                        } else {
                            panic!("Malformed shape: missing '>' in {}", tensor_shape_str);
                        }
                    } else {
                        panic!("Malformed shape: missing '<' in {}", tensor_shape_str);
                    };

                let tensor = cx.tensor(tensor_shape);

                tensor_map.insert(arg_name.to_string(), tensor);
            }
        }
    }
}

fn parse_constant(value_str: &str, cx: &mut Graph) -> GraphTensor {
    if let Some(start) = value_str.find("dense<") {
        if let Some(end) = value_str[start..].find('>') {
            let value_str = &value_str[start + 6..start + end];
            let value: f32 = value_str.parse().expect("Failed to parse constant value");
            cx.constant(value).retrieve()
        } else {
            panic!("Malformed constant: missing '>' in {}", value_str);
        }
    } else {
        panic!("Malformed constant: missing 'dense<' in {}", value_str);
    }                        
}

fn parse_hlo_op(op_line: &str, cx: &mut Graph, tensor_map: &mut HashMap<String, GraphTensor>) {
    let op_line = op_line.trim();
    if let Some((lhs, rest)) = op_line.split_once(" = ") {
        if let Some((op, args)) = rest.split_once(' ') {
            // Parse arguments from args
            let args_tokens: Vec<_> = args
                .split(&[',', ' '][..])
                .filter(|s| s.starts_with('%') || s.starts_with("dense<"))
                .map(|s| s.to_string())
                .collect();

            let result = match args_tokens.len() {
                1 => {
                    match op {
                        // Unary ops
                        "stablehlo.abs" => tensor_map[&args_tokens[0]].abs(),
                        "stablehlo.negate" => -tensor_map[&args_tokens[0]],
                        "stablehlo.sqrt" => tensor_map[&args_tokens[0]].sqrt(),
                        "stablehlo.log" => tensor_map[&args_tokens[0]].log(),
                        "stablehlo.exponential" => tensor_map[&args_tokens[0]].exp(),

                        // Other ops
                        "stablehlo.constant" => parse_constant(args, cx),

                        _ => panic!("Unsupported unary op: {}", op),
                    }
                }
                2 => {
                    let (mut op_lhs, mut op_rhs) = (tensor_map[&args_tokens[0]], tensor_map[&args_tokens[1]]);

                    if op_lhs.shape.dims().is_empty() && !op_rhs.shape.dims().is_empty() {
                        for &dim in op_rhs.shape.dims().iter() {
                            op_lhs = op_lhs.expand_dim(0, dim);
                        }
                    } else if op_rhs.shape.dims().is_empty() && !op_lhs.shape.dims().is_empty() {
                        for &dim in op_lhs.shape.dims().iter() {
                            op_rhs = op_rhs.expand_dim(0, dim);
                        }
                    }
                        
                    match op {
                        // Binary ops
                        "stablehlo.add" => op_lhs + op_rhs,
                        "stablehlo.subtract" => op_lhs - op_rhs,
                        "stablehlo.multiply" => op_lhs * op_rhs,
                        "stablehlo.divide" => op_lhs / op_rhs,
                        "stablehlo.remainder" => op_lhs % op_rhs,
                        "stablehlo.maximum" => op_lhs.maximum(op_rhs),
                        "stablehlo.minimum" => op_lhs.minimum(op_rhs),
                        _ => panic!("Unsupported binary op: {}", op),
                    }
                }
                _ => panic!(
                    "Unsupported number of arguments for op {}: {}",
                    op,
                    args_tokens.len()
                ),
            };

            tensor_map.insert(lhs.to_string(), result);
        }
    }
}

fn parse_return(return_line: &str, tensor_map: &mut HashMap<String, GraphTensor>) {
    if let Some(name) = return_line.split_whitespace().find(|s| s.starts_with('%')) {
        let lhs = name.trim();
        let result = tensor_map[lhs].retrieve();
        tensor_map.insert(lhs.to_string(), result);
    }
}
