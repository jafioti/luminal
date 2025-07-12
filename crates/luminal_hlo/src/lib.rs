use std::collections::HashMap;

use luminal::prelude::*;

mod utils;
use utils::{is_binary_op, is_unary_op};

pub fn import_hlo(path: &str) -> (Box<Graph>, HashMap<String, GraphTensor>) {
    let contents = std::fs::read_to_string(path).expect("Failed to read file.");

    let mut cx = Box::new(Graph::new());
    let mut tensor_map = HashMap::new();

    for line in contents.lines().map(str::trim) {
        if line.starts_with("func.func") {
            parse_func_args(line, &mut cx, &mut tensor_map);
        } else if line.starts_with('%') && line.contains(" = stablehlo.") {
            parse_hlo_op(line, &mut tensor_map);
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
                let arg_name = arg_name.trim().trim_start_matches('%');
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

fn parse_binary_op(op: &str, args: &[String], tensor_map: &HashMap<String, GraphTensor>) -> GraphTensor {
    let lhs = tensor_map[&args[0]];
    let rhs = tensor_map[&args[1]];
    match op {
        "stablehlo.add" => lhs + rhs,
        "stablehlo.subtract" => lhs - rhs,
        "stablehlo.multiply" => lhs * rhs,
        "stablehlo.divide" => lhs / rhs,
        "stablehlo.remainder" => lhs % rhs,
        // "stablehlo.maximum" => lhs.maximum(rhs),
        // "stablehlo.minimum" => lhs.minimum(rhs),
        _ => panic!("Unsupported binary op: {}", op),
    }
}

fn parse_unary_op(op: &str, args: &[String], tensor_map: &HashMap<String, GraphTensor>) -> GraphTensor {
    let tensor = tensor_map[&args[0]];
    match op {
        "stablehlo.abs" => tensor.abs(),
        "stablehlo.negate" => -tensor,
        "stablehlo.sqrt" => tensor.sqrt(),
        "stablehlo.log" => tensor.log(),
        "stablehlo.exponential" => tensor.exp(),
        _ => panic!("Unsupported unary op: {}", op),
    }
}

fn parse_hlo_op(op_line: &str, tensor_map: &mut HashMap<String, GraphTensor>) {
    let op_line = op_line.trim().trim_start_matches('%');
    if let Some((lhs, rest)) = op_line.split_once(" = ") {
        if let Some((op, args)) = rest.split_once(' ') {
            // Parse arguments from args
            let args: Vec<_> = args
                .split(&[',', ' '][..])
                .filter(|s| s.starts_with('%'))
                .map(|s| s.trim_start_matches('%').to_string())
                .collect();

            let result = match op {
                op if is_binary_op(op) => parse_binary_op(op, &args, tensor_map),
                op if is_unary_op(op) => parse_unary_op(op, &args, tensor_map),
                _ => panic!("Unsupported op: {}", op),
            };

            tensor_map.insert(lhs.to_string(), result);
        }
    }
}

fn parse_return(return_line: &str, tensor_map: &mut HashMap<String, GraphTensor>) {
    if let Some(name) = return_line.split_whitespace().find(|s| s.starts_with('%')) {
        let lhs = name.trim_start_matches('%');
        let result = tensor_map[lhs].retrieve();
        tensor_map.insert(lhs.to_string(), result);
    }
}
