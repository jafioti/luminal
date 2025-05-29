use std::collections::HashMap;

use luminal::prelude::*;

pub fn import_hlo(path: &str, cx: &mut Graph) -> HashMap<String, GraphTensor> {
    let contents = std::fs::read_to_string(path).expect("Failed to read file.");

    let mut tensor_map = HashMap::new();

    for line in contents.lines().map(str::trim) {
        if line.starts_with("func.func") {
            parse_func_args(line, cx, &mut tensor_map);
        } else if line.starts_with('%') && line.contains(" = stablehlo.") {
            parse_hlo_op(line, &mut tensor_map);
        } else if line.starts_with("return") {
            parse_return(line, &mut tensor_map);
        }
    }

    tensor_map
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

fn parse_hlo_op(op_line: &str, tensor_map: &mut HashMap<String, GraphTensor>) {
    let op_line = op_line.trim().trim_start_matches('%');
    if let Some((lhs, rest)) = op_line.split_once(" = ") {
        if let Some((op, args)) = rest.split_once(' ') {
            // Parse arguments from args_str
            let args: Vec<_> = args
                .split(&[',', ' '][..])
                .filter(|s| s.starts_with('%'))
                .map(|s| s.trim_start_matches('%').to_string())
                .collect();

            let result = match op {
                // Arithmetic ops
                "stablehlo.add" => tensor_map[&args[0]] + tensor_map[&args[1]],
                "stablehlo.multiply" => tensor_map[&args[0]] * tensor_map[&args[1]],
                "stablehlo.subtract" => tensor_map[&args[0]] - tensor_map[&args[1]],
                "stablehlo.divide" => tensor_map[&args[0]] / tensor_map[&args[1]],
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
