use std::collections::HashMap;

use luminal::prelude::*;
use regex::Regex;

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

                let tensor_shape = parse_tensor_shape(tensor_shape_str);

                let tensor = cx.tensor(tensor_shape);

                tensor_map.insert(arg_name.to_string(), tensor);
            }
        }
    }
}

#[derive(Debug)]
struct ReduceParts {
    input: String,
    init: String,
    apply: String,
    dims: Vec<usize>,
}

fn parse_reduce(line: &str) -> Option<ReduceParts> {
    let re = Regex::new(
        r#"stablehlo\.reduce\(\s*(?P<input>%[A-Za-z0-9_]+)\s+init:\s*(?P<init>%[A-Za-z0-9_]+)\s*\)\s*applies\s+(?P<apply>[A-Za-z0-9_.]+)\s+across\s+dimensions\s*=\s*(?P<dims>\[[^\]]*\])"#,
    ).ok()?;

    let caps = re.captures(line)?;

    let dims_str = caps.name("dims")?.as_str();
    let dims: Vec<usize> = dims_str
        .trim_matches(|c| c == '[' || c == ']')
        .split(',')
        .filter_map(|s| s.trim().parse::<usize>().ok())
        .collect();

    Some(ReduceParts {
        input: caps.name("input")?.as_str().to_string(),
        init: caps.name("init")?.as_str().to_string(),
        apply: caps.name("apply")?.as_str().to_string(),
        dims,
    })
}

pub fn parse_tensor_shape(tensor_type_str: &str) -> Vec<usize> {
    if let Some(start) = tensor_type_str.find('<') {
        if let Some(end) = tensor_type_str.find('>') {
            let shape_str = &tensor_type_str[start + 1..end];

            if !shape_str.contains('x')
                && (shape_str.ends_with("f32")
                    || shape_str.ends_with("f16")
                    || shape_str.ends_with("i32")
                    || shape_str.ends_with("i64"))
            {
                return vec![1];
            }

            let dims: Vec<usize> = shape_str
                .split('x')
                .filter_map(|s| {
                    let s = s.trim();
                    if s.ends_with("f32")
                        || s.ends_with("f16")
                        || s.ends_with("i32")
                        || s.ends_with("i64")
                    {
                        None
                    } else {
                        s.parse::<usize>().ok()
                    }
                })
                .collect();

            if dims.is_empty() {
                vec![1]
            } else {
                dims
            }
        } else {
            panic!("Malformed tensor type: missing '>' in {}", tensor_type_str);
        }
    } else {
        panic!("Malformed tensor type: missing '<' in {}", tensor_type_str);
    }
}

pub fn parse_output_shape_from_op(op_line: &str) -> Vec<usize> {
    if let Some(arrow_pos) = op_line.find("->") {
        let after_arrow = &op_line[arrow_pos + 2..].trim();

        // Find the tensor type after the arrow
        if let Some(tensor_start) = after_arrow.find("tensor<") {
            let tensor_end = after_arrow[tensor_start..]
                .find('>')
                .map(|pos| tensor_start + pos + 1)
                .unwrap_or(after_arrow.len());

            let tensor_type = &after_arrow[tensor_start..tensor_end];
            parse_tensor_shape(tensor_type)
        } else {
            panic!("No tensor type found after '->' in: {}", op_line);
        }
    } else {
        panic!("No '->' found in operation line: {}", op_line);
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
        let rest_str = rest.replace("(", " ").replace(")", " ");
        if let Some((op, args)) = rest_str.split_once(' ') {
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

                        // Movement ops
                        "stablehlo.reshape" => {
                            if let Some(start) = args.find("->") {
                                let shape_str = &args[start + 2..];
                                let shape = parse_tensor_shape(shape_str);
                                tensor_map[&args_tokens[0]].reshape(shape)
                            } else {
                                panic!("Malformed reshape op: missing '->' in {}", args);
                            }
                        }
                        "stablehlo.broadcast_in_dim" => {
                            if let Some(start) = args.find("dims = [") {
                                if let Some(end) = args[start..].find("]") {
                                    let dims_str = &args[start + 8..start + end];
                                    let dims: Vec<usize> = dims_str
                                        .trim_matches(|c| c == '[' || c == ']')
                                        .split(&[',', ' '][..])
                                        .filter(|s| !s.is_empty())
                                        .map(|s| s.trim().parse::<usize>().unwrap())
                                        .collect();
                                    tensor_map[&args_tokens[0]].expand(dims)
                                } else {
                                    panic!(
                                        "Malformed broadcast_in_dim op: missing ']' in {}",
                                        args
                                    );
                                }
                            } else {
                                panic!(
                                    "Malformed broadcast_in_dim op: missing 'dims = [' in {}",
                                    args
                                );
                            }
                        }
                        // Constants
                        "stablehlo.constant" => parse_constant(args, cx),

                        _ => panic!("Unsupported unary op: {}", op),
                    }
                }
                2 => {
                    let (mut op_lhs, mut op_rhs) =
                        (tensor_map[&args_tokens[0]], tensor_map[&args_tokens[1]]);

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

                        // Movement ops
                        "stablehlo.concatenate" => {
                            if let Some(start) = args.find("dim = ") {
                                if let Some(end) = args[start..].find(" :") {
                                    let value_str = &args[start + 6..start + end];
                                    let dim: usize =
                                        value_str.parse().expect("Failed to parse constant value");
                                    op_lhs.concat_along(op_rhs, dim)
                                } else {
                                    panic!("Malformed concat op: missing ' :' in {}", args);
                                }
                            } else {
                                panic!("Malformed concat op: missing 'dim = ' in {}", args);
                            }
                        }

                        "stablehlo.reduce" => {
                            if let Some(reduce) = parse_reduce(rest) {
                                match reduce.apply.as_str() {
                                    "stablehlo.add" => tensor_map[&reduce.input].sum(reduce.dims),
                                    _ => panic!("Unsupported reduce apply: {}", reduce.apply),
                                }
                            } else {
                                panic!("Malformed reduce op: missing 'reduce' in {}", args);
                            }
                        }

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
