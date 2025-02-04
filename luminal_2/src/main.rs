#![allow(clippy::type_complexity)]

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
enum Input {
    Inp(usize), // An input to this scope
    Ref(usize), // A reference to an earlier variable in the scope
}

#[derive(Debug, PartialEq, Eq)]
struct Block {
    n: usize,                    // The size of the block
    inputs: Vec<(Input, usize)>, // Inputs and strides
    body: Vec<Body>,
}

#[derive(Debug, PartialEq, Eq)]
enum Body {
    Block(Block),
    Expr(Expr),
}

#[derive(Debug, PartialEq, Eq)]
enum Expr {
    Add(Input, Input),
    Mul(Input, Input),
    Exp(Input),
    Sin(Input),
    SumReduce(Input),
}

fn main() {
    // let b = vec![
    //     Body::Block(Block {
    //         n: 5,
    //         inputs: vec![(Input::Inp(0), 1)],
    //         body: vec![Body::Expr(Expr::Sin(Input::Inp(0)))],
    //     }),
    //     Body::Expr(Expr::SumReduce(Input::Ref(0))),
    //     Body::Block(Block {
    //         n: 5,
    //         inputs: vec![(Input::Ref(0), 1), (Input::Ref(1), 0)],
    //         body: vec![
    //             Body::Expr(Expr::Exp(Input::Inp(0))),
    //             Body::Expr(Expr::Mul(Input::Ref(0), Input::Inp(1))),
    //         ],
    //     }),
    // ];

    create_code(vec![
        (
            vec![Input::Inp(0), Input::Inp(1)],
            "mul".to_string(),
            vec![
                (4, vec![16, 0], 16, false), // Grid x
                (4, vec![0, 2], 2, false),   // Grid y
                (1, vec![0, 0], 0, false),   // Grid z (padding)
                (4, vec![2, 16], 4, false),  // Reduced
                (2, vec![8, 0], 2, false),   // Block x
                (2, vec![0, 1], 1, false),   // Block y
                (2, vec![1, 8], 1, false),   // Reduced
            ],
        ),
        (
            vec![Input::Ref(0)],
            "sum".to_string(),
            vec![
                (4, vec![16], 16, false), // Grid x
                (4, vec![2], 2, false),   // Grid y
                (1, vec![0], 0, false),   // Grid z (padding)
                (4, vec![4], 4, false),   // Reduced
                (2, vec![2], 2, false),   // Block x
                (2, vec![1], 1, false),   // Block y
                (2, vec![1], 1, true),    // Reduced
            ],
        ),
        (
            vec![Input::Ref(1)],
            "sum".to_string(),
            vec![
                (4, vec![16], 16, false), // Grid x
                (4, vec![2], 2, false),   // Grid y
                (1, vec![0], 0, false),   // Grid z
                (2, vec![2], 8, false),   // Block x
                (2, vec![1], 1, false),   // Block y
                (4, vec![4], 4, true),    // Reduced
            ],
        ),
    ]);
}

struct Kernel {
    code: String,
    grid: (usize, usize, usize),
    threadblock: (usize, usize, usize),
    inputs: Vec<usize>,  // global buffer indexes this kernel uses
    outputs: Vec<usize>, // buffer sizes this kernel creates
}

fn create_stacks(ir: Vec<Body>) -> Vec<(Vec<Input>, String, Vec<(usize, Vec<usize>, usize)>)> {
    todo!()
}

fn create_code(
    ir: Vec<(Vec<Input>, String, Vec<(usize, Vec<usize>, usize, bool)>)>,
) -> Vec<Kernel> {
    // Merge the stacks as much as possible
    let mut merged_ir = ir
        .iter()
        .cloned()
        .map(|v| {
            (
                v.clone(),
                vec![(
                    v.0,
                    v.1,
                    v.2.into_iter()
                        .map(|i| {
                            (
                                i.0,
                                i.1.into_iter()
                                    .map(|i| (i, Option::<char>::None))
                                    .collect::<Vec<_>>(),
                                i.2,
                                i.3,
                            )
                        })
                        .collect::<Vec<_>>(),
                    None,
                )],
            )
        })
        .collect::<Vec<_>>();
    let mut no_match = false;
    let mut loop_dim = 0;
    while !no_match {
        no_match = true;
        for l in 0..(merged_ir.len() - 1) {
            // Try to match ir[l] and ir[l + 1]
            let check_match =
                |a: &[(usize, Vec<usize>, usize, bool)], b: &[(usize, Vec<usize>, usize, bool)]| {
                    a.iter().filter(|l| !l.3).zip(b.iter()).all(
                        |((a_size, _, a_out, _), (b_size, b_inp, _, _))| {
                            *a_out == b_inp[0] && b_inp.len() == 1 && a_size == b_size
                        },
                    )
                };
            let mut matched = check_match(&merged_ir[l].0 .2, &merged_ir[l + 1].0 .2);
            if !matched {
                if let Some(orig_dim) = merged_ir[l + 1].0 .2.iter().position(|i| i.3) {
                    // Try to slide the reduce until we get a match
                    let mut dim = orig_dim;
                    for i in 0..merged_ir[l + 1].0 .2.len() {
                        let e = merged_ir[l + 1].0 .2.remove(dim);
                        merged_ir[l + 1].0 .2.insert(i, e);
                        if check_match(&merged_ir[l].0 .2, &merged_ir[l + 1].0 .2) {
                            // Found a match!
                            matched = true;
                            break;
                        }
                        dim = i;
                    }
                    if !matched {
                        // No match, restore to orig dim
                        let e = merged_ir[l + 1].0 .2.remove(dim);
                        merged_ir[l + 1].0 .2.insert(orig_dim, e);
                    }
                }
            }
            if matched {
                // Merge l and l + 1
                for (b, a) in merged_ir[l]
                    .0
                     .2
                    .iter()
                    .enumerate()
                    .filter(|(_, l)| !l.3)
                    .map(|(i, _)| i)
                    .enumerate()
                    .collect::<Vec<_>>()
                {
                    // Set output strides
                    merged_ir[l].0 .2[a].2 = merged_ir[l + 1].0 .2[b].2;
                    // Set reduction dims
                    merged_ir[l].0 .2[a].3 = merged_ir[l + 1].0 .2[b].3;
                }
                if merged_ir[l + 1].0 .1 == "sum" {
                    // Need to set reduce dim
                    let dim = merged_ir[l + 1].0 .2.iter().position(|s| s.3).unwrap();
                    for (_, _, m, _) in &mut merged_ir[l].1 {
                        for k in &mut m.iter_mut().filter(|s| !s.3).nth(dim).unwrap().1 {
                            k.1 = Some((b'a' + (loop_dim % 26) as u8) as char);
                        }
                    }
                    merged_ir[l + 1].1[0].3 = Some((b'a' + (loop_dim % 26) as u8) as char);
                    loop_dim += 1;
                }
                let mut t = merged_ir.remove(l + 1).1;
                merged_ir[l].1.append(&mut t);
                no_match = false;
                break;
            }
        }
    }

    for (stack, instructions) in merged_ir {
        // Compute grid and threadblock dim assignments
        let exec_dims = stack
            .2
            .iter()
            .filter(|i| !i.3)
            .map(|d| d.0)
            .collect::<Vec<_>>();
        let grid = exec_dims
            .iter()
            .take(3)
            .cloned()
            .chain((0..(3_usize.saturating_sub(exec_dims.len()))).map(|_| 0))
            .collect::<Vec<_>>();
        let threadblock = exec_dims
            .iter()
            .skip(3)
            .cloned()
            .chain((0..(3_usize.saturating_sub(exec_dims.len().saturating_sub(3)))).map(|_| 0))
            .collect::<Vec<_>>();

        // TODO: detect when we can use shared mem

        // Write kernels
        let mut kernel = "".to_string();
        for (i, (inputs, instruction, strides, loop_dim)) in instructions.into_iter().enumerate() {
            let var_name = (b'a' + (i % 26) as u8) as char;
            if instruction == "sum" {
                let (reduce_size, _, _, _) = strides.iter().find(|(_, _, _, b)| *b).unwrap();
                let loop_dim = loop_dim.unwrap();
                kernel = format!(
                    "float {var_name} = 0.0;
for (int loop_{loop_dim} = 0; loop_{loop_dim} < {reduce_size}; ++loop_{loop_dim}) {{
{}
	{var_name} += {};
}}",
                    kernel
                        .split("\n")
                        .map(|s| format!("\t{s}"))
                        .collect::<Vec<_>>()
                        .join("\n"),
                    get_inputs(&inputs, &strides)[0]
                );
            } else {
                kernel = format!(
                    "{kernel}
{var_name} = {instruction}({});",
                    get_inputs(&inputs, &strides).join(", ")
                )
            }
            kernel = kernel.trim().to_string();
        }
        println!("---");
        println!("Grid: {:?} Threadblock: {:?}", grid, threadblock);
        println!("{kernel}");
    }

    todo!()
}

fn get_inputs(
    inputs: &[Input],
    strides: &[(usize, Vec<(usize, Option<char>)>, usize, bool)],
) -> Vec<String> {
    let mut indexes = vec![vec![]; strides[0].1.len()];
    for (_, st, _, b) in strides {
        if !*b {
            for (i, s) in st.iter().enumerate() {
                indexes[i].push(*s);
            }
        }
    }
    inputs
        .into_iter()
        .zip(indexes.into_iter())
        .map(|(inp, stride)| {
            format!(
                "{}[{}]",
                match inp {
                    Input::Inp(i) => (b'A' + (i % 26) as u8) as char,
                    Input::Ref(i) => (b'a' + (i % 26) as u8) as char,
                },
                stride
                    .iter()
                    .enumerate()
                    .filter(|(_, (s, _))| *s != 0)
                    .map(|(i, (s, loop_ind))| if let Some(l) = loop_ind {
                        format!("loop_{l} * {s}")
                    } else {
                        format!("{} * {s}", index_to_dim(i))
                    })
                    .collect::<Vec<_>>()
                    .join(" + ")
            )
        })
        .collect()
}

fn index_to_dim(index: usize) -> &'static str {
    [
        "blockIdx.x",
        "blockIdx.y",
        "blockIdx.z",
        "threadIdx.x",
        "threadIdx.y",
        "threadIdx.z",
    ][index]
}
