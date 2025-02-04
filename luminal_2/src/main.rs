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
                (4, vec![16, 0], 16, false),
                (4, vec![0, 2], 2, false),
                (4, vec![2, 16], 4, false),
                (2, vec![8, 0], 2, false),
                (2, vec![0, 1], 1, false),
                (2, vec![1, 8], 1, false),
            ],
        ),
        (
            vec![Input::Ref(0)],
            "sum".to_string(),
            vec![
                (4, vec![16], 16, false),
                (4, vec![2], 2, false),
                (4, vec![4], 4, false),
                (2, vec![2], 2, false),
                (2, vec![1], 1, false),
                (2, vec![1], 1, true),
            ],
        ),
        (
            vec![Input::Ref(1)],
            "sum".to_string(),
            vec![
                (4, vec![16], 16, false),
                (4, vec![2], 2, false),
                (4, vec![4], 4, true),
                (2, vec![2], 8, false),
                (2, vec![1], 1, false),
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
        .map(|v| (v.clone(), vec![v]))
        .collect::<Vec<_>>();
    let mut no_match = false;
    while !no_match {
        no_match = true;
        for l in 0..(merged_ir.len() - 1) {
            // Try to match ir[l] and ir[l + 1]
            if merged_ir[l]
                .0
                 .2
                .iter()
                .filter(|l| !l.3)
                .zip(merged_ir[l + 1].0 .2.iter())
                .all(|((a_size, _, a_out, _), (b_size, b_inp, _, _))| {
                    *a_out == b_inp[0] && b_inp.len() == 1 && a_size == b_size
                })
            {
                // Merge l and l + 1
                for (b, a) in merged_ir[l]
                    .0 // Set reduction dims and output strides
                     .2
                    .iter()
                    .enumerate()
                    .filter(|(_, l)| !l.3)
                    .map(|(i, _)| i)
                    .enumerate()
                    .collect::<Vec<_>>()
                {
                    merged_ir[l].0 .2[a].2 = merged_ir[l + 1].0 .2[b].2;
                    merged_ir[l].0 .2[a].3 = merged_ir[l + 1].0 .2[b].3;
                }
                let mut t = merged_ir.remove(l + 1).1;
                merged_ir[l].1.append(&mut t);
                no_match = false;
                break;
            }
        }
    }
    for (_, instructions) in &merged_ir {
        println!("---");
        for (inputs, op, stack) in instructions {
            println!("{inputs:?} {op} -> ");
        }
        println!("---");
    }
    // println!("remaining: {:?}", ir);
    // Split into kernels

    // Compute grid and threadblock dim assignments

    // Write kernels

    todo!()
}

fn codegen(ir: Vec<Body>) -> Vec<Kernel> {
    // Loop merging

    vec![]
}
