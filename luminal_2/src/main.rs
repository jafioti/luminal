enum Input {
    Inp(usize), // An input to this scope
    Ref(usize), // A reference to an earlier variable in the scope
}

struct Block {
    n: usize,                    // The size of the block
    inputs: Vec<(Input, usize)>, // Inputs and strides
    body: Vec<Body>,
}

enum Body {
    Block(Block),
    Expr(Expr),
}

enum Expr {
    Add(Input, Input),
    Mul(Input, Input),
    Exp(Input),
    Sin(Input),
    SumReduce(Input),
}

fn main() {
    let b = vec![
        Body::Block(Block {
            n: 5,
            inputs: vec![(Input::Inp(0), 1)],
            body: vec![Body::Expr(Expr::Sin(Input::Inp(0)))],
        }),
        Body::Expr(Expr::SumReduce(Input::Ref(0))),
        Body::Block(Block {
            n: 5,
            inputs: vec![(Input::Ref(0), 1), (Input::Ref(1), 0)],
            body: vec![
                Body::Expr(Expr::Exp(Input::Inp(0))),
                Body::Expr(Expr::Mul(Input::Ref(0), Input::Inp(1))),
            ],
        }),
    ];
}

struct Kernel {
    code: String,
    grid: (usize, usize, usize),
    threadblock: (usize, usize, usize),
    inputs: Vec<usize>,  // global buffer indexes this kernel uses
    outputs: Vec<usize>, // buffer sizes this kernel creates
}

fn codegen(ir: Vec<Body>) -> Vec<Kernel> {
	// Loop merging

    vec![]
}
