use crate::{GPUArch, codegen::codegen, run::run_graph, symbolic::expression_cleanup};
use petgraph::{Directed, graph::NodeIndex, prelude::StableGraph};
use std::collections::HashMap;

use crate::{
    GraphTerm,
    symbolic::{Expression, Term},
};

fn unary(
    a: NodeIndex,
    term: GraphTerm,
    graph: &mut StableGraph<GraphTerm, u8, Directed>,
) -> NodeIndex {
    let tmp = graph.add_node(term);
    graph.add_edge(a, tmp, 0);
    tmp
}

fn binary(
    a: NodeIndex,
    b: NodeIndex,
    term: GraphTerm,
    graph: &mut StableGraph<GraphTerm, u8, Directed>,
) -> NodeIndex {
    let tmp = graph.add_node(term);
    graph.add_edge(a, tmp, 0);
    graph.add_edge(b, tmp, 0);
    tmp
}

fn loop_in(
    node: NodeIndex,
    range: impl Into<Expression>,
    stride: impl Into<Expression>,
    graph: &mut StableGraph<GraphTerm, u8, Directed>,
) -> NodeIndex {
    unary(
        node,
        GraphTerm::LoopIn {
            range: range.into(),
            stride: stride.into(),
        },
        graph,
    )
}

fn loop_out(
    node: NodeIndex,
    range: impl Into<Expression>,
    stride: impl Into<Expression>,
    graph: &mut StableGraph<GraphTerm, u8, Directed>,
) -> NodeIndex {
    unary(
        node,
        GraphTerm::LoopOut {
            range: range.into(),
            stride: stride.into(),
        },
        graph,
    )
}

fn pad_in(
    mut node: NodeIndex,
    graph: &mut StableGraph<GraphTerm, u8, Directed>,
    levels: usize,
) -> NodeIndex {
    for _ in 0..levels {
        node = loop_in(node, 1, 0, graph);
    }
    node
}

fn pad_out(
    mut node: NodeIndex,
    graph: &mut StableGraph<GraphTerm, u8, Directed>,
    levels: usize,
) -> NodeIndex {
    for _ in 0..levels {
        node = loop_out(node, 1, 0, graph);
    }
    node
}

pub fn make_complex_kernel() -> (StableGraph<GraphTerm, u8, Directed>, NodeIndex) {
    let mut graph = StableGraph::new();
    let a = graph.add_node(GraphTerm::GMEM {
        label: Some("A".to_string()),
    });
    let in_a = loop_in(
        loop_in(
            pad_in(a, &mut graph, 6),
            50,
            Expression::from('z') * 5,
            &mut graph,
        ),
        5,
        'z',
        &mut graph,
    );

    // Acc
    let slin0 = graph.add_node(GraphTerm::GMEM {
        label: Some("acc".to_string()),
    });
    let in_exp_acc = loop_in(
        loop_in(pad_in(slin0, &mut graph, 6), 50, 'z', &mut graph),
        5,
        Term::Acc('z'),
        &mut graph,
    );

    // Exp-acc
    let exp = unary(in_a, GraphTerm::Exp, &mut graph);
    let add_acc = binary(exp, in_exp_acc, GraphTerm::Add, &mut graph);
    let add_acc_out = loop_out(add_acc, 5, Term::Acc('z'), &mut graph);

    // Sin
    let sin = unary(add_acc_out, GraphTerm::Sin, &mut graph);

    // Other input
    let b = graph.add_node(GraphTerm::GMEM {
        label: Some("B".to_string()),
    });
    let in_b = loop_in(pad_in(b, &mut graph, 6), 50, 'z', &mut graph);

    // Mul
    let mul = binary(sin, in_b, GraphTerm::Mul, &mut graph);

    let mul_out = loop_out(mul, 50, 'z', &mut graph);

    let mut out = pad_out(mul_out, &mut graph, 6);
    out = unary(out, GraphTerm::GMEM { label: None }, &mut graph);
    (graph, out)
}

#[test]
fn test_sum_reduce() {
    let mut graph = StableGraph::new();
    let mut a = graph.add_node(GraphTerm::GMEM {
        label: Some("A".to_string()),
    });
    a = pad_in(a, &mut graph, 6);
    a = loop_in(a, 5, 'z', &mut graph);

    let mut acc = graph.add_node(GraphTerm::NewAcc {
        starting_value: "0.0".to_string(),
    });
    acc = loop_in(acc, 5, Term::Acc('z'), &mut graph);

    let mut out = binary(a, acc, GraphTerm::Add, &mut graph);
    out = loop_out(out, 5, Term::Acc('z'), &mut graph);
    out = pad_out(out, &mut graph, 6);
    out = unary(out, GraphTerm::GMEM { label: None }, &mut graph);

    let kernels = codegen(graph, out, GPUArch::Metal(HashMap::new())).unwrap();
    let input = vec![0., 1., 2., 3., 4.];
    let outputs = run_graph(&[input], &kernels).0;
    assert_eq!(outputs[0], vec![10.0]);
    expression_cleanup();
}

#[test]
fn test_matmul() {
    let mut graph = StableGraph::new();
    let (m, k, n) = (3, 4, 5);

    let mut a = graph.add_node(GraphTerm::GMEM {
        label: Some("A".to_string()),
    });
    a = loop_in(a, m, Expression::from('z') * k, &mut graph);
    a = loop_in(a, n, 0, &mut graph);
    a = pad_in(a, &mut graph, 4);
    a = loop_in(a, k, 'z', &mut graph);

    let mut b = graph.add_node(GraphTerm::GMEM {
        label: Some("B".to_string()),
    });
    b = loop_in(b, m, 0, &mut graph);
    b = loop_in(b, n, 'z', &mut graph);
    b = pad_in(b, &mut graph, 4);
    b = loop_in(b, k, Expression::from('z') * n, &mut graph);

    let mut acc = graph.add_node(GraphTerm::NewAcc {
        starting_value: "0.0".to_string(),
    });
    acc = loop_in(acc, k, Term::Acc('a'), &mut graph);

    let mut out = binary(
        binary(a, b, GraphTerm::Mul, &mut graph),
        acc,
        GraphTerm::Add,
        &mut graph,
    );

    out = loop_out(out, k, Term::Acc('a'), &mut graph);
    out = pad_out(out, &mut graph, 4);
    out = loop_out(out, n, 'z', &mut graph);
    out = loop_out(out, m, Expression::from('z') * n, &mut graph);
    out = unary(out, GraphTerm::GMEM { label: None }, &mut graph);
    let kernels = codegen(graph, out, GPUArch::Metal(HashMap::new())).unwrap();
    let a = vec![
        [1.5410, -0.2934, -2.1788, 0.5684],
        [-1.0845, -1.3986, 0.4033, 0.8380],
        [-0.7193, -0.4033, -0.5966, 0.1820],
    ]
    .into_flattened();
    let b = vec![
        [0.4681, -0.1577, 1.4437, 0.2660, -0.1740],
        [-0.6787, 0.9383, 0.4889, -0.6731, 0.8728],
        [1.0554, 0.1778, -0.5181, -0.3067, -1.5810],
        [1.7066, -0.4462, 0.7440, 1.5210, 3.4105],
    ]
    .into_flattened();
    let outputs = run_graph(&[a, b], &kernels).0.pop().unwrap();
    let pt_output = vec![
        [-0.4088, -1.1595, 3.6329, 2.1403, 4.8591],
        [2.2975, -1.4434, -1.8349, 1.8038, 1.1883],
        [-0.3819, -0.4523, -0.7910, 0.5400, 1.3372],
    ]
    .into_flattened();
    for (a, b) in outputs.into_iter().zip(pt_output) {
        assert!((a - b).abs() < 1e-3);
    }
    expression_cleanup();
}

#[test]
fn test_tiled_matmul_basic() {
    let (m, k, n) = (16, 8, 32);
    let mut graph = StableGraph::new();
    let mut a = graph.add_node(GraphTerm::GMEM {
        label: Some("A".to_string()),
    });
    a = loop_in(
        a,
        Expression::from(m) / 8,
        Expression::from('z') * k * 8,
        &mut graph,
    );
    a = loop_in(a, Expression::from(n) / 8, 0, &mut graph);
    a = pad_in(a, &mut graph, 1);
    a = loop_in(a, 8, Expression::from('z') * k, &mut graph);
    a = loop_in(a, 8, 0, &mut graph);
    a = pad_in(a, &mut graph, 1);
    a = loop_in(
        a,
        Expression::from(k) / 8,
        Expression::from('z') * 8,
        &mut graph,
    );
    a = loop_in(a, 8, 'z', &mut graph);

    let mut b = graph.add_node(GraphTerm::GMEM {
        label: Some("B".to_string()),
    });
    b = loop_in(b, Expression::from(m) / 8, 0, &mut graph);
    b = loop_in(
        b,
        Expression::from(n) / 8,
        Expression::from('z') * 8,
        &mut graph,
    );
    b = pad_in(b, &mut graph, 1);
    b = loop_in(b, 8, 0, &mut graph);
    b = loop_in(b, 8, 'z', &mut graph);
    b = pad_in(b, &mut graph, 1);
    b = loop_in(
        b,
        Expression::from(k) / 8,
        Expression::from('z') * n * 8,
        &mut graph,
    );
    b = loop_in(b, 8, Expression::from('z') * n, &mut graph);

    let mut acc = graph.add_node(GraphTerm::NewAcc {
        starting_value: "0.0".to_string(),
    });
    acc = loop_in(acc, Expression::from(k) / 8, Term::Acc('a'), &mut graph);
    acc = loop_in(acc, 8, Term::Acc('a'), &mut graph);

    let mut out = binary(
        binary(a, b, GraphTerm::Mul, &mut graph),
        acc,
        GraphTerm::Add,
        &mut graph,
    );

    out = loop_out(out, Expression::from(k) / 8, Term::Acc('a'), &mut graph);
    out = loop_out(out, 8, Term::Acc('a'), &mut graph);
    out = pad_out(out, &mut graph, 1);
    out = loop_out(out, 8, 'z', &mut graph);
    out = loop_out(out, 8, Expression::from('z') * n, &mut graph);
    out = pad_out(out, &mut graph, 1);
    out = loop_out(
        out,
        Expression::from(n) / 8,
        Expression::from('z') * 8,
        &mut graph,
    );
    out = loop_out(
        out,
        Expression::from(m) / 8,
        Expression::from('z') * n * 8,
        &mut graph,
    );
    out = unary(out, GraphTerm::GMEM { label: None }, &mut graph);
    let kernels = codegen(graph, out, GPUArch::Metal(HashMap::new())).unwrap();
    let a = vec![
        [
            -1.1258e+00,
            -1.1524e+00,
            -2.5058e-01,
            -4.3388e-01,
            8.4871e-01,
            6.9201e-01,
            -3.1601e-01,
            -2.1152e+00,
        ],
        [
            3.2227e-01,
            -1.2633e+00,
            3.4998e-01,
            3.0813e-01,
            1.1984e-01,
            1.2377e+00,
            1.1168e+00,
            -2.4728e-01,
        ],
        [
            -1.3527e+00,
            -1.6959e+00,
            5.6665e-01,
            7.9351e-01,
            5.9884e-01,
            -1.5551e+00,
            -3.4136e-01,
            1.8530e+00,
        ],
        [
            7.5019e-01,
            -5.8550e-01,
            -1.7340e-01,
            1.8348e-01,
            1.3894e+00,
            1.5863e+00,
            9.4630e-01,
            -8.4368e-01,
        ],
        [
            -6.1358e-01,
            3.1593e-02,
            -4.9268e-01,
            2.4841e-01,
            4.3970e-01,
            1.1241e-01,
            6.4079e-01,
            4.4116e-01,
        ],
        [
            -1.0231e-01,
            7.9244e-01,
            -2.8967e-01,
            5.2507e-02,
            5.2286e-01,
            2.3022e+00,
            -1.4689e+00,
            -1.5867e+00,
        ],
        [
            -6.7309e-01,
            8.7283e-01,
            1.0554e+00,
            1.7784e-01,
            -2.3034e-01,
            -3.9175e-01,
            5.4329e-01,
            -3.9516e-01,
        ],
        [
            -4.4622e-01,
            7.4402e-01,
            1.5210e+00,
            3.4105e+00,
            -1.5312e+00,
            -1.2341e+00,
            1.8197e+00,
            -5.5153e-01,
        ],
        [
            -5.6925e-01,
            9.1997e-01,
            1.1108e+00,
            1.2899e+00,
            -1.4782e+00,
            2.5672e+00,
            -4.7312e-01,
            3.3555e-01,
        ],
        [
            -1.6293e+00,
            -5.4974e-01,
            -4.7983e-01,
            -4.9968e-01,
            -1.0670e+00,
            1.1149e+00,
            -1.4067e-01,
            8.0575e-01,
        ],
        [
            -9.3348e-02,
            6.8705e-01,
            -8.3832e-01,
            8.9182e-04,
            8.4189e-01,
            -4.0003e-01,
            1.0395e+00,
            3.5815e-01,
        ],
        [
            -2.4600e-01,
            2.3025e+00,
            -1.8817e+00,
            -4.9727e-02,
            -1.0450e+00,
            -9.5650e-01,
            3.3532e-02,
            7.1009e-01,
        ],
        [
            1.6459e+00,
            -1.3602e+00,
            3.4457e-01,
            5.1987e-01,
            -2.6133e+00,
            -1.6965e+00,
            -2.2824e-01,
            2.7995e-01,
        ],
        [
            2.4693e-01,
            7.6887e-02,
            3.3801e-01,
            4.5440e-01,
            4.5694e-01,
            -8.6537e-01,
            7.8131e-01,
            -9.2679e-01,
        ],
        [
            -2.1883e-01,
            -2.4351e+00,
            -7.2915e-02,
            -3.3987e-02,
            9.6252e-01,
            3.4917e-01,
            -9.2146e-01,
            -5.6195e-02,
        ],
        [
            -6.2270e-01,
            -4.6372e-01,
            1.9218e+00,
            -4.0255e-01,
            1.2390e-01,
            1.1648e+00,
            9.2337e-01,
            1.3873e+00,
        ],
    ]
    .into_flattened();
    let b = vec![
        [
            -0.8834, -0.4189, -0.8048, 0.5656, 0.6104, 0.4669, 1.9507, -1.0631, -0.0773, 0.1164,
            -0.5940, -1.2439, -0.1021, -1.0335, -0.3126, 0.2458, -0.2596, 0.1183, 0.2440, 1.1646,
            0.2886, 0.3866, -0.2011, -0.1179, 0.1922, -0.7722, -1.9003, 0.1307, -0.7043, 0.3147,
            0.1574, 0.3854,
        ],
        [
            0.9671, -0.9911, 0.3016, -0.1073, 0.9985, -0.4987, 0.7611, 0.6183, 0.3140, 0.2133,
            -0.1201, 0.3605, -0.3140, -1.0787, 0.2408, -1.3962, -0.0661, -0.3584, -1.5616, -0.3546,
            1.0811, 0.1315, 1.5735, 0.7814, -1.0787, -0.7209, 1.4708, 0.2756, 0.6668, -0.9944,
            -1.1894, -1.1959,
        ],
        [
            -0.5596, 0.5335, 0.4069, 0.3946, 0.1715, 0.8760, -0.2871, 1.0216, -0.0744, -1.0922,
            0.3920, 0.5945, 0.6623, -1.2063, 0.6074, -0.5472, 1.1711, 0.0975, 0.9634, 0.8403,
            -1.2537, 0.9868, -0.4947, -1.2830, 0.9552, 1.2836, -0.6659, 0.5651, 0.2877, -0.0334,
            -1.0619, -0.1144,
        ],
        [
            -0.3433, 1.5713, 0.1916, 0.3799, -0.1448, 0.6376, -0.2813, -1.3299, -0.1420, -0.5341,
            -0.5234, 0.8615, -0.8870, 0.8388, 1.1529, -1.7611, -1.4777, -1.7557, 0.0762, -1.0786,
            1.4403, -0.1106, 0.5769, -0.1692, -0.0640, 1.0384, 0.9068, -0.4755, -0.8707, 0.1447,
            1.9029, 0.3904,
        ],
        [
            -0.0394, -0.8015, -0.4955, -0.3615, 0.5851, -1.1560, -0.1434, -0.1947, -0.0856, 1.3945,
            0.5969, -0.4828, -0.3661, -1.3271, 1.6953, 2.0655, -0.2340, 0.7073, 0.5800, 0.2683,
            -2.0589, 0.5340, -0.5354, -0.8637, -0.0235, 1.1717, 0.3987, -0.1987, -1.1559, -0.3167,
            0.9403, -1.1470,
        ],
        [
            0.5588, 0.7918, -0.1847, -0.7318, -0.0807, -0.9801, 0.0605, -0.4890, -0.8137, 0.8200,
            -0.6332, 1.2948, 1.4628, -0.6204, 0.9884, -0.4322, -0.6232, -0.2162, -0.4887, 0.7870,
            0.1076, -1.0715, -0.1166, -1.0170, -1.1980, 0.4784, -1.2295, -1.3700, 1.5435, -0.0332,
            -0.4186, -0.2556,
        ],
        [
            -0.1292, -0.0546, 0.4083, 1.1264, 1.9351, 1.0077, 1.0046, -0.4335, -1.2426, 1.2846,
            0.2438, 0.5304, -0.0145, -2.2357, 1.4660, -1.2191, 0.6442, 3.9300, -0.1244, 0.2953,
            0.3827, -0.5497, -0.9940, 1.3459, 1.9457, -1.2904, -2.3495, -2.0689, 0.9094, -0.6946,
            1.9595, -1.1038,
        ],
        [
            0.5411, 1.5390, 1.0860, 1.2464, 0.1151, 1.6193, 0.4637, 1.3007, 0.8732, 0.0651, 0.7732,
            -0.9701, -0.8877, -0.3183, -0.3344, 0.4543, 0.4990, 0.8780, 0.3894, 1.4625, 0.4795,
            -0.5334, -0.0347, 0.6573, -0.3112, -0.5620, -0.4835, -1.2721, -0.1740, 0.5541, -0.1817,
            -0.2345,
        ],
    ]
    .into_flattened();
    let outputs = run_graph(&[a, b], &kernels).0.pop().unwrap();
    let pt_output = vec![
        [
            -5.8128e-01,
            -2.5720e+00,
            -2.6012e+00,
            -4.5824e+00,
            -2.2321e+00,
            -5.8501e+00,
            -4.2573e+00,
            -2.3126e+00,
            -2.2847e+00,
            1.3359e+00,
            -7.0825e-01,
            2.8330e+00,
            3.2795e+00,
            2.1692e+00,
            1.7889e+00,
            3.1117e+00,
            -1.1726e+00,
            -1.6314e+00,
            6.2006e-01,
            -3.0596e+00,
            -4.6896e+00,
            2.2768e-01,
            -1.8609e+00,
            -3.6252e+00,
            9.5380e-03,
            3.8499e+00,
            1.4707e+00,
            1.8278e+00,
            4.9804e-01,
            -5.0718e-01,
            9.0723e-01,
            4.9808e-01,
        ],
        [
            -1.3994e+00,
            2.2303e+00,
            -5.3943e-01,
            5.7371e-01,
            1.0536e+00,
            6.5700e-01,
            5.4495e-01,
            -2.6102e+00,
            -3.1125e+00,
            1.8216e+00,
            -6.9492e-01,
            1.9941e+00,
            2.2922e+00,
            -2.4791e+00,
            3.3092e+00,
            -6.5214e-01,
            -2.4888e-01,
            3.9730e+00,
            1.6415e+00,
            1.7594e+00,
            -1.0725e+00,
            -1.4744e+00,
            -3.3582e+00,
            -1.5480e+00,
            2.5036e+00,
            8.6159e-01,
            -6.4024e+00,
            -3.9702e+00,
            1.5935e+00,
            3.9879e-01,
            3.5957e+00,
            8.6781e-02,
        ],
        [
            -8.8058e-01,
            4.9557e+00,
            2.8233e+00,
            2.7886e+00,
            -2.5081e+00,
            4.7050e+00,
            -3.9789e+00,
            3.1151e+00,
            2.6736e+00,
            -2.3199e+00,
            3.5056e+00,
            -2.1895e+00,
            -3.7918e+00,
            3.5530e+00,
            -3.6840e-01,
            3.4949e+00,
            1.4881e+00,
            1.5502e-01,
            4.7961e+00,
            1.9255e-01,
            -2.4336e+00,
            9.1083e-01,
            -2.0833e+00,
            -2.0420e-01,
            2.6679e+00,
            3.1751e+00,
            2.4753e+00,
            -3.4078e-01,
            -4.4315e+00,
            2.4826e+00,
            2.9210e+00,
            1.4049e+00,
        ],
        [
            -9.4201e-01,
            -7.4579e-01,
            -2.3271e+00,
            -1.1604e+00,
            2.2360e+00,
            -2.9661e+00,
            1.4722e+00,
            -4.1345e+00,
            -3.5774e+00,
            4.4527e+00,
            -1.1361e+00,
            1.6142e+00,
            2.3768e+00,
            -4.4558e+00,
            5.3234e+00,
            1.4209e+00,
            -1.7553e+00,
            3.5775e+00,
            5.2862e-01,
            1.4044e+00,
            -2.6672e+00,
            -1.0063e+00,
            -2.7208e+00,
            -2.4486e+00,
            7.6908e-01,
            1.4507e+00,
            -5.2168e+00,
            -3.5825e+00,
            7.2155e-01,
            -7.6680e-01,
            3.9975e+00,
            -1.7649e+00,
        ],
        [
            9.6443e-01,
            7.3376e-01,
            8.5263e-01,
            5.7995e-01,
            1.0755e+00,
            1.6618e-01,
            -3.0919e-01,
            -6.4255e-03,
            -4.8137e-01,
            1.8979e+00,
            7.2612e-01,
            5.4085e-01,
            -8.9135e-01,
            -8.2353e-01,
            1.8350e+00,
            -8.3975e-02,
            -3.2683e-01,
            2.6242e+00,
            -3.6255e-01,
            -3.6685e-01,
            3.9606e-01,
            -1.2199e+00,
            -3.4069e-01,
            1.3455e+00,
            3.2595e-01,
            -4.2925e-01,
            8.4076e-02,
            -2.5963e+00,
            2.6639e-01,
            -5.1575e-01,
            2.4036e+00,
            -1.4647e+00,
        ],
        [
            1.5980e+00,
            -1.7725e+00,
            -2.7938e+00,
            -5.7431e+00,
            -2.2332e+00,
            -7.5735e+00,
            -1.6752e+00,
            -2.4216e+00,
            -1.2075e+00,
            1.0722e+00,
            -2.9060e+00,
            3.7745e+00,
            4.1293e+00,
            1.3113e+00,
            1.6464e+00,
            8.9310e-02,
            -3.7378e+00,
            -7.7104e+00,
            -2.7944e+00,
            -1.5026e+00,
            -8.8581e-01,
            -7.6075e-01,
            2.4077e+00,
            -4.8187e+00,
            -6.2889e+00,
            3.6916e+00,
            3.1966e+00,
            1.8158e+00,
            2.3608e+00,
            -9.0384e-01,
            -3.6132e+00,
            -1.2820e-01,
        ],
        [
            2.9318e-01,
            -5.0400e-01,
            1.2476e+00,
            4.9899e-01,
            1.5186e+00,
            8.4619e-01,
            -6.2974e-01,
            1.5838e+00,
            -4.5928e-01,
            -1.1101e+00,
            5.5312e-01,
            2.2080e+00,
            1.8995e-01,
            -1.9099e+00,
            1.4177e+00,
            -3.4230e+00,
            1.5410e+00,
            1.1082e+00,
            -6.6059e-01,
            -1.1860e+00,
            1.3297e-01,
            1.0852e+00,
            7.3198e-01,
            4.4612e-01,
            1.5806e+00,
            4.9353e-01,
            1.3258e+00,
            6.2565e-01,
            1.4292e+00,
            -1.5996e+00,
            -8.4255e-01,
            -1.4973e+00,
        ],
        [
            -2.0714e+00,
            4.9217e+00,
            2.9867e+00,
            4.3826e+00,
            2.8991e+00,
            6.8479e+00,
            1.7166e-02,
            -2.6519e+00,
            -1.9368e+00,
            -4.2218e+00,
            -1.1284e+00,
            5.3073e+00,
            -2.9874e+00,
            -4.1068e-01,
            4.2111e+00,
            -1.3085e+01,
            -1.1675e+00,
            -3.0773e-01,
            -2.7193e-01,
            -4.8352e+00,
            7.1328e+00,
            8.4760e-01,
            1.6495e+00,
            2.7698e+00,
            5.5729e+00,
            8.7938e-01,
            9.2042e-01,
            -1.6835e+00,
            -1.0582e-01,
            -1.4812e+00,
            6.6622e+00,
            2.8793e-01,
        ],
        [
            2.0636e+00,
            5.7056e+00,
            1.8644e+00,
            -9.5123e-01,
            -1.3740e+00,
            3.3033e-01,
            -1.0444e+00,
            2.6764e-01,
            -1.0145e+00,
            -2.3144e+00,
            -2.3756e+00,
            6.2726e+00,
            3.3664e+00,
            6.5767e-01,
            1.7870e+00,
            -7.7373e+00,
            -1.9098e+00,
            -5.7188e+00,
            -2.3295e+00,
            5.2769e-01,
            4.5951e+00,
            -2.6046e+00,
            2.7073e+00,
            -2.6078e+00,
            -4.1890e+00,
            2.4598e+00,
            6.8436e-02,
            -2.4779e+00,
            5.3934e+00,
            -4.6922e-02,
            -3.3615e+00,
            5.3964e-01,
        ],
        [
            2.4670e+00,
            3.1719e+00,
            1.9950e+00,
            -8.2606e-01,
            -2.4471e+00,
            7.8235e-02,
            -2.8657e+00,
            2.3382e+00,
            1.2244e-01,
            -2.1784e-01,
            3.5314e-01,
            2.2153e+00,
            1.7728e+00,
            3.2189e+00,
            -1.6731e+00,
            -6.3854e-01,
            5.0206e-01,
            -6.4953e-03,
            -8.7179e-01,
            1.6121e-01,
            1.4667e+00,
            -3.2373e+00,
            -3.5226e-02,
            5.9067e-01,
            -1.9816e+00,
            -4.6841e-01,
            2.9880e-01,
            -2.4475e+00,
            3.7642e+00,
            8.2262e-01,
            -1.9359e+00,
            7.9457e-01,
        ],
        [
            1.0186e+00,
            -1.5847e+00,
            4.1153e-01,
            1.1486e+00,
            3.0626e+00,
            -7.3825e-02,
            1.6467e+00,
            -2.8672e-01,
            -4.4017e-01,
            3.2554e+00,
            9.3001e-01,
            -8.5447e-01,
            -1.9886e+00,
            -2.9396e+00,
            2.1224e+00,
            2.8225e-01,
            -1.0357e-01,
            4.7410e+00,
            -1.2092e+00,
            -3.1591e-01,
            5.6110e-01,
            -6.5737e-01,
            6.5304e-02,
            2.9375e+00,
            8.1057e-01,
            -2.2459e+00,
            -4.0938e-02,
            -2.5224e+00,
            -4.2577e-01,
            -1.4614e+00,
            2.9909e+00,
            -2.8561e+00,
        ],
        [
            3.4009e+00,
            -2.0897e+00,
            1.5966e+00,
            8.5292e-01,
            1.4456e+00,
            3.8577e-01,
            2.2817e+00,
            1.4092e+00,
            2.3353e+00,
            3.9210e-01,
            -3.0282e-01,
            -1.4305e+00,
            -3.5475e+00,
            1.6779e+00,
            -3.4742e+00,
            -3.6215e+00,
            -1.0021e+00,
            -7.2744e-01,
            -5.3384e+00,
            -2.6153e+00,
            7.1075e+00,
            -1.5742e+00,
            5.1878e+00,
            6.6381e+00,
            -3.3105e+00,
            -6.0612e+00,
            5.3992e+00,
            1.0816e-01,
            8.4892e-01,
            -1.5786e+00,
            -1.5192e+00,
            -1.4131e+00,
        ],
        [
            -3.8048e+00,
            2.8538e+00,
            3.2410e-01,
            3.6884e+00,
            -2.1714e+00,
            6.9871e+00,
            2.1027e+00,
            -1.1285e+00,
            1.4783e+00,
            -6.0630e+00,
            -1.2762e+00,
            -3.2122e+00,
            -1.7439e+00,
            4.7283e+00,
            -6.5688e+00,
            -3.0597e+00,
            9.5925e-01,
            -2.3297e+00,
            2.3477e+00,
            4.3379e-01,
            4.5664e+00,
            1.1383e+00,
            -5.2752e-01,
            2.0721e+00,
            3.6419e+00,
            -3.0446e+00,
            -3.4414e+00,
            2.7473e+00,
            -2.2737e+00,
            3.1318e+00,
            2.5493e-01,
            6.0417e+00,
        ],
        [
            -1.5930e+00,
            -1.8057e+00,
            -7.0506e-01,
            6.3038e-01,
            1.9621e+00,
            2.6923e-01,
            5.5266e-01,
            -1.6840e+00,
            -1.1997e+00,
            3.0413e-01,
            3.3288e-02,
            2.8540e-01,
            -8.5038e-01,
            -1.8860e+00,
            2.0452e+00,
            -1.0875e+00,
            1.2848e-01,
            2.0040e+00,
            5.3020e-01,
            -1.6289e+00,
            -7.9422e-01,
            1.6250e+00,
            -7.2197e-01,
            4.4824e-01,
            3.0929e+00,
            2.9367e-01,
            -3.1056e-01,
            6.8576e-01,
            -1.4132e+00,
            -1.1165e+00,
            2.9444e+00,
            -8.0610e-01,
        ],
        [
            -1.8634e+00,
            1.8816e+00,
            -1.5732e+00,
            -1.6156e+00,
            -3.8270e+00,
            -1.4477e+00,
            -3.3184e+00,
            -1.3340e+00,
            -8.1063e-03,
            -5.9155e-03,
            4.9688e-01,
            -1.1250e+00,
            9.9056e-01,
            3.4964e+00,
            4.3305e-02,
            6.3808e+00,
            -8.8175e-01,
            -2.1661e+00,
            4.1567e+00,
            7.6281e-01,
            -4.9769e+00,
            2.0345e-01,
            -3.4093e+00,
            -4.2413e+00,
            3.0079e-01,
            4.3110e+00,
            -1.0013e+00,
            5.8339e-01,
            -2.8628e+00,
            2.6426e+00,
            1.8380e+00,
            2.6601e+00,
        ],
        [
            4.4171e-01,
            4.0207e+00,
            2.6733e+00,
            2.1750e+00,
            1.4698e+00,
            3.2596e+00,
            -3.8247e-01,
            3.6845e+00,
            -1.0777e+00,
            3.4896e-01,
            2.0239e+00,
            1.9953e+00,
            2.2526e+00,
            -4.9052e+00,
            3.0374e+00,
            -5.9108e-01,
            3.5699e+00,
            5.6693e+00,
            2.3210e+00,
            4.7399e+00,
            -2.7813e+00,
            -7.9015e-01,
            -2.9555e+00,
            -1.8235e+00,
            2.2085e+00,
            1.5952e+00,
            -5.3663e+00,
            -4.2273e+00,
            3.2858e+00,
            1.9215e-01,
            -1.1670e+00,
            -1.8468e+00,
        ],
    ]
    .into_flattened();
    for (a, b) in outputs.into_iter().zip(pt_output) {
        assert!((a - b).abs() < 1e-3);
    }
    expression_cleanup();
}

#[test]
fn test_tiled_matmul_smem() {
    let (m, k, n) = (16, 8, 32);
    let mut graph = StableGraph::new();
    let mut a = graph.add_node(GraphTerm::GMEM {
        label: Some("A".to_string()),
    });
    a = loop_in(
        a,
        Expression::from(m) / 8,
        Expression::from('z') * k * 8,
        &mut graph,
    );
    a = loop_in(a, Expression::from(n) / 8, 0, &mut graph);
    a = pad_in(a, &mut graph, 1);
    a = loop_in(a, 8, Expression::from('z') * k, &mut graph);
    a = loop_in(a, 8, 'z', &mut graph);
    a = pad_in(a, &mut graph, 1);
    a = loop_in(
        a,
        Expression::from(k) / 8,
        Expression::from('z') * 8,
        &mut graph,
    );

    let smem_a_orig = graph.add_node(GraphTerm::SMEM);
    let mut smem_a = loop_in(smem_a_orig, 8, Expression::from('z') * 8, &mut graph);
    smem_a = loop_in(smem_a, 8, 'z', &mut graph);
    smem_a = pad_in(smem_a, &mut graph, 1);
    smem_a = loop_in(smem_a, k / 8, 0, &mut graph);
    smem_a = binary(smem_a, a, GraphTerm::SMEMLoad, &mut graph);

    let mut smem_read_a = loop_in(smem_a_orig, 8, Expression::from('z') * 8, &mut graph);
    smem_read_a = loop_in(smem_read_a, 8, 0, &mut graph);
    smem_read_a = pad_in(smem_read_a, &mut graph, 1);
    smem_read_a = loop_in(smem_read_a, k / 8, 0, &mut graph);
    a = binary(smem_read_a, smem_a, GraphTerm::SMEMRead, &mut graph);
    a = loop_in(a, 8, 'z', &mut graph);

    let mut b = graph.add_node(GraphTerm::GMEM {
        label: Some("B".to_string()),
    });
    b = loop_in(b, Expression::from(m) / 8, 0, &mut graph);
    b = loop_in(
        b,
        Expression::from(n) / 8,
        Expression::from('z') * 8,
        &mut graph,
    );
    b = pad_in(b, &mut graph, 1);
    b = loop_in(b, 8, Expression::from('z') * n, &mut graph);
    b = loop_in(b, 8, 'z', &mut graph);
    b = pad_in(b, &mut graph, 1);
    b = loop_in(
        b,
        Expression::from(k) / 8,
        Expression::from('z') * n * 8,
        &mut graph,
    );
    let smem_b_orig = graph.add_node(GraphTerm::SMEM);
    let mut smem_b = loop_in(smem_b_orig, 8, Expression::from('z') * 8, &mut graph);
    smem_b = loop_in(smem_b, 8, 'z', &mut graph);
    smem_b = pad_in(smem_b, &mut graph, 1);
    smem_b = loop_in(smem_b, k / 8, 0, &mut graph);
    smem_b = binary(smem_b, b, GraphTerm::SMEMLoad, &mut graph);

    let mut smem_read_b = loop_in(smem_b_orig, 8, 0, &mut graph);
    smem_read_b = loop_in(smem_read_b, 8, Expression::from('z'), &mut graph);
    smem_read_b = pad_in(smem_read_b, &mut graph, 1);
    smem_read_b = loop_in(smem_read_b, k / 8, 0, &mut graph);
    b = binary(smem_read_b, smem_b, GraphTerm::SMEMRead, &mut graph);
    b = loop_in(b, 8, Expression::from('z') * 8, &mut graph);

    let mut acc = graph.add_node(GraphTerm::NewAcc {
        starting_value: "0.0".to_string(),
    });
    acc = loop_in(acc, Expression::from(k) / 8, Term::Acc('a'), &mut graph);
    acc = loop_in(acc, 8, Term::Acc('b'), &mut graph);

    let mut out = binary(
        binary(a, b, GraphTerm::Mul, &mut graph),
        acc,
        GraphTerm::Add,
        &mut graph,
    );

    out = loop_out(out, 8, Term::Acc('b'), &mut graph);
    out = loop_out(out, Expression::from(k) / 8, Term::Acc('a'), &mut graph);
    out = pad_out(out, &mut graph, 1);
    out = loop_out(out, 8, 'z', &mut graph);
    out = loop_out(out, 8, Expression::from('z') * n, &mut graph);
    out = pad_out(out, &mut graph, 1);
    out = loop_out(
        out,
        Expression::from(n) / 8,
        Expression::from('z') * 8,
        &mut graph,
    );
    out = loop_out(
        out,
        Expression::from(m) / 8,
        Expression::from('z') * n * 8,
        &mut graph,
    );
    out = unary(out, GraphTerm::GMEM { label: None }, &mut graph);
    let kernels = codegen(graph, out, GPUArch::Metal(HashMap::new())).unwrap();
    let a = vec![
        [
            -1.1258e+00,
            -1.1524e+00,
            -2.5058e-01,
            -4.3388e-01,
            8.4871e-01,
            6.9201e-01,
            -3.1601e-01,
            -2.1152e+00,
        ],
        [
            3.2227e-01,
            -1.2633e+00,
            3.4998e-01,
            3.0813e-01,
            1.1984e-01,
            1.2377e+00,
            1.1168e+00,
            -2.4728e-01,
        ],
        [
            -1.3527e+00,
            -1.6959e+00,
            5.6665e-01,
            7.9351e-01,
            5.9884e-01,
            -1.5551e+00,
            -3.4136e-01,
            1.8530e+00,
        ],
        [
            7.5019e-01,
            -5.8550e-01,
            -1.7340e-01,
            1.8348e-01,
            1.3894e+00,
            1.5863e+00,
            9.4630e-01,
            -8.4368e-01,
        ],
        [
            -6.1358e-01,
            3.1593e-02,
            -4.9268e-01,
            2.4841e-01,
            4.3970e-01,
            1.1241e-01,
            6.4079e-01,
            4.4116e-01,
        ],
        [
            -1.0231e-01,
            7.9244e-01,
            -2.8967e-01,
            5.2507e-02,
            5.2286e-01,
            2.3022e+00,
            -1.4689e+00,
            -1.5867e+00,
        ],
        [
            -6.7309e-01,
            8.7283e-01,
            1.0554e+00,
            1.7784e-01,
            -2.3034e-01,
            -3.9175e-01,
            5.4329e-01,
            -3.9516e-01,
        ],
        [
            -4.4622e-01,
            7.4402e-01,
            1.5210e+00,
            3.4105e+00,
            -1.5312e+00,
            -1.2341e+00,
            1.8197e+00,
            -5.5153e-01,
        ],
        [
            -5.6925e-01,
            9.1997e-01,
            1.1108e+00,
            1.2899e+00,
            -1.4782e+00,
            2.5672e+00,
            -4.7312e-01,
            3.3555e-01,
        ],
        [
            -1.6293e+00,
            -5.4974e-01,
            -4.7983e-01,
            -4.9968e-01,
            -1.0670e+00,
            1.1149e+00,
            -1.4067e-01,
            8.0575e-01,
        ],
        [
            -9.3348e-02,
            6.8705e-01,
            -8.3832e-01,
            8.9182e-04,
            8.4189e-01,
            -4.0003e-01,
            1.0395e+00,
            3.5815e-01,
        ],
        [
            -2.4600e-01,
            2.3025e+00,
            -1.8817e+00,
            -4.9727e-02,
            -1.0450e+00,
            -9.5650e-01,
            3.3532e-02,
            7.1009e-01,
        ],
        [
            1.6459e+00,
            -1.3602e+00,
            3.4457e-01,
            5.1987e-01,
            -2.6133e+00,
            -1.6965e+00,
            -2.2824e-01,
            2.7995e-01,
        ],
        [
            2.4693e-01,
            7.6887e-02,
            3.3801e-01,
            4.5440e-01,
            4.5694e-01,
            -8.6537e-01,
            7.8131e-01,
            -9.2679e-01,
        ],
        [
            -2.1883e-01,
            -2.4351e+00,
            -7.2915e-02,
            -3.3987e-02,
            9.6252e-01,
            3.4917e-01,
            -9.2146e-01,
            -5.6195e-02,
        ],
        [
            -6.2270e-01,
            -4.6372e-01,
            1.9218e+00,
            -4.0255e-01,
            1.2390e-01,
            1.1648e+00,
            9.2337e-01,
            1.3873e+00,
        ],
    ]
    .into_flattened();
    let b = vec![
        [
            -0.8834, -0.4189, -0.8048, 0.5656, 0.6104, 0.4669, 1.9507, -1.0631, -0.0773, 0.1164,
            -0.5940, -1.2439, -0.1021, -1.0335, -0.3126, 0.2458, -0.2596, 0.1183, 0.2440, 1.1646,
            0.2886, 0.3866, -0.2011, -0.1179, 0.1922, -0.7722, -1.9003, 0.1307, -0.7043, 0.3147,
            0.1574, 0.3854,
        ],
        [
            0.9671, -0.9911, 0.3016, -0.1073, 0.9985, -0.4987, 0.7611, 0.6183, 0.3140, 0.2133,
            -0.1201, 0.3605, -0.3140, -1.0787, 0.2408, -1.3962, -0.0661, -0.3584, -1.5616, -0.3546,
            1.0811, 0.1315, 1.5735, 0.7814, -1.0787, -0.7209, 1.4708, 0.2756, 0.6668, -0.9944,
            -1.1894, -1.1959,
        ],
        [
            -0.5596, 0.5335, 0.4069, 0.3946, 0.1715, 0.8760, -0.2871, 1.0216, -0.0744, -1.0922,
            0.3920, 0.5945, 0.6623, -1.2063, 0.6074, -0.5472, 1.1711, 0.0975, 0.9634, 0.8403,
            -1.2537, 0.9868, -0.4947, -1.2830, 0.9552, 1.2836, -0.6659, 0.5651, 0.2877, -0.0334,
            -1.0619, -0.1144,
        ],
        [
            -0.3433, 1.5713, 0.1916, 0.3799, -0.1448, 0.6376, -0.2813, -1.3299, -0.1420, -0.5341,
            -0.5234, 0.8615, -0.8870, 0.8388, 1.1529, -1.7611, -1.4777, -1.7557, 0.0762, -1.0786,
            1.4403, -0.1106, 0.5769, -0.1692, -0.0640, 1.0384, 0.9068, -0.4755, -0.8707, 0.1447,
            1.9029, 0.3904,
        ],
        [
            -0.0394, -0.8015, -0.4955, -0.3615, 0.5851, -1.1560, -0.1434, -0.1947, -0.0856, 1.3945,
            0.5969, -0.4828, -0.3661, -1.3271, 1.6953, 2.0655, -0.2340, 0.7073, 0.5800, 0.2683,
            -2.0589, 0.5340, -0.5354, -0.8637, -0.0235, 1.1717, 0.3987, -0.1987, -1.1559, -0.3167,
            0.9403, -1.1470,
        ],
        [
            0.5588, 0.7918, -0.1847, -0.7318, -0.0807, -0.9801, 0.0605, -0.4890, -0.8137, 0.8200,
            -0.6332, 1.2948, 1.4628, -0.6204, 0.9884, -0.4322, -0.6232, -0.2162, -0.4887, 0.7870,
            0.1076, -1.0715, -0.1166, -1.0170, -1.1980, 0.4784, -1.2295, -1.3700, 1.5435, -0.0332,
            -0.4186, -0.2556,
        ],
        [
            -0.1292, -0.0546, 0.4083, 1.1264, 1.9351, 1.0077, 1.0046, -0.4335, -1.2426, 1.2846,
            0.2438, 0.5304, -0.0145, -2.2357, 1.4660, -1.2191, 0.6442, 3.9300, -0.1244, 0.2953,
            0.3827, -0.5497, -0.9940, 1.3459, 1.9457, -1.2904, -2.3495, -2.0689, 0.9094, -0.6946,
            1.9595, -1.1038,
        ],
        [
            0.5411, 1.5390, 1.0860, 1.2464, 0.1151, 1.6193, 0.4637, 1.3007, 0.8732, 0.0651, 0.7732,
            -0.9701, -0.8877, -0.3183, -0.3344, 0.4543, 0.4990, 0.8780, 0.3894, 1.4625, 0.4795,
            -0.5334, -0.0347, 0.6573, -0.3112, -0.5620, -0.4835, -1.2721, -0.1740, 0.5541, -0.1817,
            -0.2345,
        ],
    ]
    .into_flattened();
    let outputs = run_graph(&[a, b], &kernels).0.pop().unwrap();
    let pt_output = vec![
        [
            -5.8128e-01,
            -2.5720e+00,
            -2.6012e+00,
            -4.5824e+00,
            -2.2321e+00,
            -5.8501e+00,
            -4.2573e+00,
            -2.3126e+00,
            -2.2847e+00,
            1.3359e+00,
            -7.0825e-01,
            2.8330e+00,
            3.2795e+00,
            2.1692e+00,
            1.7889e+00,
            3.1117e+00,
            -1.1726e+00,
            -1.6314e+00,
            6.2006e-01,
            -3.0596e+00,
            -4.6896e+00,
            2.2768e-01,
            -1.8609e+00,
            -3.6252e+00,
            9.5380e-03,
            3.8499e+00,
            1.4707e+00,
            1.8278e+00,
            4.9804e-01,
            -5.0718e-01,
            9.0723e-01,
            4.9808e-01,
        ],
        [
            -1.3994e+00,
            2.2303e+00,
            -5.3943e-01,
            5.7371e-01,
            1.0536e+00,
            6.5700e-01,
            5.4495e-01,
            -2.6102e+00,
            -3.1125e+00,
            1.8216e+00,
            -6.9492e-01,
            1.9941e+00,
            2.2922e+00,
            -2.4791e+00,
            3.3092e+00,
            -6.5214e-01,
            -2.4888e-01,
            3.9730e+00,
            1.6415e+00,
            1.7594e+00,
            -1.0725e+00,
            -1.4744e+00,
            -3.3582e+00,
            -1.5480e+00,
            2.5036e+00,
            8.6159e-01,
            -6.4024e+00,
            -3.9702e+00,
            1.5935e+00,
            3.9879e-01,
            3.5957e+00,
            8.6781e-02,
        ],
        [
            -8.8058e-01,
            4.9557e+00,
            2.8233e+00,
            2.7886e+00,
            -2.5081e+00,
            4.7050e+00,
            -3.9789e+00,
            3.1151e+00,
            2.6736e+00,
            -2.3199e+00,
            3.5056e+00,
            -2.1895e+00,
            -3.7918e+00,
            3.5530e+00,
            -3.6840e-01,
            3.4949e+00,
            1.4881e+00,
            1.5502e-01,
            4.7961e+00,
            1.9255e-01,
            -2.4336e+00,
            9.1083e-01,
            -2.0833e+00,
            -2.0420e-01,
            2.6679e+00,
            3.1751e+00,
            2.4753e+00,
            -3.4078e-01,
            -4.4315e+00,
            2.4826e+00,
            2.9210e+00,
            1.4049e+00,
        ],
        [
            -9.4201e-01,
            -7.4579e-01,
            -2.3271e+00,
            -1.1604e+00,
            2.2360e+00,
            -2.9661e+00,
            1.4722e+00,
            -4.1345e+00,
            -3.5774e+00,
            4.4527e+00,
            -1.1361e+00,
            1.6142e+00,
            2.3768e+00,
            -4.4558e+00,
            5.3234e+00,
            1.4209e+00,
            -1.7553e+00,
            3.5775e+00,
            5.2862e-01,
            1.4044e+00,
            -2.6672e+00,
            -1.0063e+00,
            -2.7208e+00,
            -2.4486e+00,
            7.6908e-01,
            1.4507e+00,
            -5.2168e+00,
            -3.5825e+00,
            7.2155e-01,
            -7.6680e-01,
            3.9975e+00,
            -1.7649e+00,
        ],
        [
            9.6443e-01,
            7.3376e-01,
            8.5263e-01,
            5.7995e-01,
            1.0755e+00,
            1.6618e-01,
            -3.0919e-01,
            -6.4255e-03,
            -4.8137e-01,
            1.8979e+00,
            7.2612e-01,
            5.4085e-01,
            -8.9135e-01,
            -8.2353e-01,
            1.8350e+00,
            -8.3975e-02,
            -3.2683e-01,
            2.6242e+00,
            -3.6255e-01,
            -3.6685e-01,
            3.9606e-01,
            -1.2199e+00,
            -3.4069e-01,
            1.3455e+00,
            3.2595e-01,
            -4.2925e-01,
            8.4076e-02,
            -2.5963e+00,
            2.6639e-01,
            -5.1575e-01,
            2.4036e+00,
            -1.4647e+00,
        ],
        [
            1.5980e+00,
            -1.7725e+00,
            -2.7938e+00,
            -5.7431e+00,
            -2.2332e+00,
            -7.5735e+00,
            -1.6752e+00,
            -2.4216e+00,
            -1.2075e+00,
            1.0722e+00,
            -2.9060e+00,
            3.7745e+00,
            4.1293e+00,
            1.3113e+00,
            1.6464e+00,
            8.9310e-02,
            -3.7378e+00,
            -7.7104e+00,
            -2.7944e+00,
            -1.5026e+00,
            -8.8581e-01,
            -7.6075e-01,
            2.4077e+00,
            -4.8187e+00,
            -6.2889e+00,
            3.6916e+00,
            3.1966e+00,
            1.8158e+00,
            2.3608e+00,
            -9.0384e-01,
            -3.6132e+00,
            -1.2820e-01,
        ],
        [
            2.9318e-01,
            -5.0400e-01,
            1.2476e+00,
            4.9899e-01,
            1.5186e+00,
            8.4619e-01,
            -6.2974e-01,
            1.5838e+00,
            -4.5928e-01,
            -1.1101e+00,
            5.5312e-01,
            2.2080e+00,
            1.8995e-01,
            -1.9099e+00,
            1.4177e+00,
            -3.4230e+00,
            1.5410e+00,
            1.1082e+00,
            -6.6059e-01,
            -1.1860e+00,
            1.3297e-01,
            1.0852e+00,
            7.3198e-01,
            4.4612e-01,
            1.5806e+00,
            4.9353e-01,
            1.3258e+00,
            6.2565e-01,
            1.4292e+00,
            -1.5996e+00,
            -8.4255e-01,
            -1.4973e+00,
        ],
        [
            -2.0714e+00,
            4.9217e+00,
            2.9867e+00,
            4.3826e+00,
            2.8991e+00,
            6.8479e+00,
            1.7166e-02,
            -2.6519e+00,
            -1.9368e+00,
            -4.2218e+00,
            -1.1284e+00,
            5.3073e+00,
            -2.9874e+00,
            -4.1068e-01,
            4.2111e+00,
            -1.3085e+01,
            -1.1675e+00,
            -3.0773e-01,
            -2.7193e-01,
            -4.8352e+00,
            7.1328e+00,
            8.4760e-01,
            1.6495e+00,
            2.7698e+00,
            5.5729e+00,
            8.7938e-01,
            9.2042e-01,
            -1.6835e+00,
            -1.0582e-01,
            -1.4812e+00,
            6.6622e+00,
            2.8793e-01,
        ],
        [
            2.0636e+00,
            5.7056e+00,
            1.8644e+00,
            -9.5123e-01,
            -1.3740e+00,
            3.3033e-01,
            -1.0444e+00,
            2.6764e-01,
            -1.0145e+00,
            -2.3144e+00,
            -2.3756e+00,
            6.2726e+00,
            3.3664e+00,
            6.5767e-01,
            1.7870e+00,
            -7.7373e+00,
            -1.9098e+00,
            -5.7188e+00,
            -2.3295e+00,
            5.2769e-01,
            4.5951e+00,
            -2.6046e+00,
            2.7073e+00,
            -2.6078e+00,
            -4.1890e+00,
            2.4598e+00,
            6.8436e-02,
            -2.4779e+00,
            5.3934e+00,
            -4.6922e-02,
            -3.3615e+00,
            5.3964e-01,
        ],
        [
            2.4670e+00,
            3.1719e+00,
            1.9950e+00,
            -8.2606e-01,
            -2.4471e+00,
            7.8235e-02,
            -2.8657e+00,
            2.3382e+00,
            1.2244e-01,
            -2.1784e-01,
            3.5314e-01,
            2.2153e+00,
            1.7728e+00,
            3.2189e+00,
            -1.6731e+00,
            -6.3854e-01,
            5.0206e-01,
            -6.4953e-03,
            -8.7179e-01,
            1.6121e-01,
            1.4667e+00,
            -3.2373e+00,
            -3.5226e-02,
            5.9067e-01,
            -1.9816e+00,
            -4.6841e-01,
            2.9880e-01,
            -2.4475e+00,
            3.7642e+00,
            8.2262e-01,
            -1.9359e+00,
            7.9457e-01,
        ],
        [
            1.0186e+00,
            -1.5847e+00,
            4.1153e-01,
            1.1486e+00,
            3.0626e+00,
            -7.3825e-02,
            1.6467e+00,
            -2.8672e-01,
            -4.4017e-01,
            3.2554e+00,
            9.3001e-01,
            -8.5447e-01,
            -1.9886e+00,
            -2.9396e+00,
            2.1224e+00,
            2.8225e-01,
            -1.0357e-01,
            4.7410e+00,
            -1.2092e+00,
            -3.1591e-01,
            5.6110e-01,
            -6.5737e-01,
            6.5304e-02,
            2.9375e+00,
            8.1057e-01,
            -2.2459e+00,
            -4.0938e-02,
            -2.5224e+00,
            -4.2577e-01,
            -1.4614e+00,
            2.9909e+00,
            -2.8561e+00,
        ],
        [
            3.4009e+00,
            -2.0897e+00,
            1.5966e+00,
            8.5292e-01,
            1.4456e+00,
            3.8577e-01,
            2.2817e+00,
            1.4092e+00,
            2.3353e+00,
            3.9210e-01,
            -3.0282e-01,
            -1.4305e+00,
            -3.5475e+00,
            1.6779e+00,
            -3.4742e+00,
            -3.6215e+00,
            -1.0021e+00,
            -7.2744e-01,
            -5.3384e+00,
            -2.6153e+00,
            7.1075e+00,
            -1.5742e+00,
            5.1878e+00,
            6.6381e+00,
            -3.3105e+00,
            -6.0612e+00,
            5.3992e+00,
            1.0816e-01,
            8.4892e-01,
            -1.5786e+00,
            -1.5192e+00,
            -1.4131e+00,
        ],
        [
            -3.8048e+00,
            2.8538e+00,
            3.2410e-01,
            3.6884e+00,
            -2.1714e+00,
            6.9871e+00,
            2.1027e+00,
            -1.1285e+00,
            1.4783e+00,
            -6.0630e+00,
            -1.2762e+00,
            -3.2122e+00,
            -1.7439e+00,
            4.7283e+00,
            -6.5688e+00,
            -3.0597e+00,
            9.5925e-01,
            -2.3297e+00,
            2.3477e+00,
            4.3379e-01,
            4.5664e+00,
            1.1383e+00,
            -5.2752e-01,
            2.0721e+00,
            3.6419e+00,
            -3.0446e+00,
            -3.4414e+00,
            2.7473e+00,
            -2.2737e+00,
            3.1318e+00,
            2.5493e-01,
            6.0417e+00,
        ],
        [
            -1.5930e+00,
            -1.8057e+00,
            -7.0506e-01,
            6.3038e-01,
            1.9621e+00,
            2.6923e-01,
            5.5266e-01,
            -1.6840e+00,
            -1.1997e+00,
            3.0413e-01,
            3.3288e-02,
            2.8540e-01,
            -8.5038e-01,
            -1.8860e+00,
            2.0452e+00,
            -1.0875e+00,
            1.2848e-01,
            2.0040e+00,
            5.3020e-01,
            -1.6289e+00,
            -7.9422e-01,
            1.6250e+00,
            -7.2197e-01,
            4.4824e-01,
            3.0929e+00,
            2.9367e-01,
            -3.1056e-01,
            6.8576e-01,
            -1.4132e+00,
            -1.1165e+00,
            2.9444e+00,
            -8.0610e-01,
        ],
        [
            -1.8634e+00,
            1.8816e+00,
            -1.5732e+00,
            -1.6156e+00,
            -3.8270e+00,
            -1.4477e+00,
            -3.3184e+00,
            -1.3340e+00,
            -8.1063e-03,
            -5.9155e-03,
            4.9688e-01,
            -1.1250e+00,
            9.9056e-01,
            3.4964e+00,
            4.3305e-02,
            6.3808e+00,
            -8.8175e-01,
            -2.1661e+00,
            4.1567e+00,
            7.6281e-01,
            -4.9769e+00,
            2.0345e-01,
            -3.4093e+00,
            -4.2413e+00,
            3.0079e-01,
            4.3110e+00,
            -1.0013e+00,
            5.8339e-01,
            -2.8628e+00,
            2.6426e+00,
            1.8380e+00,
            2.6601e+00,
        ],
        [
            4.4171e-01,
            4.0207e+00,
            2.6733e+00,
            2.1750e+00,
            1.4698e+00,
            3.2596e+00,
            -3.8247e-01,
            3.6845e+00,
            -1.0777e+00,
            3.4896e-01,
            2.0239e+00,
            1.9953e+00,
            2.2526e+00,
            -4.9052e+00,
            3.0374e+00,
            -5.9108e-01,
            3.5699e+00,
            5.6693e+00,
            2.3210e+00,
            4.7399e+00,
            -2.7813e+00,
            -7.9015e-01,
            -2.9555e+00,
            -1.8235e+00,
            2.2085e+00,
            1.5952e+00,
            -5.3663e+00,
            -4.2273e+00,
            3.2858e+00,
            1.9215e-01,
            -1.1670e+00,
            -1.8468e+00,
        ],
    ]
    .into_flattened();
    for (a, b) in outputs.into_iter().zip(pt_output) {
        assert!((a - b).abs() < 1e-3);
    }
    expression_cleanup();
}

#[test]
fn test_flash_attention() {
    let mut graph = StableGraph::new();

    let n_qkv = 4;
    let d = 5;

    // inputs
    let mut q = graph.add_node(GraphTerm::GMEM {
        label: Some("Q".to_string()),
    });
    q = loop_in(q, n_qkv, Expression::from('z') * d, &mut graph);
    q = pad_in(q, &mut graph, 5);
    q = loop_in(q, n_qkv, 0, &mut graph);
    q = loop_in(q, d, 'z', &mut graph);
    let mut k = graph.add_node(GraphTerm::GMEM {
        label: Some("K".to_string()),
    });
    k = loop_in(k, n_qkv, 0, &mut graph);
    k = pad_in(k, &mut graph, 5);
    k = loop_in(k, n_qkv, Expression::from('z') * d, &mut graph);
    k = loop_in(k, d, 'z', &mut graph);
    let mut v = graph.add_node(GraphTerm::GMEM {
        label: Some("V".to_string()),
    });
    v = loop_in(v, n_qkv, 0, &mut graph);
    v = pad_in(v, &mut graph, 5);
    v = loop_in(v, n_qkv, Expression::from('z') * d, &mut graph);
    v = loop_in(v, d, 'z', &mut graph);

    // accumulators
    let mut dot_acc = graph.add_node(GraphTerm::NewAcc {
        starting_value: "0.0".to_string(),
    });
    dot_acc = loop_in(dot_acc, d, Term::Acc('d'), &mut graph);
    let mut score_max_acc = graph.add_node(GraphTerm::NewAcc {
        starting_value: "-INFINITY".to_string(),
    });
    score_max_acc = loop_in(score_max_acc, n_qkv, Term::Acc('m'), &mut graph);
    let mut exp_sum_acc = graph.add_node(GraphTerm::NewAcc {
        starting_value: "0.0".to_string(),
    });
    exp_sum_acc = loop_in(exp_sum_acc, n_qkv, Term::Acc('e'), &mut graph);
    let mut output_acc = graph.add_node(GraphTerm::NewAcc {
        starting_value: "0.0".to_string(),
    });
    output_acc = loop_in(output_acc, n_qkv, Term::Acc('o'), &mut graph);
    output_acc = loop_in(output_acc, d, 'z', &mut graph);

    // get dot products
    let dots = loop_out(
        binary(
            binary(q, k, GraphTerm::Mul, &mut graph),
            dot_acc,
            GraphTerm::Add,
            &mut graph,
        ),
        d,
        Term::Acc('d'),
        &mut graph,
    );
    let new_max = binary(score_max_acc, dots, GraphTerm::Max, &mut graph);
    loop_out(new_max, n_qkv, Term::Acc('m'), &mut graph); // This is needed so we know to feed new max back in for score max acc

    let rescale = unary(
        binary(
            score_max_acc,
            unary(new_max, GraphTerm::Neg, &mut graph),
            GraphTerm::Add,
            &mut graph,
        ),
        GraphTerm::Exp,
        &mut graph,
    );
    let weight = unary(
        binary(
            dots,
            unary(new_max, GraphTerm::Neg, &mut graph),
            GraphTerm::Add,
            &mut graph,
        ),
        GraphTerm::Exp,
        &mut graph,
    );
    let exp_sum_new = binary(
        binary(exp_sum_acc, rescale, GraphTerm::Mul, &mut graph),
        weight,
        GraphTerm::Add,
        &mut graph,
    );
    let weight_b = loop_in(weight, d, 0, &mut graph);
    let rescale_b = loop_in(rescale, d, 0, &mut graph);
    let mut partial_output = binary(
        binary(output_acc, rescale_b, GraphTerm::Mul, &mut graph),
        binary(weight_b, v, GraphTerm::Mul, &mut graph),
        GraphTerm::Add,
        &mut graph,
    );
    partial_output = loop_out(partial_output, d, 'z', &mut graph);
    partial_output = loop_out(partial_output, n_qkv, Term::Acc('o'), &mut graph);
    partial_output = loop_in(partial_output, d, 'z', &mut graph);
    let exp_sum = loop_out(exp_sum_new, n_qkv, Term::Acc('e'), &mut graph);
    let exp_sum_b = loop_in(exp_sum, d, 0, &mut graph);
    let mut output = binary(
        partial_output,
        unary(exp_sum_b, GraphTerm::Recip, &mut graph),
        GraphTerm::Mul,
        &mut graph,
    );
    output = loop_out(output, d, 'z', &mut graph);
    output = pad_out(output, &mut graph, 5);
    output = loop_out(output, n_qkv, Expression::from('z') * d, &mut graph);
    output = unary(output, GraphTerm::GMEM { label: None }, &mut graph);
    let kernels = codegen(graph, output, GPUArch::Metal(HashMap::new())).unwrap();
    let q = vec![
        [-1.1258, -1.1524, -0.2506, -0.4339, 0.5988],
        [-1.5551, -0.3414, 1.8530, 0.4681, -0.1577],
        [1.4437, 0.2660, 1.3894, 1.5863, 0.9463],
        [-0.8437, 0.9318, 1.2590, 2.0050, 0.0537],
    ]
    .into_flattened();
    let k = vec![
        [0.4397, 0.1124, 0.6408, 0.4412, 0.2055],
        [-0.4503, -0.5731, -0.5554, 0.5943, 1.5419],
        [0.5073, -0.5910, -1.3253, 0.1886, -0.0691],
        [-0.4949, -1.4959, -0.1938, 0.4455, 1.3253],
    ]
    .into_flattened();
    let v = vec![
        [1.5091, 2.0820, 1.7067, 2.3804, 1.9415],
        [0.7915, -0.0203, -0.4372, 1.6459, -1.3602],
        [0.3446, 0.5199, -0.3656, -1.3024, 0.0994],
        [0.4418, 0.2469, 0.0769, 0.3380, 0.4544],
    ]
    .into_flattened();
    let outputs = run_graph(&[q, k, v], &kernels).0.pop().unwrap();
    let pt_output = vec![
        [0.5441, 0.2194, -0.0533, 0.6271, -0.0108],
        [0.8770, 0.8527, 0.5614, 1.2643, 0.6696],
        [1.2617, 1.5422, 1.1725, 1.9658, 1.2628],
        [1.2003, 1.3576, 0.9888, 1.9141, 0.9768],
    ]
    .into_flattened();
    for (a, b) in outputs.into_iter().zip(pt_output) {
        assert!((a - b).abs() < 1e-3);
    }
    expression_cleanup();
}

#[test]
fn test_naive_attention() {
    let mut graph = StableGraph::new();

    let n_qkv = 4;
    let d = 5;

    // inputs
    let mut q = graph.add_node(GraphTerm::GMEM {
        label: Some("Q".to_string()),
    });
    q = loop_in(q, n_qkv, Expression::from('z') * d, &mut graph);
    q = loop_in(q, n_qkv, 0, &mut graph);
    q = pad_in(q, &mut graph, 4);
    q = loop_in(q, d, 'z', &mut graph);
    let mut k = graph.add_node(GraphTerm::GMEM {
        label: Some("K".to_string()),
    });
    k = loop_in(k, n_qkv, 0, &mut graph);
    k = loop_in(k, n_qkv, Expression::from('z') * d, &mut graph);
    k = pad_in(k, &mut graph, 4);
    k = loop_in(k, d, 'z', &mut graph);
    let mut v = graph.add_node(GraphTerm::GMEM {
        label: Some("V".to_string()),
    });
    v = loop_in(v, n_qkv, 0, &mut graph);
    v = pad_in(v, &mut graph, 5);
    v = loop_in(v, n_qkv, Expression::from('z') * d, &mut graph);
    v = loop_in(v, d, 'z', &mut graph);

    // accumulators
    let mut dot_acc = graph.add_node(GraphTerm::NewAcc {
        starting_value: "0.0".to_string(),
    });
    dot_acc = loop_in(dot_acc, d, Term::Acc('d'), &mut graph);
    let mut score_max_acc = graph.add_node(GraphTerm::NewAcc {
        starting_value: "-INFINITY".to_string(),
    });
    score_max_acc = loop_in(score_max_acc, n_qkv, Term::Acc('m'), &mut graph);
    let mut exp_sum_acc = graph.add_node(GraphTerm::NewAcc {
        starting_value: "0.0".to_string(),
    });
    exp_sum_acc = loop_in(exp_sum_acc, n_qkv, Term::Acc('e'), &mut graph);
    let mut output_acc = graph.add_node(GraphTerm::NewAcc {
        starting_value: "0.0".to_string(),
    });
    output_acc = loop_in(output_acc, n_qkv, Term::Acc('o'), &mut graph);
    output_acc = loop_in(output_acc, d, 'z', &mut graph);

    // get dot products
    let mut dots = binary(
        binary(q, k, GraphTerm::Mul, &mut graph),
        dot_acc,
        GraphTerm::Add,
        &mut graph,
    );
    dots = loop_out(dots, d, Term::Acc('d'), &mut graph);
    dots = pad_out(dots, &mut graph, 4);
    dots = loop_out(dots, n_qkv, 'z', &mut graph);
    dots = loop_out(dots, n_qkv, Expression::from('z') * n_qkv, &mut graph);

    // get max
    let mut dots_in = loop_in(dots, n_qkv, Expression::from('z') * n_qkv, &mut graph);
    dots_in = pad_in(dots_in, &mut graph, 5);
    dots_in = loop_in(dots_in, n_qkv, 'z', &mut graph);
    let mut max = binary(score_max_acc, dots_in, GraphTerm::Max, &mut graph);
    max = loop_out(max, n_qkv, Term::Acc('m'), &mut graph);
    max = pad_out(max, &mut graph, 5);
    max = loop_out(max, n_qkv, 'z', &mut graph);

    // get exp sum
    let mut max_in = loop_in(max, n_qkv, 'z', &mut graph);
    max_in = pad_in(max_in, &mut graph, 5);
    max_in = loop_in(max_in, n_qkv, 0, &mut graph);
    let mut exp_sum = binary(
        unary(
            binary(
                dots_in,
                unary(max_in, GraphTerm::Neg, &mut graph),
                GraphTerm::Add,
                &mut graph,
            ),
            GraphTerm::Exp,
            &mut graph,
        ),
        exp_sum_acc,
        GraphTerm::Add,
        &mut graph,
    );
    exp_sum = loop_out(exp_sum, n_qkv, Term::Acc('e'), &mut graph);
    exp_sum = pad_out(exp_sum, &mut graph, 5);
    exp_sum = loop_out(exp_sum, n_qkv, 'z', &mut graph);

    // get final scores
    let mut exp_sum_in = loop_in(exp_sum, n_qkv, 'z', &mut graph);
    exp_sum_in = pad_in(exp_sum_in, &mut graph, 5);
    exp_sum_in = loop_in(exp_sum_in, n_qkv, 0, &mut graph);
    let mut final_scores = binary(
        unary(
            binary(
                dots_in,
                unary(max_in, GraphTerm::Neg, &mut graph),
                GraphTerm::Add,
                &mut graph,
            ),
            GraphTerm::Exp,
            &mut graph,
        ),
        unary(exp_sum_in, GraphTerm::Recip, &mut graph),
        GraphTerm::Mul,
        &mut graph,
    );
    final_scores = loop_out(final_scores, n_qkv, 'z', &mut graph);
    final_scores = pad_out(final_scores, &mut graph, 5);
    final_scores = loop_out(
        final_scores,
        n_qkv,
        Expression::from('z') * n_qkv,
        &mut graph,
    );

    // get output
    let mut final_scores_in = loop_in(
        final_scores,
        n_qkv,
        Expression::from('z') * n_qkv,
        &mut graph,
    );
    final_scores_in = pad_in(final_scores_in, &mut graph, 5);
    final_scores_in = loop_in(final_scores_in, n_qkv, 'z', &mut graph);
    final_scores_in = loop_in(final_scores_in, d, 0, &mut graph);
    let mut output = binary(
        binary(v, final_scores_in, GraphTerm::Mul, &mut graph),
        output_acc,
        GraphTerm::Add,
        &mut graph,
    );
    output = loop_out(output, d, 'z', &mut graph);
    output = loop_out(output, n_qkv, Term::Acc('o'), &mut graph);
    output = pad_out(output, &mut graph, 5);
    output = loop_out(output, n_qkv, Expression::from('z') * d, &mut graph);
    output = unary(output, GraphTerm::GMEM { label: None }, &mut graph);
    let kernels = codegen(graph, output, GPUArch::Metal(HashMap::new())).unwrap();
    let q = vec![
        [-1.1258, -1.1524, -0.2506, -0.4339, 0.5988],
        [-1.5551, -0.3414, 1.8530, 0.4681, -0.1577],
        [1.4437, 0.2660, 1.3894, 1.5863, 0.9463],
        [-0.8437, 0.9318, 1.2590, 2.0050, 0.0537],
    ]
    .into_flattened();
    let k = vec![
        [0.4397, 0.1124, 0.6408, 0.4412, 0.2055],
        [-0.4503, -0.5731, -0.5554, 0.5943, 1.5419],
        [0.5073, -0.5910, -1.3253, 0.1886, -0.0691],
        [-0.4949, -1.4959, -0.1938, 0.4455, 1.3253],
    ]
    .into_flattened();
    let v = vec![
        [1.5091, 2.0820, 1.7067, 2.3804, 1.9415],
        [0.7915, -0.0203, -0.4372, 1.6459, -1.3602],
        [0.3446, 0.5199, -0.3656, -1.3024, 0.0994],
        [0.4418, 0.2469, 0.0769, 0.3380, 0.4544],
    ]
    .into_flattened();
    let outputs = run_graph(&[q, k, v], &kernels).0.pop().unwrap();
    let pt_output = vec![
        [0.5441, 0.2194, -0.0533, 0.6271, -0.0108],
        [0.8770, 0.8527, 0.5614, 1.2643, 0.6696],
        [1.2617, 1.5422, 1.1725, 1.9658, 1.2628],
        [1.2003, 1.3576, 0.9888, 1.9141, 0.9768],
    ]
    .into_flattened();
    for (a, b) in outputs.into_iter().zip(pt_output) {
        assert!((a - b).abs() < 1e-3);
    }
    expression_cleanup();
}
