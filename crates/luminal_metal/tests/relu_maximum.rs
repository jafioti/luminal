#![cfg(target_os = "macos")]

mod support;
use luminal::{dtype::DType, graph::Graph, prelude::NodeIndex};
use luminal_metal::{MetalRuntime, harness_search_options};
use support::TestTransfers;

const A: [f32; 8] = [-3.5, -0.0, 0.0, 2.25, 9.0, 7.0, f32::NEG_INFINITY, 1.0];
const B: [f32; 8] = [-1.0, 0.0, -0.0, 2.25, 4.0, -9.0, 5.0, f32::INFINITY];
const A_NAN: [f32; 4] = [7.0, f32::NAN, 1.0, f32::NAN];
const B_NAN: [f32; 4] = [f32::NAN, 2.0, 3.0, f32::NAN];

fn run(graph: &Graph, inputs: Vec<(NodeIndex, Vec<f32>)>, output: NodeIndex) -> Vec<f32> {
    let mut runtime = MetalRuntime::load(graph).unwrap();
    runtime
        .search(
            &Default::default(),
            &Default::default(),
            &inputs
                .iter()
                .map(|(id, values)| (*id, values.clone().into()))
                .collect(),
            &harness_search_options(),
        )
        .unwrap();
    let mut arena = support::allocate_arena(&runtime).expect("allocate execution arena");
    for (id, values) in inputs {
        runtime.upload(&mut arena, id, values).unwrap();
    }
    runtime.execute(arena.buffer()).unwrap();
    runtime.read_f32(&arena, output).unwrap()
}

// NaN-propagating, and numeric about signed zero: the frontend does not
// promise which zero a tie returns.
fn assert_same(actual: &[f32], expected: &[f32], what: &str) {
    assert_eq!(actual.len(), expected.len(), "{what}: length");
    for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (a.is_nan() && e.is_nan()) || a == e,
            "{what} element {i}: got {a}, expected {e}"
        );
    }
}

fn max(a: f32, b: f32) -> f32 {
    if a.is_nan() || b.is_nan() {
        f32::NAN
    } else {
        a.max(b)
    }
}

fn min(a: f32, b: f32) -> f32 {
    if a.is_nan() || b.is_nan() {
        f32::NAN
    } else {
        a.min(b)
    }
}

#[test]
fn relu_lowers_and_matches_scalar() {
    let mut graph = Graph::new();
    let x = graph.tensor(vec![A.len()], DType::F32);
    let out = x.relu();
    let actual = run(&graph, vec![(x.id, A.to_vec())], out.id);
    let expected: Vec<f32> = A.iter().map(|&v| max(v, 0.)).collect();
    assert_same(&actual, &expected, "relu");
}

#[test]
fn maximum_matches_scalar() {
    let mut graph = Graph::new();
    let a = graph.tensor(vec![A.len()], DType::F32);
    let b = graph.tensor(vec![B.len()], DType::F32);
    let out = a.maximum(b);
    let actual = run(&graph, vec![(a.id, A.to_vec()), (b.id, B.to_vec())], out.id);
    let expected: Vec<f32> = A.iter().zip(B).map(|(&x, y)| max(x, y)).collect();
    assert_same(&actual, &expected, "maximum");
}

#[test]
fn minimum_matches_scalar() {
    let mut graph = Graph::new();
    let a = graph.tensor(vec![A.len()], DType::F32);
    let b = graph.tensor(vec![B.len()], DType::F32);
    let out = a.minimum(b);
    let actual = run(&graph, vec![(a.id, A.to_vec()), (b.id, B.to_vec())], out.id);
    let expected: Vec<f32> = A.iter().zip(B).map(|(&x, y)| min(x, y)).collect();
    assert_same(&actual, &expected, "minimum");
}

// `GraphTensor::ne` is `lt + gt`, which is false for NaN, so the
// `x.ne(x)` NaN guards in `maximum`/`minimum` never fire. ReferenceRuntime
// gives the same answers, so this is a frontend bug, not a Metal one.
#[test]
#[ignore = "maximum/minimum drop a NaN in the right operand: ne() is false for NaN"]
fn maximum_minimum_propagate_nan_from_either_operand() {
    let mut graph = Graph::new();
    let a = graph.tensor(vec![A_NAN.len()], DType::F32);
    let b = graph.tensor(vec![B_NAN.len()], DType::F32);
    let (hi, lo) = (a.maximum(b), a.minimum(b));
    let inputs = vec![(a.id, A_NAN.to_vec()), (b.id, B_NAN.to_vec())];
    let expected_hi: Vec<f32> = A_NAN.iter().zip(B_NAN).map(|(&x, y)| max(x, y)).collect();
    let expected_lo: Vec<f32> = A_NAN.iter().zip(B_NAN).map(|(&x, y)| min(x, y)).collect();
    assert_same(&run(&graph, inputs.clone(), hi.id), &expected_hi, "maximum");
    assert_same(&run(&graph, inputs, lo.id), &expected_lo, "minimum");
}

#[test]
fn clip_matches_scalar() {
    let mut graph = Graph::new();
    let x = graph.tensor(vec![A.len()], DType::F32);
    let out = x.clip(-1., 2.);
    let actual = run(&graph, vec![(x.id, A.to_vec())], out.id);
    let expected: Vec<f32> = A.iter().map(|&v| min(max(v, -1.), 2.)).collect();
    assert_same(&actual, &expected, "clip");
}
