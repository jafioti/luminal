use super::test_support::*;
use luminal_reference::{ReferenceRuntime, TypedBuffer, harness_search_options};
use rustc_hash::FxHashMap;
fn run(t: &super::Translation, values: &[TypedBuffer]) -> ReferenceRuntime {
    let data: FxHashMap<_, _> = t
        .inputs
        .iter()
        .zip(values)
        .map(|(i, v)| (i.tensor, v.clone()))
        .collect();
    let mut rt = ReferenceRuntime::load(&t.graph).unwrap();
    rt.search(
        &Default::default(),
        &Default::default(),
        &data,
        &harness_search_options(),
    )
    .unwrap();
    for (input, value) in t.inputs.iter().zip(values) {
        rt.set_data(input.tensor, value.clone());
    }
    rt.execute().unwrap();
    rt
}
#[test]
fn integer_boolean_translations_preserve_all_encoding_bits() {
    let a = [i64::MIN, i64::MAX, (1 << 54) + 3, -1, 0, 1];
    let b = [-1, 0x1234_5678_abcd_ef01, 7];
    for (op, which) in [
        ("bitwise_and.Tensor", 0),
        ("bitwise_or.Tensor", 1),
        ("bitwise_xor.Tensor", 2),
    ] {
        let t = translate_one(
            node(
                &format!("torch.ops.aten.{op}"),
                vec![tensor_input("self", "a"), tensor_input("other", "b")],
                &["y"],
            ),
            &[("a", 5, &[2, 3]), ("b", 5, &[3]), ("y", 5, &[2, 3])],
            &["a", "b"],
            &["y"],
        );
        let rt = run(
            &t,
            &[TypedBuffer::I64(a.to_vec()), TypedBuffer::I64(b.to_vec())],
        );
        let want = a
            .iter()
            .enumerate()
            .map(|(i, a)| match which {
                0 => *a & b[i % 3],
                1 => *a | b[i % 3],
                _ => *a ^ b[i % 3],
            })
            .collect::<Vec<_>>();
        assert_eq!(rt.get_i64(t.outputs[0].tensor).unwrap(), &want, "{op}");
    }
}
#[test]
fn integer_boolean_scalar_and_complement_are_exact() {
    let values = [i64::MIN, i64::MAX, (1 << 54) + 3, -1, 0, 1];
    let mask = (1 << 54) + 1;
    for target in [
        "bitwise_and.Scalar",
        "bitwise_or.Scalar",
        "bitwise_xor.Scalar",
        "bitwise_not.default",
    ] {
        let mut args = vec![tensor_input("self", "a")];
        if target.contains("Scalar") {
            args.push(scalar_int("other", mask));
        }
        let t = translate_one(
            node(&format!("torch.ops.aten.{target}"), args, &["y"]),
            &[("a", 5, &[6]), ("y", 5, &[6])],
            &["a"],
            &["y"],
        );
        let rt = run(&t, &[TypedBuffer::I64(values.to_vec())]);
        let want = values.map(|x| match target {
            "bitwise_and.Scalar" => x & mask,
            "bitwise_or.Scalar" => x | mask,
            "bitwise_xor.Scalar" => x ^ mask,
            _ => !x,
        });
        assert_eq!(rt.get_i64(t.outputs[0].tensor).unwrap(), &want, "{target}");
    }
}
#[test]
fn logical_operations_convert_each_operand_to_truth() {
    let a = [0.0f32, -0.0, 2.0, -2.0, f32::NAN, f32::from_bits(1)];
    let b = [1.0f32, 0.0, -2.0, 2.0, 0.0, f32::INFINITY];
    for target in [
        "logical_and.default",
        "logical_or.default",
        "logical_xor.default",
        "logical_not.default",
    ] {
        let unary = target.starts_with("logical_not");
        let mut args = vec![tensor_input("self", "a")];
        if !unary {
            args.push(tensor_input("other", "b"));
        }
        let inputs = if unary { vec!["a"] } else { vec!["a", "b"] };
        let t = translate_one(
            node(&format!("torch.ops.aten.{target}"), args, &["y"]),
            &[("a", FLOAT, &[6]), ("b", FLOAT, &[6]), ("y", BOOL, &[6])],
            &inputs,
            &["y"],
        );
        let mut values = vec![TypedBuffer::F32(a.to_vec())];
        if !unary {
            values.push(TypedBuffer::F32(b.to_vec()));
        }
        let rt = run(&t, &values);
        let want = a
            .iter()
            .zip(b)
            .map(|(a, b)| {
                let (a, b) = (*a != 0.0, b != 0.0);
                u8::from(match target {
                    "logical_and.default" => a && b,
                    "logical_or.default" => a || b,
                    "logical_xor.default" => a ^ b,
                    _ => !a,
                })
            })
            .collect::<Vec<_>>();
        assert_eq!(
            rt.get_bool8(t.outputs[0].tensor).unwrap(),
            &want,
            "{target}"
        );
    }
}
