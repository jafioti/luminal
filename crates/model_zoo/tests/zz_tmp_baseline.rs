// THROWAWAY: dumps yolo input specs for refactor no-op diffing. Delete before commit.
use luminal::prelude::*;
use model_zoo::yolo_v11::model::{IMG_SIZE, YoloV11};
use std::io::Write;

#[test]
fn dump_yolo_specs() {
    let mut cx = Graph::new();
    let image = cx.tensor(vec![1usize, 3usize, IMG_SIZE, IMG_SIZE], DType::F32);
    let model = YoloV11::init(&mut cx);
    let out = model.forward(image);
    let mut lines: Vec<String> = cx
        .logical
        .input_specs()
        .into_iter()
        .map(|s| format!("{} {:?} {:?}", s.label, s.dims, s.dtype))
        .collect();
    lines.sort();
    lines.push(format!("OUTPUT {:?}", out.dims()));
    let path = std::env::var("YOLO_DUMP").expect("YOLO_DUMP");
    let mut f = std::fs::File::create(path).unwrap();
    writeln!(f, "{}", lines.join("\n")).unwrap();
}
