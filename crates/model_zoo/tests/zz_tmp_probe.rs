// THROWAWAY: probe smallest buildable yolo configs. Delete before PR.
use luminal::prelude::*;
use model_zoo::yolo_v11::model::{YoloConfig, YoloV11};

#[test]
fn probe_sizes() {
    for widths in [[16usize, 32, 64, 128, 256], [4, 8, 16, 32, 64], [4, 8, 8, 16, 16], [2, 4, 8, 8, 8]] {
        let r = std::panic::catch_unwind(|| {
            let cfg = YoloConfig::new(32, widths);
            let mut cx = Graph::new();
            let img = cx.tensor(vec![1usize, 3, 32, 32], DType::F32);
            let m = YoloV11::with_config(&mut cx, &cfg);
            let out = m.forward(img);
            format!("{:?}", out.dims())
        });
        println!("PROBE widths={widths:?} -> {:?}", r.map_err(|e| e.downcast_ref::<String>().cloned()));
    }
}
