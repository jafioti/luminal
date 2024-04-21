fn main() -> Result<(), Box<dyn std::error::Error>>{
    let mut config = prost_build::Config::new();
    config.btree_map(&["."]);
    prost_build::compile_protos(&["src/onnx/proto/onnx.proto"], &["src/onnx/proto/"])?;
    Ok(())
}