**Qwen 3 4B**
```bash
cd ./examples/qwen
# Download the model
bash ./setup/setup.sh
# Run the model
cargo run --release --features metal    # MacOS
cargo run --release --features cuda     # Nvidia
cargo run --release                     # CPU
```
