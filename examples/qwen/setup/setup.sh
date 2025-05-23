#!/usr/bin/env bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

echo "Downloading Model and Tokenizer..."
curl --location https://huggingface.co/Qwen/Qwen3-4B/resolve/main/tokenizer.json?download=true --output $SCRIPT_DIR/tokenizer.json
curl --location https://huggingface.co/unsloth/Qwen3-4B-GGUF/resolve/main/Qwen3-4B-Q8_0.gguf?download=true --output $SCRIPT_DIR/qwen3-4b.gguf
echo "Done!"
