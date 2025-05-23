#!/usr/bin/env bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

echo "Downloading Model and Tokenizer..."
curl --location https://huggingface.co/vikhyatk/moondream2/resolve/main/tokenizer.json?download=true --output $SCRIPT_DIR/tokenizer.json
curl --location https://huggingface.co/vikhyatk/moondream2/resolve/main/model.safetensors?download=true --output $SCRIPT_DIR/moondream2.safetensors
echo "Done!"
