#!/usr/bin/env bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

echo "Downloading Model and Tokenizer..."
curl --location https://huggingface.co/NousResearch/Meta-Llama-3-8B-Instruct/resolve/main/tokenizer.json?download=true --output $SCRIPT_DIR/tokenizer.json
curl --location https://huggingface.co/QuantFactory/Meta-Llama-3-8B-GGUF/resolve/main/Meta-Llama-3-8B.Q8_0.gguf?download=true --output $SCRIPT_DIR/llama3-8b.gguf
echo "Done!"
