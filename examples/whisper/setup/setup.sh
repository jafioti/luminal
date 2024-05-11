#!/usr/bin/env bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

echo "Downloading Model and Tokenizer..."
curl --location https://huggingface.co/openai/whisper-tiny/resolve/main/tokenizer.json?download=true --output $SCRIPT_DIR/tokenizer.json
curl --location https://huggingface.co/FL33TW00D-HF/whisper-tiny/resolve/main/tiny_f32.gguf?download=true --output $SCRIPT_DIR/whisper-tiny.gguf
echo "Done!"
