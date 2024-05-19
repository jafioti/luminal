#!/usr/bin/env bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

echo "Downloading Model, Tokenizer and Sample File..."
curl --location https://huggingface.co/datasets/Narsil/candle-examples/resolve/main/samples_jfk.wav?download=true --output $SCRIPT_DIR/jfk.wav
curl --location https://huggingface.co/openai/whisper-tiny.en/resolve/main/tokenizer.json?download=true --output $SCRIPT_DIR/tokenizer.json
curl --location https://huggingface.co/openai/whisper-tiny.en/resolve/main/model.safetensors?download=true --output $SCRIPT_DIR/whisper-tiny.safetensors
echo "Done!"
