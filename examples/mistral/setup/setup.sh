#!/usr/bin/env bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

echo "Downloading Tokenizer"
curl --location https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2/resolve/main/tokenizer.model?download=true --output $SCRIPT_DIR/mistral_tokenizer.model
echo "Downloading Model"
curl --location https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q8_0.gguf?download=true --output $SCRIPT_DIR/mistral-7b-instruct-v0.2.Q8_0.gguf
echo "Done Downloading Model"
