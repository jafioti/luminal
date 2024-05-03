#!/usr/bin/env bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

echo "Downloading Model and Tokenizer..."
curl --location https://luminal-public.s3.amazonaws.com/phi3/tokenizer.json --output $SCRIPT_DIR/tokenizer.json
curl --location https://luminal-public.s3.amazonaws.com/phi3/phi3.gguf --output $SCRIPT_DIR/phi3.gguf
echo "Done!"
