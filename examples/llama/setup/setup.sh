#!/usr/bin/env bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# Setup git LFS
echo "Setting up git LFS..."
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
	sudo apt install git-lfs
elif [[ "$OSTYPE" == "darwin"* ]]; then
	brew install git-lfs
fi
git lfs install

echo "Downloading Model..."
git lfs clone https://huggingface.co/decapoda-research/llama-7b-hf $SCRIPT_DIR/llama-7b-hf

# Convert the model
echo "Converting Model..."
python3 $SCRIPT_DIR/convert.py $SCRIPT_DIR/llama-7b-hf

echo "Done!"
