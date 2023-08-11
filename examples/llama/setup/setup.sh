#!/usr/bin/env bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# Setup git LFS
echo "Setting up git LFS..."
sudo apt install git-lfs
git lfs install

echo "Downloading Model..."
git clone https://huggingface.co/decapoda-research/llama-7b-hf $SCRIPT_DIR/llama-7b-hf

# Convert the model
echo "Converting Model..."
python3 $SCRIPT_DIR/convert.py $SCRIPT_DIR/llama-7b-hf

echo "Done!"