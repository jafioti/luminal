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
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2 $SCRIPT_DIR/mistral-7b-hf
echo "Done Downloading Model"