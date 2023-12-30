#!/usr/bin/env bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# Check if git LFS is installed
if command -v git-lfs &> /dev/null; then
    echo "git LFS is already installed."
else
    # Install git LFS based on the OS type
    echo "Installing git LFS..."
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        sudo apt install git-lfs
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        brew install git-lfs
    else
        echo "Unsupported operating system."
        exit 1
    fi
fi

# Setup git LFS
echo "Configuring git LFS..."
git lfs install

echo "Downloading Model..."
# Clone the repo
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2 $SCRIPT_DIR/mistral-7b-hf

# Download the big boi files
echo "Downloading Tokenizer"
curl --location https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2/resolve/main/tokenizer.model?download=true --output $SCRIPT_DIR/mistral-7b-hf/tokenizer.model


echo "Downloading Model Files" 

curl\
	--parallel --parallel-immediate --parallel-max 3\
 	--location https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2/resolve/main/model-00001-of-00003.safetensors?download=true --output $SCRIPT_DIR/mistral-7b-hf/model-00001-of-00003.safetensors\
	--location https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2/resolve/main/model-00002-of-00003.safetensors?download=true --output $SCRIPT_DIR/mistral-7b-hf/model-00002-of-00003.safetensors\
	--location https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2/resolve/main/model-00003-of-00003.safetensors?download=true --output $SCRIPT_DIR/mistral-7b-hf/model-00003-of-00003.safetensors

echo "Done Downloading Model" 