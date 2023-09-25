#!/bin/bash

# Command to rerun
command_to_run="./target/release/examples/llama"

# Number of times to rerun the command
count=100

for ((i=1; i<=count; i++)); do
  eval "$command_to_run"
done
