#!/bin/bash
set -exu

bindgen \
  --whitelist-type="^curand.*" \
  --whitelist-var="^curand.*" \
  --whitelist-function="^curand.*" \
  --default-enum-style=rust \
  --no-doc-comments \
  --with-derive-default \
  --with-derive-eq \
  --with-derive-hash \
  --with-derive-ord \
  --size_t-is-usize \
  --use-core \
  wrapper.h -- -I/usr/local/cuda/include \
  > sys.rs