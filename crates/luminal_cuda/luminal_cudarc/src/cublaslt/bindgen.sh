#!/bin/bash
# Requires rust-bindgen 0.68.1 or superior
set -exu
BINDGEN_EXTRA_CLANG_ARGS="-D__CUDA_BF16_TYPES_EXIST__" \
bindgen \
  --allowlist-type="^cublasLt.*" \
  --allowlist-var="^cublasLt.*" \
  --allowlist-function="^cublasLt.*" \
  --default-enum-style=rust \
  --no-doc-comments \
  --with-derive-default \
  --with-derive-eq \
  --with-derive-hash \
  --with-derive-ord \
  --use-core \
  wrapper.h -- -I/usr/local/cuda/include \
  > sys.rs
