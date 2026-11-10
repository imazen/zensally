#!/usr/bin/env bash
set -euo pipefail
source_root=$1
build_root=$2
ops_config=$3
python_bin=$4
export TMPDIR="${HOME}/tmp"
export EM_CACHE="${build_root}/emscripten-cache"
"${python_bin}" "${source_root}/tools/ci_build/build.py" \
  --build_dir "${build_root}" --config MinSizeRel --cmake_generator Ninja \
  --update --build --skip_tests --skip_submodule_sync --parallel 6 \
  --build_wasm --enable_wasm_simd --enable_wasm_threads \
  --minimal_build --include_ops_by_config "${ops_config}" \
  --enable_reduced_operator_type_support --disable_ml_ops --disable_rtti \
  --disable_wasm_exception_catching --emsdk_version 4.0.23
