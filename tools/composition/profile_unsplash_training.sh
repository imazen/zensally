#!/usr/bin/env bash
set -euo pipefail
python=$1
cache=$2
checkpoint=$3
output=$4
if [[ -e $output ]]; then
    echo 'refusing existing diagnostic output' >&2
    exit 1
fi
mkdir -p "$output"
export TMPDIR="${HOME}/tmp"
export NSYS_TMPDIR="$TMPDIR"
nsys profile --trace=cuda,nvtx,cudnn --sample=none --cpuctxsw=none \
    --capture-range=cudaProfilerApi --capture-range-end=stop \
    --force-overwrite=false --output="$output/trace" \
    "$python" tools/composition/profile_unsplash_training.py "$cache" "$checkpoint" "$output"
nsys stats --report cuda_gpu_kern_sum,cuda_api_sum,nvtx_sum --format csv \
    "$output/trace.nsys-rep" > "$output/statistics.csv"
