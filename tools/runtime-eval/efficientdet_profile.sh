#!/usr/bin/env bash
set -euo pipefail
root=$1
python=$2
mkdir "$root/profiles"
for model in int8-efficientdet_lite0 tf-efficientdet_lite0 tf-efficientdet_lite1 tf-efficientdet_lite2 int8-efficientdet_lite2; do
    sudo perf stat --cputype=core -M tma_memory_bound,tma_core_bound \
        -e cycles,instructions,cache-misses -o "$root/profiles/$model.stat" -- \
        sudo -u lilith env TMPDIR=/home/lilith/tmp OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
        /usr/bin/time -v -o "$root/profiles/$model.time" taskset -c 2 "$python" \
        tools/runtime-eval/mediapipe_probe.py profile "$root/manifest.json" "$model" \
        "$root/profiles/$model" --repeats 100
done
