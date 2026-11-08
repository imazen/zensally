#!/usr/bin/env bash
set -euo pipefail
root=$1
python=$2
binary=$3
mkdir -p "$root/profiles"
for model in float16-blaze_face_full_range_sparse float16-selfie_segmenter; do
    sudo perf stat --cputype=core -M tma_memory_bound,tma_core_bound -o "$root/profiles/$model.stat" -- \
        sudo -u lilith env TMPDIR=/home/lilith/tmp OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
        /usr/bin/time -v -o "$root/profiles/$model.time" taskset -c 2 "$python" \
        tools/runtime-eval/mediapipe_probe.py profile "$root/manifest.json" "$model" "$root/profiles/$model" --repeats 300
 done
for model in ultraface selfie; do
    sudo perf stat --cputype=core -M tma_memory_bound,tma_core_bound -o "$root/profiles/$model.stat" -- \
        sudo -u lilith env TMPDIR=/home/lilith/tmp RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 \
        /usr/bin/time -v -o "$root/profiles/$model.time" taskset -c 2 "$binary" \
        profile "$model" "$root/inputs.tsv" "$root/profiles/$model" 300
done
