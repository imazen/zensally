#!/usr/bin/env bash
set -euo pipefail
root=$1
python=$2
profile_name=${3:-profiles}
profile_root="$root/$profile_name"
mkdir "$profile_root"
for model in int8-efficientdet_lite0 tf-efficientdet_lite0 tf-efficientdet_lite1 tf-efficientdet_lite2 int8-efficientdet_lite2; do
    # Perf's startup /proc FD scan can block on unrelated stale NFS handles.
    # A private PID/mount namespace limits that scan without changing host mounts.
    sudo unshare --mount --pid --fork --mount-proc --propagation private \
        perf stat --cputype=core -M tma_memory_bound,tma_core_bound \
        -e cycles,instructions,cache-misses -o "$profile_root/$model.stat" -- \
        sudo -u lilith env TMPDIR=/home/lilith/tmp OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
        /usr/bin/time -v -o "$profile_root/$model.time" taskset -c 2 "$python" \
        tools/runtime-eval/mediapipe_probe.py profile "$root/manifest.json" "$model" \
        "$profile_root/$model" --repeats 100
done
