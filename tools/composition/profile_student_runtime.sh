#!/usr/bin/env bash
set -euo pipefail
python=$1
cache=$2
weights=$3
export_root=$4
output=$5
mkdir "$output"
runner=(sudo -u "$(id -un)" env TMPDIR="$HOME/tmp" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
    /usr/bin/time -v -o "$output/target.time" taskset -c 2 "$python"
    tools/composition/bench_unsplash_student.py "$cache" "$weights" "$export_root"
    --analysis-threads 1 --phase bank)
sudo unshare --mount --pid --fork --mount-proc --propagation private \
    perf stat --cputype=core -M tma_memory_bound,tma_core_bound \
    -e cycles,instructions,cache-misses -o "$output/counters.stat" -- "${runner[@]}" --repeats 20
sudo unshare --mount --pid --fork --mount-proc --propagation private \
    perf record -F 199 --call-graph dwarf,4096 -o "$output/calls.perf" -- "${runner[@]}" --repeats 10
sudo perf report -i "$output/calls.perf" --stdio --no-children --percent-limit 1 > "$output/calls.txt"
