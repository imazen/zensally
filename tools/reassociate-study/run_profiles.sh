#!/usr/bin/env bash
# Run under run-heavy. Profile the saved std driver, not Cargo's mutable output.
set -euo pipefail
driver="$(realpath "${1:?saved driver required}")"
output="$(realpath -m "${2:?fresh output directory required}")"
test ! -e "$output"
mkdir -p "$output"
for arm in baseline candidate; do
    just reduction-perf "$driver" "$arm" "$output/$arm-tma" 2 1001
done
for cpus in 2 0,1,2,3,4,5,6,7; do
    for arm in batch-baseline batch; do
        just scaling-perf "$driver" "$output/threads-$cpus-$arm" "$cpus" 160 "$arm"
    done
done
