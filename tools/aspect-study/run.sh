#!/usr/bin/env bash
# Invoke from repository root through run-heavy --mem 6G --jobs 4.
set -euo pipefail
executable="$(realpath "${1:?executable}")"
dataset="$(realpath "${2:?dataset}")"
shards="$(realpath "${3:?shards}")"
output="$(realpath -m "${4:?new output}")"
test ! -e "$output"
mkdir -p "$output"
cp "$executable" "$output/aspect_crops"
sha256sum "$output/aspect_crops" > "$output/executable.sha256"
result=0
for kind in natural stress; do
    pids=()
    for shard in 0 1 2 3; do
        "$output/aspect_crops" "$dataset" "$shards/shard-$shard.tsv" "$output/$kind-$shard" "$kind" 999999 - > "$output/$kind-$shard.log" 2>&1 &
        pid=$!
        pids+=("$pid")
        printf '%s\t%s\t%s\n' "$kind" "$shard" "$pid" >> "$output/pids.tsv"
    done
    for pid in "${pids[@]}"; do
        if wait "$pid"; then :; else result=1; fi
    done
    if [ "$result" != 0 ]; then exit "$result"; fi
done
printf 'all eight partitions complete\n' > "$output/complete.txt"
