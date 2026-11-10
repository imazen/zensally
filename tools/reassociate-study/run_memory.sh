#!/usr/bin/env bash
# Run from the repository root under run-heavy, with an isolated Cargo target.
# Retain each attempt in a fresh output directory; never overwrite old traces.
set -euo pipefail
output="$(realpath -m "${1:?output directory required}")"
cpu="${2:-2}"
test ! -e "$output"
mkdir -p "$output/std" "$output/alloc-only"
for mode in std alloc-only; do
    options=()
    if [ "$mode" = alloc-only ]; then
        options=(--no-default-features)
    fi
    cargo build --release --manifest-path tools/reassociate-study/Cargo.toml --bins "${options[@]}"
    for name in memory_baseline memory_candidate zensally-reassociate-study; do
        cp "${CARGO_TARGET_DIR:?}/release/$name" "$output/$mode/"
    done
    "$output/$mode/zensally-reassociate-study" parity 32
    for arm in baseline candidate; do
        executable="$output/$mode/memory_$arm"
        just memory-perf "$executable" 101 "$output/$mode/$arm" "$cpu"
        for repeats in 1 101; do
            just allocation-record "$executable" '' "$repeats" "$output/$mode/$arm-$repeats"
        done
        just allocation-diff "$output/$mode/$arm-101.zst" "$output/$mode/$arm-1.zst" "$output/$mode/$arm-diff"
        just allocation-summary "$output/$mode/$arm-diff.stacks" 100 "$output/$mode/$arm-allocations.json"
    done
done
