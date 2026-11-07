#!/usr/bin/env bash
# Invoke through run-heavy. Linux CPU placement needs a real cpuset controller;
# user systemd scopes can accept AllowedCPUs without enabling that controller.
set -euo pipefail
worker_user="$(id -un)"
if [ "$#" = 0 ]; then
    set -- 2 2,3 2,3,4,5 0,1,2,3,4,5,6,7
fi
cargo_options=()
if [ -n "${W24_CARGO_CONFIG:-}" ]; then
    cargo_options=(--config "$W24_CARGO_CONFIG")
fi
result=0
for cpus in "$@"; do
    if sudo systemd-run --scope --quiet \
        -p "AllowedCPUs=$cpus" -p "MemoryMax=${W24_BENCH_MEMORY:-6G}" -p MemorySwapMax=0 \
        runuser -u "$worker_user" -- env \
        PATH="$PATH" HOME="$HOME" TMPDIR="${TMPDIR:?}" \
        CARGO_HOME="${CARGO_HOME:?}" CARGO_TARGET_DIR="${CARGO_TARGET_DIR:?}" \
        CARGO_PROFILE_DEV_DEBUG=0 CARGO_PROFILE_RELEASE_DEBUG=1 \
        CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:?}" \
        RAYON_NUM_THREADS="${RAYON_NUM_THREADS:?}" OMP_NUM_THREADS="${OMP_NUM_THREADS:?}" \
        W24_CPU_SET="$cpus" just scaling-bench "${cargo_options[@]}"; then
        :
    else
        result=1
    fi
done
exit "$result"
