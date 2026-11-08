# Real-photo ranking evidence — 2026-11-08

The following packets were frozen from `36b8b4cc`. Every listed member has
independent NAS membership, byte-count and SHA256 verification. Paths were
checked on i265 locally and through dev for NAS; `/mnt/v` is host-specific.
Local prefix: `/mnt/v/output/zensally/`; NAS prefix:
`/mnt/tower/output/zensally/`.

| Relative directory | Members | Payload bytes | SHA256SUMS.json SHA256 |
|---|---:|---:|---|
| unsplash-relationships/2026-11-08 |184|10,319,962|`d0e134561fcb57f80ec5a60b28cdea977283686f77b8b630b1af0afcd64ea001`|
| unsplash-roles/2026-11-08 |172|1,105,894|`83619144625bb4dd38bbfe02d21095596910e1ca03cf93d74ec3850f6c2c8dfa`|
| unsplash-rank-distillation/2026-11-08 |1,138|813,039,045|`d2a9e3c1b2f67a0db01088c52b865895467b1e86c6fbc5588f9e2a15a49b1b8e`|
| unsplash-rank-training-profile/2026-11-08 |367|225,728,125|`eb21f3868819923638ef544d3eb882e0afa071d20e9e2fd80b23365c48a51f97`|
| unsplash-rank-training-profile/2026-11-08-scoped |363|221,848,547|`be197e2fc31ddf1c57ccaae05da3a1b5b79a790aa71f8ef060adb84f6a50482c`|

The rank packet retains the failed empty-pair attempt, full fit/development
and TEST reference caches, all12 histories/checkpoints, materialized best/latest
aliases, development-only frozen selection plan, held-out predictions and paired
family analysis. No alias depends on a live symlink. The two profiler packets
retain original and corrected target-memory accounting separately; only the
scoped repeat verifies the actual target's4GiB cgroup ceiling.

Producer source, owned fingerprint origin/kernel, protocols and logs are retained.
The older original/owned-preview/teacher-weight packets are linked by identity.
See [findings and limits](unsplash_rank_distillation_2026-11-08.md).
