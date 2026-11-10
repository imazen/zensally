# Current-JXL crop-fitting evidence

Primary study: `/mnt/v/output/zensally/crop-fitting/2026-11-08/` on i265.
Inputs: `/mnt/v/input/zensally-crop-fitting-2026-11-08/` on the same host.
The source inventory and image bytes are immutable and hash-identified.

The study retains current/historical pinned manifests, family assignments,
source acquisition and exact-duplicate audits, all original/derived hashes,
native API requests/responses/usage, the invalid and corrected containment
requests, 7,320 candidate crops, owned/independent pixel checks, teacher/Sol
reviews, the admission manifest and its linked provenance sidecar.

Model evidence includes all epoch checkpoints, the frozen pre-evaluation
selection, canonical evaluation, fixed-policy per-ratio metrics, source
snapshots, resume/data-order guards, CPU perf and both Nsight reports/SQLite
exports. The resident replay is a performance-path experiment, not a promoted
checkpoint. The retained zenbench plan failed admission before starting; no
paired latency data is claimed. The report owning the findings is
[crop_fitting_2026-11-08.md](crop_fitting_2026-11-08.md).

Independent NAS mirrors are `/mnt/tower/input/zensally-crop-fitting-2026-11-08/`
and `/mnt/tower/output/zensally/crop-fitting/2026-11-08/`, accessible from dev.
Complete membership and streamed SHA256 readback pass on both local and NAS
copies, including symlink-target contents:

| Tree | Files | SHA256 of `SHA256SUMS.json` |
|---|---:|---|
| Inputs | 962 | `5aa2ddacca5e3bee73accb47aa35adc7c45b303d278fe233bd1d907a94399692` |
| Study | 20,431 | `2db9fb0487cc618950cddfed6fdd5b778ca6c381d25a3ad7956e4c41fccf4a07` |

Post-freeze verification logs and the environment observation are retained in
`/mnt/v/output/zensally/crop-fitting/2026-11-08-verification/` on i265, with the
corresponding independent NAS output mirror. These logs are separate from the
immutable study. Original fit source identity is retained under `source/`;
`source-final/` captures subsequent research code at `dc7a27c28984`.
Provider API cache entries remain independently available outside the study.

The review gallery is served at
http://dev.lan:3300/zensally/crop-fitting/2026-11-08/index.html
