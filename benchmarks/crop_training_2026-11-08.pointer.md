# Preliminary crop training and direct API artifacts

Frozen 2026-11-08. [Training findings](crop_training_2026-11-08.md) and
[direct API findings](crop_direct_api_2026-11-08.md).

Training artifacts are separated by measurement host:

- i265 source: `/mnt/v/output/zensally/crop-training/2026-11-08/`.
  NAS: `/mnt/tower/output/zensally/crop-training/2026-11-08/i265/`.
  Payload: 238 files. SHA256SUMS.json SHA256:
  `695a5d40d01068a8cd57797278659af78a96dd2e3b9e31462c45ed78e11abad1`.
- i270 source: the same local path on i270.
  NAS: `/mnt/tower/output/zensally/crop-training/2026-11-08/i270/`.
  Payload: 111 files. Manifest SHA256:
  `a1f7d98a73188fb9c6400ea25ebdd64263da60a797854cd5bb0f8858a2b37a05`.

Both mirrors pass complete membership/hash readback. Packets retain successful
and invalid attempts, all plans/results/progress/PIDs, telemetry, profiler
reports/SQLite/CSV, sudo perf counters, logs, source snapshots, requirements,
zenbench packets/A/A and the GPU compiler-cache archive. Benchmark checks cover
fixture/finite-gradient execution, not artistic or bit-exact model equivalence.

The direct API frozen snapshot lives at
`/mnt/tower/output/zensally/crop-direct-api/2026-11-08/`, with matching served
copy `/mnt/v/output/zensally/crop-direct-api/2026-11-08/` on dev. Both copies
verify all 154 payload files. Manifest SHA256:
`5ade5354592a485084b633e8eb1810fd672f5f3573a11934718043396018d1b8`.

The **ongoing response cache** is separate from that immutable study snapshot:
`/mnt/tower/output/zensally/api-response-cache/roi-crop-zones-v1/`.
Four credential-free repeat reads from this persistent cache produce four hits
and zero network requests. New cache entries must not modify the frozen study.

Verify with the retained `artifacts.py verify ROOT`. Resolve the NAS mount using
`ssh dev findmnt -T /mnt/tower`; do not assume its backing storage. No private
credentials, full agent sessions, binaries or large generated packets enter git.

Review gallery:

http://dev.lan:3300/zensally/crop-direct-api/2026-11-08/index.html
