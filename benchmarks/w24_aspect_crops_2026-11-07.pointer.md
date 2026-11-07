# Aspect/crop study artifacts

Frozen on 2026-11-07 UTC. The primary capture is
`/mnt/v/output/zensally/aspect/2026-11-07/` on i270. The independent
mirror is `/mnt/tower/output/zensally/aspect/2026-11-07/`.

All **5,388 manifest-listed files** passed SHA256 readback verification from
the independent mirror. SHA256 of `SHA256SUMS`:
`6507993f5e0116eed3811aa101061f3c96bf156df4f91c604da7980b94019982`.
The manifest does not list itself. The listed files total 649,207,643 bytes.

The packet includes:

- All eight completed inference partitions, their exact executable, binary
  hash, PID ledger and completion markers; both failed earlier runs remain.
- Initial and corrected source-clustered analysis, complete Parquet+zstd
  tables, calibration policy, paired bootstrap intervals and gallery selection.
- The 5,019-pair frozen input/shard manifests and a complete DUTS-TE input
  archive under `reproduction-inputs/`. Original per-file hashes are retained.
- The frozen renderer, the 29-case gallery, all extracted crops/maps and a
  byte-identical renderer reproduction; three real unlabelled sources retain
  original URLs and hashes in `visuals/real-inputs.tsv`.
- Independent native-mask/PNG/crop checks, browser screenshots/logs, the
  corrected browser run and its failed shutdown-handler predecessor.
- Root source archive at 1e17cef01efe and shared zenlayout source at
  841ab33fe953, consumer Cargo.lock, compiler identity, meaningful failing and
  passing crop-bounds regressions, and public API verification logs.

Complete inference source is 5dac616943444a30e03641a49d403fae681e5b7c;
renderer source is 0fa21350bfb727f51a184875da7b13bf7267e43c;
analysis source is 1e17cef01efe9a56d9d18f9e739eade8b811fb2a. The source
archive contains the later analysis and rendering tools; the separately
captured executables identify the exact inference/rendering computations.

`compact-summary.json` is byte-identical to
[w24_aspect_crops_2026-11-07.json](w24_aspect_crops_2026-11-07.json).
The [report](w24_aspect_crops_2026-11-07.md) describes the data roles,
selection rule, corrected defects and limits. Results do not promote a new
model or preprocessing default.

To reproduce, extract the source and input archives outside the frozen packet,
retain the supplied input roles/hashes, and use the `aspect-run`,
`aspect-summary`, `aspect-render-inputs`, `aspect-gallery`, `aspect-oracle`
and `aspect-browser-check` just recipes. The frozen executable can rerun
`tools/aspect-study/run.sh` without rebuilding. Original absolute paths in
manifests identify the captured host; relocate path prefixes without changing
source bytes or roles. Analysis requires Pillow 13.0.0 and PyArrow 26.0.1;
the browser checker uses Node and Chromium through the DevTools protocol.
Use fresh output directories and the resource wrapper.

Resolve the mirror mount with `findmnt /mnt/tower`, then run
`sha256sum -c SHA256SUMS` there. The full verification log is retained outside
the packet at `~/tmp/zensally-aspect-mirror-verification.log` on the verification
host. No new model binaries, source images, generated PNGs or large result
matrices were committed to git.
