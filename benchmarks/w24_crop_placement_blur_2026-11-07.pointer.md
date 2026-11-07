# Corrected crop and blurred-padding artifacts

Frozen on 2026-11-07 UTC. Primary capture on i270:
`/mnt/v/output/zensally/aspect-corrected/2026-11-07/`.
Independent mirror:
`/mnt/tower/output/zensally/aspect-corrected/2026-11-07/`.

All **4,148 manifest-listed files** passed SHA256 readback from the mirror.
Listed file bytes: 486,801,017. SHA256 of `SHA256SUMS`:
`efc597dc8be72cf5e3d77c9169601dc9a838c6976bcd6e11ba69d21b9573312d`.
The manifest does not list itself. Verification log, outside the packet:
`~/tmp/zensally-crop-corrected-mirror-verification.log` on the verification host.

The original solver packet remains immutable and independently verified at
`/mnt/tower/output/zensally/aspect/2026-11-07/`; see its
[pointer](w24_aspect_crops_2026-11-07.pointer.md). Its complete DUTS-TE source
archive remains the input for both studies. The corrected packet retains
exact source/shard manifests and all roles and hashes, without another data
copy or new labels.

The new packet includes:

- All eight complete inference partitions, the frozen executable and its hash,
  PID/completion records, exact consumer lockfile and compiler identity.
- Original/corrected metric and crop-size equality checks, complete typed
  Parquet+zstd tables, both six-arm/eight-arm calibration policies, paired
  source-byte-cluster bootstrap results and all fixed-method rows.
- The original reported case, 80 independently checked crops, 66,456 source
  origin comparisons, full-precision selected maps, source-bounded extracted
  crops and all exact model inputs for the one-pass strategies.
- All 29 matched review cases, original crop coordinates, preserved pre-fit
  gallery versions, independent PNG/mask readbacks and both successful browser
  captures (before/after whole-map fit), with six screenshots in total.
- Exact root inference source 95c7a8ca6314, analysis/gallery tooling archives
  6c23fa0994c0 and 6d02c36de6b1, and shared zenlayout 6ed32dea4df1. The
  archived later root revisions have identical Cargo/Rust sources to 95c7a8ca.
- The meaningful pre-fix placement failure and passing shared solver std,
  alloc-only, oracle, Clippy, doctest and workspace API-check logs.

`compact-summary.json` matches
[w24_crop_placement_blur_2026-11-07.json](w24_crop_placement_blur_2026-11-07.json).
The [report](w24_crop_placement_blur_2026-11-07.md) states the corrected
interpretation and limits; model weights and preprocessing defaults do not change.

Resolve the NAS mount with `findmnt /mnt/tower`, then run
`sha256sum -c SHA256SUMS` from the mirror directory. Reproduce into fresh
outputs using the source archive, frozen executable, manifests and `aspect-*`
just recipes. Python analysis uses Pillow 13.0.0 and PyArrow 26.0.1. The
original inputs are available through the original packet above; three real
unlabelled sources retain their URLs/hashes in `visuals/real-inputs.tsv`.
No generated imagery, float maps or model binaries were committed to git.
