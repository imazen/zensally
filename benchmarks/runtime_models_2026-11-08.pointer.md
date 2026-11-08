# Existing-model runtime evidence

- Primary on i265: `/mnt/v/output/zensally/runtime-models/2026-11-08/`.
- Independent NAS via dev: `/mnt/tower/output/zensally/runtime-models/2026-11-08/`.
- Served gallery on dev: `/mnt/v/output/zensally/runtime-models/2026-11-08/gallery/`.
- All 4,001 NAS members pass full streamed SHA256 readback.
- SHA256SUMS.json digest:
  `b6bf576e6f3539f1ae9d6231e78ded484c0a35b35d980dd7eb101549be063230`.

Initial evaluation worker: `084432cf7f5d88e3c6be0c3c167d3c04e049090d`.
Anchor repair: `d05dd5029874aa57fccd4f505a598b53ff35389e`.
Final pair/raw-graph tools: `160613abc121cc7d2e666784c9879055787c6e30`.
The packet retains exact source snapshots, original/current native executables,
zenbench executable identity, five complete packets (124 invocations), input
manifest/benchmark RGB, raw detections/masks, visual gallery, face-point audit,
original failures, four sudo perf/GNU time profiles and environment versions.
The source-model files remain in the separately mirrored
[model inventory packet](cached_evaluation_2026-11-08.pointer.md), manifest digest
`4a58cd6aa5aa966bbaba3cd02506d7cdba545bde9371974421242c0effd4689b`.

The initial gallery is preserved as a backup; use current gallery/index.html.
The corrected gallery displays multiclass foreground as 1 minus background,
rather than the original background-class overlay. Raw class maps are unchanged.
Full native image/response packets remain at the existing source-study pointers;
manifest hashes identify the owned RGB used here. No R2 upload or paid API call.

http://dev.lan:3300/zensally/runtime-models/2026-11-08/gallery/index.html
