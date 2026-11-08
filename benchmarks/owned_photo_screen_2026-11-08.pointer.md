# Owned-photo screening evidence — 2026-11-08

Immutable output packet built from `8445a93d0c5e`:

- i265: `/mnt/v/output/zensally/owned-photo-evidence/2026-11-08/`
- NAS: `/mnt/tower/output/zensally/owned-photo-evidence/2026-11-08/`
- 887 members, 260,311,276 payload bytes; checksum file is additional.
- SHA256SUMS.json SHA256:
  `6594edba891d16c16baa7304058e8008374eb041a52dc6de0545ac9fa42de934`.

The full NAS membership/size/hash readback passes using the packet's retained
`source/tools/crop-first/artifacts.py`; run-heavy records 3 s / 0.02 GiB peak
RSS. The first attempt referred to a missing project checkout and failed before
reading packet members; its log remains retained separately.

Original inputs are a separately frozen, fully NAS-verified 1,012-member cache:

- i265: `/mnt/v/input/zensally/owned-photo-screen/2026-11-08/`
- NAS: `/mnt/tower/input/zensally/owned-photo-screen/2026-11-08/`
- SHA256SUMS.json SHA256:
  `1d884d295ed9e958a74d18aaf105aedb896aa493b88eb3542dc60004ccc5e16f`.

The packet includes 101-source coordinate proposals, exact source and PNG
oracles, 2,222-request runtime gates, six bounded neural image witnesses,
scalar/SIMD comparisons, person-region retention diagnostics, 18 owned ROI
views / 54 model calls, corrected and retained-invalid SVG viewers, screenshot
clipping controls, exact source snapshots and logs. Earlier model/runtime
assets are referenced by frozen manifest digests. No new labels, paid API
calls, canonical holdout reads or quality model promotion.

See the [readable findings](owned_photo_screen_2026-11-08.md), including the
false face box on a skier's torso and the limits of pose/face counts.
