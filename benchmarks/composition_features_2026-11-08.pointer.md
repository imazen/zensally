# Composition-feature evidence

Completed first-gate packet on i265:
`/mnt/v/output/zensally/composition-features-evidence/2026-11-08/`.

Independent NAS copy via dev:
`/mnt/tower/output/zensally/composition-features-evidence/2026-11-08/`.

All **2,297 members** pass independent streamed SHA256 readback. Manifest digest:
`d19651ba88ace98b07175459520b683879cc9bea73db657b490908a67edb5c76`.

The packet retains the 136-source input manifest and owned RGB teacher inputs,
pinned teacher/critic sources and weights, compact pose/gaze/scanpath sidecars,
frozen descriptor cache, all 15 head fits and 91 recovered epoch predictions,
the geometry-only admission and all nine joint fits, both successful critic
precision probes, original failures, source snapshots, environment versions and
complete logs available when packed. It excludes mutable goal progress and HF
transfer metadata. Paths inside metadata retain the original source locations;
packet membership/hashes provide the independent copy identity.

Core workers: `853c2c68` (heads), `757dfa7b` (critic/replay), `3f0d34fd`
(joint fits), `0382102a` (new gradient guard correction), `5766c0ef` (report).
Source snapshots record exact hashes in addition to the parent revision.

The later FocalNet/FrameLite study and subsequent GPU profiles are separate
ongoing packets, outside this frozen membership. No new quality result is
silently added to the old experiment.
