# Photographic scope audit

The metadata-only audit packet is at
`/mnt/v/output/zensally/photo-focus/2026-11-08/` on i265. Its independent NAS
mirror is at `/mnt/tower/output/zensally/photo-focus/2026-11-08/` via dev.
The packet contains the initial audit, the audit-v2 revision with explicit
exact-development-hash exclusions, and legacy proposal/review request identity
checks against `5be7bb1d811d`.

Input identities are the pinned current/historical manifests in the
[fitting evidence packet](crop_fitting_2026-11-08.pointer.md). The substantive
findings and source-admission limits are in
[DATA_PROVENANCE.md](../DATA_PROVENANCE.md#photographic-successor-scope-2026-11-08)
and the [photographic protocol](../docs/experiments/PHOTO_CROP_CORPUS.md).
No new photos, labels or fitting are claimed. All 121 entries are screening
candidates; visual population coverage and external teacher rights remain
unverified.

All six files pass complete membership and SHA256 verification locally and
on the independent NAS mirror. The SHA256 of `SHA256SUMS.json` is
`263f1c4ce95b13192915fc3baa330c089417382d6a948ed31fe942849c5e6807`.
