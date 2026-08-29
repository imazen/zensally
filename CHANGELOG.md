# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/), and the crates adhere to
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Changed

- Dependency requirements written out in full instead of truncated, at the
  versions already locked and tested: `serde` 1 → 1.0.229 and `zenlayout` 0.2 →
  0.2.2 (`zensally`); `anyhow` 1 → 1.0.104, `image` 0.25 → 0.25.10, `criterion`
  0.8 → 0.8.2, `zenlayout` 0.2 → 0.2.2 (`zensally-tract`); `anyhow` 1 → 1.0.104
  (`zensally-zentract`). The resolved graph does not move — a lock diff shows
  zero package changes — so there is no saliency or detection output to re-hash.
  The `tract-onnx`, `zenflate` and `zentract-api` git pins are left untouched
  (see issue #1), and no blanket `cargo update` was run, since that would drag
  the unpinned git dependencies to their branch HEAD.

### Fixed

- `clippy::unnecessary_map_or` in `zensally-zentract`: `map_or(false, …)` →
  `is_some_and(…)`. Newer clippy releases error on this under CI's `-D warnings`,
  so it would have turned the currently-green CI red on the next run.

### Changed
- `zensally-tract`: migrated from `tract-onnx` 0.22 to 0.23.5. `TypedRunnableModel` is no longer generic and `into_runnable()` now yields an `Arc`; tensor slice access moved to `Tensor::as_plain()` views, wrapped by the crate-private `output::plain_f32` helper so every detector keeps its `TractResult<&[f32]>` shape. No public API change.
- Workspace MSRV raised from 1.89 to 1.91 (required by tract 0.23) and `resolver = "3"` so the MSRV CI job (which deletes `Cargo.lock`) resolves MSRV-compatible dependency versions instead of the newest release.

### Fixed
- **Pushes to `main` now cancel their superseded CI runs.** `ci.yml` keyed its concurrency group on `${{ github.head_ref || github.run_id }}`. `github.head_ref` is populated only for `pull_request` events, so on a push it was empty and the group fell through to `github.run_id` — unique per run, so no two pushes ever shared a group and `cancel-in-progress` could never fire. Every push started a full matrix that ran to completion even when several commits landed seconds apart. Now keyed on `${{ github.ref }}`, which is set for both event types (`refs/heads/main` on push, `refs/pull/N/merge` on a PR), so PR cancellation is unchanged and consecutive pushes supersede each other.
- `windows-11-arm` CI (#1): `zensally-tract` now depends on `tract-onnx` at sonos/tract commit `9f6e4061` (PR #2718, merged 2026-08-25), whose `tract-linalg` build.rs assembles the ARM64 SIMD kernels with clang on `aarch64-pc-windows-msvc` instead of handing `.S` files to `cl.exe` (D9024 → LNK1181). Temporary git pin: cargo refuses a `[patch.crates-io]` override because the git tree is `0.23.6-pre`; revert to `tract-onnx = "0.23.6"` when it ships.
- MSRV CI job: tract 0.23 drops the `liquid`/`kstring` dependency chain that required Rust 1.96, which was failing `cargo hack check --rust-version`.

### Added
- `README.md` (GitHub) for the `zensally` workspace: badge row, quick-start smart-crop flow, core API overview, backend/detector tables, and the shared crosslink footer.
- Generated `README.crates.md` (CI-badge-only, crates.io surface) plus `readme = "../../README.crates.md"` in `crates/zensally/Cargo.toml` so crates.io renders the trimmed README.
- This `CHANGELOG.md`.
