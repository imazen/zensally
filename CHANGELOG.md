# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/), and the crates adhere to
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### QUEUED BREAKING CHANGES

None.

## [0.2.0] - 2026-11-11

### QUEUED BREAKING CHANGES

None.

### Added

- `zensally-micro` (`publish = false`, `#![no_std]`, `#![forbid(unsafe_code)]`):
  NHWC micro-graph inference kernels (`conv3x3_s2_nhwc`, `depthwise_conv3x3_nhwc`
  with stride 1/2 and dilation 1/2/4/8, `pointwise_conv1x1_nhwc`,
  `pointwise_conv1x1_residual_nhwc`, `bilinear_upsample2x_add_nhwc`, `se_gate_nhwc`,
  `centernet_peaks_3x3`) dispatched via `archmage` + `magetypes` (`v4x`/`v4`/`v3`/
  `neon`/`wasm128`/`scalar`) over a zero-allocation `ScratchArena` (`F32` and `F16`
  ping-pong modes), plus `zenbench` kernel tier benchmarks and multi-tier parity tests.

- `zensally` (`pub(crate)` v0.2 Phase 1 internals): fused single-pass strided/streaming
  NHWC downsample + OKLab chroma dispersion + 3x3 Sobel gradient + horizon continuity
  (`stream_prep`) and dual summed-area table (`DualIntegralMap`) $O(1)$ arbitrary-aspect
  `Minimal`/`Maximal` crop solver + distant-subject Tile-on-Demand trigger (`sat_crop`),
  with zero additional core dependencies (`core: 0`) and detached `tools/sat-crop-bench`
  `zenbench` suite.

- `zensally-micro` (v0.2 Phase 2 `W24MicroModel` fixed-graph NHWC executor):
  pure-Rust NHWC execution of `microsalnet_v3ds_w24.onnx.gz` (`W24MicroModel`,
  `W24Workspace`, `W24Stride8Features`, `tools/export_w24_nhwc.py`, SHA-256 verified
  `w24_plan`), high-precision Cody-Waite + degree-7 Taylor `exp_hp` (`1.25e-7` max
  relative error through `Sigmoid`), `Activation::HardSigmoid`, transposed 2x2 stride-2
  decoder convolutions with fused skip concat (`conv_transpose2x2_s2_nhwc`,
  `conv_transpose2x2_s2_cat_nhwc`, `conv1x1_head_sigmoid_nhwc`), bit-exact cross-tier
  parity (`max_abs_diff <= 4.89e-6` vs `tract-onnx`, 0 warm heap allocations, `7.8 ms`
  AVX2 forward pass).

- `zensally` (`pub(crate)` v0.2 Phase 2 internals): `DualIntegralMap::from_crop_inputs`
  fusing w24 saliency (top-0.5% robust peak normalization), SelfieSeg matte gated by
  saliency support, YuNet faces + 5-point landmarks, manual focus regions, classical
  energy and horizon cues into a unified foreground/energy map plus dual fixed-point
  SATs. `solve_crop_sat` now protects padded landmark cores (eye+brow hard veto,
  covered-fraction fallback anchoring the top edge at the brow, horizontal symmetry on
  narrow crops), interior salient-object bisection penalties, and subject top-apex
  decapitation protection. Detached `tools/sat-crop-bench` gained the `eval_crop_v2`
  real-image evaluation harness (8 corpus images × 8 aspect/mode cells vs the v0.1
  `zenlayout::smart_crop` path) and `eval_corpus_v2`, a rayon-parallel evaluator over
  the frozen 741-photo corpus manifest (all 8 cells per image, YuNet-landmark oracle
  metrics, per-image/per-cell `corpus_results.json` + `corpus_data.js` +
  `SUMMARY.md` for the client-rendered split-switching `index.html` viewer).

- `zensally` (v0.2 public API): the `sat_crop` and `stream_prep` modules are
  now public. `sat_crop` gains `solve_crops_sat` for batching several
  `(aspect, mode)` requests over one shared `DualIntegralMap`, and
  `AxisMap`, `TargetToSrc`, `FaceLandmarkInput`, `SatCropMode` and
  `SatCropConfig` derive `serde` traits under the `serde` feature.

- `zensally-tract` (v0.2 detectors): `YuNetDetector::detect_with_landmarks`
  decodes the model's 5-point facial landmarks into
  `sat_crop::FaceLandmarkInput`; `MicroSalNetV3` wraps the committed
  `microsalnet_v3ds_w24` model via `zensally_micro::W24MicroModel`;
  `MicroCropAnalyzer` (also available directly in `zensally-micro` without
  `tract-onnx`) runs the ~5.3 ms fast v0.2 pipeline (`stream_prep` →
  `downsample_512x384_to_256x256_nhwc` → `W24MicroModel` saliency + `d3`
  foreground matte → `DualIntegralMap` → `solve_crop(s)_sat`); and the
  `analyzer_v2` feature (`yunet + microsalnet + selfie_seg`) adds
  `CropAnalyzerV2` — the full v0.2 pipeline (`stream_prep` → `W24MicroModel` +
  `SelfieSeg` + `YuNetDetector` → `DualIntegralMap` → `solve_crop(s)_sat`)
  with reusable 512×384/256×256/64×48 buffers.

  On the frozen 741-photo corpus evaluation (8 aspect/mode cells per image,
  YuNet-landmark oracle), the v0.2 pipeline cut protected face cores in 25
  cells vs the v0.1 stack's 379, raised primary-face retention to 92.9%
  from 85.7%, and improved Maximal-crop saliency concentration to 2.08 from
  1.55 — at ~5.4 ms `stream_prep`, ~0.4 ms map build and ~0.4 ms for all
  eight solves per image.

- Local native CUDA/WebGPU inference probes, paired numerical checks and
  profiler evidence for YuNet; separate from the Rust detector backends.

- Standalone native reduced ORT/YuNet deployment probes and manual Windows
  x64/ARM64 checks; [measurement contract](docs/NATIVE_ORT.md).

- Primary-source reading list for using existing depth maps as a saliency cue,
  plus verified all-feature i686 validation evidence.


- Reproducible dependency/model payload report (`just dependency-weight`) and
  [saliency/runtime assessment](docs/SALIENCY_RUNTIMES.md). Correct the README
  MSRV and the historical claim that the committed v3ds models are selected
  by the detector; both backends still load v1.

### Changed

- CI tests all detector features on supported platforms, adds i686 via cross,
  removes the macOS Intel job, and updates checkout/cache action majors.


- `zensally-tract`: use published tract-onnx 0.23.10 with optional transformer
  operators disabled; adapt the private tensor accessor to `as_plain_ram()`.
  The Windows ARM assembler fix is included in this release.
- Both backends now use published zenflate 0.4.1 and require enough 0.4.5.
  Pin the plugin API to the reviewed zentract revision instead of branch HEAD.
- Use zenbench 0.1.11 for both benchmark targets, replacing Criterion in the
  face benchmark. No model, public API or test expectation changes.
- `just verify-update` covers all features, examples and benchmarks through
  Clippy, all-feature tests, doctests and no-default-feature core checks.

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

- `solve_crop_sat` protected only the *padded* landmark core, so a crop too
  small to fit the padding on an axis earned the same intersect-without-contain
  veto for shaving expendable forehead/brow margin as for slicing actual eye or
  mouth features — and, since coverage reward was unreachable there, the solver
  preferred clipping the mouth over trimming padding. `FaceCellBox` now also
  stores the raw landmark band and the per-axis containment target falls back
  to it when the padded core cannot fit: a hard cut penalty applies when the
  band itself is sliced, a softer one when only padding is shaved, and the
  covered-fraction reward fires on every intersecting candidate. Found by the
  741-image corpus evaluation: all single-face `16:9`/`3:4` Maximal misses are
  eliminated (remaining cuts are forced crowd overlap or frame-edge pad shaves).

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

### Development profiling

- Add an isolated Coz preprocessing worker with full tensor-bit and letterbox-metadata checks; inference and postprocessing are not measured by this worker.
