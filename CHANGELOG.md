# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/), and the crates adhere to
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### QUEUED BREAKING CHANGES

None.

### Added

- Budgeted GPT-Image-2.6 Sunburst ecommerce-photo exploration, native response
  caching, exhaustive calculator parity and source-preserving review gallery.
  Synthetic fixtures remain separate from admitted training and real holdouts.

- Photographic crop-study scope for people, pets/living subjects, scenery and
  news/sales/ecommerce use, with opt-in proposal/review prompts and a metadata
  coverage audit that excludes document categories and preserves holdouts.
  New source candidates still require visual and rights admission.

- Current-JXL source-family reconciliation, TRAIN-only rendered weak fitting,
  independent Sol extreme-ratio audits, resumable bounded input loading and an
  optional bounded GPU input cache. Research checkpoints and human-review
  tooling do not change the shipping crop solver or certify face coverage.

- Direct, image-attached Terra ROI/crop-zone proposals in percentages, immutable
  response caching, rendered previews and native token-usage evidence. Cached
  responses remain distinct from admitted training labels or face certification.

- Experimental multiscale crop-region pooling and independent value/gradient
  checks, plus bounded synthetic training probes for CUDA compilation, graphs,
  precision/layout and optimizer choices. No artistic quality claim is made.

- Expand private crop annotation research to 32 imazen-26 sources and seven
  target ratios in both modes, with frozen roles, rendered-pixel provenance,
  separate placement-coverage auditing and a recorded Astra design consultation.
  No trained model or shipping crop behavior change is claimed.

- Replace the private annotation pilot's endpoint-only crop pair with
  center/interior and directly proposed Terra compositions, inspect rendered
  proposal pools, and record teacher usage separately from actual billing.

- Private crop-first research prototype: shared encoder, ratio/mode-conditioned
  candidate ranking, independent hard known-face constraints and explicit
  weak-label admission. Add an owned-source annotation pilot; no trained model
  or shipping crop behavior change is claimed.

- Extend the private crop study with two fixed blurred-letterbox backgrounds,
  exact model-input views, original/corrected overlays and links to cases.
  Eight methods cover 616,560 labelled crops; no dependency or weight added.

- Private aspect/crop evaluation example: six fixed strategies, human-mask
  crop oracles, overlapping refinement, bounded aspect-covering views, and
  a 29-case interactive gallery. Full results cover 5,019 natural sources and
  2,688 stress cases; defaults and model weights are unchanged.

- `zensally-micro`: opt-in ordered channel tiles and weight load-ahead, exact
  panel/tail checks, persistent image-worker study and heaptrack recipes
  (`6f0c862d`, `b6e32879`). Record throughput scaling and allocation patterns;
  no kernel promotion or allocation reduction is claimed.

- `zensally-micro`: opt-in split-reduction, algebraic-array and depthwise
  arithmetic studies, with a frozen same-process baseline and sudo perf
  counter/load-latency recipes (`ed0ce598`). Default arithmetic is unchanged.


- `zensally-micro` (unpublished): complete fixed w24 graph, strided NCHW input,
  scalar/SIMD contract, optional fast sigmoid and offline exporter. Uses existing
  owned inference kernels and specialized pointwise/depthwise operations.
- `zensally-depth` (unpublished): JPEG/HEIC embedded geometry extraction,
  strided depth/confidence views, bounded foreground cue and RGB fusion.
- Frozen validation-corpus rectangle comparison, operator checks, independent
  HEIC oracle and paired runtime measurements. Three map pixels per edge is the
  approved geometry tolerance; tested rectangle movement was zero.

- Primary-source reading list for using existing depth maps as a saliency cue,
  plus verified all-feature i686 validation evidence.


- Reproducible dependency/model payload report (`just dependency-weight`) and
  [saliency/runtime assessment](docs/SALIENCY_RUNTIMES.md). Correct the README
  MSRV and the historical claim that the committed v3ds models are selected
  by the detector; both backends still load v1.

### Changed

- `zensally-micro`: keep border math inside the dispatched SIMD context;
  correct an intermediate CPU regression (`84d927d5`). Final std and alloc-only
  allocation/memory measurements and unchanged rectangle checks are recorded
  in `benchmarks/w24_memory_cpu_2026-11-07.md`.

- `zensally-micro`: remove full depthwise padding tensors through direct
  interior/border reads while preserving tap order and zero-padding arithmetic.
  Bound alloc-only idle activation retention to 2 MiB; the std runtime keeps
  its per-execution reuse policy.
- `zensally-micro` + std: share immutable packed kernels by lane layout between
  active instances using weak cache entries. Each instance owns its execution
  buffers; dropping the final instance releases packed weights.

- `zensally-micro`: retain execution-owned activation/padding storage under
  both std modes, use fixed graph bookkeeping and stack concat references,
  and pack RGB input in one traversal. Update the owned NN kernels to remove
  serial row scheduling allocations, redundant packing clears, and fresh
  padding/cat buffers; new zeroed allocations are fallible through bytemuck.
  Ordered arithmetic and public signatures are unchanged. The first std
  heaptrack differential records one returned-map allocation per image.

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

- Pin zenlayout 6ed32dea's heatmap-mass placement repair. The reported stretch
  00084919 9:16 crop moves y=637 to y=60 and retains the labelled subject;
  original prediction metrics and crop sizes match throughout the complete
  rerun. Correct the original crop-quality interpretation.

- Pin the shared zenlayout Maximal crop integer-bounds correction and restored
  published `alloc` feature (`8cc30c38`; zenpipe `0007edcf`, `841ab33f`). Keep
  rounded crop extents while constraining their integer origins.

- Keep JPEG primary/auxiliary geometry in stored coordinates, retain EXIF
  orientation explicitly and check complete float/confidence output budgets.
  Unset decoder limits use consistent documented bounds.

- Pack partial 16-lane channel blocks when concatenating w24 skip connections. Preserve the shared aligned path and scalar/SIMD bits; complete-graph AVX-512 execution passes.

- Make the core's existing std feature effective. Default-free builds use alloc;
  preprocessing preserves standard float rounding without a new dependency.
  Bare-metal checks cover the depth and inference crates, with opt-in std
  propagated through their owned SIMD dependencies.

- Pass an explicit project-relative corpus cache into cross so i686 runs the real HEIC depth regression with a writable cache.

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
