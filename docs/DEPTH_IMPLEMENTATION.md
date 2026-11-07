# Embedded depth and fixed-graph saliency implementation

The requested scope includes JPEG and HEIC embedded-depth extraction, a reusable
geometric saliency experiment, current-main codec integration with patches in
PRs, recovery/search of earlier depth abstractions, and a custom w24 runtime
verified faster than tract on the same model and inputs.

## Source baseline

Resolved on 2026-11-06:
- zencodec main: `1fb997bb88e7405e6a315bf1f0b22f71e99be272`.
- zenjpeg main: `cae3a9f38786803a20575474792ead2a7d418f3b`.
- heic main: `4d72caddb4c1d664237a8d3693f35d69ac80caaf`.

Searched all fetched branches of these three repositories for shared decoded
DepthMap/DepthRepresentation types. Existing reusable types are JPEG's
`DepthMapData` (introduced by `b3a693c9`) and HEIC's `DepthMap` and representation
metadata. zencodec has `Supplements::depth_map`, but no decoded-depth payload on
the branches searched. Also searched zenpipe/retired zenimage source.

## Acceptance gates

- Decode real embedded JPEG and HEIC depth payloads with existing codec code;
  exercise GDepth/DDF/MPF, confidence, representation and alignment semantics.
- Validate strided depth views, missing/invalid-depth behavior, orientation and
  meaningful foreground/background examples. This remains an experimental cue;
  do not claim demonstrated crop-quality gains without measuring them.
- Reuse existing owned inference kernels before extending missing operations.
- Run the committed w24 weights, all operators and complete fixed graph; compare
  outputs against a separately implemented reference and tract. Report numerical
  disagreements explicitly rather than hiding them behind crop thresholds.
- Profile and benchmark custom and tract w24 under identical preprocessing,
  release settings, thread count and inputs. Record full commands and paired
  measurements. Faster-than-tract claims require those measurements to pass.
- Submit public API and dependency changes as PRs assigned to lilith. Do not
  publish crates as part of this work.

## Implemented and verified on 2026-11-06

Both new crates default to `no_std + alloc`; codec extraction opts into `std`
through `jpeg` and `heic`. They are unpublished workspace crates with Rust 1.93
requirements. Root patches pin the codec/pixel cohort, including review commits,
so no private path configuration is needed to reproduce the consumer build.

The complete w24 graph executes all 36 Conv, 26 ReLU, seven Add, two ReduceMean,
three Sigmoid, two Mul, three ConvTranspose and three Concat nodes. Batch
normalization was already folded in the committed ONNX. The exporter validates
the supported attributes, shapes and initializer table; `just micro-check`
checks regeneration and independent planar operator references. No training
optimizer or deployed general graph optimizer is needed.

The user selected a fixed scalar/SIMD arithmetic contract and numerical tract
validation, then approved limited arithmetic approximation when rectangle edges
stay within **three saliency-map pixels**. This is evaluated on the 128×128 map:
source x edges multiply by 128/source_width, y edges by 128/source_height.
Salient bounds (relative thresholds 0.3 and 0.5) and final shared-solver crops
(nine aspect ratios, Minimal/Maximal) are checked separately. Presence mismatch
fails. This gate does not authorize approximation in the depth/image decoders.

Across 257 validation images and 992 size cases, both fixed and fast-sigmoid
paths had zero measured rectangle movement against tract. The fast sigmoid's
largest map-value difference was 1.043081284e-5. Runtime and evidence limits are
recorded in [the measurement report](../benchmarks/w24_2026-11-06.md).

JPEG integration tests use the existing encoder to create GDepth, Extended XMP,
Dynamic Depth appended PNG/confidence and MPF disparity carriers. Native 16-bit
codes, Google inverse range, default format, millimeter normalization, missing
maps and malformed payloads are covered. A real portrait HEIC's 576×768 native
codes match the independent libheif oracle exactly after a deblocking QP fix.
The corpus source and hashes are in `tests/heic_corpus.rs`; no images were added
to this repository.

There is no measured depth-driven crop-quality improvement, no learned depth
fusion, no model quantization, and no warm inference-only heap comparison.
Separate executable-size and cold-process RSS probes are recorded below.
General ONNX backend constructors continue to select v1.

## Dependency review status

Draft reviews: [HEIC #67](https://github.com/imazen/heic/pull/67),
[JPEG #259](https://github.com/imazen/zenjpeg/pull/259),
[PNG #57](https://github.com/imazen/zenpng/pull/57),
[NN kernels #2](https://github.com/imazen/zenjpegai/pull/2),
[zenflate #20](https://github.com/imazen/zenflate/pull/20),
[ultrahdr #44](https://github.com/imazen/ultrahdr/pull/44), and
[zentone #32](https://github.com/imazen/zentone/pull/32).

The consumer's all-feature tests and Clippy pass with public git pins. Several
standalone dependency checks remain blocked by other dev/optional crates' SIMD
requirements, including zensim's explicit upper bound `<=0.9.31-beta.1`, whereas
the current pixel cohort requires beta.2. Those bounds and their numerical
contracts have not been relaxed. This is the earlier dependency-cohort audit;
it does not claim all standalone reviews green. The NN follow-up at 677d69be
now passes every required CI platform, independent operators, all 50 stored
decoder-reference tests and all 16 CLI model roundtrips. Its full encoder
reference suite was not rerun.


The default depth/runtime libraries also pass a real `aarch64-unknown-none`
bare-metal check with public git dependencies. This required making the core's
existing std feature effective, preserving its resize rounding through a private
core-only conversion, and opting std into the NN dependency chain. No new core
runtime dependency was introduced. A sampled float-bit oracle preserves the
existing rounding expectation; full model scalar/SIMD tests pass with and
without std. CI now repeats the bare-metal check.

The initial consumer i686 job compiled successfully but could not create its
corpus cache under the container HOME. A caller-selected project-relative cache
and the same commit's tar archive resolve this without skipping the real-image
regression. Local cross execution passed the complete depth hash and all adapter
samples; fixture image SHA256 and native-depth expectations are unchanged.


## Complete-image fusion example

`just depth-crops heic INPUT REPORT.tsv` (or `jpeg`) decodes the primary RGB
image and embedded geometry, runs w24, and compares shared-solver crops at
unfitted fusion amounts 0, 0.25, 0.5 and 1. This is a reproducible experiment,
not a model-quality calibration or a recommended default blend.

On the pinned portrait HEIC, the primary was 3024×4032, depth 576×768 and cue
192×256. Mean valid depth confidence was 1.0. Of 18 aspect/mode crops, amounts
0.25, 0.5 and 1 changed 11, 13 and 16 crops respectively; maximum source-edge
movements from RGB-only were 189, 252 and 945 pixels. These are intentional
fusion changes and are separate from the runtime's three-map-pixel numerical
approximation gate. Without crop ground truth, movement is not evidence of
improvement. The square Minimal crop was unchanged at 0.25/0.5, and its top edge
moved from 1008 to 687 at amount 1. Full coordinates are retained in
[the small result table](../benchmarks/depth_fusion_portrait_2026-11-06.tsv).

This run used the same source image/hash as the depth-oracle regression. It
completed under run-heavy with peak RSS 0.63 GiB, minimum available 25992 MiB,
peak load 1.39. RSS covers release compilation plus the complete-image command,
not a standalone inference measurement.


Final deployment probes link the fixed runtime and tract into separate programs,
with the same w24 payload, input and complete first inference. Their stripped
Linux executable sizes are 1,895,232 and 23,759,936 bytes; cold-process max RSS
is 26,636 and 29,900 KiB respectively (one process per arm, GNU time -v).
See [final measurements](../benchmarks/w24_final_2026-11-06.md).

JPEG Dynamic Depth and GDepth now share meter/millimeter normalization. Stored
JPEG primary and auxiliary coordinates retain an explicit EXIF orientation;
a non-square Rotate90 case verifies the convention. Unset decoder caps are
64 megapixels, 512 MiB per-codec memory and 256 MiB encoded input; returned
float/confidence output budgets are checked together. JPEG extraction currently
uses a primary decode to obtain preserved extras; image-level speed savings
from a metadata-only extraction path have not been measured.

Named hardware validation uses `ZENSALLY_REQUIRED_TIER` through
`just micro-check-tier`: a requested v4 run fails if it selects another tier.
The full graph and bit contracts passed with v4 explicitly required.


The [2026-11-07 memory/CPU follow-up](../benchmarks/w24_memory_cpu_2026-11-07.md)
implements inference-owned scratch, direct padding reads, shared active packed
weights and fixed graph bookkeeping. Matched std standalone RSS is now
90,732 versus 15,304 KiB, with 190.78 versus one allocation per image.
Those measurements supersede the earlier runtime's memory/allocation figures;
the earlier executable-size probes remain historical. Updated paired scaling
confirmations fail shared-host resource gates and do not establish a new
throughput improvement. Both modes retain zero rectangle movement over the
same 1,984 frozen comparisons.

The [corrected crop and blurred-padding study](../benchmarks/w24_crop_placement_blur_2026-11-07.md)
compares eight fixed inference strategies after repairing shared heatmap-only
placement in zenpipe 6ed32dea. All 616,560 labelled crop comparisons enforce
positive, source-bounded rectangles. The reported stretch 9:16 failure was a
solver defect; its prediction is unchanged while retention rises from 0% to
100%. The original six-arm prediction metrics and crop sizes are preserved.
The one-pass default stays stretch; refinement has a small natural crop gain
and a larger controlled-extreme gain. Blurred backgrounds remain tool-private
and do not improve on stretch. No weights or public runtime APIs change.
