# Existing-model runtime evaluation, 2026-11-08

Use UltraFace plus the existing v1 saliency adapter for an immediately available
Rust baseline: the measured serial pair takes 9.32 ms. Where person retention
matters, UltraFace plus SelfieSeg takes 12.22 ms. These outputs provide geometry
and soft crop features; they do not grade artistic composition. Reuse an analysis
across ratio requests rather than rerunning the models for each crop.

No detector here satisfies an unconditional never-cut-face guarantee. Keep
provided protected regions and containment fallback. Do not certify complete
face coverage from empty detections or the small diagnostic below.

## Scope and timing contract

Evaluated 18 model/runtime combinations on 76 sources: all 56 generated AI
photos and 20 photographic imazen-26 sources already in TRAIN fit/development.
No canonical VALIDATE/TEST image is opened, no new fit runs, and no paid API is
called. Existing owned-decoded native AI RGB and owned aspect-preserving real
previews are used; exact paths, dimensions and hashes are pinned in the manifest.

Warm timings use six native AI images: dog-harness-white, speaker-pair-white,
watch-face-context, selfie-couple-night, selfie-family-hike and selfie-mirror-outfit.
Each fresh process warms one complete six-image sweep, then measures 20 sweeps.
Numbers below are means of four process means, not p95 latency. They include
adapter/Task preprocessing, inference and returned postprocessing. Model/image
load, file IO and result visualization are excluded. MediaPipe image construction
is outside timing; Rust ImageRef construction is included.

All five zenbench Iterate studies complete, with A/A controls and enforced
cpuset CPU 2 on i265 (Core Ultra 7 265K performance core). No target-cpu=native.
Background CPU load is observed in all 124 invocations; cleanliness is not
certified. Different model outputs are explicitly unchecked for equivalence.
These are exploratory deployment-cost observations, not Confirm acceptance or
cross-machine speed claims. See the [packet pointer](runtime_models_2026-11-08.pointer.md).

## Measured warm calls

| Runtime / model | Mean ms | Role / finding |
|---|---:|---|
| Rust tract UltraFace | 6.990 | Available general face baseline |
| Rust tract front BlazeFace, corrected | 1.782 | Fast but misses group/distant faces |
| Rust tract BlazeFace-320 | 17.489 | No speed advantage in this study |
| Rust tract YuNet | 24.709 | Better diagnostic center coverage, extra detections |
| Rust tract SelfieSeg square | 6.121 | Person retention cue |
| Rust tract MicroSalNet v1 | 2.941 | Existing default saliency, not w24 |
| Rust tract U2Netp | 584.331 | Too costly for this fast path |
| MediaPipe CPU short BlazeFace | 0.872 | Near/selfie face specialization |
| MediaPipe CPU full-range BlazeFace | 2.476 | Candidate face adapter |
| MediaPipe CPU sparse full-range BlazeFace | 2.480 | Smaller file, no demonstrated timing gain |
| MediaPipe CPU Selfie Segmenter square | 3.108 | Better distant-person retention in inspected hike |
| MediaPipe CPU Selfie Segmenter landscape | 1.863 | Faster; largely loses the fourth hiking person |
| MediaPipe CPU EfficientDet-Lite0 int8 | 13.801 | Useful objects/pets, limited product categories |
| MediaPipe CPU SSD MobileNetV2 f16 | 14.850 | Similar category limits |
| MediaPipe CPU Pose Landmarker Lite, up to 8 poses | 12.111 | Optional soft joint-cut features |
| MediaPipe CPU Face Landmarker, up to 16 faces | 4.692 | Extra facial detail, not distant-face coverage |
| MediaPipe CPU Hair Segmenter | 14.470 | Extra cost; no crop-quality gain established |
| MediaPipe CPU Selfie Multiclass | 66.706 | Extra cost; no crop-quality gain established |

MediaPipe 1.2.2 uses its CPU reference Task implementation, with XNNPACK/LiteRT
reported in the face logs. These C++ runtime numbers do not predict a Rust port's
latency. Rust detector confidence/NMS defaults and MediaPipe confidence 0.5
are preserved; object score threshold is 0.3. This is not a matched-threshold
comparison or numerical backend parity test.

Separate, actually executed Rust serial pairs: UltraFace + v1 saliency 9.320 ms;
UltraFace + square SelfieSeg 12.221 ms. Their matched UltraFace baseline is
6.854 ms. No pair number is obtained by adding separately measured rows.

## Visible failures and face fallback

A posthoc agent visual audit marks 19 face centers in six synthetic scenes:
two birthday groups, the four-person hiking selfie, couple selfie, mirror selfie
and solo bus selfie. A one-to-one point-inside-detection check yields:

| Adapter | Matched centers / 19 | Returned detections in those scenes |
|---|---:|---:|
| Corrected Rust front BlazeFace | 8 | 8 |
| MediaPipe short BlazeFace | 9 | 9 |
| Rust BlazeFace-320 | 14 | 15 |
| MediaPipe full-range | 16 | 17 |
| MediaPipe sparse full-range | 17 | 18 |
| Rust UltraFace | 17 | 18 |
| Rust YuNet | 18 | 21 |
| Sparse full-range with five views | 18 | 19 |

These are approximate agent-marked points, not human labels, IoU recall, head
containment or a representative statistical benchmark. Counts alone do not
establish detection precision. The marked centers remain training-ineligible.

Five views means the whole image plus four overlapping 75%-width/height windows,
then source-coordinate remapping and greedy NMS at IoU 0.3. Its 12.508 ms mean
includes view copies and merge work; the matched single-view baseline is
2.411 ms. The extra center costs substantial work. Prefer a conditional fallback
to indiscriminate tiling; its trigger and general quality remain unvalidated.

Square SelfieSeg retains the distant fourth hiking person where landscape
largely misses them. It also includes the carried tote in one commerce scene,
so person masks are soft retention cues, not precise product segmentation.
EfficientDet recognizes the dog and bottles but calls a speaker a vase and
returns only a person for the tote-model image. Do not use these COCO classes
as hard product/branding constraints. No brand-text protection is established.

## Correctness repair found during evaluation

The existing front decoder generated stride-16 anchors in layer order. The
converted ONNX has 6-score/96-regressor channels, transposes NCHW to NHWC and
then flattens, requiring six anchors per cell. The correction merges equal-stride
layers before visiting cells, matching the
[MediaPipe anchor generator](https://raw.githubusercontent.com/google-ai-edge/mediapipe/master/mediapipe/calculators/tflite/ssd_anchors_calculator.cc).
The new regression fails on the original decoder, then passes; all four crate
unit tests and scoped clippy pass. No threshold or established assertion changes.

On the real close-up source 12008, three misplaced overlapping detections become
one centered detection. The six-scene center score remains 8/19. Pre-fix outputs
and binary are retained, and corrected outputs are separate. Initial Rust table
rows except corrected front use the pre-repair binary; pair followups use the
updated example. No model weights or default detector selection changes.

## Native TFLite feasibility

An isolated development tool uses tract-tflite 0.23.10, outside shipped dependency
graphs. Two of five targeted imports execute finite zero-input forwards:
short-range [1,896,16]/[1,896,1] and dense full-range
[1,2304,16]/[1,2304,1]. Sparse full-range fails while reading a compressed sparse
F16 constant (154 bytes versus 512 expected). Both raw selfie files fail at custom
operator import; the existing Selfie ONNX adapter remains available.

Raw graph timing is a distinct study: short-range 1.371 ms, dense full-range
5.010 ms, one preallocated zero tensor, one warm forward then 200 forwards per
process. It excludes resizing, normalization, box decode and NMS and supplies
no numerical reference comparison. Dense full-range is a plausible next Rust
adapter, not a completed or faster face detector. Preserve native aspect/box
mapping and validate actual outputs against the reference before adoption.

## Hardware counters and resources

Separate sudo perf stat runs use tma_memory_bound/tma_core_bound, with 1,800 warm
calls per model, one performance core and GNU time -v RSS. Counters cover the
whole process including initialization/import, not just model kernels:

| Profile | Memory-bound slots | Core-bound slots |
|---|---:|---:|
| MediaPipe sparse full-range | 8.6% | 38.5% |
| MediaPipe square Selfie | 10.0% | 28.5% |
| Rust UltraFace | 11.9% | 29.0% |
| Rust Selfie | 13.2% | 18.8% |

Core-bound exceeds memory-bound in each observation; this does not show a
memory-latency-dominated workload. TMA categories are slot classifications, not
DRAM latency measurements or predicted speedups. Frontend/retiring/speculation
also matter; no measured optimization is claimed.

Resource wrapper observations: reference comparison peak-RSS 0.33 GiB; initial
Rust comparison 0.58 GiB; face followup 0.35 GiB; pair followup 0.36 GiB; raw-graph
comparison 0.02 GiB. Profiling wrapper:
`run-heavy: done rc=0 61s | peak-RSS 0.36GiB | min-avail 24041MiB | peak-load 13.64`.
Full logs and each GNU time maximum RSS are retained. These are complete worker
footprints with input caches, imports and multiple embedded models, not isolated
model activation memory or shipping binary size.

The initial nested root scopes did not inherit the outer wrapper's memory cap.
The Rust scope received an explicit cap during that run; later root scopes
start with MemoryMax=4G/MemorySwapMax=0. This operational correction and all
original attempts remain in the packet. An incomplete default Cargo source
cache was bypassed with a separate clean Cargo home, without changing versions.
The first reference mask-summary attempt expected 2D data; the returned HxWx1
view is now handled explicitly, and all 11 models complete evaluation. None of
these original failures is hidden or treated as timing/quality evidence.
