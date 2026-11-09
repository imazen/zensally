# Selected photographic student runtime — 2026-11-08

Human artistic quality and complete face preservation remain unmeasured. This
study qualifies arithmetic for the pre-TEST-selected relative-rank student;
it does not promote a crop model or certify teacher-equivalent rectangles.
[Training/quality findings](unsplash_rank_distillation_2026-11-08.md) remain separate.

## Export and crop gate

The selected 238,629-parameter seed 911/epoch 64 student exports as static batch 1
FP32 ONNX opset18, including all exported weights:

| Graph | Bytes | Nodes |
|---|---:|---:|
| Encoder |667,688|75|
| Query |339,949|448|
| Complete |1,007,616|523|

These are graph sizes, not executable/dependency sizes. ONNX Runtime remains
required for this experiment. The existing native FocalNet query benchmark
does not measure this newly trained student's head.

All 144 sources ×22 ratio/mode requests are compared against the same frozen
student on CUDA, FP32 with TF32 disabled, batch 1. The unchanged gate allows
at most3 source-normalized128-grid pixels on every edge independently.

| Runtime | Requests | Changed choices | Gate violations | Maximum edge movement | Maximum absolute score error |
|---|---:|---:|---:|---:|---:|
| Torch CPU |3,168|1|0|0.068966|0.000726522|
| ORT cached encoder/query |3,168|3|0|0.017106|0.000600577|
| ORT complete graph |3,168|3|0|0.017106|0.000600577|

Coordinates normalize x and y by their respective source dimensions. This
comparison never uses teacher ranking scores, updates weights or reselects a
model. A subsequent1,100-request gate checks all 50 original development/TEST
sources against their frozen batched predictions: CUDA batch 1 changes one
choice, ORT batch 1 changes two, maximum movement0.053602 map pixels for both,
and zero gate violations. This passes for both gather variants described below.
All face/acceptance guards remain separate; raw ranking alone cannot authorize
an offered crop.

The first export wrapper accidentally restored training mode recursively after
export, invalidating its Torch CPU reference. That attempt is retained. Fresh
wrappers now enter evaluation mode explicitly, and export must preserve every
parameter, batch-normalization buffer and evaluation flag. The corrected full
gate takes 41 s: run-heavy peak RSS1.40 GiB, minimum available20489 MiB,
peak load13.81. CUDA peak allocated18,293,760 bytes/reserved27,262,976 bytes.

## Validation launch reduction

A private CUDA graph replays only the fixed-shape candidate query. Each
evaluation copies current scorer parameters into retained storage and refreshes
feature buffers once. The original encoder, masking, nonfinite checks,
reductions, CPU metrics and prediction construction remain unchanged.
Every selected model is checked, including changed head parameters and
geometry/permutation controls:844,800 FP32 scores match by bits. All 12 original
metric/prediction records, model state and RNG state remain unchanged.
This qualification takes 2 s: run-heavy peak RSS1.56 GiB,
minimum available23460 MiB, peak load13.04. CUDA allocated269,370,368 bytes,
reserved408,944,640 bytes.

Four paired exploratory zenbench blocks on i265, CPU boundary2–3 and RTX4080,
measure the whole25-photo/22-query development evaluation at 30.2563 ms eager
and 23.0328 ms with query replay, a23.8737% reduction. A/A change is+0.6524%.
Twelve invocations record background CPU load above5%; this is not a clean-resource
confirmation. CUDA event intervals include submission gaps and explicit end
synchronization. Model/cache loading and graph capture/warmup are excluded.
The run-heavy wrapper reports 33 s, peak RSS1.57 GiB, minimum available23408 MiB,
peak load12.54; its outer accounting does not replace per-target memory telemetry.
No complete training-speed improvement is claimed from validation timing alone.

A second bounded training capture includes the qualified replay: the same 200
optimizer steps and 33 development evaluations span2.909138920 s between first
and last kernels, versus3.342195929 s in the eager capture. Development host
ranges total0.924297981 s versus1.380006412 s. This is instrumented attribution,
not paired whole-training timing. CUDA peak allocated602,239,488 bytes and
reserved887,095,296 bytes increase; the actual target retains its verified4 GiB
host cap. Target run-heavy reports 35 s, peak RSS2.91 GiB, minimum available22092 MiB,
peak load13.76; GNU time maximum RSS3,047,680 KiB. Graph replay does not produce
the original per-query host NVTX ranges, and kernel counts exclude memcpy/memset
events, so they are not a measure of total work or SM occupancy.

## Whole CPU pipeline and sampler specialization

All following means measure25 fixed development photos, one encoder per photo,
then either one square-Minimal query or the whole11-ratio/22-mode request bank.
ORT input/output bindings and buffers are retained. Decode, resize, model/cache
load and shipping face/acceptance guards are excluded. Image analysis can use
one/two/four threads while the query executor uses one. These are whole-pipeline
measurements, never sums of separately timed means.

The first four-block exploratory study measures5.4373/4.8720/4.7313 ms for one
query with 1/2/4 analysis threads, and 88.6809/89.1603/90.6830 ms for the 22-query
bank. Adding encoder threads does not help the query-dominated bank. A/A changes
are−0.3100% and+0.0635%. Run-heavy reports 277 s, peak RSS0.71 GiB,
minimum available21063 MiB, peak load15.26. All 32 invocations observed background
CPU load above5%; no clean-resource certification is claimed.

Sudo perf on one P core reports 2.6% memory-bound and 12.5% core-bound slots for
the complete profiling command, including setup/warmup. GNU-time maximum RSS is
737,928 KiB for the counter pass and 738,680 KiB for call sampling, including Python,
Torch reference/setup and cached inputs; these are not standalone deployment
memory costs. Run-heavy reports 79 s, peak RSS0.70 GiB, minimum available23779 MiB,
peak load13.20. The wheel is stripped, so sampled ORT offsets alone do not name
the operator. A separate six-source/132-request ORT node trace identifies
GridSample at 468,247 of 553,062 query operator microseconds (84.66%); dense Gemm
accounts for 16,793 us. The instrumented trace is attribution evidence, not timing
used for the speed claim. It takes 2 s, peak RSS0.78 GiB, minimum available24540 MiB,
peak load13.01.

A fixed finite-coordinate, border, align-corners bilinear gather replacement
is bound to the exact original graph hashes. It keeps all weights and passes
the same 3,168-request gate, including the same three small ORT choice shifts.
The complete graph is1,040,870 bytes; encoder667,688, query373,203.
Four paired blocks compare the original and gather pipelines directly:

| Query workload | Original, one analysis thread | Gather, one analysis thread | Gather, two | Gather, four |
|---|---:|---:|---:|---:|
| One crop |5.4716 ms|3.1932 ms|2.6170 ms|2.3491 ms|
| All 22 requests |89.0534 ms|37.8157 ms|37.6801 ms|38.0746 ms|

Single-thread gather reduces one-query time41.6392% and bank time57.5356%.
A/A changes are+0.8899%/+0.5935%. Run-heavy reports 265 s, peak RSS0.72 GiB,
minimum available23357 MiB, peak load14.29. The arithmetic qualification takes 28 s,
peak RSS1.42 GiB, minimum available22549 MiB, peak load14.23. All results remain
exploratory under the retained background observations.

A further contiguous-channel Gather variant removes the per-channel index
broadcast and also passes all numerical gates. It is retained as an unsuccessful
optimization: its one-thread query bank measures39.7940 ms against37.6251 ms for
the first gather in the same paired study (+5.7692%; A/A+0.6863%). One-query
time is3.2489 ms against3.1819 ms (+2.1591%; A/A−1.7471%). The first gather remains
the preferred variant in this screen. The second graph is1,036,876 bytes;
size reduction alone does not establish faster execution. Its qualification
takes 26 s, peak RSS1.41 GiB, minimum available23053 MiB, peak load13.63; paired
timing takes 189 s, peak RSS0.72 GiB, minimum available23701 MiB, peak load14.24.
Both original-batched-prediction gates take5 s each, peak RSS1.34 GiB; minimum
available23115/23175 MiB and peak load14.22/13.96 respectively.

The post-change132-request node trace totals264,387 query operator microseconds,
with 57,111 us in GatherElements and 17,133 us in Gemm. Sampling is no longer one
generic GridSample operator; elementwise work and expanded indices are visible
in the retained per-operator profile. The instrumented trace takes 1 s, run-heavy
peak RSS0.87 GiB, minimum available21638 MiB, peak load13.21. Both sampler-layout
profiles are retained separately from the uninstrumented timing packets.

## Bounded visual review

The fixed review shows all three detector-positive development sources and
three non-positive sources chosen by a registered source-ID hash: six photos,
132 ratio/mode requests and 396 teacher/student/geometry comparison crops.
Browser checks cover180 crop views across five ratios and two modes, with six
source images decoded and no clipping/stretching or JavaScript failures.
Building takes 1 s, peak RSS0.61 GiB, minimum available22424 MiB, peak load13.57;
browser checking1 s, peak RSS0.20 GiB, minimum available24635 MiB, peak load12.76.

An unblinded Codex visual spot-check finds a catalogue failure on
`unsplash:MqOGVcLcJ3E`: at 3:1 Minimal both teacher and student choose native
`[0,970,4233,2381]`, centering the hand and removing most of the wallet and its
brand. The full wallet is taller than the feasible3:1 crop. Catalogue use
therefore needs containment/padding under the approved whole-product contract.
This is a development witness, not a human rating, training label or complete
face certificate. Better relative teacher agreement does not resolve it.

All finished and failed attempts have [complete NAS verification](unsplash_student_runtime_2026-11-08.pointer.md).
