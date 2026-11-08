# Local crop teachers and constrained search — 2026-11-08

The useful result is candidate recovery, not a promoted artistic model. Exact
integer containment recovers 129 of 477 coarse-bank abstentions while retaining
all supplied head bounds. Local direct-crop and extent teachers still need
independent photographic quality evaluation. No paid API was called; no new
inference dependency or canonical holdout assignment changes.

## Models, inputs and numerical screen

Pinned [FocalNet](https://github.com/appwrite/focalnet/tree/f3c55aeb5827f302d1bbe9f0f41da38139f3dc36)
has 4,757,694 parameters, including a 7,069-parameter ranking head. Its ONNX file
is 20,398,992 bytes; the Torch checkpoint is 19,596,175 bytes. The comparison
executes on the 136 previously admitted fit/development/annotation sources.
No canonical VALIDATE/TEST image enters this study.

Inputs use owned zenresize Lanczos aspect-preserving 256-square letterboxes,
ImageNet normalization on content and zero normalized padding. This differs
from upstream bilinear preprocessing. CPU ONNX Runtime 1.31.1 uses one thread;
16 Torch-reference witnesses have maximum absolute score difference
0.000005841255187988281 and no argmax change. Maps required internally by the
model are transient; no diagnostic maps are saved.

On the old exact candidate bank, FocalNet chooses 119 teacher-accepted crops
among 338 judged development requests. The frozen old student chooses 118/337.
Across the essential nine ratios both choose 110 accepted among 274 judged.
These noisy cached labels do not establish an improvement or calibrated
acceptability. No FocalNet scores become new composition labels.

[FrameLite](https://github.com/JianzeLi-114/FrameLite/tree/e0f28dd2b6e137264a0a5a9996c1cca17c330657)
requires its 223,576,720-byte task weights plus a separate 1,746,942,600-byte
Qwen3.5-0.8B backbone. Eight free-ratio outputs parse, but one excludes part of
a known padded head. A second probe uses explicit source dimensions, exact
requested ratios 3:1/1:3/6:1/1:6 and a no-face-cut/abstention instruction.
All 32 answers parse; **zero meet the requested ratio within 1%**, zero explicitly
abstain, and one cuts a known head. No geometric repair is applied. This tests
that prompt/checkpoint/preprocessing setting, not all possible fine-tuning.
Owned inputs are retained 768-long-axis RGB rounded to multiples of 32, with
processor resizing disabled. CUDA BF16/SDPA uses the reference Torch fallbacks
for optional causal-convolution/delta-rule extensions; upstream environment
parity and CPU latency are not claimed.

## Candidate coverage versus genuine infeasibility

The full photographic bank has 56 sources × 11 ratios × two crop amounts:
1,232 requests. Upstream grid candidates use five positions per free axis,
scales 1/.9/.8/.7/.65, retention within 0.05 of the best before ranking.
Minimal uses the largest area; the initial Maximal bank excludes that size.
Raw selections cut a known padded head in 516 requests. Restricting that grid
to known-head containment produces 477 abstentions.

An independent integer geometry audit rounds supplied head bounds outward.
For reduced ratio p:q, crop sizes are exactly kp×kq; origin intervals directly
intersect source bounds and every supplied head. The largest feasible size
decides whether any integer crop can fit. This is conditional geometry, not
evidence of complete face detection.

| Ratio | Requests | Old abstentions | Genuine integer infeasibility | Missed feasible amount | Least-crop fallback |
|---|---:|---:|---:|---:|---:|
| 3:1 | 112 | 64 | 36 | 25 | 3 |
| 1:3 | 112 | 55 | 48 | 6 | 1 |
| 6:1 | 112 | 82 | 76 | 6 | 0 |
| 1:6 | 112 | 74 | 70 | 4 | 0 |
| All eleven | 1,232 | 477 | 348 | 112 | 17 |

The new research bank places five origin anchors within each feasible interval,
at the same five scales, bounded by 128 slots. It scores those candidates with
the unchanged FocalNet retention/ranking rule. All candidates pass independent
integer aspect, source-bound and protected-head checks. The result is **884
crops / 348 contains**, recovering all 129 feasible old abstentions; 17 use
the least-crop fallback. The authoritative crop coordinates are integer pixels;
f32 normalized coordinates are model scoring inputs only. Artistic quality,
detector completeness and renderer/export integration remain unestablished.

Ten sources have no detected head, covering 220 requests. They are not certified
face-free. Visual inspection still shows missing people and discarded featured
products. Unknown face coverage keeps the deployment guard closed.

Eight additional agent-inspected witnesses supply 31 approximate product/story/
head regions. Their conservative whole-product profile yields 26 crops and
150 contains in 176 requests, with nine least-crop fallbacks. Seventy-nine old
face-guarded choices violate those added regions. These are unreviewed region
constraints, not exhaustive acceptable-crop annotations or a global policy for
hero/detail shots. The private geometry refactor preserves all 300 prior full
crop plans exactly; no face assertion or numerical tolerance is weakened.

A later network-disabled retention evaluator accepts arbitrary crop coordinates
and checks those same region lists, independently of the cached exact-candidate
aesthetic evaluator. It reports rectangle-area retention and explicit region
clipping, never extrapolated artistic grades:

| Eight-scene witness profile | Requests | Offered crops | Offered crops losing an annotated region | Contains |
|---|---:|---:|---:|---:|
| Original known-head grid | 176 | 92 | 79 | 84 |
| Constrained integer grid | 176 | 100 | 88 | 76 |
| Explicit whole-product/story constraints | 176 | 26 | 0 | 150 |

The search correction alone does not fix semantic retention. Of offered product
region instances, 72/80 are clipped in the original grid and 80/88 in the
constrained grid. These are conservative agent region boxes, not segmentation
pixels or independent human acceptability. Zero clipping in the constrained
whole-product profile is by construction and costs many contains. The evaluator
runs all three comparisons inside a disabled network namespace; it needs no
post-training API call. Source admission hashes and exact bank bytes are bound
in its outputs; no region grade becomes a training label automatically.

## Runtime cost and hardware counters

Four-block zenbench CPU studies use enforced performance core 2 on Core Ultra 7
265K, six preloaded images and 128 slots. Mean process means: graph optimizations
disabled **22.59443850 ms**, optimized **20.81808971 ms**, optimized with reused
output bindings **20.82640825 ms**. This includes both graph outputs, excludes
resize, candidate generation, ranking/guards, loading and IO. Reused bindings
match allocating optimized outputs bit for bit on all six inputs before timing.
There is no demonstrated speed gain from output-buffer reuse. All 16 invocations
observe background CPU load; this is exploratory, with cross-graph equivalence
explicitly unchecked. `run-heavy`: 83 s, peak RSS 0.67 GiB, minimum available
24,833 MiB, peak load 14.00.

Separate sudo perf counts cover 6,000 optimized bound forwards plus process
initialization. Intel Arrow Lake definitions are pinned at
[`8da58930a226`](https://github.com/intel/perfmon/tree/8da58930a226ab3eac72f309bd0fb1d5abfc6245/ARL)
and byte-identical to the retained definitions. All six counters are scheduled
100%. Memory-bound slots are **12.5601%**, core/dependency-bound slots **47.7422%**,
IPC **3.1138**, retired L3 load misses **39,955,729**. Core-bound includes execution
dependencies/resource pressure; these classifications are not measured DRAM
latency or predictions of a tiling gain. Named-metric initialization hung;
that failed attempt is retained, and raw PMU events supply these measurements.
`run-heavy`: 127 s, peak RSS 0.64 GiB, minimum available 25,182 MiB, peak load
14.07; target GNU time maximum RSS 673,856 KiB.

Resources for complete diagnostic jobs, not isolated per-image timings:
FocalNet 136-source probe: 6 s / 0.96 GiB peak RSS; full grid: 17 s / 0.67 GiB;
constrained bank: 12 s / 0.67 GiB; FrameLite free-ratio: 9 s / 2.68 GiB;
ratio probe: 20 s / 2.69 GiB, CUDA peak allocated/reserved
2,068,041,728 / 2,103,443,456 bytes. All use explicit run-heavy caps and two
CPU threads, with ORT inference restricted to one.

## Review and next gate

The gallery preserves source aspect, displays actual crop frames/contain,
shows detected heads and exports local human ratings without API calls or
automatic training admission. Its constrained study has a separate rating
namespace. Browser checks decode all 56 images and verify all sixteen extreme
ratio/mode/guard combinations with no exceptions; a bounded eight-scene
screenshot is inspected. This does not establish a representative human score.

Primary-object supervision and its negative small-model results are recorded in
[the extent study](primary_extents_2026-11-08.md). Keep large local teachers
offline and retain explicit caller regions/contain. Pose, gaze and scanpaths
remain optional teacher features until a controlled crop-quality benefit is
shown; they add measured cost without an established improvement here.
