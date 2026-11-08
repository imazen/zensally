# Composition training paths — 2026-11-08

Compiled BF16 integral pooling is the fastest tested steady training path.
FP32 remains the evaluation/inference contract: BF16 changes offered crop edges
beyond the owner's three-saliency-map-pixel limit. No trained checkpoint is
promoted by a speed measurement.

## Corrected paired training timing

The corrected task-specific graph computes only the active auxiliary output;
inactive crop-head gradients remain absent. The [earlier report](composition_features_2026-11-08.md)
records the original AdamW confound and the nine corrected quality repeats.
All corrected quality runs still select epoch zero. Old combined-head profiles
remain useful cost diagnostics but do not establish an auxiliary learning gain.

The corrected four-block zenbench study completes 24 process invocations,
including A/A and an explicit combined-head diagnostic. Fixed inputs are the
same 48 synthetic TRAIN sources, batch 48 resident in GPU memory, bounded by
128 MiB input cache. Two CPU threads are enforced on performance cores 2/3 of
Core Ultra 7 265K; GPU is RTX 4080. Each process performs 20 actual gradient/
AdamW warmup steps, then 300 measured steps with CUDA events. Loss, backward,
finite checks, clipping, optimizer and scalar reads are included; model/data
load, compile setup, epoch validation and checkpoint IO are excluded.

| Candidate versus FP32 integral | Paired latency change | Between-block log-ratio SD |
|---|---:|---:|
| FP32 combined head | +1.3988% | 0.00080635 |
| BF16 integral | -36.8436% | 0.00080278 |
| BF16 matmul pooling | -22.1867% | 0.00079670 |
| BF16 integral, compiled reduce-overhead | **-52.2130%** | 0.00525690 |
| A/A | +0.0332% | 0.00106445 |

Positive means slower. Dispersion is not a confidence interval. Every invocation
observes background CPU load; comparisons are exploratory with numerical
optimizer-trajectory equivalence explicitly unchecked. Each worker requires
finite losses/gradients. The plan hashes its worker, training modules and cache.
`run-heavy`: 260 s, peak host RSS **2.13 GiB**, minimum available 23,635 MiB,
peak load 13.18, hard host cap 6 GiB with no cgroup swap. This is steady-step
cost, not a measured end-to-end time-to-quality gain.

The preceding combined-head-only comparison remains retained: 20 invocations,
compiled BF16 -52.3817%, A/A -0.0751%, 191 s / 2.00 GiB peak RSS. It is explicitly
superseded for the cost of the corrected active-task graph, not overwritten.

## Nsight suitability probes

These diagnostic traces use the earlier combined-output graph: 20 warmup and
200 profiled steps, frozen BatchNorm running statistics, TF32 disabled. They
are not substituted for the uninstrumented paired results above.

| Batch / precision / pooling | Instrumented ms/step | CUDA peak allocated / reserved bytes |
|---|---:|---:|
| 16 / FP32 / integral | 9.13321472 | 508,957,184 / 717,225,984 |
| 48 / FP32 / integral | 25.58121582 | 1,388,090,368 / 2,164,260,864 |
| 48 / FP32 / matmul | 29.26462158 | 2,422,339,072 / 2,491,416,576 |
| 48 / BF16 / integral | 16.23938843 | 807,865,856 / 1,042,284,544 |
| 48 / BF16 / matmul | 19.95489136 | 1,839,168,512 / 1,883,242,496 |
| 96 / BF16 / matmul | 39.11109863 | 3,655,365,120 / 4,865,392,640 |
| 48 / BF16 / integral, compiled | 12.25088867 | 898,210,816 / 1,216,348,160 |

Batch 96 repeats the 48 fitting sources twice; it supplies no new information
or effective independent batch count. Matmul pooling is slower and consumes
more memory here. The first compiled invocation takes 50 s including setup/
compilation, target GNU-time max RSS 2,981,648 KiB. CUDA allocated bytes do not
include all private/driver allocation; reserved bytes are retained separately.
The nsys launcher-only run-heavy RSS is not training memory. Matched traces
retain target GNU-time records, native reports and result summaries. An older
unmatched BatchNorm-training profile is preserved and excluded from this table.

The initial trace identifies BatchNorm backward and grid-sampler backward among
large costs. A saturation claim requires a matched utilization trace of the
selected corrected path; the numbers here do not establish maximum GPU use.

## Three-pixel inference gate

One frozen checkpoint is evaluated on the 64 fitting /16 development sources,
1,760 ratio/mode requests, batch 16 channels-last, FP32/TF32-off reference.
Displacements are normalized source-coordinate edge movements ×128, matching
the owner's saliency-map-pixel units. Raw research acceptance is not trusted
face coverage.

FP32 matmul has maximum absolute logit difference 0.00069618225, but **zero
selected-choice/action changes and zero crop-edge movements in all 1,760
requests**. This does not make its slower, larger-memory training path useful.

BF16 integral changes 23 fitting and four development raw choices, with three
research action changes overall. Fifteen offered fitting crops and three
offered development crops move an edge by more than three map pixels; maxima
are 47.27716728 and 21.42912123 respectively. BF16 matmul also fails: 20/5 raw
choice changes, two action changes overall, 12/3 offered movements beyond the
limit. Both are rejected as inference replacements under the unchanged gate.

Use mixed precision/compilation as a training-cost candidate only, followed by
FP32 validation and a measured quality comparison. Preserve exact task-specific
optimizer participation and immutable checkpoints. CPU memory/core counters and
the larger composition model's cost are in [the local-teacher report](local_crop_teachers_2026-11-08.md).
