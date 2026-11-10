# Accelerated crop-training suitability: preliminary findings

Measured 2026-11-08, starting 01:06 UTC. Still missing: reviewed composition
targets, an independent crop-quality TEST set, corpus fitting, trained-model
export and a Rust crop scorer. All training below uses synthetic diagnostic
targets on a resident repeated fixture. Loss reduction establishes functioning
gradients/updates, not artistic quality or generalization. Existing teacher
labels remain training-ineligible; no source roles changed.

## Suitability

The experimental region model has **238,726 measured parameters**. It reuses
the w24 encoder, projects stride-8/16 maps to 16 channels each, and pools nine
retained bins, four discarded bands, eight boundary strips and global context.
Geometry/area/presence features feed ranking and acceptability logits. The
point-sampling model remains available; its rank-only objective differs, so
its timings cannot isolate the cost of the new representation.

Fractional integral pooling matches an independent cell-overlap value/gradient
oracle, including empty bands. Thin content, cached chunks, candidate order,
encoder/projection/head gradients and hard known-face masking pass. A denser
matrix pooling alternative passes the same mathematical checks. The default
remains integral pooling.

CUDA FP64 operator errors against the CPU reference are at most 2.89e-14 in
values and 1.38e-14 in gradients. On the small thin-box fixture, FP32 integral
pooling has maximum value error 1.42e-5; FP32 matrix pooling 7.94e-8. BF16 matrix
pooling reaches 0.001282; gradients remain finite. These are operator diagnostics,
not whole-network/crop equivalence or permission to promote approximate serving.

## GPU: RTX 4080 on i265

PyTorch 2.14.1+cu130, Python 3.12.14, input 256×256, 64 candidates/image,
BF16 autocast, channels-last, fused AdamW. Each step includes forward,
ranking/acceptability losses, backward and optimizer update. Compilation and
warmup are outside the timed window. Ordinary FP32 probes use framework
convolution defaults; they are not strict FP64 numerical references.

| Batch | Execution | ms/step | Images/s |
|---:|---|---:|---:|
| 16 | Eager | 5.536 | 2,890.1 |
| 16 | Compiled | 3.937 | 4,063.5 |
| 16 | Manual CUDA graph, corrected | 3.554 | 4,502.1 |
| 64 | Eager | 15.276 | 4,189.7 |
| 64 | Compiled | 13.402 | 4,775.4 |
| 64 | Manual CUDA graph | 14.809 | 4,321.7 |
| 256 | Eager | 79.228 | 3,231.2 |
| 256 | Compiled | 58.972 | 4,341.1 |
| 256 | Manual CUDA graph | 78.863 | 3,246.2 |

These are exploratory single-process scouts. The initial choice is compiled
BF16/channels-last at batch 64; larger batches consume more memory without
improving observed image throughput. Manual graph capture helps the small-batch
scout but does not beat compilation at batches 64/256. No GPU serving-speed or
trained-quality improvement is claimed.

A fresh zenbench comparison completes four blocks / twelve processes at batch
64: mean eager 15.347 ms/step, compiled 13.366 ms/step; paired geometric latency
change **−12.9038%**, A/A **−0.0170%**. Wrapper: 267 s, peak RSS **2.38 GiB**,
minimum available 20,221 MiB. This is exploratory Iterate with unrestricted
placement and observed background CPU load; it is not a confirmed promotion.
All attempts, including the failed virtualenv-launcher packet, remain retained.

The six-cell batch-64/256 wrapper completes in 205 s, peak RSS **2.37 GiB**,
minimum available 21,950 MiB (14 GiB hard host cap, four configured jobs).
The batch-64 compiled case peaks at 962,358,272 tensor-allocated bytes and
1,396,703,232 reserved bytes; sampled whole-device peak is **1,798 MiB**.
Batch-256 compiled samples **5,836 MiB** whole-device usage. Sampling is every
200 ms and can miss brief peaks. CUDA tensor allocations are capped at 70% of
device memory; this is not a hard cap on all driver/library memory.

CUDA graphs retain private pools. Their low live-allocation counter is not a
total-memory saving: batch-256 graph reserves 5,133,828,096 bytes and samples
5,242 MiB whole-device usage. Memory samples, allocator peaks, device telemetry
and host RSS remain separately labelled. The best batch-64 compiled scout's
five post-step samples are flat at 72,529,920 allocated / 1,396,703,232 reserved
bytes over steps 0, 60, 120, 180 and 240. This bounds the observed window; it is
not a full-duration data-loader or memory-leak study.

## Profiling findings

Nsight captures 1,000 warmed batch-64 compiled steps. It records 608,000 GPU
kernels. Sample means: SMs active **88.99%**, SM issue **17.91%**, tensor activity
**3.02%**, DRAM read **39.22%**, DRAM write **26.84%**. The GPU is busy, but the
workload does not approach peak tensor arithmetic. Pooling backward accounts
for 7.6% of summed kernel time; normalization/backward and small convolutions
also contribute. This motivates pooling/normalization fusion experiments.
The 34-second profiler wrapper reports 0.25 GiB peak RSS, minimum available
20,480 MiB. That wrapper observation does not establish the target's host peak;
training RSS is measured separately by the scout/paired wrappers above.

The resident timed window contains device-to-device copies and memset, not
host-image uploads. Decode, augmentation, data-loader prefetch and upload overlap
are therefore still unmeasured. A final two-cell batch-64 compiled GPU scout
measures 13.306 ms/step for integral pooling and 13.332 ms for matrix pooling.
Matrix pooling reaches 1,142,832,128 tensor-allocated bytes versus 963,324,928
for integral pooling. It supplies no observed speed gain here; retain integral
pooling. Wrapper: 66 s, peak RSS **2.33 GiB**, minimum available 21,547 MiB
(14 GiB host cap). These two scouts are not a paired confirmation.

`sudo perf` on i265's four-thread CPU probe records 56,743,332,080 cycles,
73,612,144,979 instructions, and 487,492,265 cache misses. Top-down slots are
38.4% memory-bound, 70.9% backend-bound and 16.3% retiring. This covers the whole
4.489-second process, including startup/profiling, not an isolated kernel.
Wrapper peak RSS is **1.13 GiB**; the cap is 8 GiB.

## CPU: Core Ultra 7 270K Plus on i270

PyTorch 2.14.1+cpu, Python 3.12.15, batch 4, 256×256, 64 candidates, FP32.

| Threads | Layout/execution | ms/step | Images/s |
|---:|---|---:|---:|
| 1 | NCHW eager | 71.981 | 55.57 |
| 1 | Channels-last eager | 64.063 | 62.44 |
| 4 | NCHW eager | 31.414 | 127.33 |
| 4 | Channels-last eager | 28.065 | 142.52 |
| 8 | NCHW eager | 27.597 | 144.94 |
| 8 | Channels-last eager | 21.342 | 187.42 |
| 16 | NCHW eager | 34.406 | 116.26 |
| 16 | Channels-last eager | 21.597 | 185.21 |
| 4 | Channels-last compiled | 30.128 | 132.77 |
| 8 | Channels-last compiled | 23.682 | 168.90 |
| 4 | Channels-last eager, matrix pool | 32.744 | 122.16 |

The 11-cell wrapper completes in 87 s, peak RSS **0.71 GiB**, minimum available
25,551 MiB (12 GiB hard cap, eight configured jobs; explicit thread counts above).
Eight channels-last threads are the initial CPU choice; sixteen add no observed
throughput, and compilation is slower in these scouts.

A separate zenbench run completes four blocks / twelve fresh processes with
A/A, soft affinity 0–7 and eight training threads. Mean NCHW step latency is
26.028 ms versus 21.409 ms channels-last; paired geometric latency change is
**−17.7325%**, with A/A **−0.0753%**. Wrapper: 64 s, peak RSS **0.56 GiB**,
minimum available 26,197 MiB. This is Iterate, not Confirm: hard hybrid-core
placement could not be verified. Checks establish fixture/finite-gradient
execution, not trained-model quality or bit-exact numerical equivalence.

## Reproduction and limits

Use `just crop-train-probe`, `crop-train-matrix`, `crop-regions-check` and
`crop-pool-numeric`. Retain the explicit plans, frozen source/requirements,
CUDA events, telemetry, progress/PIDs, profiler traces, complete logs and
zenbench packets. No native-CPU Rust flags, model-weight changes or corpus fits.

The first manual graph attempt read a captured loss before replay; that initial
loss is invalid and the attempt is retained. Replay/synchronization and releasing
the side-stream warmup graph correct it. The first GPU paired attempt also failed
before importing Torch because the launcher canonicalized the virtualenv Python
symlink. The corrected launcher uses `/usr/bin/env` with the virtualenv path as
an argument. Failed attempts are not timing evidence.

An intervening training job used the shared GPU. No foreign job was stopped.
The GPU owner later confirmed a free window for the corrected paired study.
The numerical setup check finishes before its first measured window; both
timestamps are retained. Resource/placement limits remain explicit.
