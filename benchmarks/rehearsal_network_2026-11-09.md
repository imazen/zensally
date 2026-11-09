# Complete neural student qualification and gather runtime

The quality-preserving contrastive student qualifies from owned cached image
tensors through encoder, pooling and final scores on191 FIT/DEV sources.
Reusing the existing guarded gather lowering reduces its complete-neural
reference cost40.88% for one query and56.42% for22 requests. No learned weight
or tested candidate crop choice changes. Artistic quality remains unproven;
no model is promoted.

## Paired complete-neural measurements

Four exploratory zenbench paired blocks/A/A on i265, enforced performance-core
CPU2, ORT analysis/query threads1. All49 DEV sources (25 old/24 new),128
candidates per request, one encoder per image, retained graph bindings and
score/mask buffers. Exclude model/cache loading, warmup, image decode/resize,
candidate generation and face/acceptance guards. These are complete-neural
reference measurements, not native Rust deployment costs.

| Same selected student, direct paired study | Raw | Gather |
|---|---:|---:|
| One query |5.452543ms|3.223568ms|
| All22 requests |87.978749ms|38.339248ms|

Paired geometric changes:-40.8806%/-56.4231%, A/A+0.1333%/+0.3361%.
Values in the table are arithmetic means of four process means; both
workloads are actually executed, not summed from head-only timings. All24
invocations have an observed CPU-load sample over5%; admission is not a
clean-resource certificate or a confirmation interval. Raw data is retained in
[the selected paired rows](rehearsal_network_gather_2026-11-09.csv).

A separate raw-graph comparison measures original C002 versus the student:
one query5.452551/5.417562ms;22 requests87.962240/88.131221ms. The residual
changes this reference pipeline's cost little; do not attribute the gather
speedup to new training. Its paired changes are-0.6406%/+0.1921%, A/A-0.1536%/
-0.2853%, retained in [raw comparison rows](rehearsal_network_raw_2026-11-09.csv).

## Fixed inputs, numerical behavior and weights

Use only the frozen94FIT/25DEV old panel and48FIT/24DEV new panel:142FIT,
49DEV sources,4,202 ratio/mode requests. TEST is never opened. Every role,
family, image tensor, candidate and selected model stays fixed. Reuse the
original exporter wrappers without invoking its TEST-reading main routine.

Both raw and gather layouts produce the following same qualification totals:

| Comparison | Changed choices / requests | Three-grid violations | Max edge movement |
|---|---:|---:|---:|
| Candidate ORT versus full Torch CUDA |0/4202|0|0|
| Candidate Torch CPU versus full Torch CUDA |1/4202|0|0.056165|
| Original ORT versus its Torch CPU model |3/4202|0|0.017105|

The candidate maximum score error is0.0010538101 for ORT and0.0009043217
for CPU. These are score differences, not pixel errors. Reference CUDA is
FP32/TF32-off, batch1, with the complete encoder/query; the previous standalone
native head check uses cached hidden values and remains a distinct scope.
The original old quality gate already passes for this selected checkpoint;
numerical qualification does not create artistic or face-safety evidence.

The independent binding checker instruments the actual timing runner and
compares every bound output with unbound graph execution:4,508 calls/4,704
arrays per layout are exactly equal, across both arms and all49 DEV images.
Its RESULT timings include duplicate reference work and are never performance
evidence. Export preserves every parameter, BN buffer and evaluation flag.

All74 original encoder initializers, six baseline-query initializers and13
candidate-query initializers remain byte-exact, with identical public graph
tensors. Each query replaces exactly two fixed GridSample nodes with the
existing finite border/align-corners gather implementation. The original
two-hash entry-point guard is unchanged; a separate rehearsal entry point
admits only raw candidate graph2468fda8. A rejection test verifies neither
entry point mutates an unregistered graph.

Encoder bytes667,688; raw candidate query359,837; gather candidate query393,091.
The gather encoder/query total1,060,779 bytes includes model weights and needs
ONNX Runtime; it is not a dependency-free Rust binary. The encoder SHA stays
83c95070, raw candidate2468fda8, gather candidate1794fcd1. Exact hashes and all
operator counts remain in the packets. Software: Torch2.14.1+cu130,
ONNX1.23.2, ORT1.31.1, NumPy2.5.4.

## Counters, resources and remaining work

sudo perf stat covers the complete profiling command including setup/warmup.
Raw memory/core-bound slots are4.6%/12.1%; gather27.5%/21.5%. The gather
transformation changes the bottleneck; the isolated native head's low memory
bound percentage cannot characterize this complete path. No tiling or batching
gain is claimed without another controlled measurement.

run-heavy: raw qualification48s/peak RSS1.42GiB; gather31s/1.44GiB under4GiB/
two threads. Paired runs250s/0.73GiB/min available22265MiB and209s/0.74GiB/
22501MiB,4GiB/one thread. Binding checks35s and16s, both0.82GiB under3GiB;
counters9s/0.73GiB and5s/0.74GiB under3GiB. These include framework imports
and corpus tensors and are not deployment-memory claims. Preserve full logs,
GNU-time RSS, hardware counters and process-level timing packets.

Evidence roots under `/mnt/v/output/zensally/`: `rehearsal-network` and
`rehearsal-network-gather`, each dated2026-11-09. Recipes:
composition-export-rehearsal-network (optionally --gather),
composition-rehearsal-network-plan (optionally --baseline-export), runtime-pair,
composition-verify-rehearsal-bindings and composition-verify-rehearsal-gather.
No API calls, extra teacher inference, new fitting, TEST inference or promotion.
The selected student's changed crops still need artistic review; neural speed
does not establish safe or well-composed offered crops.
