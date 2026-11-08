# Cached crop queries and CPU scaling — 2026-11-08

Reusing image analysis removes most of the measured multi-ratio cost without
changing the model's decisions. This is an inference optimization for the
research FocalNet model, not an artistic-quality improvement or a promoted
backend. All canonical holdouts remain unopened in this study.

## Graph split and unchanged outputs

ONNX 1.23.2 extracts the dependency frontier between image-only work and
candidate-dependent work. Original weights/operators remain unchanged:

| Graph | Nodes | ONNX bytes | Numeric initializer bytes |
|---|---:|---:|---:|
| Original | 527 | 20,398,992 | 18,993,968 |
| Image analysis | 414 | 20,143,753 | 18,960,568 |
| Candidate query | 113 | 138,423 | 33,404 |

The analysis returns one 1×1024×17 frontier tensor plus the required importance
output. The query graph consumes that frontier, content bounds and 128 candidate
slots. Runtime context owns fixed input/output arrays and rejects stale/foreign
source tokens. Both optimized and unoptimized ORT runs compare all 56 sources,
616 complete score arrays and 1,232 crop decisions. Every score/importance
array matches bit for bit; every choice/action and rectangle is unchanged.
Four-encoder/one-query-thread and four/four with spinning disabled repeat the
same gate with identical results. Each configuration rejects all 55 stale-token
probes. No diagnostic maps are saved.

The context's retained tensor payload is 875,024 bytes, measured from allocated
array sizes, including its normalized image buffer. This is **not** total
process memory: graph sessions, weights, arenas and Python dependencies are
additional. Complete verification jobs peak at 0.69–0.72 GiB host RSS under
3 GiB run-heavy caps, taking 16–17 s. Strided input, output-buffer reuse and
token invalidation are covered by a small independent ONNX fixture.

## Multi-ratio timing and scaling

Each source receives all eleven tested target ratios using the same fixed
candidate banks. The repeated full graph computes its encoder eleven times;
the cached path performs one image analysis and eleven small query forwards.
Both reuse bound output buffers. Cached input validation and copies are timed;
resize, candidate generation, final selection, loading and IO are excluded.
All timings are warm process means, not p95 or standalone additional-query cost.

Six native synthetic photos, three measured sweeps/process, one warm sweep,
four randomized paired blocks and A/A run on enforced performance cores 2–5 of
Core Ultra 7 265K. Every invocation checks the same 132 exact selected integer
rectangles/actions outside timing. No target-cpu=native is used. Background
CPU load is observed throughout; all three studies are exploratory.

| Initial study: threads for every graph | Mean ms/source for eleven ratios | Paired change versus full/one |
|---|---:|---:|
| Full, one | 228.78095 | reference |
| Full, two | 139.60123 | -38.9812% |
| Full, four | 96.66814 | -57.7480% |
| Cached, one | 23.32673 | -89.8039% |
| Cached, two | 15.53357 | -93.2121% |
| Cached, four | 26.53449 | -88.4206% |

All 28 invocations complete; A/A +0.0802%. run-heavy: 106 s, peak RSS
0.70 GiB, minimum available 24,771 MiB, peak load 13.27. More threads help the
full encoder workload but regress the cached workload at four/four.

The next matched 20-invocation study uses cached two/two as baseline:

| Encoder/query threads | Mean ms/source | Paired change versus two/two |
|---|---:|---:|
| Two/two | 15.29252 | reference |
| Two/one | 15.28660 | -0.0444% |
| Four/one | **11.45542** | **-25.0926%** |
| One/one | 23.47800 | +53.5217% |

A/A +0.4207%. run-heavy: 31 s, peak RSS 0.68 GiB, minimum available
24,608 MiB, peak load 12.96. This demonstrates stage-specific scaling;
it does not justify spreading every small operator across four threads.

## Waiting rather than useful arithmetic

Separate sudo perf stat/record probes cover six scenes ×60 sweeps, including
startup/validation as well as inference. No samples are lost. In the four/four
profile, **43.33% of sampled performance-core cycles** land at ORT offset
0x10f0004. The retained binary disassembly identifies `TPAUSE`; the fallback
nearby is `PAUSE`. Four/one reduces that sampled share to 16.29%.
The stripped library does not supply a reliable full symbolic call tree; no
precise caller attribution is invented.

The timing workers observe eight OS threads for four/four and five for
four/one, on the same four-core cpuset. This includes all process threads;
thread count alone is not a utilization measure.

The raw Arrow Lake counters classify memory-bound slots as 12.2042% and
core/dependency-bound as 62.8099% for four/four; four/one gives 21.2138% and
46.5358%. Counters are scheduled 99.82%/99.73%. The wait hotspot shows why
core-bound must not be interpreted as useful arithmetic. These measurements
are not DRAM-latency tests. Target GNU-time records, event definitions and
native perf data are retained. run-heavy peak RSS is 0.65/0.68 GiB,
minimum available 24,824/24,839 MiB; profiler teardown is included in its
12/28 s elapsed times, which are not paired speed evidence.

[ORT documents per-session pools and spinning controls](https://onnxruntime.ai/docs/performance/tune-performance/threading.html).
A third 24-invocation study directly tests those controls:

| Configuration | Mean ms/source | Paired change versus four/one, spinning |
|---|---:|---:|
| Four/one, spinning | 11.58716 | reference |
| Four/four, no spinning | 11.77046 | +1.6018% |
| Four/one, no spinning | 12.22918 | +5.5633% |
| Four/four, spinning | 25.05335 | +115.8025% |
| Two/one, no spinning | 15.66544 | +35.2247% |

Disabling spinning removes the large four/four regression but does not beat
the simpler four/one setup in this observation. A/A -1.2113%; between-block
dispersion/background load limit small differences. run-heavy: 36 s, peak RSS
0.70 GiB, minimum available 24,653 MiB, peak load 13.00. The default research
helper remains explicit/configurable; no shipping runtime is changed.

## Runtime implications and remaining work

The encoder still carries the large weights and convolution cost. Query-only
work requires region means, geometry/log features, one LayerNorm and three
dense layers with ReLU. Its dense portion fits the owned zenpredict runtime;
pooling/normalization still need an explicit matching contract. A custom query
executor can be evaluated separately from replacing the whole CNN. A smaller
query file alone does not remove the encoder or ORT dependency.

First-analysis and already-cached single-query latency are not separately
measured yet. The reported 11.46 ms is the combined eleven-ratio workload and
must not be presented as isolated per-query latency. The next runtime gate
isolates those costs and checks any native pooling/dense path against the same
three-map-pixel/action contract. The model's weak product/face retention remains
documented in [the teacher/search report](local_crop_teachers_2026-11-08.md).
