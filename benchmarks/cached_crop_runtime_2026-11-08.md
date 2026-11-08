# Cached crop queries and CPU scaling — 2026-11-08

Reusing image analysis removes most of the measured multi-ratio cost without
changing the model's decisions. This is an inference optimization for the
research FocalNet model, not an artistic-quality improvement or a promoted
backend. All canonical holdouts remain unopened in this study.
The complete packet passes independent NAS membership/hash verification;
see its [evidence pointer](cached_crop_runtime_2026-11-08.pointer.md).

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

## Isolated analysis and native query scoring

A separate 16-invocation study isolates normalized RGB → cached features and
importance output on six fixed image witnesses. Bound outputs match the cached
reference by raw float32 bits before timing. Five measured sweeps/process,
four randomized blocks and A/A use the same performance-core cpuset:

| Encoder threads | Mean ms/source | Paired change versus one |
|---|---:|---:|
| One | 20.40893 | reference |
| Two | 12.37308 | -39.3720% |
| Four | **8.42649** | **-58.7102%** |

A/A -0.8692%. run-heavy: 16 s, peak RSS 0.11 GiB, minimum available
25,004 MiB, peak load 12.33. These process footprints use the smaller standalone
phase worker; they do not replace the earlier full-worker memory measurements.

The standalone `tools/query-native` experiment implements region means through
f64 summed-area tables, ordered f32 geometry/log features, a fixed 62-element
LayerNorm array and the existing zenpredict 0.2.0 dense/ReLU runtime. Its
build-time baker embeds 6,145 unchanged dense parameters in 25,348 ZNPR bytes;
124 normalization parameters add 496 bytes of constants. The baker is
unpublished and explicitly patched to a frozen owned source snapshot. JSON and
the baker are build dependencies. No quantization or model fitting is involved.

The dispatched executable is 630,288 bytes, or 508,520 after stripping symbols;
these sizes include the query weights. It links system libc, libm and libgcc_s,
without an ORT runtime dependency. **Image analysis still needs its separate
20,143,753-byte graph and an encoder executor.** This experiment does not remove
ORT from the complete image-to-crop pipeline or change shipping dependencies.

`ordered` and Rust 1.98 algebraic normalization both pass the frozen 56-source,
616-query-array / 1,232-decision gate: zero changed choices/actions, zero edge
movement and zero violations of the supplied protected-head bounds. Largest
absolute score errors against ORT are 3.814697265625e-6 and
3.337860107421875e-6. Scores are not bit-identical to ORT. Scalar and dispatched
builds match bit for bit for all 78,848 scores in each arithmetic mode.
The measured compiler is rustc 1.99.1; a Rust 1.98 compiler was not tested.

The query-only study measures all eleven ratios on each of the 56 cached
sources, including fixed eligibility-mask selection and a checked decision
digest. Native timing additionally builds one summed-area context per source.
Loading, image analysis, preprocessing and candidate planning are excluded.
The ORT/dispatched arms use five sweeps/process; the slower scalar arm uses one.
Every invocation performs one untimed warm sweep. Four randomized blocks and
A/A run on one enforced performance core:

| Query executor | Mean ms/source for eleven ratios | Paired change versus ORT |
|---|---:|---:|
| ORT, one thread | 2.52422 | reference |
| Native dispatched, ordered | 0.87137 | -65.4797% |
| Native dispatched, algebraic | **0.81491** | **-67.7160%** |
| Native scalar, ordered | 10.75228 | +325.9707% |

All 20 invocations complete; A/A -0.4851%. run-heavy: 21 s, peak RSS
0.07 GiB, minimum available 25,062 MiB, peak load 12.60. Background load occurs
in 17 query and all 16 analysis invocations; these remain exploratory results.
The analysis and query studies are separate measurements: their means must not
be added and presented as a measured combined native pipeline. Individual
single-query latency and p95 remain unmeasured.

## Allocation behavior

Heaptrack observes **83 allocation calls** both for one and for fifty timed
sweeps, after one warm sweep in each process: 157,696 versus 4,021,248 candidate
scores. Peak live heap is 5.57 MB as reported by heaptrack. The 57 input-loading allocations plus one summed-area allocation account for
5.47 MB; model/predictor and summed-area storage are initialized once. Repeated queries add no observed allocation
calls. The 548 bytes remaining at process exit are attributed to Rust's
`std::sys::pal::unix::stack_overflow::thread_info::set_current_info` (4 + 544
bytes), rather than candidate scoring. Heaptrack reports two temporary calls.
Its instrumented RSS includes profiler overhead and is not a production-memory
claim. run-heavy reports peak RSS 0.08 GiB for both runs, taking 0/3 s,
minimum available 25,183/25,069 MiB and peak load 12.25/12.11.

A reusable predictor plus a per-source summed-area context addresses the
observed query allocation pattern; a general allocation pool is unnecessary
for this experiment. Multiple concurrently active images would need separate
contexts and an explicit memory budget. No concurrency behavior is inferred
from the single-context study.

## Native hardware profile

A separate sudo perf run of 500 complete 56-source sweeps attributes 65.91%
of sampled cycles to zenpredict's dispatched AVX2/FMA dense multiply, 11.86% to
crop descriptors, 7.41% to its generic forward executor, and 3.67% to summed-area
reads. No samples are lost. The raw Arrow Lake events give memory-bound
6.0484% and core/dependency-bound 24.2253% of slots, IPC 5.0969 and 28,001 L3
load misses, with counters scheduled 99.73%. This is a cached small-head
workload, not a DRAM-latency measurement. The previous event-definition
identities remain pinned in the teacher/performance evidence.

Target GNU time reports 22.95 s wall, 22.92 s user and 8,524 KiB maximum RSS.
The wrapper takes 66 s including profiler metadata work, peak RSS 0.02 GiB,
minimum available 25,183 MiB and peak load 13.25. These are profiling/resource
observations, not paired latency results. Two earlier profiler attempts are
retained as interrupted: the native target succeeded, but perf's process-file
metadata scan blocked in NFS getattr. Disabling build IDs alone did not avoid
it. The successful command also disables unrelated BPF/event synthesis while
retaining child exec/mmap events; native symbol attribution is available.
Manual termination produced those earlier rc=137 values; no memory cap fired.

## Remaining quality and runtime limits

The encoder still carries the large weights and convolution cost. Replacing
its executor is separate work; the small query implementation is not a full
CNN runtime. Faster unchanged scoring does not repair weak product retention,
missed faces or artistic composition. Those limitations remain documented in
[the teacher/search report](local_crop_teachers_2026-11-08.md). No crop network,
face certification, global product-preservation policy or public API is promoted.
