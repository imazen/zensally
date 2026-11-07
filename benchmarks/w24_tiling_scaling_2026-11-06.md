# w24 tiling, image workers and allocations, 2026-11-06

The [2026-11-07 memory/CPU follow-up](w24_memory_cpu_2026-11-07.md)
implements the allocation recommendations below and measures the result.
This report retains the original source, pool behavior and historical data.


The default kernel remains unchanged. Ordered channel tiling and weight
load-ahead are opt-in experiments without a valid speed confirmation.
Image-level parallelism reaches 4.7101× throughput with eight performance
cores. Single-image inference remains serial. Allocation profiling recommends
removing metadata allocations and introducing per-execution tensor storage
before expanding the existing shared pool.

Sources: initial tiles/scaling
[6f0c862d](https://github.com/imazen/zensally/commit/6f0c862d5416ea21988603576b1889e4d09aa7cf),
refined weight pipeline
[b6e32879](https://github.com/imazen/zensally/commit/b6e32879583e961a337faf2ca831744c1c992871).
Both use the owned NN kernels at
[084fadeb](https://github.com/imazen/zenjpegai/commit/084fadeb0b62309593dbc216f591ca6b73865a4f).
The independently linked ordered oracle is zensally 4dd26ecc.

Intel Core Ultra 7 270K Plus, Linux x86_64, Rust 1.99.1, release debug=1,
runtime AVX2 dispatch, no target-cpu=native. These results do not measure
AVX-512 speed or no_std allocation behavior.

## Throughput

Persistent workers each hold a model, receive a cyclic partition of the same
32 preprocessed 256×256 inputs, and validate all maps bit for bit against
the independent ordered runtime before timing. Loading, construction, thread
creation and validation are excluded; input packing and output allocation are
included. Workers and the enclosing scope enforce the selected P-core CPU set.
The library does not add a thread-pool dependency.

Five fresh processes × ten rounds per width, zenbench 0.1.11 admission:

| Workers | Batch of 32, ms | Images/s | Descriptive speedup |
|---|---:|---:|---:|
| 1 | 122.5283 | 261.16 | 1.0000× |
| 2 | 72.1645 | 443.43 | 1.6979× |
| 4 | 42.8613 | 746.60 | 2.8587× |
| 8 | 26.0140 | 1230.11 | 4.7101× |

All selected populations pass resource admission and have 50 clean rounds.
Cross-width ratios are descriptive, not a paired speedup confirmation. Batch
duration divided by 32 is not individual-image latency. The selected logs and
all rejected attempts are identified in
[the machine-readable summary](w24_tiling_scaling_2026-11-06.json).
No gate was relaxed. Frequency verification is unavailable in these packets.
The run-heavy wrappers report peak RSS 0.16/0.17/0.20/0.25 GiB for the selected
1/2/4/8-worker runs; consult the embedded wrapper lines for exact observations.

## CPU and memory evidence

Separate `sudo perf stat --cputype=core` runs use 160 batches (5120 images)
plus construction and validation. GNU time encloses the actual driver.
These profiles diagnose behavior; their wall times are not benchmark results.

| Observation | One worker | Eight workers |
|---|---:|---:|
| Memory-bound slots | 27.0% | 40.1% |
| Core-bound slots | 20.7% | 19.1% |
| Process maximum RSS, KiB | 128588 | 232956 |
| Minor page faults | 52732 | 3640025 |
| System CPU seconds | 0.06 | 5.80 |
| run-heavy peak RSS, GiB | 0.12 | 0.22 |

Eight-worker userspace cycle samples put memmove at 12.25% and memset at
9.89%; pointwise and dense convolution account for 28.51% and 18.68%.
Mutex code is not a leading sampled hotspot. Memory pressure grows with
concurrency; these counters alone do not prove a DRAM-bandwidth ceiling,
and page faults are not allocation counts.

## Actual allocation patterns

Heaptrack intercepts allocator calls in the immutable default driver. Compare
`candidate 101` against `candidate 1` to remove identical model construction:
19708 minus 630 = **19078 calls for 100 additional images**. For eight workers,
`batch 11` minus `batch 1` is 81290 minus 20441 = **60849 calls for 320
additional images**. These are observed averages, not a constant-per-call
guarantee. Heaptrack overhead is excluded from speed claims. Recording wrappers
report peak RSS 0.11 GiB and 0.21 GiB respectively.

| Allocator caller | One worker, calls/image | Eight workers, calls/image | Recommended treatment |
|---|---:|---:|---|
| Dense convolution row schedule | 129 | 129 | Remove nested row vectors on the serial path; use direct row slices and reusable metadata where parallel scheduling needs it |
| Graph values/use counts | 2 | 2 | Fixed graph arrays or constructor-owned execution metadata |
| Concatenation reference lists | 3 | 3 | Fixed stack arrays for the known inputs |
| Packed RGB input | 1 | 1 | Reuse per-context storage; overwrite real lanes and preserve zero padding |
| Depthwise padded inputs | 10 | 10 | Reuse per-context scratch first; boundary-aware kernels can eliminate the copy |
| Concatenated activations | 3 | 3 | Reuse a live-range slot; later consider consumer fusion |
| Spatial means | 2 | 2 | Keep small fixed/context channel buffers |
| Activation pool misses | 36.91 | 36.346875 | Fixed-shape live-range storage in each execution context |
| Transpose padding pool misses | 2.87 | 2.8 | Context scratch; investigate boundary addressing separately |
| Caller-owned map | 1 | 1 | Retain under the current returning API; an approved caller-buffer API could avoid it |
| Other | 0 | 0.00625 | Two additional batch/channel infrastructure allocations in the eight-worker comparison |
| **Total** | **190.78** | **190.153125** | |

The 129-call hotspot is `PackedConv::forward_blis`: 128 row vectors and
one outer vector for the first dense convolution. It is scheduling metadata,
not 129 feature maps. Its im2col buffer already uses thread-local reuse under
std, but `border_buf` clears it for every row. Eliminating mallocs alone does
not eliminate that clearing or the packing writes.

### What a pool helps, and what it leaves behind

There already is a process-wide `Mutex<Vec<Vec<f32>>>` pool in NN `BTensor`:
24 parked buffers, default 1 GiB, minimum 16384 floats, best fit with at most
2× capacity. Oversubscribed pools evict smaller buffers first. `scratch` and
clone can consume it. `zeros`, serial `pad`, and aligned `cat` allocate fresh
storage instead, although their dropped buffers can enter that pool. Repeated
fresh zeros and concatenations therefore coexist with reuse misses. The
profiles establish churn; the exact contribution of each eviction policy to
the scaling loss is not isolated.

A larger or sharded pool can reduce malloc/free and page refaults on misses.
It cannot remove memcpy from padding/concatenation, zero filling, input
packing, or output materialization. Pooling the 129 small row vectors would
add management to metadata that the serial path does not need.

For this fixed shape and graph, prefer immutable shared model weights plus a
separate execution context per concurrent inference. Plan reusable tensor
slots from existing last-use information; keep input, small channel scratch
and padding scratch in that context. Reuse only after all consumers finish,
and distinguish buffers that kernels fully overwrite from buffers requiring
zero borders/padding lanes. Keep the general bounded pool as a fallback for
variable shapes and uncommon paths. This is a recommendation, not an
implemented allocation reduction or a measured speed gain.

Do not replace the global pool with unconstrained TLS caches: existing
`set_pool_limit`/`release_buffers` promises are process-wide. Any context or
sharding implementation must define and preserve memory accounting and limits.
The no_std path currently lacks this shared pool and std TLS scratch; its
allocation count was not measured here.

### Fallible zeroed allocation

Verified by `cargo read bytemuck --path-only`, resolving 1.25.2:
`bytemuck::allocation::try_zeroed_vec::<f32>(n)` is available with
`extern_crate_alloc`, supports alloc without std, and returns `Result<Vec<f32>, ()>`
for allocation/layout failure. It uses `alloc_zeroed`, then transfers the
allocation through a boxed slice into a vector without copying.

Use that API when allocating new zeroed numeric buffers, mapping failure into
the existing allocation error path. This is particularly useful because
`BTensor::zeros` currently returns a Result for shape overflow but uses an
infallible `vec![0.0; n]` allocation. That existing expression already reaches
the zeroed allocator; bytemuck improves fallibility without reserve-then-fill,
not allocation count or a demonstrated speedup. It does not make vector
growth, clones or `Vec::with_capacity` fallible automatically. Reused buffers
still need correct border initialization, and fully overwritten scratch
should not be cleared each call.

## Tiling and load-ahead decision

The 6-spatial × 16-channel kernel reuses input broadcasts across two output
groups with bounded spatial panels (96 positions, or 48 for cin≥96). The
4×16 alternative loads the next weight pair ahead of use. Both preserve
bias-first, ascending-input-channel FMA order and physical padding writes.
The original load-ahead prototype also retained the next input vectors;
its exploratory result was 5.65 ms versus 3.92 ms, so the refined version
reduces live values and only loads weights ahead.

Fixed nine-process × 100-round confirmations, including A/A, all fail
resource-admission reliability. Traced follow-ups also fail (32 noisy rounds
for tiles, 18 for load-ahead). Their apparent means are not accepted gains.
The experiments remain feature-gated. Operator tails/panels, full-model
scalar/SIMD and strided inputs pass std and no_std checks, including actual
AVX-512 tier v4 on Ryzen 9 7900X. Its wrapper reports 1.27 GiB peak RSS.
That host needed an isolated exact-version cc 1.11.1 source patch because
its shared package cache lacked source files; runtime dependencies are unchanged.

Each variant also passes 1984 frozen rectangle comparisons over 257 source
images, including 992 cases with its fast path active. Both ROI and final-crop
edges move zero map pixels against tract; presence agrees throughout. The
approved gate remains three map pixels per edge. The two result tables have
identical bytes; see [their identities](w24_tiling_rectangles_2026-11-06.json).
The combined crop command reports run-heavy peak RSS 1.78 GiB; its final
bare-metal check initially failed because the target was absent. After target
installation, x86_64-unknown-none with load-ahead passes (peak RSS 0.35 GiB).
API regeneration and check pass with feature-list changes only.

## Reproduction and retained evidence

Use the project justfile: `tiling-check`, `scaling-bench`, `scaling-perf`,
`scaling-record`, `allocation-record`, `allocation-diff`, `allocation-summary`
and `tiling-summary`, each heavy invocation through run-heavy. Set
`W24_CPU_SET` explicitly for batch profiling; sudo otherwise strips it.
Heaptrack differential stacks use allocation-count cost, not peak cost.
Classification is specific to this frozen driver's call sites.

Raw logs, profiler traces, executable captures and complete zenbench object
storage live outside git on i270 at `/mnt/v/output/zensally/tiling/2026-11-06/`.
The archive mirror is
`/mnt/tower/output/zensally/fixed-w24-2026-11-06/tiling/2026-11-06/`.
Invalid attempts and allocator-retention
diagnostics are retained. The latter never passed admission and do not
support an allocator-configuration speed recommendation.
