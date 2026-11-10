# Reassociation, algebraic arrays and memory profiling, 2026-11-06

No arithmetic alternative is promoted. Split pointwise reductions regress;
algebraic pointwise and three-chain depthwise reductions have no confirmed
full-model speed improvement. All remain opt-in experiments. The ordered path
and the default feature set retain their arithmetic contracts.

Frozen kernels and study driver:
[ed0ce598](https://github.com/imazen/zensally/commit/ed0ce59878eacfcaeac538d6051725fe6436353c).
The independently linked ordered baseline is
[4dd26ecc](https://github.com/imazen/zensally/commit/4dd26ecc24ab68ecba133c7688d20ee6fdbec9fa).
Both arms use identical committed w24 weights, deterministic 3×256×256 input,
fast sigmoid, input packing and output materialization; resize is excluded.
Intel Core Ultra 7 270K Plus, Linux x86_64, performance CPU 2, Rust 1.99.1,
release with debug=1, no target-cpu or LTO override. No AVX-512 performance claim.

## Experiments and fixed confirmation

`tools/reassociate-study` links both runtimes in the same process. Default
features provide an unchanged local ordered implementation as an A/A control.
`split` uses two interleaved channel reductions with six spatial positions;
`algebraic` keeps the original twelve-position loop, converting magetypes
vectors to `[f32; 8]`, applying Rust `algebraic_mul`/`algebraic_add`, and
converting back. `depthwise` splits nine taps into three chains. The combined
mode applies algebraic pointwise and algebraic three-chain depthwise arithmetic.

The array operations use the stable
[Rust 1.98 algebraic API](https://doc.rust-lang.org/core/primitive.f32.html#algebraic-operators).
That optional feature requires 1.98; the default library MSRV remains 1.93.
No global fast-math flag is used. Compiler-permitted reassociation is separate
from an explicit change to the accumulation tree.

Five fresh processes × 30 fixed rounds, 1% margin, workload
`w24-reduction256-v1`, with zenbench 0.1.11 admission and a hard CPU-2 cpuset:

| Candidate | Ordered mean | Candidate mean | Process interval | Decision |
|---|---:|---:|---|---|
| Split pointwise | 3.84 ms | 6.01 ms | +37.9638% to +77.4052% | Regression |
| Algebraic pointwise | 3.87 ms | 3.79 ms | -10.7986% to +7.4579% | Inconclusive |
| A/A control | 3.90 ms | 3.77 ms | -8.9909% to +2.3671% | Inconclusive |

Depthwise and combined five-process runs were invalidated by resource
observations. Their descriptive means are not accepted evidence of speed.

A separate, fixed nine-process × 100-round follow-up used the same kernels,
input, 1% margin and CPU boundary. It addresses the initial control's broad
interval, without changing the acceptance gate or deleting earlier results:

| Candidate | Ordered mean | Candidate mean | Process interval | Decision |
|---|---:|---:|---|---|
| A/A control | 3.82 ms | 3.82 ms | -3.6040% to +3.3678% | Inconclusive |
| Algebraic pointwise | 3.82 ms | 3.81 ms | -2.2411% to +2.1146% | Inconclusive |
| Three-chain depthwise | 3.80 ms | 3.82 ms | -1.7373% to +3.2280% | Inconclusive |

The combined follow-up again failed the resource-observation gate. No baseline
was promoted. Frequency was unverified; these process intervals do not account
for exploratory selection, all prior studies, or persistent code-layout effects.
The A/A control is retained to expose that limitation. Confirmation wrappers
used run-heavy with hard 8 GiB caps, peaking at 0.14–0.18 GiB; detailed lines
and every raw child/aggregate packet remain external.

## sudo perf: compute versus memory

Run-heavy contains `sudo perf`, which profiles only the selected child on CPU 2.
The final driver executes both arms from the same executable. For 1,500 complete
inferences per arm, non-multiplexed core-PMU Top-down metrics were:

| Arm | Memory-bound slots | Core-bound slots |
|---|---:|---:|
| Ordered | 29.0% | 20.8% |
| Algebraic pointwise | 28.9% | 20.7% |

An earlier independent ordered sample measured 28.7% and 20.9%, also without
multiplexing. These are microarchitecture slot metrics, not elapsed-time shares.
A separate memory-detail group measured DRAM-bound 0.4%, L3-bound 3.9%, L2-bound
1.3%, L1-bound 7.4% and store-bound 16.3%. That group multiplexed at 20%; its
cycle-based metrics are diagnostic and should not be added to the slot metrics.

User-only PEBS load sampling (`sudo perf mem record -U -t load --ldlat=10`)
recorded 20K core-PMU samples. Pointwise loads accounted for 23.12% of sampled
latency weight on L1 hits, 20.35% on LFB/MAB hits and 9.14% on L2 hits. This
supports investigating cache/load-store traffic; it does not establish that
DRAM latency is the main bottleneck. Weighted load samples are not total CPU
percentages. Initial captures with broader scope remain preserved separately.

The initial generalized algebraic tile emitted packed FMA but spilled
accumulators in its channel loop. `sudo perf record` attributed 58.66% of its
sampled user cycles to pointwise. Moving algebraic operations into the original
loop reduced that sample share to 31.23%, without a confirmed gain over the
ordered implementation. Final sampled shares also included dense convolution
21.80%, memset 10.86%, memmove 9.48% and depthwise 4.87%; unknown samples were
retained. These are profile observations, not wall-clock speed comparisons.
Disassembly confirms packed FMA in the array/algebraic path.

Each final perf wrapper peaked at 0.09 GiB (4 GiB hard cap). The initial
candidate test/build wrapper peaked at 1.28 GiB; later isolated checks peaked
at 0.56 GiB. These include supervision/builds and are not inference heap claims.

## Numerical and geometry gates

All four experimental configurations passed the frozen three-map-pixel gate.
Each checked 992 source-size cases from the same 257 originals, with a repeated
ordered control (1,984 report rows per configuration). Across the 3,968
experimental comparisons, salient rectangles and final crop edges moved zero
pixels; there were no presence mismatches. Fixed and fast paths are checked
separately by the existing harness. Largest absolute map differences versus
tract were 1.078844070e-5 for split pointwise, 1.043081284e-5 for algebraic,
and 1.022219658e-5 for depthwise and combined. These are runtime-equivalence
checks, not human-ground-truth crop-quality claims.

[Machine-readable results](w24_reductions_2026-11-06.json) retain report/log
hashes, process verdicts and every wrapper line. The original scalar/SIMD bit
assertions and ordered operator references remain unchanged. Isolated std and
no_std tests pass; all-feature tests also passed on actual AVX-512 hardware
with v4 explicitly required. The latter wrapper peaked at 0.55 GiB.
The default-free algebraic/depthwise libraries also compile for the bare-metal
`aarch64-unknown-none` target. API snapshot signatures are unchanged; their
feature inventories include the new opt-in experiments.

## Reproduction and evidence

Run `just reduction-check --all-features`, then select `--features split`,
`--features algebraic`, `--features depthwise`, or `--features algebraic,depthwise`
with `just reduction-bench`, `just reduction-confirm`, or the independent
`just reduction-confirm-long` plan. Run through run-heavy with a zenbench-verified
CPU boundary. Use `just reduction-perf`, `just reduction-record`, and
`just reduction-memory` for counters, sampled cycles and load latency.

All raw logs, counter files, traces, disassembly, executable identities and
geometry reports are retained on i270 under
`/mnt/v/output/zensally/reassociate/2026-11-06/`. Scientific inputs and their roles
are unchanged; see [data provenance](../DATA_PROVENANCE.md).

Verified mirror, accessed on dev:
`/mnt/tower/output/zensally/fixed-w24-2026-11-06/reassociate/2026-11-06/`.
All 389 files in `SHA256.json` were read back and matched. Manifest SHA256:
`98af102e66872ee1a67d666693c65ab3a638914a1241b1765c4fad99680b079b`.
`frozen/references.json` identifies all five confirmation executables; each
matches its raw packet's executable hash. `source.json` verifies all eight
kernel/driver files against the committed source. The retained `zenbench/`
includes content-addressed objects, child logs and references, not references
alone. No new executable-size comparison was made. Counter recipes use the
Intel hybrid PMU; other CPUs need their own metric/event selections.
