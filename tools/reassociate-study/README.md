# w24 reduction study

This standalone test tool links the frozen ordered runtime at 4dd26ecc and the
local candidate in one process. Both use the same input, weights and fast sigmoid.
The timing boundary includes input packing/output materialization, excludes resize.

Default study: unchanged ordered local implementation (A/A control).
`--features split`: two interleaved FMA channel reductions, six spatial outputs.
`--features algebraic`: the original twelve-output tile, with magetypes vectors converted
to fixed `[f32; 8]` arrays for Rust `algebraic_mul`/`algebraic_add`, then back.
`--features depthwise`: three independent tap reductions in the 3×3 depthwise kernels.
Rust 1.98 is required for the algebraic feature; default library MSRV stays 1.93.
These candidates do not change the ordered library path. Only the explicitly enabled
experimental features affect `.with_fast_sigmoid()`. These are experiments,
not accepted performance improvements; scalar/SIMD, numerical and rectangle
checks must pass before promotion. No general fast-math compiler flags are used.

From the repository root: `just reduction-check --all-features`, then
`W24_CPU_SET=2 just reduction-bench` and the same with `--features split`, `--features algebraic`, or `--features depthwise`.
Use `just reduction-confirm` only after freezing source and the workload.
Run through run-heavy and zenbench CPU admission. The profiler driver accepts
`baseline|candidate REPEATS`. `just reduction-perf` and `just reduction-record`
use sudo perf and a caller-selected CPU; raw traces belong outside git.

`--features tiles` uses ordered 6×16 channel tiles and bounded spatial panels.
`--features load-ahead` uses ordered 4×16 tiles with the next weight pair loaded
ahead. Both preserve each output's FMA order; neither has a valid speed
confirmation. The `scaling` bench uses persistent workers over the same 32
inputs at every width; `W24_CPU_SET` selects and pins one worker per CPU.
Single-image inference stays serial. See the
[tiling, scaling and allocation measurements](../../benchmarks/w24_tiling_scaling_2026-11-06.md).

The driver also accepts `batch REPEATS`. `just allocation-record` captures
heaptrack data; compare `candidate 101` against `candidate 1` or `batch 11`
against `batch 1` with eight workers using `just allocation-diff`. Feed the
resulting allocation-count stacks and additional image count (100 or 320)
to `just allocation-summary`. Differential runs remove identical startup;
pool misses can vary between runs. Instrumented profiles do not establish speed.

`just memory-parity 32` checks both sigmoid modes against the frozen runtime,
with independent old/current NN sources. The `memory_baseline` and
`memory_candidate` binaries link only their respective runtime and accept a
repeat count; use `just memory-process` under run-heavy for GNU-time RSS.
Default tool features enable both libraries' std mode; `--no-default-features`
exercises their alloc-only implementations in an ordinary host process.
