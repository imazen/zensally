# w24 reduction study

This standalone test tool links the frozen ordered runtime at 4dd26ecc and the
local candidate in one process. Both use the same input, weights and fast sigmoid.
The timing boundary includes input packing/output materialization, excludes resize.

Default study: two interleaved FMA channel reductions, four spatial outputs.
`--features algebraic`: twelve spatial outputs, with magetypes vectors converted
to fixed `[f32; 8]` arrays for Rust `algebraic_mul`/`algebraic_add`, then back.
Rust 1.98 is required for the algebraic feature; default library MSRV stays 1.93.
Neither candidate changes the ordered library path. Only the explicitly enabled
experimental features affect `.with_fast_sigmoid()`. These are experiments,
not accepted performance improvements; scalar/SIMD, numerical and rectangle
checks must pass before promotion. No general fast-math compiler flags are used.

From the repository root: `just reduction-check --all-features`, then
`W24_CPU_SET=2 just reduction-bench` and the same with `--features algebraic`.
Use `just reduction-confirm` only after freezing source and the workload.
Run through run-heavy and zenbench CPU admission. The profiler driver accepts
`baseline|candidate REPEATS`. `just reduction-perf` and `just reduction-record`
use sudo perf and a caller-selected CPU; raw traces belong outside git.
