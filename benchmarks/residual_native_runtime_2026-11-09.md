# Native quality-preserving residual head

The first registered contrastive-rehearsal seed31 exports to embedded
zenpredict0.2.0 models and matches every tested crop choice. Its extra native
head work is measurable; no whole-pipeline speed or artistic-quality claim
follows. Encoder, pooling and the earlier scorer layer remain outside this
study. No model is promoted.

| Measured workload,128 candidates/query | Original scalar | Scalar + residual |
|---|---:|---:|
| One query |0.011898ms|0.034065ms|
| Full22-query bank |0.405550ms|1.266587ms|

Four zenbench paired blocks, A/A included, on i265's enforced performance-core
CPU2. Values are arithmetic means of the four process means. The bank row
comes from the separately executed22-query workload, reversing its per-query
normalization; it is not an extrapolation from the one-query case. Both use
the same binary, actual frozen hidden values, persistent predictors and
preallocated input rows. Model/fixture loading and one warmup sweep are excluded.
Paired geometric latency increases are186.29% and212.32%; A/A changes0.71%
and0.18%. These are exploratory results, not confirmation intervals. Every
invocation has an observed CPU-load sample over5%; admission and uniform
placement do not certify clean measurement conditions.

## Numerical and dependency evidence

The selected checkpoint is replay-contrastive-seed31, SHA256
`4b801ad420688a167056ee6439d5c40c0f51fba31db46554f2c68113bf60bb3b`.
Seed31 is chosen by registration order for this runtime study, not as a new
best-seed claim. The preceding original full-model old quality gate passes for
this head; it is not an artistic or complete-face validation.

Native scores versus the same uncompressed Torch CPU student on all72 sources/
22 queries change0/1584 eligible crop choices and0 edge pixels. Maximum absolute
score difference1.1920928955078125e-6; zero violations of the unchanged three-grid
gate. The head uses a96-input base scalar plus96->32 ReLU->1 compact residual,
with normalization folded into its first layer and feature-teacher weights
folded into its last. It requires no pose/gaze/scanpath model at inference.
Rust encoder and complete-query parity remain unmeasured for this new head.

Both model payloads are literal-byte verified inside the601,952-byte release
executable: base1,332 bytes, residual13,540 bytes,14,872 total including format
headers. No weights are downloaded by the executable. The runtime dependency
is zenpredict0.2.0 with std/SIMD; serde_json and the retained owned
zenpredict-bake are build-only. Build uses Rust1.99.1, thin LTO, no
target-cpu=native. Formatting, release compilation and clippy -D warnings pass.
The standalone tool has forbid(unsafe_code) and bounds/fallible fixture loading.

The first export writes Torch output-major weights into the baker's input-major
format. It fails1000 numerical crop gates before any timing is admitted. The
rectangular dense-layer regression fails with[38,48] versus expected[27,55]
before the transpose correction and passes afterward. All failed model, native
scores and qualification output remain retained separately. Timing-plan creation
refuses the failed qualification. No numeric or quality expectation is relaxed.

## Profiling and allocations

sudo perf stat on the22-query residual workload reports1.2% memory-bound and
8.2% core-bound slots; raw slot counters show32.2% front-end bound and27.8% bad
speculation. Counters cover the process including startup, unlike the internal
timing boundary. sudo perf record captures1K cycle samples with zero lost;
72.81% self samples are in the AVX2 saxpy f32 kernel,21.54% in forward and1.67%
in layer materialization. This points toward kernel/control-flow work rather
than memory-latency tiling. The owned kernel contains a zero-input skip branch;
its contribution is a hypothesis, not a demonstrated causal improvement.

Heaptrack records34 total allocation calls for both one and20000 measured
22-query sweeps, each after the same warmup. No allocation growth appears with
repeated predictor evaluation. Both traces report77.95MB peak heap, including
the full77,856,780-byte corpus fixture; this is not a deployment-memory claim.
/usr/bin/time -v reports maximum RSS78,368KiB for the counter run. Contextual
predictor buffers already avoid per-query growth in this workload; adding a
pool is not supported by these traces.

run-heavy records: pair22s/peak RSS0.07GiB/min available22951MiB under4GiB;
counters25s/0.07GiB under3GiB; call stacks6s/0.07GiB under3GiB; allocation
short0s and long26s, both0.08GiB under3GiB. These include loaders/profilers,
not standalone model memory. Preserve all native counters, stacks and traces.

Evidence: `/mnt/v/output/zensally/residual-native-runtime/2026-11-09-v2/`.
The failed first export remains under sibling2026-11-09. The packet storage
budget is256MiB, with no metric/heat maps. Reproduction recipes in justfile:
composition-export-residual-native, composition-build-residual-native,
composition-qualify-residual-native, composition-residual-runtime-plan,
runtime-pair, composition-profile-residual-runtime,
composition-record-residual-runtime and composition-heap-residual-runtime.
No API calls, TEST inference, crate release or model promotion.
