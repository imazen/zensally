# Saliency models and runtime cost

Source and dependency audit: 2026-11-06, following the dependency refresh
[65d8f879](https://github.com/imazen/zensally/commit/65d8f879).

## What is implemented

Both backends' `MicroSalNet::new()` load **v1**,
`microsalnet.onnx.gz`. The better v3ds w24/w28 models are committed but have
no selectable detector constructor or feature. `ContentAnalyzer` therefore
also uses v1. The backend's default feature enables UltraFace; `analyzer`
adds MicroSalNet. The plugin backend enables both by default.

The [March exploration](../training/SALIENCY-EXPLORATION.md) records this
DUTS-TE evaluation (5,019 images). These are historical results, not a new
quality evaluation or timings of tract 0.23.10. The raw evaluation logs were
not available in this checkout, so the scores have not been independently
reproduced in this audit. Lower MAE and higher F-beta are better.

| Model | MAE | F-beta | Recorded tract ms | Current integration |
|---|---:|---:|---:|---|
| MicroSalNet v1 | 0.120 | 0.550 | 7.5 | Both backends |
| MicroSalNet v3ds w24 | 0.080 | 0.652 | 15.7 | Committed model only |
| MicroSalNet v3ds w28 | 0.078 | 0.654 | 21 | Committed model only |
| U2-Netp | 0.054 | 0.744 | 1828 | `u2netp`, tract backend |
| SelfieSeg | 0.107 | 0.564 | 31 | `selfie_seg`, tract backend |

**Best compact candidate: v3ds w24.** It improves the recorded v1 quality
and has a smaller embedded model. Its recorded 15.7 ms exceeds a strict
15 ms budget. **Higher-quality compact candidate: w28**, with a small quality
gain over w24. **Best recorded mask accuracy: U2-Netp**, at much higher
recorded CPU cost. SelfieSeg targets people rather than general subjects.
CSNet was explored but has no production detector here; the recorded
360 ms (201 ms after an upsampling rewrite) did not meet the target budget.

DUTS salient-object mask quality does not establish eye-fixation accuracy
or downstream crop quality. A model promotion needs held-out crop evaluation
as well as mask evaluation and current runtime measurements.

## Model payloads

These byte counts come from the committed gzip files and decompression,
not from parameter-count estimates. Each feature embeds its selected model;
merely storing the w24/w28 files in the repository does not link them.

| Model | Embedded gzip bytes | ONNX bytes after decompression |
|---|---:|---:|
| MicroSalNet v1 | 878,253 | 953,851 |
| MicroSalNet v3ds w24 | 706,570 | 769,236 |
| MicroSalNet v3ds w28 | 950,093 | 1,031,040 |
| U2-Netp | 4,232,846 | 4,553,647 |
| SelfieSeg | 265,116 | 459,047 |
| UltraFace (faces) | 1,064,589 | 1,270,727 |

`analyzer` embeds both UltraFace and MicroSalNet v1. These payloads exclude
runtime code, allocation overhead, optimized weights and activation buffers.
They are neither final executable size nor peak inference memory.

Reproduce payload sizes and the resolved graphs with `just dependency-weight`.
For package identities and model hashes, use
`python3 tools/dependency_weight.py --json`. Resolve first with `cargo update`;
the workspace's root lockfile is not tracked.

## Dependency weight

Unique resolved normal/build dependencies, excluding the root package and
all dev-only tooling, on Linux x86_64, resolved 2026-11-06:

| Consumer | Dependencies | What it buys |
|---|---:|---|
| `zensally`, default | 0 | Types, preprocessing, output decoding |
| `zensally` + `serde` | 7 | Serialization and derive tooling |
| `zensally` + `zenlayout` | 2 | Smart-crop geometry bridge |
| `zensally-tract`, default | 97 | General ONNX parser, optimizer and CPU engine |
| `zensally-zentract`, default | 14 | Plugin discovery/loading and host-side API |
| `zensr-micro`, default | 10 | Specialized super-resolution CNN kernels |
| `zenpredict` 0.2.0, default | 11 | Packed dense-MLP runtime |

The tract graph was 118 before the refresh. The update selects published
tract-onnx 0.23.10, disables its optional transformer operators, and uses
published zenflate 0.4.1 without compression/SIMD defaults. Benchmark tooling
now uses zenbench 0.1.11 instead of Criterion. Benchmark dependencies are
excluded from this table.

The sibling-runtime counts were resolved from `zensr-micro` at zensr
`c1bcd6a` and `zenpredict` at zenanalyze `208ee7e`, using a separate consumer
manifest and each package's defaults. Their direct runtime dependencies are:

- micro: archmage and magetypes; pixel integration is optional.
- predict: archmage, bytemuck, libm and lz4_flex; archmage is optional with
  `default-features = false`.

Counts include build/proc-macro packages, and vary by target and resolution.
They do not measure code bytes or compilation time. The plugin host still
requires a deployed `libzentract_abi` carrying the inference runtime: moving
tract into a library removes it from the application's compile graph, not
from the deployed system. A second model sharing tract pays the model and
activation cost, without adding another independent engine.

## Could a custom runtime help?

**Yes: a fixed-graph CNN runtime is a plausible way to remove the general
ONNX machinery.** Reuse the kernel and packing approach from zensr-micro,
export a fixed plan plus weights offline, preallocate scratch, and keep
zensally's preprocessing/decoding and trait boundary. This can avoid loading
and optimizing an ONNX graph at startup. No saliency-specific executable-size,
startup, speed or memory savings have been measured yet.

It is not a direct model swap into the existing SR runtime. The v3ds
[model definition](../training/model_v3ds.py) requires strided/depthwise and
*dilated* convolutions, residual adds, squeeze/excitation reductions and
broadcast multiplication, concatenation, transposed convolutions, ReLU and
sigmoid. The current micro entry points load particular SR architectures;
their dense-convolution and activation kernels are useful starting points,
not an implemented v3ds executor. Batch normalization may be folded offline
only with output-parity verification.

**Zenpredict cannot execute these CNN graphs.** It accepts feature vectors
and dense layers, not spatial convolution tensors. It could score crop
candidates or predict a coarse map from extracted spatial features, but
that requires a different model, a feature extractor, training and crop
validation. It is not a conversion of the current MicroSalNet weights.

The next concrete runtime experiment is a private w24 fixed-graph executor,
validated operator-by-operator and end-to-end against the existing ONNX
path before any promotion. Measure release bytes, cold load, warm inference
and peak memory on the same hardware, with unchanged weights and
preprocessing. Quantization and approximate activations are separate quality
changes; neither follows from removing ONNX. Keep new public signatures out
of that experiment until their API is reviewed.

## Dependency-refresh validation

Linux x86_64: `just verify-update` passed formatting, all-target/all-feature
Clippy, all-feature workspace tests, doctests, and core checks without default
features (including serde/zenlayout). All seven tract detectors were exercised
by their existing tests. One existing doctest remains ignored. Plugin-host
compilation passed; dynamic plugin inference was not exercised. The full
DUTS/crop evaluation and cross-version tensor-bit parity were not measured.
