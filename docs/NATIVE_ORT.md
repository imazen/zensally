# Native reduced ORT deployment experiment

This is a standalone C++ deployment probe, separate from the Rust detector API.
It embeds the YuNet ORT-format weights and statically links a model-specific
ORT 1.31.1 CPU runtime. It does not add an ORT dependency to the Rust crates.

The fixed diagnostic input is one 640×640 FP32 BGR tensor. Both stock and reduced
probes use the same model, sequential execution, one inference thread, disabled
graph optimization at load, three warm forwards, and three batches of ten
forwards. Timing includes output allocation and destruction, but excludes session
construction, input generation, preprocessing, NMS, crop selection and evidence IO.
The retained 134,400 output values must satisfy
`abs(reduced-stock) <= 1e-5 + 1e-5*abs(stock)`; complete output payloads are required.
This fixture tests execution agreement, not face-detection or composition quality.

`tools/ort-native/prepare.py` converts the existing compressed ONNX model once
using fixed optimization without platform-specific conversion. Its generated
operator/type configuration drives `build.py`. The build uses MinSizeRel,
minimal ORT, reduced operators/types, static dependencies, disabled telemetry
and RTTI. Exceptions remain enabled. Windows uses the static MSVC runtime and disables optional KleidiAI because
the tested Windows ARM assembler rejects its architecture flag.
Pinned source and stock-package SHA256 values are checked before compilation.
See the upstream [custom-build guide](https://onnxruntime.ai/docs/build/custom.html).

Manual workflow `native-ort.yml` prepares the common model, then builds and runs
native Windows x64 and ARM64 executables. `windows.py prove` copies only the
reduced executable into a fresh directory, inspects imports, restricts PATH to
Windows System32, and requires successful inference with no companion files.
`mac-prove.py` performs the equivalent test with a system-only PATH and rejects
non-system dylib dependencies. These are local execution checks; distribution
signing, notarization and installation on older OS releases are separate work.

The paired benchmark uses zenbench with four blocks and an A/A control. Hosted
Windows and remote Mac runs explicitly permit unverified CPU placement and are
exploratory. They do not establish a dedicated-core performance ranking.
Changing the model or adding detectors requires regenerating the operator list
and repeating deployment and numerical checks. GPU providers are not covered.

Reduced ORT retains native C/C++ implementation and memory-safety obligations.
Static linkage and fewer registered operators reduce deployment size; they do
not provide a Rust memory-safety guarantee or a safe GPU abstraction.

Measurements and artifact identities belong in
[`benchmarks/native_ort_2026-11-11.md`](../benchmarks/native_ort_2026-11-11.md).

All three deployment checks passed: macOS ARM64, Windows x64 and Windows ARM64.
All paired diagnostic outputs matched stock bit for bit. Measured raw executable
sizes, including weights, are 3,689,680 / 3,427,328 / 3,731,456 bytes respectively.
The tested MinSizeRel builds trade deployment size for higher measured
inference latency. Full Rust application integration remains untested.
