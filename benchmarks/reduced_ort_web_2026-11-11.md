# Reduced ORT Web for YuNet

Completed2026-11-11 from a2026-11-10 run directory. A model-specific WASM CPU
build runs the existing YuNet graph with all36 output tensors bit-identical to
stock ORT Web across three retained photo tensors. No detector, crop, Lite2 or
GPU performance claim follows. No shipped backend or default changes.

## Actual runtime weight

Both arms use ORT1.31.1 and the same preconverted ORT model. Three files are
counted per runtime: common JS bundle, Emscripten loader and WASM binary.
Model data is separate. Stored gzip uses level9; Brotli uses quality9. Each
compressed candidate is retained alongside raw bytes; these are artifact sizes,
not a claim about HTTP server compression configuration.

|Runtime|Raw bytes|gzip9 bytes|Brotli9 bytes|
|---|---:|---:|---:|
|stock|14,950,614|3,862,290|2,831,215|
|reduced|1,360,173|516,000|435,025|

WASM alone is14,848,852 bytes stock and1,263,636 bytes reduced. The common JS
bundle stays77,024 bytes; loader becomes19,513 bytes versus24,738. The ORT-format
model is295560 bytes. This experiment changes runtime packaging, not model weights.
Stock is the published package; reduced is a MinSizeRel source build with
model-specific operator/type registration. The size difference therefore
includes compiler/toolchain/build-setting differences, not solely op removal.
No smaller-runtime latency or memory advantage has been measured.

## Model and correctness scope

Compressed YuNet source lives in crates/zensally-tract/models. Its decompressed
ONNX SHA256 is8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4.
The1.31.1 converter uses fixed optimizations without an architecture-specific
target, excluding x86 NCHWc. The generated config retains eight operator kinds:
Transpose, Reshape, Sigmoid, Add, Conv, MaxPool, Resize and FusedConv.
The minimal runtime loads ORT format; it does not include a general ONNX importer.

Three existing source-bound DEV photographs use the owned square640 resize,
BGR float32 packing and identical input tensors for both runtimes. These are
backend diagnostic fixtures, not a reproduction of the native detector's
letterbox preprocessing. No image encoder, metric map, new label or TEST is used.
One browser process runs separate pages for stock/reduced, one inference thread,
WASM CPU only. All403200 scalar outputs are finite and bit-identical; max absolute
difference0. All shapes and output names match. Exact output bytes/hashes are
retained. No NMS, face-box, crop rectangle, cross-browser or GPU qualification.
Only YuNet is covered; Lite2 remains a TFLite asset needing separate integration.

## Build and resources

ORT v1.31.1 commit6af1a932e2286d5c58166085d24c68783033edee; emsdk submodule
c0bb220cb6e6f4e0fabb6f6db9efd53390ef5e56; Emscripten4.0.23, Node22.16.0.
Build uses MinSizeRel, SIMD and pthread capability, reduced ops/types,
no RTTI, no ML ops and disabled WASM exception catching. See
[reproduction protocol](../tools/ort-web/README.md) and tracked build/verify tools.

Initial failures are retained: missing Python FlatBuffers; Node PATH shim without
adjacent npm; LLVM missed-vectorization warning promoted to error for unused
DeformConv code at MinSizeRel. Repairs isolate the existing pure-Python package,
select SDK Node/npm and retain that specific compiler diagnostic without making
it fatal. No model arithmetic, numeric tolerance or quality threshold is relaxed.
No source patch is applied to ORT. Upstream unit tests are excluded from this
custom WASM packaging build; actual browser inference is checked separately.

Successful build: run-heavy175s, peak RSS1.11GiB, minimum available20433MiB,
peak load5.99,12GiB/six-job cap. Browser parity/compression:2s, peak RSS0.21GiB,
minimum available22683MiB, peak load3.52,3GiB/two-job cap. These are wrapper
process RSS observations, not aggregate browser memory or inference timings.
Full failed and successful logs are retained.

## GPU safety boundary

[ORT WebGPU](https://onnxruntime.ai/docs/tutorials/web/ep-webgpu.html) already
executes through the browser WebGPU API. It is distinct from native ORT
accelerator providers; the host ORT implementation in WASM remains separate
from browser GPU validation and driver trust.
[Burn WGPU](https://github.com/tracel-ai/burn/tree/main/crates/burn-wgpu) is a
Rust-side inference path using wgpu and supporting native graphics backends and
browser WebGPU. It is a plausible alternative, not qualified here. Safe APIs do
not imply unsafe-free transitive implementations, correct shader math, or absence
of driver boundaries. This reduced CPU build says nothing about GPU speed/safety.

## Artifacts

Checked on i265: `/mnt/v/output/zensally/reduced-ort-web/2026-11-10/`.
The packet retains raw/compressed stock/reduced runtimes, source-bound inputs,
ORT/ONNX models, config, outputs, identities and all logs. External source/SDK
build trees remain local; archive URLs and hashes are in the protocol. No
binaries or generated tensors are committed to git. No paid API calls.

Frozen packet:108 members,42,804,951 member bytes; manifest SHA256
`d423d59c64c13d76017b22c1b4363e26cb1d848953361694276d883afc504eec`.
All108 members independently verified on dev under
`/mnt/tower/output/zensally/reduced-ort-web/2026-11-10/`.
