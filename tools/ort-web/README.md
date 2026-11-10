# Reduced ORT Web experiment

Build a model-specific WASM CPU runtime and compare its raw outputs with stock
ORT Web. This does not port Lite2, change the shipping crop path, benchmark
latency, or qualify a GPU backend.

Pinned inputs:

- ONNX Runtime v1.31.1, commit `6af1a932e2286d5c58166085d24c68783033edee`.
- Source archive: <https://codeload.github.com/microsoft/onnxruntime/tar.gz/6af1a932e2286d5c58166085d24c68783033edee>,
  SHA256 `ca15bee9ccf04f17da55d78834cce95168e3e2178edf402bd38d2e5eb95b1e26`.
- Its emsdk submodule commit `c0bb220cb6e6f4e0fabb6f6db9efd53390ef5e56`;
  SDK4.0.23 and its paired Node22.16.0/npm installation.
- The existing compressed YuNet model and three owned-photo RGB previews,
  source/hash-bound by `prepare.py`. Square-stretched BGR640 inputs are diagnostic
  backend fixtures, not the native detector's letterbox preprocessing.

`composition-ort-web-prepare PYTHON OUTPUT FLATBUFFERS_SITE` converts the model
to portable fixed ORT format and generates its operator/type configuration.
No x86-specific target is supplied to the converter. It also copies the
already installed stock1.31.1 WASM-only package files for a matched-version
comparison. The Python interpreter must already provide NumPy/ORT1.31.1;
the extra site supplies the pure-Python FlatBuffers package without modifying
existing environments.

`composition-ort-web-build SOURCE BUILD CONFIG PYTHON FLATBUFFERS_SITE` builds
MinSizeRel with SIMD, pthread capability, reduced operators/types and no ONNX
importer. Run through run-heavy with12GiB/six jobs. LLVM's missed-vectorization
diagnostic remains visible but is not fatal; no numeric test is disabled.
All other compiler-warning policy remains upstream's.

Copy the generated `ort-wasm-simd-threaded.mjs/.wasm` into `OUTPUT/reduced/`,
then run `composition-ort-web-verify OUTPUT PLAYWRIGHT_PACKAGE CHROMIUM` through
run-heavy. It launches a bounded local server/browser, selects one inference
thread, compares all outputs on all three tensors and retains exact output
bytes, hashes and scalar differences. The diagnostic tolerance is
`abs(delta) <= 1e-5 + 1e-5*abs(stock)`; exact-bit agreement is reported separately.
This tolerance is not a shipping detector/crop gate. Output shape differences
and nonfinite values fail explicitly. No images enter the chat context.

Raw, gzip9 and Brotli9 file sizes are measured; both compressed artifacts are
retained. Stock is the published package, while reduced is a MinSizeRel source
build, so the comparison measures delivered size rather than isolating operator
reduction from toolchain/compiler settings. Model and common JS costs are separate.

GPU alternatives: [ORT WebGPU](https://onnxruntime.ai/docs/tutorials/web/ep-webgpu.html)
already uses browser WebGPU. [Burn WGPU](https://github.com/tracel-ai/burn/tree/main/crates/burn-wgpu)
provides a Rust inference path through wgpu; adopting it still needs model/operator
and numerical qualification. A safe API is not a claim that every transitive
implementation or driver is unsafe-free, or that shader arithmetic is correct.
