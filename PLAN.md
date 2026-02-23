# zenfaces — Fast Pure-Rust Face Detection

## Goal

Ship face detection for imageflow's smart crop that runs in **<5ms on CPU** for typical web images (800x600 to 2048x1536), with no C/C++ dependencies, `#![forbid(unsafe_code)]` in application code, and a model small enough to embed in the binary.

## Current State

rustface (SeetaFace FuSt cascade) is the only pure-Rust face detector. It runs at **~800ms** on 1666x1136. That's 250x slower than our saliency engine (~3ms). Unusable for a hot path.

rust-faces shows BlazeFace320 at **2.77ms** via ONNX Runtime (ort), but ort wraps a C library.

## Architecture Decision: tract + BlazeFace

**tract** is a pure-Rust ONNX inference engine (sonos/tract, 2.1k stars, actively maintained). It passes 85%+ of ONNX backend tests including all major vision models. CPU-only, no GPU — which is exactly our use case.

**BlazeFace** (Google, 2019) is designed for sub-millisecond inference on mobile. Two variants:
- **BlazeFace320**: 320x320 input, ~0.3M params, ~100M FLOPs
- **BlazeFace128**: 128x128 input, even fewer FLOPs (front-camera variant)

At 320x320, BlazeFace is ~170x fewer FLOPs than a single SeetaFace pyramid pass on 1080p. With tract's pure-Rust inference, we should hit **2-5ms** on modern x86 CPUs.

## Three Branches to Benchmark

### Branch A: `rustface-optimized` — Fork and optimize rustface

**Purpose:** Establish how fast the SeetaFace cascade can go with mechanical optimizations, no algorithm changes. This is the "known quantity" baseline.

**Source:** Fork from https://github.com/atomashpolskiy/rustface (BSD-2-Clause)

**Optimizations (in priority order):**

1. **Stop cloning model per call** — `create_detector_with_model(model.clone())` copies 1.2MB every call. Make detector reusable.

2. **Uncomment `set_max_scale`** — `detector/mod.rs:47`. Currently commented out; enabling it skips pyramid levels where no face can exist at `min_face_size`. For a 1080p image with min_face=40, this could eliminate 3-4 large-scale levels.

3. **Disable Rayon** — Rayon parallelizes 38 rows of a 40x40 window and small MLP layers. The README itself says `RAYON_NUM_THREADS=2` is optimal. For imageflow (concurrent request processing), single-threaded detection is better. Benchmark with `default-features = false`.

4. **Fix heap allocations in hot loops:**
   - `surf_mlp_featmap.rs:299`: 4x `Vec<*const i32>` of 8 elements per feature per candidate → `[*const i32; 8]`
   - `surf_mlp_featmap.rs:221`: `Vec<u32>` for constant XOR mask per pixel → `const [u32; 4]`
   - `detector/mod.rs:339`: `Vec::insert(len, x)` → `push(x)`

5. **f64 → f32 bilinear resize** — `image_pyramid.rs:193-220` uses f64 throughout. f32 doubles SIMD lanes for identical u8-output quality. This is the hottest single function (called at every pyramid level).

6. **Integer grayscale conversion** — Replace f32 BT.709 with fixed-point: `(54*R + 183*G + 19*B) >> 8`. Autovectorizes to 16 pixels/iteration with AVX2.

7. **Add `#[multiversed]`** to the math kernels (`vector_add`, `vector_sub`, `vector_inner_product`, `copy_u8_to_i32`, `square`, `abs`) and the resize function.

8. **Downsample input before detection** — Our saliency engine already downsamples to 256x256. For face detection, downsample to ~640x480 max before building the pyramid. Cuts pyramid levels from ~17 to ~12 and reduces per-level work by 4-6x.

**Expected result:** ~100-200ms (4-8x faster than stock). Still too slow for a hot path but useful as a no-model-dependency fallback.

**Effort:** 2-3 days

---

### Branch B: `tract-blazeface` — Pure Rust BlazeFace via tract

**Purpose:** The primary candidate. Pure Rust, no C dependencies, sub-5ms target.

**Components:**

1. **Model acquisition:**
   - Export BlazeFace320 to ONNX from the MediaPipe model zoo or the rust-faces project's model downloader
   - Alternatively, use the pre-exported ONNX models from https://github.com/nicholasStrategworx/blazeface-onnx or similar
   - Verify model size (should be ~400KB-1MB)
   - Embed via `include_bytes!`

2. **tract integration:**
   ```toml
   [dependencies]
   tract-onnx = "0.21"  # or latest
   ```
   tract-onnx loads ONNX models and runs inference entirely in Rust. No unsafe in application code.

3. **Preprocessing pipeline:**
   - Input: BGRA u8 pixels from imageflow
   - Resize to 320x320 (or 128x128 for speed mode) using our existing bilinear downsampler
   - Convert BGRA → RGB f32 normalized to [-1, 1] (BlazeFace's expected input)
   - Shape into NCHW tensor: `[1, 3, 320, 320]`

4. **Postprocessing:**
   - BlazeFace outputs: bounding box regressions + confidence scores
   - Apply anchor decoding (BlazeFace uses fixed anchor grids)
   - Non-maximum suppression (NMS) with IoU threshold ~0.3
   - Convert pixel coordinates to percentage FocusRects

5. **Anchor generation:**
   - BlazeFace320 uses a 2-level anchor grid (8x8 and 16x16)
   - Pre-compute anchors at startup, store as `const` or `lazy_static`

6. **Optimizations:**
   - `#[multiversed]` on preprocessing (BGRA→RGB conversion, normalization)
   - tract already uses optimized kernels internally; profile to see if further tuning needed
   - Optional: use tract's `TypedModel::optimize()` and `declutter()` passes
   - Optional: quantize to int8 via tract's quantization support for further speedup

**Expected result:** **2-5ms** for 320x320 inference on x86 AVX2. The preprocessing (resize + color convert) adds ~0.5ms. Total: ~3-6ms.

**Model size:** ~400KB-1MB embedded. Acceptable.

**Effort:** 3-5 days

---

### Branch C: `ort-blazeface` — ONNX Runtime BlazeFace (speed ceiling)

**Purpose:** Establish the theoretical speed ceiling using ONNX Runtime's hand-optimized kernels. This tells us how much room tract has to improve.

**Components:**

1. **Same BlazeFace320 ONNX model** as Branch B
2. **ort crate** (latest, with `load-dynamic` feature to avoid static linking)
3. **Same pre/postprocessing** as Branch B
4. **Benchmark against Branch B** on identical hardware and images

**Expected result:** **1-3ms** — ort uses hand-tuned assembly for conv2d on x86. This is the floor.

**Why this branch exists:** If tract is >2x slower than ort, we know the bottleneck is inference kernel quality and can decide whether the C dependency is worth it. If tract is within 1.5x, pure Rust wins.

**Effort:** 1-2 days (reuses Branch B's pre/postprocessing code)

**Note:** ort pulls in ONNX Runtime (~20MB shared library). Not ideal for deployment but acceptable for benchmarking. This branch is for measurement, not shipping.

---

## Benchmark Protocol

All branches benchmarked with Criterion on identical test images:

| Image | Resolution | Expected Faces |
|-------|-----------|----------------|
| Synthetic face-colored rect | 800x600 | 0-1 (detection sanity) |
| Group photo | 1920x1080 | 5-10 |
| Portrait | 800x1200 | 1 |
| No faces (landscape) | 2048x1536 | 0 |

Metrics:
- **Latency** (p50, p99) — must be <5ms for the shipping candidate
- **Accuracy** — detect all visible faces in test images (manual verification)
- **Binary size** — model + inference engine contribution
- **Memory** — peak RSS during detection

## Decision Matrix

| Factor | Weight | Branch A (rustface) | Branch B (tract) | Branch C (ort) |
|--------|--------|-------------------|-----------------|---------------|
| Latency | 40% | ~150ms ❌ | ~4ms ✅ | ~2ms ✅ |
| Pure Rust | 25% | ✅ | ✅ | ❌ (C dep) |
| Binary size | 15% | +1.2MB model | +0.5MB model | +20MB runtime |
| Accuracy | 10% | Good (SeetaFace) | Good (BlazeFace) | Good (BlazeFace) |
| Maintenance | 10% | We own the fork | tract is active | ort is active |

**Predicted winner:** Branch B (tract + BlazeFace). Pure Rust, <5ms, small binary.

## Project Structure

```
zenfaces/
├── Cargo.toml              # workspace
├── PLAN.md                 # this file
├── crates/
│   ├── zenfaces/           # public API crate (facade)
│   │   ├── Cargo.toml
│   │   └── src/lib.rs      # FaceDetector trait, FaceRect, common types
│   ├── zenfaces-rustface/  # Branch A: optimized rustface fork
│   │   ├── Cargo.toml
│   │   └── src/
│   ├── zenfaces-tract/     # Branch B: tract + BlazeFace
│   │   ├── Cargo.toml
│   │   ├── models/         # embedded ONNX models
│   │   └── src/
│   └── zenfaces-ort/       # Branch C: ort + BlazeFace (benchmark only)
│       ├── Cargo.toml
│       └── src/
└── benches/
    └── face_bench.rs       # comparative Criterion benchmarks
```

The `zenfaces` facade crate defines the trait:

```rust
#![forbid(unsafe_code)]

pub struct FaceRect {
    pub x1: f32,  // percentage 0-100
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
    pub confidence: f32,
}

pub trait FaceDetector {
    fn detect(&mut self, pixels: &[u8], width: u32, height: u32) -> Vec<FaceRect>;
}
```

Each backend implements this trait. imageflow_focus depends on `zenfaces` and picks the backend via cargo features.

## Integration with imageflow_focus

Once a winner is chosen:

```toml
# imageflow_focus/Cargo.toml
[dependencies]
zenfaces = { path = "../zenfaces/crates/zenfaces" }
zenfaces-tract = { path = "../zenfaces/crates/zenfaces-tract", optional = true }

[features]
faces = ["zenfaces-tract"]
```

Replace the current `rustface` dependency entirely. The `faces.rs` module becomes a thin wrapper calling the `FaceDetector` trait.

## Open Questions

1. **BlazeFace ONNX model source:** MediaPipe's official models are in TFLite format. Need to verify a reliable ONNX export exists, or convert ourselves using `tf2onnx` or `onnx-simplifier`. The rust-faces project may have usable exports.

2. **tract maturity for BlazeFace ops:** BlazeFace uses standard ops (Conv2d, DepthwiseConv2d, ReLU, Reshape, Concat). tract handles all of these. But need to verify no edge cases with the specific model.

3. **Anchor format:** BlazeFace anchor definitions are model-specific. The rust-faces crate has working anchor generation code (MIT licensed) we can reference.

4. **128x128 variant:** BlazeFace128 (front-camera) would be even faster (~1ms) but less accurate for multi-face detection at distance. Worth benchmarking as a "fast mode" option.

5. **SCRFD alternative:** SCRFD (insightface, ICLR 2022) offers better accuracy than BlazeFace at similar speed. The SCRFD-500MF model at 640x480 runs at ~46ms — slower than BlazeFace but potentially better detection. Worth trying in Branch B if BlazeFace accuracy is insufficient.

## Execution Order

1. **Week 1:** Branch B (tract + BlazeFace) — this is the likely winner, start here
2. **Week 1 parallel:** Branch C (ort + BlazeFace) — quick to set up, gives speed ceiling
3. **Week 2 if needed:** Branch A (rustface optimized) — only if tract proves too slow
4. **Week 2:** Integration into imageflow_focus, replace rustface dependency

## License

zenfaces: MIT OR Apache-2.0 (dual license, compatible with imageflow's AGPL)
- rustface fork: BSD-2-Clause (compatible)
- BlazeFace model: Apache-2.0 (Google)
- tract: MIT OR Apache-2.0
- ort: MIT (benchmark only, not shipped)
