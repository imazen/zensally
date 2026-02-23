# zenfaces — Fast Pure-Rust Face Detection & Neural Saliency

## Goal

Ship face detection and optional neural saliency for imageflow's smart crop. Both running via **tract** (pure-Rust ONNX inference), **<5ms each on CPU** for typical web images, `#![forbid(unsafe_code)]`, models small enough to embed in the binary.

## Why Both in One Project

If we embed tract for face detection, the marginal cost of adding a saliency model is just the model weights — the inference engine is already paid for. A neural saliency model replaces our hand-tuned composite engine (edge + skin + saturation) with a single forward pass trained on where humans actually look. This captures semantic content — dogs, text, hands, cars — that pixel-level heuristics can't.

## Current State

| Detector | Approach | Latency | Quality |
|----------|----------|---------|---------|
| imageflow_focus saliency | Hand-tuned (edges + skin + saturation) | ~3ms @ 256x256 | Good for faces/bright objects, blind to semantics |
| rustface | SeetaFace cascade (pure Rust) | ~800ms @ 1080p | Untested — pure Rust, no C deps, needs speed optimization |
| rust-faces BlazeFace | ONNX Runtime (C dep) | ~2.8ms @ 320x320 | Poor reliability — designed for selfie framing, not general photos |

### BlazeFace / MediaPipe Reliability Problems

BlazeFace (used by MediaPipe) is optimized for mobile selfie speed, not general-purpose face detection:
- <68% on WiderFace Hard — worst among modern face detectors
- Fixed tiny input (128x128 front / 256x256 back) limits resolution
- No multi-scale feature pyramid — struggles with small faces and groups
- Designed for single-face selfie/video-call framing
- MediaPipe inherits all these limitations

BlazeFace is not sufficient for production smart cropping where we need to find faces in group photos, portraits with varied angles, and real-world lighting conditions.

### Face Detector Comparison (WiderFace Hard = hardest benchmark, most relevant to real use)

| Model | Params | ONNX Size | WF Hard | Est. CPU ms | tract Risk | License (weights) |
|-------|--------|-----------|---------|-------------|------------|-------------------|
| BlazeFace | ~0.1M | ~0.4 MB | <68% | <1ms | Known OK | Apache-2.0 |
| **YuNet** | **0.083M** | **337 KB** | **70.8%** | **~2ms** | **Low** | **MIT** |
| UltraFace RFB-640 | ~1M | ~1 MB | 66.3% | ~5ms | **Proven** | MIT |
| SCRFD_2.5G | 0.67M | 3.1 MB | 77.9% | ~45ms | Low-Med | Non-commercial |
| RetinaFace-Mob0.25 | ~0.44M | ~1.7 MB | 81.0% | ~20ms | Medium | MIT |

**YuNet** (OpenCV Zoo, MIT) is the top candidate: smaller than BlazeFace, more accurate, and simple anchor-free architecture ideal for tract. UltraFace is already proven working with tract-onnx in multiple Rust projects. SCRFD has the best accuracy but non-commercial weight licensing is a blocker unless we retrain or license commercially.

## Architecture: tract as Unified Inference Engine

**tract** (sonos/tract, MIT/Apache-2.0, 2.1k stars, actively maintained) is a pure-Rust ONNX inference engine. CPU-only, no GPU — which is our use case. Passes 85%+ of ONNX backend tests including all major vision models.

tract adds ~2MB to binary size. After that, each model is just embedded weights.

---

## Part 1: Face Detection

### Branch A: `rustface-optimized` — Fork and optimize rustface

**Purpose:** Establish how fast SeetaFace can go with mechanical optimizations. "Known quantity" baseline. No-model-dependency fallback.

**Source:** Fork from https://github.com/atomashpolskiy/rustface (BSD-2-Clause)

**Optimizations (priority order):**

1. **Stop cloning model per call** — `create_detector_with_model(model.clone())` copies 1.2MB every call. Make detector reusable.
2. **Uncomment `set_max_scale`** — `detector/mod.rs:47`. Skips pyramid levels where no face can exist at `min_face_size`.
3. **Disable Rayon** — parallelizes 38 rows of a 40x40 window. Overhead dominates. Benchmark `default-features = false`.
4. **Fix heap allocations in hot loops:**
   - `surf_mlp_featmap.rs:299`: 4x `Vec<*const i32>` of 8 elements → `[*const i32; 8]`
   - `surf_mlp_featmap.rs:221`: `Vec<u32>` for constant XOR mask per pixel → `const [u32; 4]`
   - `detector/mod.rs:339`: `Vec::insert(len, x)` → `push(x)`
5. **f64 → f32 bilinear resize** — `image_pyramid.rs:193-220`. Doubles SIMD lanes for identical u8-output quality.
6. **Integer grayscale conversion** — `(54*R + 183*G + 19*B) >> 8`. Autovectorizes to 16 px/iter with AVX2.
7. **`#[multiversed]`** on math kernels and resize function.
8. **Downsample input** to ~640x480 max before building the pyramid.

**Expected:** ~100-200ms. Still too slow for hot path. Useful as fallback when tract isn't available.

**Effort:** 2-3 days

---

### Branch B: `tract-face` — Pure Rust Face Detection via tract

**Purpose:** Primary face detection. Pure Rust, sub-10ms, reliable on real-world photos.

#### B1: YuNet (Top Choice)

**YuNet** (OpenCV Zoo, 2023) — anchor-free face detector, 83K params, 337KB ONNX.
- **Paper:** "YuNet: A Tiny Millisecond-level Face Detector" (Machine Intelligence Research, 2023)
- **Source:** [opencv/opencv_zoo](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet)
- **Input:** 320x320 (dynamic supported). Detects faces from ~10x10 to 300x300 pixels.
- **Output:** Bounding boxes + 5-point landmarks + confidence
- **Accuracy:** 83.4% Easy / 82.4% Medium / 70.8% Hard (WiderFace)
- **Speed:** ~1.6ms at 320x320 on CPU
- **Architecture:** Depthwise separable convolutions + TFPN neck. Anchor-free (no anchor decoding needed). Standard ONNX operators only — Conv, BN, ReLU. No Resize in the model graph for fixed input.
- **License:** MIT (code AND weights)
- **ONNX:** Pre-exported at [opencv_zoo](https://github.com/opencv/opencv_zoo/blob/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx) and [HuggingFace](https://huggingface.co/opencv/face_detection_yunet)

#### B2: UltraFace RFB-640 (Proven tract Fallback)

**UltraFace** — already proven working with tract-onnx in multiple Rust projects ([infercam_onnx](https://github.com/sgasse/infercam_onnx), [dfinity face-recognition](https://github.com/dfinity/examples/tree/master/rust/face-recognition)).
- **Source:** [Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB](https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB)
- **Input:** 640x480
- **Accuracy:** 81.6% Easy / 80.2% Medium / 66.3% Hard
- **Size:** ~1MB ONNX
- **License:** MIT
- **ONNX:** Pre-exported in the repo

Lower accuracy than YuNet but **guaranteed tract compatibility**. Use as fallback if YuNet hits tract operator issues.

#### B3: BlazeFace (Speed Baseline Only)

Keep BlazeFace as a speed baseline to contextualize the other results, but do not plan to ship it given its <68% WF Hard.

**Implementation (shared across B1/B2):**

1. **Model:** Download pre-exported ONNX. Embed via `include_bytes!`.

2. **Preprocessing:** BGRA u8 → resize to model input size → RGB f32 normalized → NCHW tensor.

3. **Postprocessing:** YuNet outputs boxes directly (anchor-free). NMS with IoU ~0.3. Convert to percentage FocusRects.

4. **Optimizations:** `#[multiversed]` on preprocessing. `TypedModel::optimize()` + `declutter()`.

**Expected:** 2-5ms total. **Target: <10ms.**

**Effort:** 3-5 days

---

### Branch C: `ort-face` — ONNX Runtime (speed ceiling measurement)

Same models, same pre/postprocessing, but using **ort** (ONNX Runtime C bindings) to measure the theoretical floor. If tract is within 1.5x of ort, pure Rust wins.

**Expected:** 1-3ms. **This branch is for benchmarking, not shipping.**

**Effort:** 1-2 days

---

## Part 2: Neural Saliency

### The Problem with Our Composite Engine

Our hand-tuned saliency engine (edge + YCbCr skin + HSL saturation) works for:
- Photos with faces (skin detection)
- Photos with colorful foreground on neutral background (saturation)
- Photos with sharp subjects on blurry backgrounds (edge detection)

It fails for:
- Dog on a beach (edges everywhere, no skin, moderate saturation)
- Text/sign in a scene (edges but no color signal)
- Person in dark clothing (skin only on face/hands)
- Product photos with neutral colors (no saturation signal)

A neural model trained on eye-tracking data captures **semantic** saliency — it knows a dog is interesting even when it's the same color as the sand.

### Candidate Models

#### CSNet (100K params, ~400KB) — Top Choice

- **Paper:** ECCV 2020 / TPAMI 2021
- **Architecture:** Generalized OctConv (gOctConv) for multi-scale features with 80% parameter reduction
- **Training:** From scratch on DUTS/ECSSD (manual salient object annotations), no ImageNet pre-training needed
- **Input:** 256x256 (matches our existing working size)
- **Output:** Single-channel saliency heatmap, 0.0-1.0
- **Size:** ~400KB ONNX (100K params × 4 bytes, plus structure)
- **Expected inference:** <2ms at 256x256 on tract (100K params is tiny)
- **License:** Open source
- **ONNX:** PyTorch model, standard conversion path

This is ideal: the model is smaller than a JPEG, inference is trivial, and it replaces three hand-tuned signal functions with a single forward pass.

#### U2-Netp (small variant, 4.7MB) — Backup

- **Paper:** Pattern Recognition 2020
- **Architecture:** Nested U-structure, small variant trained from scratch
- **Training:** DUTS dataset
- **Input:** 320x320
- **Output:** Full-resolution saliency mask
- **Size:** 4.7MB ONNX
- **Expected inference:** 5-10ms at 320x320 on tract
- **License:** MIT (Apache-2.0 for weights)
- **ONNX:** Well-established export path, many community ports

Fallback if CSNet accuracy is insufficient. 4.7MB is heavier but still embeddable.

### Branch D: `tract-csnet` — Neural Saliency via tract

**Implementation:**

1. **Model:** Export CSNet to ONNX from PyTorch. Optionally INT8 quantize (~100KB). Embed via `include_bytes!`.

2. **Preprocessing:** BGRA u8 → resize to 256x256 (already done for our composite engine) → RGB f32 normalized [0, 1] → NCHW `[1, 3, 256, 256]`.

3. **Postprocessing:** Output is a 256x256 heatmap. Find peak region (threshold at 50% of max, bounding box — same approach as our current `extract_focus_rects`). Convert to percentage FocusRects.

4. **Integration strategy:**
   - When tract is available: use CSNet saliency instead of the composite engine
   - Composite engine remains as fallback (no-model mode)
   - Face detection results (if enabled) still override/dominate saliency with weight 10.0

**Expected:** <2ms inference + ~0.5ms pre/postprocessing = **~2.5ms total**. Comparable to the composite engine but with semantic understanding.

**Effort:** 2-3 days (reuses tract setup from Branch B)

### Branch E: `tract-u2netp` — U2-Netp Saliency (accuracy comparison)

Same approach as Branch D but with U2-Netp small. Benchmarked against CSNet on the same test images for accuracy comparison. If CSNet is "good enough," this branch is unnecessary.

**Effort:** 1 day (trivial variant of Branch D)

---

## Combined Pipeline (Final Architecture)

```
Input BGRA image
    │
    ├── Downsample to 320x320 (shared)
    │
    ├── YuNet (tract)     →  face FocusRects (weight 10.0)    ─┐
    │                                                           │
    ├── CSNet (tract)     →  saliency FocusRects (weight 1.0)  ├── Merge → Smart Crop
    │                                                           │
    └── User &c.focus=    →  manual FocusRects (weight varies)  ─┘
```

Both models share the same tract runtime. The preprocessing resize is done once. Two inference calls back-to-back: ~4ms faces + ~2.5ms saliency = **~6.5ms total** for full analysis.

Without the `faces` feature: just CSNet saliency at ~2.5ms — faster than our current composite engine and more accurate.

Without any neural models: fall back to the composite engine (~3ms) that we already ship.

---

## Benchmark Protocol

All branches benchmarked with Criterion on identical test images:

**Face detection images:**
| Image | Resolution | Expected Faces |
|-------|-----------|----------------|
| Synthetic face-colored rect | 800x600 | 0-1 |
| Group photo | 1920x1080 | 5-10 |
| Portrait | 800x1200 | 1 |
| Landscape (no faces) | 2048x1536 | 0 |

**Saliency images:**
| Image | Content | Expected Salient Region |
|-------|---------|------------------------|
| Dog on beach | Animal on sand | Dog |
| Text on sign | Sign in landscape | Sign text |
| Red car on road | Vehicle in scene | Car |
| Portrait | Person on background | Face/upper body |
| Product photo | Object on white | Product |

**Comparison:** Neural saliency (CSNet/U2-Netp) vs composite engine on the same images. Measure whether the neural model's focus region is more semantically correct.

**Metrics:**
- **Latency** (p50, p99) — must be <5ms per model
- **Accuracy** — manual verification of focus regions
- **Binary size** — tract runtime + model weights
- **Memory** — peak RSS during inference

## Decision Matrix

| Factor | Weight | A (rustface) | B1 (tract+YuNet) | B2 (tract+UltraFace) | D (tract+CSNet) |
|--------|--------|-------------|-------------------|-----------------------|-----------------|
| Latency | 30% | ~150ms ❌ | ~4ms ✅ | ~7ms ✅ | ~2.5ms ✅ |
| Pure Rust | 20% | ✅ | ✅ | ✅ | ✅ |
| Binary size | 10% | +1.2MB | +2MB tract +337KB | +2MB tract +1MB | +0.4MB (shared tract) |
| Quality | 30% | Decent frontal | 70.8% WF Hard ✅ | 66.3% WF Hard | Semantic saliency |
| License | 10% | BSD-2 ✅ | MIT ✅ | MIT ✅ | Verify ✅ |

**Predicted shipping configuration:** B1 + D (tract shared). YuNet for faces, CSNet for saliency. Total model weight: ~737KB. Total added binary: ~3MB. Total latency: ~6.5ms for both, ~2.5ms saliency-only.

## Project Structure

```
zenfaces/
├── Cargo.toml                  # workspace
├── PLAN.md                     # this file
├── crates/
│   ├── zenfaces/               # public API crate (facade)
│   │   ├── Cargo.toml
│   │   └── src/lib.rs          # FaceDetector + SaliencyDetector traits, FocusRect
│   ├── zenfaces-tract/         # Branches B + D: tract-based face + saliency
│   │   ├── Cargo.toml
│   │   ├── models/             # embedded ONNX models (BlazeFace + CSNet)
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── blazeface.rs    # face detection
│   │       ├── csnet.rs        # saliency detection
│   │       └── preprocess.rs   # shared resize + color conversion
│   └── zenfaces-ort/           # Branch C: ort benchmark only
│       ├── Cargo.toml
│       └── src/
└── benches/
    ├── face_bench.rs           # face detection comparison
    └── saliency_bench.rs       # saliency comparison (neural vs composite)
```

The facade crate:

```rust
#![forbid(unsafe_code)]

pub struct FocusRect {
    pub x1: f32,  // percentage 0-100
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
    pub confidence: f32,
    pub kind: FocusKind,
}

pub enum FocusKind {
    Face,
    Saliency,
}

pub trait FaceDetector {
    fn detect_faces(&mut self, pixels: &[u8], width: u32, height: u32) -> Vec<FocusRect>;
}

pub trait SaliencyDetector {
    fn detect_saliency(&mut self, pixels: &[u8], width: u32, height: u32) -> Vec<FocusRect>;
    fn saliency_map(&mut self, pixels: &[u8], width: u32, height: u32) -> Vec<f32>;
}
```

## Integration with imageflow_focus

```toml
# imageflow_focus/Cargo.toml
[dependencies]
zenfaces = { path = "../zenfaces/crates/zenfaces" }
zenfaces-tract = { path = "../zenfaces/crates/zenfaces-tract", optional = true }

[features]
default = ["saliency"]
saliency = []                           # composite engine (no model, always available)
neural-saliency = ["zenfaces-tract"]    # CSNet replaces composite engine
faces = ["zenfaces-tract"]              # BlazeFace face detection
```

The `analyze_all` function checks features at compile time:
- `neural-saliency` enabled → use CSNet via tract, skip composite engine
- `neural-saliency` disabled → use composite engine (current code)
- `faces` enabled → also run BlazeFace, merge face rects with weight 10.0

## Open Questions

1. **CSNet ONNX availability:** Need to verify a pre-exported ONNX model exists or export from PyTorch ourselves. The architecture is standard (Conv2d, BatchNorm, ReLU, OctConv) but gOctConv may need custom export handling.

2. **tract + OctConv:** Generalized OctConv decomposes into standard ops (grouped convolutions + upsampling + downsampling). Need to verify tract handles the decomposed graph efficiently.

3. **INT8 quantization:** Both CSNet and YuNet can be quantized to INT8, halving model size and potentially doubling inference speed. Need to test accuracy impact.

4. **YuNet tract compatibility:** YuNet uses only standard ops (Conv, BN, ReLU, depthwise separable convolutions) and is anchor-free. Should be straightforward for tract, but needs verification. UltraFace is the proven-working fallback.

5. **Saliency vs attention:** CSNet is trained on salient object detection (DUTS — human-annotated object masks). This finds "the subject." An alternative is fixation prediction (SALICON/MIT1003 — eye-tracking data). This finds "where people look first." For smart cropping, SOD is probably better (we want to keep the subject in frame, not just the first fixation point).

6. **Downsampling strategy:** Both models want small inputs (256x256 or 320x320). We already downsample for the composite engine. Share the downsampled buffer between saliency and face detection to avoid redundant work.

7. **SCRFD licensing:** SCRFD_2.5G has the best accuracy (77.9% WF Hard) but pretrained weights are non-commercial only. Options: (a) contact InsightFace for commercial license, (b) retrain from scratch using MIT-licensed training code on WiderFace, or (c) accept YuNet's 70.8% as good enough. Probably (c) — the gap between 70.8% and 77.9% matters less for smart crop (where we just need to find *some* faces) than for security/recognition applications.

## Execution Order

1. **Week 1:** Branch B (tract + YuNet, with UltraFace fallback) and Branch D (tract + CSNet) in parallel — they share infrastructure
2. **Week 1:** Branch C (ort benchmark) — quick speed ceiling measurement for YuNet + UltraFace
3. **Week 1:** Head-to-head comparisons: YuNet vs UltraFace vs BlazeFace on face detection; CSNet vs composite engine on saliency
4. **Week 2:** Integration into imageflow_focus
5. **Week 2 if needed:** Branch A (rustface optimized) as no-model-dependency fallback
6. **Week 2 if needed:** Branch E (U2-Netp) if CSNet accuracy is insufficient

## License

zenfaces workspace: MIT OR Apache-2.0 (dual, compatible with imageflow's AGPL)
- YuNet model weights: MIT (OpenCV Zoo)
- UltraFace model weights: MIT
- BlazeFace model weights: Apache-2.0 (Google) — benchmark only
- rustface fork: BSD-2-Clause (compatible)
- CSNet model weights: verify license (ECCV 2020 paper, likely open)
- U2-Netp model weights: Apache-2.0
- tract: MIT OR Apache-2.0
- ort: MIT (benchmark only)
