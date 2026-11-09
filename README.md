# zensally [![CI](https://img.shields.io/github/actions/workflow/status/imazen/zensally/ci.yml?style=flat-square&label=CI)](https://github.com/imazen/zensally/actions/workflows/ci.yml) [![crates.io](https://img.shields.io/crates/v/zensally?style=flat-square)](https://crates.io/crates/zensally) [![lib.rs](https://img.shields.io/crates/v/zensally?style=flat-square&label=lib.rs&color=blue)](https://lib.rs/crates/zensally) [![docs.rs](https://img.shields.io/docsrs/zensally?style=flat-square)](https://docs.rs/zensally) [![MSRV](https://img.shields.io/badge/MSRV-1.91-blue?style=flat-square)](https://doc.rust-lang.org/cargo/reference/manifest.html#the-rust-version-field) [![license](https://img.shields.io/badge/license-AGPL--3.0%20%2F%20Commercial-blue?style=flat-square)](#license)

zensally is face detection and neural saliency for content-aware image cropping — find the faces and the parts predicted to contain the main subject, then crop to any aspect ratio without cutting off the subject. The `zensally` crate is the shared, pure-Rust core: the result types, the detector traits, model preprocessing, non-maximum suppression, output decoding, and a bridge into [zenlayout]'s smart-crop solver. Ready-to-run detectors with embedded ONNX models live in sibling backend crates that plug into those traits. Pure Rust, `#![forbid(unsafe_code)]`.

The default detector path runs entirely in Rust via [tract] — no ONNX Runtime, no C dependency, models small enough to embed in the binary.

Two unpublished experiments are available in this workspace: `zensally-micro`
runs the fixed MicroSalNet w24 graph, and `zensally-depth` extracts embedded
JPEG/HEIC geometry and computes a confidence-weighted cue. Both require Rust
1.93; their defaults use `no_std + alloc`. Enable `std` for runtime CPU detection,
shared packed weights between active runtime instances, and the runtime tools,
and `jpeg`/`heic` for container extraction. See the
[implementation and measured limits](docs/DEPTH_IMPLEMENTATION.md).
Optional arithmetic-study features are documented in
[the reduction study](tools/reassociate-study/README.md); algebraic math requires
Rust 1.98 and remains disabled by default.
Without std, CPU detection uses compile-time features; a generic x86 build
uses the scalar path. The fixed runtime retains bounded activation scratch
between calls in that configuration.

An [experimental crop-first research prototype](docs/experiments/CROP_FIRST_PROTOCOL.md)
explores composition preferences conditioned on aspect ratio and crop mode.
The [current-corpus weak fitting study](benchmarks/crop_fitting_2026-11-08.md)
retains trained research checkpoints, teacher audits and a human-review gallery.
It is separate from the current detector/solver and has no exported Rust model
or human-rated quality claim.
The [successor corpus scope](docs/experiments/PHOTO_CROP_CORPUS.md) focuses on
people, pets/living subjects and scenery for news, sales and ecommerce photos.
The [synthetic photo study](benchmarks/products_selfies_2026-11-08.md) retains
56 sources, including 24 branded ecommerce shots and 16 smartphone candids/selfies, with
exact prompts, measured API usage and verified lossless JXL dataset blobs.
Experimental [pooled features and training-path probes](benchmarks/crop_training_2026-11-08.md)
exercise GPU/CPU training without claiming artistic quality.

The [existing-model runtime screen](benchmarks/runtime_models_2026-11-08.md)
compares face, person-mask and saliency costs and documents missed-face limits.
The [local crop-teacher study](benchmarks/local_crop_teachers_2026-11-08.md)
separates candidate-search failures from composition-model quality; its
[small-model extent fits](benchmarks/primary_extents_2026-11-08.md) remain
experimental, with no promoted crop network.
The [cached-query runtime study](benchmarks/cached_crop_runtime_2026-11-08.md)
measures encoder reuse, separate thread policies and an embedded zenpredict
query executor, with unchanged frozen-bank crop decisions. It remains research.
The [frozen-encoder preference fits](benchmarks/focal_preference_heads_2026-11-08.md)
also remain unpromoted: better mixed-corpus weak scores do not improve the
photographic subset.

The [additional owned-photo screen](benchmarks/owned_photo_screen_2026-11-08.md)
adds 101 review-only photos, real-photo arithmetic checks and detector-miss
witnesses. Its corrected SVG viewers clip the selected source rectangle.
The [EfficientDet-Lite family comparison](benchmarks/efficientdet_family_2026-11-08.md)
measures Lite0/1/2 reference CPU costs and person detections on those photos.
The [ROI-head controls](benchmarks/synthetic_person_rois_2026-11-08.md) find
useful weak localization in larger frozen features, with incomplete protection.
[Unsplash preparation](benchmarks/unsplash_preparation_2026-11-08.md) adds144
owned working previews. The subsequent [registered relative-rank study](benchmarks/unsplash_rank_distillation_2026-11-08.md)
fixes94/25/25 fit/development/TEST photos and measures image-dependent teacher
agreement, with no human artistic-quality or complete-face claim.
The same five detector checkpoints are also
[screened on all144 Unsplash previews](benchmarks/unsplash_detectors_2026-11-08.md).
The selected photo student's [runtime study](benchmarks/unsplash_student_runtime_2026-11-08.md)
qualifies a smaller ONNX export and faster pooling without promoting its crop
quality; a fixed review still exposes whole-product framing failures.
An explicit [catalogue/hero/detail policy study](benchmarks/unsplash_purpose_witnesses_2026-11-09.md)
preserves supplied regions or uses containment. A separate local
[phrase-grounding and extent study](benchmarks/florence_extents_2026-11-09.md)
finds weak localization signal in larger frozen features, while small features
fail controls and visible product-bound errors remain. Neither promotes a model.
The selected photographic student's [frozen-feature transfer](benchmarks/photo_student_extent_transfer_2026-11-09.md)
also fails the extent controls.
The [matched encoder-training and augmentation follow-ups](benchmarks/florence_extent_finetune_2026-11-09.md)
show overfitting. [Content coverage review](benchmarks/unsplash_content_audit_2026-11-09.md)
and a [supplementary development cohort](benchmarks/supplementary_development_2026-11-09.md)
address evaluation gaps before more fitting; no new artistic labels are claimed.
The [fixed-model supplementary evaluation](benchmarks/supplementary_extents_2026-11-09.md)
finds that none of the extent predictors beats geometry-only controls on the
additional people/wildlife scenes; the earlier FocalNet gain does not carry over.
The [photographic subject-reference audit](benchmarks/photo_subject_retention_2026-11-09.md)
documents prop substitutions, missed people and candidate-bank gaps before
candidate-retention fitting; its boxes remain incomplete weak supervision.
The [nine retention-head controls](benchmarks/photo_retention_heads_2026-11-09.md)
show uncertain classification gains and continued known-box violations, so they
remain research diagnostics.

## Quick start

The full smart-crop flow: detect faces and saliency, then compute crops for several aspect ratios from one analysis.

```toml
[dependencies]
zensally = { version = "0.1", features = ["zenlayout"] }   # core toolkit + smart-crop bridge
# Batteries-included detectors with embedded models (workspace crate; consumed via git):
zensally-tract = { git = "https://github.com/imazen/zensally", features = ["analyzer"] }
zenlayout = { version = "0.2", features = ["smart-crop"] }  # crop geometry solver
```

```rust
use zensally::{ImageRef, PixelFormat};
use zensally_tract::ContentAnalyzer;          // UltraFace (faces) + MicroSalNet (saliency)

let mut analyzer = ContentAnalyzer::new()?;   // loads the embedded ONNX models
let img = ImageRef::new(&rgba, width, height, PixelFormat::Rgba)?;
let analysis = analyzer.analyze(&img);        // faces (percentage coords) + saliency heatmap
println!("found {} face(s)", analysis.faces.len());
```

```rust
use zensally::bridge::build_smart_crop_input;
use zenlayout::smart_crop::{AspectRatio, CropMode};

// Fold faces + saliency (and any manual focus regions) into one crop input:
let input = build_smart_crop_input(analysis, &[]);

// One crop per requested aspect ratio — Vec<Option<Rect>>:
let crops = input.compute_crops(width, height, &[
    (AspectRatio { w: 1,  h: 1  }, CropMode::Minimal),
    (AspectRatio { w: 16, h: 9  }, CropMode::Minimal),
    (AspectRatio { w: 9,  h: 16 }, CropMode::Minimal),
]);
```

Coordinates returned in [`FaceRect`](https://docs.rs/zensally/latest/zensally/struct.FaceRect.html) are percentages (0–100) of image dimensions, so they survive a later resize. `CropMode::Minimal` keeps the subject visible at the largest crop; `CropMode::Maximal` zooms in tight.

## What `zensally` provides

The published `zensally` crate is codec-, runtime-, and model-agnostic. It owns everything around the neural network except the network itself:

| Module | Contents |
|--------|----------|
| (root) | [`FaceRect`], [`SaliencyMap`], [`AnalysisOutput`], [`ImageRef`], [`PixelFormat`], and the [`FaceDetector`] / [`SaliencyDetector`] traits |
| [`preprocess`](https://docs.rs/zensally/latest/zensally/preprocess/) | Bilinear resize to NCHW RGB f32 with `Letterbox`/`Stretch` modes and `CenterScale`/`UnitScale`/`MeanSubtract` normalization; returns `LetterboxInfo` for coordinate reversal |
| [`nms`](https://docs.rs/zensally/latest/zensally/nms/) | IoU and greedy non-maximum suppression over raw detections |
| [`decode`](https://docs.rs/zensally/latest/zensally/decode/) | Turn raw model output tensors into typed results (`decode_ultraface`, `decode_microsalnet`) |
| [`bridge`](https://docs.rs/zensally/latest/zensally/bridge/) | `From` conversions and `build_smart_crop_input` into [zenlayout]'s solver (feature `zenlayout`) |

Serializable records — [`DetectionSummary`], [`SmartCropResult`], [`WhitespaceCropResult`], [`FocusRegion`], [`CropRect`] — derive `serde` traits under the `serde` feature, handy for logs, UI overlays, and debugging.

### Features

| Feature | Default | Effect |
|---------|:-------:|--------|
| `std` | yes | Standard library support |
| `serde` | no | `Serialize` / `Deserialize` on the result types |
| `zenlayout` | no | The `bridge` module and `From` impls into `zenlayout::smart_crop` |

### Bring your own runtime

If you already run ONNX (or any other inference engine), use the core directly: preprocess into the model's input tensor, run your network, decode the outputs. No backend crate required.

```rust
use zensally::{ImageRef, PixelFormat};
use zensally::preprocess::{preprocess_nchw, ResizeMode, Normalization};
use zensally::decode::decode_ultraface;

// 1. Build the model's NCHW RGB f32 input (UltraFace RFB-320 is 320x240):
let mut input = vec![0.0f32; 3 * 320 * 240];
let lb = preprocess_nchw(
    &rgba, width, height, PixelFormat::Rgba,
    320, 240, ResizeMode::Letterbox, Normalization::CenterScale,
    &mut input,
);

// 2. Run `input` through your ONNX runtime → `scores`, `boxes` output slices.

// 3. Decode to FaceRects (letterbox reversed, NMS applied):
let faces = decode_ultraface(
    &scores, &boxes, 320.0, 240.0, &lb,
    width as f32, height as f32,
    0.7,   // score threshold
    0.3,   // NMS IoU threshold
);
```

Or implement [`FaceDetector`] / [`SaliencyDetector`] over your engine and feed the results straight into the `bridge`.

## Backends

Two crates implement the traits with embedded models so you don't have to wire up inference yourself:

| Crate | Inference | Notes |
|-------|-----------|-------|
| [`zensally-tract`](https://github.com/imazen/zensally/tree/main/crates/zensally-tract) | [tract] (pure-Rust ONNX, compiled in) | Models embedded as gzip'd bytes; no C dependency; `#![forbid(unsafe_code)]` |
| [`zensally-zentract`](https://github.com/imazen/zensally/tree/main/crates/zensally-zentract) | [zentract] plugin (loaded at runtime) | Skips compiling tract; loads ONNX through `libzentract_abi` instead |

`zensally-tract` uses published runtime dependencies (tract-onnx 0.23.10 and
zenflate 0.4.1). The backend itself is consumed from this repository until
released. `zensally-zentract` still uses a pinned git dependency for its plugin API.

### Detectors (`zensally-tract` feature flags)

| Detector | Feature | Task |
|----------|---------|------|
| [`UltraFaceDetector`](https://github.com/imazen/zensally/blob/main/crates/zensally-tract/src/ultraface.rs) | `ultraface` *(default)* | Faces — UltraFace RFB-320, ~1 MB model, the recommended general-purpose detector |
| [`MicroSalNet`](https://github.com/imazen/zensally/blob/main/crates/zensally-tract/src/microsalnet.rs) | `microsalnet` | Saliency — compact MobileNetV3-style encoder/decoder |
| [`ContentAnalyzer`](https://github.com/imazen/zensally/blob/main/crates/zensally-tract/src/analyzer.rs) | `analyzer` | Faces + saliency in one pass (UltraFace + MicroSalNet) |
| `BlazeFaceDetector` | `blazeface320` | Faces — BlazeFace-320 (heavier RetinaFace-style) |
| `MediaPipeBlazeFaceDetector` | `mediapipe` | Faces — MediaPipe BlazeFace |
| `YuNetDetector` | `yunet` | Faces — YuNet (anchor-free) |
| `U2NetpDetector` | `u2netp` | Saliency — U²-Netp |
| `SelfieSeg` | `selfie_seg` | Person segmentation matte |

`zensally-zentract` exposes the same `UltraFaceDetector` / `MicroSalNet` / `ContentAnalyzer` surface through the `ultraface`, `microsalnet`, and `analyzer` features.

Both `MicroSalNet::new()` implementations load the original v1 model. The
committed v3ds w24/w28 models are evaluated candidates, not selectable through
the current detector API. See the [saliency and runtime assessment](https://github.com/imazen/zensally/blob/main/docs/SALIENCY_RUNTIMES.md)
for the recorded quality results, model sizes and dependency costs.

<!-- crates.io:skip-start -->
## Evaluation

The `zensally-tract` crate ships evaluation and benchmark examples. They depend on test corpora and are not part of the published package; run them from a clone:

```sh
git clone https://github.com/imazen/zensally && cd zensally

# WIDER FACE recall validation (downloads the dataset; also a CI job on main):
bash scripts/download_wider_face.sh
cargo run --release -p zensally-tract --example wider_validate

# Crop / saliency evaluation harnesses:
cargo run --release -p zensally-tract --features analyzer  --example eval_crop
cargo run --release -p zensally-tract --features microsalnet --example eval_microsalnet
```

No benchmark numbers are quoted here — measure on your own hardware and corpus. New comparative benchmarks should follow the [zen benchmarking conventions](https://github.com/imazen/zenbench) (no `-C target-cpu=native`, reproducible from the committed command).
<!-- crates.io:skip-end -->

## License

Dual-licensed, your choice of either:

- **[AGPL-3.0-only](https://github.com/imazen/zensally/blob/main/LICENSE-AGPL3)** — for open-source use, or
- **[Imazen Commercial License](https://github.com/imazen/zensally/blob/main/LICENSE-COMMERCIAL)** — for use in closed-source or proprietary products.

SPDX: `AGPL-3.0-only OR LicenseRef-Imazen-Commercial`. The embedded model weights originate from third-party projects and carry their own upstream terms; review them before redistribution.

[tract]: https://github.com/sonos/tract
[`FaceRect`]: https://docs.rs/zensally/latest/zensally/struct.FaceRect.html
[`SaliencyMap`]: https://docs.rs/zensally/latest/zensally/struct.SaliencyMap.html
[`AnalysisOutput`]: https://docs.rs/zensally/latest/zensally/struct.AnalysisOutput.html
[`ImageRef`]: https://docs.rs/zensally/latest/zensally/struct.ImageRef.html
[`PixelFormat`]: https://docs.rs/zensally/latest/zensally/enum.PixelFormat.html
[`FaceDetector`]: https://docs.rs/zensally/latest/zensally/trait.FaceDetector.html
[`SaliencyDetector`]: https://docs.rs/zensally/latest/zensally/trait.SaliencyDetector.html
[`DetectionSummary`]: https://docs.rs/zensally/latest/zensally/struct.DetectionSummary.html
[`SmartCropResult`]: https://docs.rs/zensally/latest/zensally/struct.SmartCropResult.html
[`WhitespaceCropResult`]: https://docs.rs/zensally/latest/zensally/struct.WhitespaceCropResult.html
[`FocusRegion`]: https://docs.rs/zensally/latest/zensally/struct.FocusRegion.html
[`CropRect`]: https://docs.rs/zensally/latest/zensally/struct.CropRect.html

## Image tech I maintain

| | |
|:--|:--|
| **Codecs** ¹ | [zenjpeg] · [zenpng] · [zenwebp] · [zengif] · [zenavif] · [zenjxl] · [zenjxl-decoder] · [jxl-encoder] · [zenbitmaps] · [heic] · [zentiff] · [zenpdf] · [zensvg] · [zenjp2] · [zenraw] · [ultrahdr] |
| Codec internals | [zenrav1e] · [rav1d-safe] · [zenravif] · [zenavif-parse] · [zenavif-serialize] |
| Compression | [zenflate] · [zenzop] · [zenzstd] |
| Processing | [zenresize] · [zenquant] · [zenblend] · [zenfilters] · **zensally** · [zentone] |
| Pixels & color | [zenpixels] · [zenpixels-convert] · [linear-srgb] · [garb] · [zenyuv] |
| Pipeline & framework | [zenpipe] · [zencodec] · [zencodecs] · [zenlayout] · [zennode] · [zenwasm] · [zentract] |
| Metrics | [zensim] · [fast-ssim2] · [butteraugli] · [zenmetrics] · [resamplescope-rs] |
| Pickers & ML | [zenanalyze] · [zenpredict] · [zenpicker] · [zenanalyze-api] |
| Test corpora | [codec-corpus] · [imazen-26] |
| Products | [Imageflow] image engine ([.NET][imageflow-dotnet] · [Node][imageflow-node] · [Go][imageflow-go]) · [Imageflow Server] · [ImageResizer] (C#) |

<sub>¹ pure-Rust, `#![forbid(unsafe_code)]` codecs, as of 2026</sub>

### General Rust awesomeness

[zenbench] · [archmage] · [magetypes] · [enough] · [whereat] · [cargo-copter] · [zenutils]

[Open source](https://www.imazen.io/open-source) · [@imazen](https://github.com/imazen) · [@lilith](https://github.com/lilith) · [lib.rs/~lilith](https://lib.rs/~lilith)

[zenjpeg]: https://github.com/imazen/zenjpeg
[zenpng]: https://github.com/imazen/zenpng
[zenwebp]: https://github.com/imazen/zenwebp
[zengif]: https://github.com/imazen/zengif
[zenavif]: https://github.com/imazen/zenavif
[zenjxl]: https://github.com/imazen/zenjxl
[zenjxl-decoder]: https://github.com/imazen/zenjxl-decoder
[jxl-encoder]: https://github.com/imazen/jxl-encoder
[zenbitmaps]: https://github.com/imazen/zenbitmaps
[heic]: https://github.com/imazen/heic
[zentiff]: https://github.com/imazen/zenextras
[zenpdf]: https://github.com/imazen/zenextras
[zensvg]: https://github.com/imazen/zenextras
[zenjp2]: https://github.com/imazen/zenextras
[zenraw]: https://github.com/imazen/zenraw
[ultrahdr]: https://github.com/imazen/ultrahdr
[zenrav1e]: https://github.com/imazen/zenrav1e
[rav1d-safe]: https://github.com/imazen/rav1d-safe
[zenravif]: https://github.com/imazen/cavif-rs
[zenavif-parse]: https://github.com/imazen/zenavif
[zenavif-serialize]: https://github.com/imazen/zenavif
[zenflate]: https://github.com/imazen/zenflate
[zenzop]: https://github.com/imazen/zenzop
[zenzstd]: https://github.com/imazen/zenzstd
[zenresize]: https://github.com/imazen/zenresize
[zenquant]: https://github.com/imazen/zenquant
[zenblend]: https://github.com/imazen/zenblend
[zenfilters]: https://github.com/imazen/zenpipe
[zentone]: https://github.com/imazen/zentone
[zenpixels]: https://github.com/imazen/zenpixels
[zenpixels-convert]: https://github.com/imazen/zenpixels
[linear-srgb]: https://github.com/imazen/linear-srgb
[garb]: https://github.com/imazen/garb
[zenyuv]: https://github.com/imazen/zenjpeg
[zenpipe]: https://github.com/imazen/zenpipe
[zencodec]: https://github.com/imazen/zencodec
[zencodecs]: https://github.com/imazen/zenpipe
[zenlayout]: https://github.com/imazen/zenpipe
[zennode]: https://github.com/imazen/zennode
[zenwasm]: https://github.com/imazen/zenwasm
[zentract]: https://github.com/imazen/zentract
[zensim]: https://github.com/imazen/zensim
[fast-ssim2]: https://github.com/imazen/fast-ssim2
[butteraugli]: https://github.com/imazen/butteraugli
[zenmetrics]: https://github.com/imazen/zenmetrics
[resamplescope-rs]: https://github.com/imazen/resamplescope-rs
[zenanalyze]: https://github.com/imazen/zenanalyze
[zenpredict]: https://github.com/imazen/zenanalyze
[zenpicker]: https://github.com/imazen/zenanalyze
[zenanalyze-api]: https://github.com/imazen/zenanalyze
[codec-corpus]: https://github.com/imazen/codec-corpus
[imazen-26]: https://github.com/imazen/imazen-26
[zenbench]: https://github.com/imazen/zenbench
[archmage]: https://github.com/imazen/archmage
[magetypes]: https://github.com/imazen/archmage
[enough]: https://github.com/imazen/enough
[whereat]: https://github.com/lilith/whereat
[cargo-copter]: https://github.com/imazen/cargo-copter
[zenutils]: https://github.com/imazen/zenutils
[Imageflow]: https://github.com/imazen/imageflow
[Imageflow Server]: https://github.com/imazen/imageflow-dotnet-server
[ImageResizer]: https://github.com/imazen/resizer
[imageflow-dotnet]: https://github.com/imazen/imageflow-dotnet
[imageflow-node]: https://github.com/imazen/imageflow-node
[imageflow-go]: https://github.com/imazen/imageflow-go
