<!-- GENERATED FROM README.md by zenutils gen-readme-crates.sh — DO NOT EDIT. -->

# zensally

zensally is face detection and neural saliency for content-aware image cropping — find the faces and the parts of an image people actually look at, then crop to any aspect ratio without cutting off the subject. The `zensally` crate is the shared, pure-Rust core: the result types, the detector traits, model preprocessing, non-maximum suppression, output decoding, and a bridge into [zenlayout]'s smart-crop solver. Ready-to-run detectors with embedded ONNX models live in sibling backend crates that plug into those traits. Pure Rust, `#![forbid(unsafe_code)]`.

The default detector path runs entirely in Rust via [tract] — no ONNX Runtime, no C dependency, models small enough to embed in the binary.

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

Both backends currently pull in git-only dependencies, so they're consumed via `git = "…"` rather than from crates.io.

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
