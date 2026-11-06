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

## Existing depth maps as a saliency cue

If an image already carries a depth/disparity map, the relevant literature is
**RGB-D salient-object detection**. Embedded metadata is an input acquisition
question; these papers generally consume aligned RGB/depth pairs rather than
parse photo containers.

- [Exploiting Global Priors for RGB-D Saliency Detection (CVPR Workshops 2015)](https://www.cv-foundation.org/openaccess/content_cvpr_workshops_2015/W14/html/Ren_Exploiting_Global_Priors_2015_CVPR_paper.html)
  combines contrast with depth, orientation and background priors. This is a
  classical approach to study when avoiding an inference engine.
- [Local Background Enclosure for RGB-D Salient Object Detection (CVPR 2016)](https://www.cv-foundation.org/openaccess/content_cvpr_2016/html/Feng_Local_Background_Enclosure_CVPR_2016_paper.html)
  uses surrounding background geometry to reduce false positives from depth
  contrast in background regions. Depth contrast alone is insufficient.
- [Depth Quality-Inspired Feature Manipulation / DFM-Net (ACM MM 2021)](https://arxiv.org/abs/2107.01779)
  gates depth features according to their quality before RGB/depth fusion.
  It is a learned CNN, so it still requires weights and inference operators.
- [A2dele (CVPR 2020)](https://openaccess.thecvf.com/content_CVPR_2020/html/Piao_A2dele_Adaptive_and_Attentive_Depth_Distiller_for_Efficient_RGB-D_Salient_CVPR_2020_paper.html)
  transfers depth knowledge during training and uses RGB alone at inference.
  It concerns avoiding the deployed depth branch, rather than consuming a
  photo's embedded depth map.

The existing `heic` source at `8d40c94` has
`DecoderConfig::decode_depth`, depth-representation metadata, and
`decode_matte`/`decode_mattes` for portrait and semantic mattes. This is source
inspection, not fresh decoder fixture validation; the inspected portrait
extraction test is ignored. None of these inputs is wired into zensally.

[Google's depth metadata reference](https://developers.google.com/depthmap-metadata/reference)
also specifies embedded JPEG/PNG depth and confidence payloads, linear versus
inverse range, and near/far metadata. The depth image must be aligned with
RGB, including crop/scale/rotation. Do not treat encoded grayscale values as
metric distance or assume every file has usable depth.

For this crate, the first experiment I would choose is an optional classical
depth cue, fused with the existing RGB result and disabled when depth is
missing or unreliable. It can consume already-decoded depth without adding
another neural runtime. A portrait matte is another available subject cue,
with a different meaning from depth or general saliency. Actual crop-quality,
latency and code-size effects remain unmeasured.

## Operations and optimizers required

The proposed small depth cue is an experiment, not a port of either complete
classical system. It needs depth decoding and interpretation, validity and
confidence handling, aligned resampling, local depth comparisons/reductions,
normalization, and fusion with RGB saliency. It does not inherently need
training or a numerical optimizer. Any coefficients need held-out crop
validation before becoming defaults; a regular spatial grid would simplify
implementation but would differ from the papers' superpixel methods.

The full [LBE system, sections 3–4](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Feng_Local_Background_Enclosure_CVPR_2016_paper.pdf)
uses SLIC superpixels, mean patch depth, neighborhood searches, angular
occupancy histograms and largest-gap statistics over depth thresholds,
depth/spatial/background priors, and GrabCut boundary refinement. The latter
adds foreground/background appearance modeling and graph-cut optimization.
The full [global-priors method, section 3.2](https://www.cv-foundation.org/openaccess/content_cvpr_workshops_2015/W14/papers/Ren_Exploiting_Global_Priors_2015_CVPR_paper.pdf)
uses PageRank-style neighborhood propagation and a quadratic MRF restoration
objective in addition to its RGB/depth/surface-orientation features. Those
solvers are distinct from a neural training optimizer.

For a fixed-graph MicroSalNet v3ds w24 runtime, the model definition requires:

| Operation | Required form |
|---|---|
| Dense convolution | 3×3 stem at stride 2; 1×1 channel mixing |
| Depthwise convolution | 3×3, strides 1/2; bottleneck dilations 1/2/4/8 |
| Transposed convolution | 2×2 kernel, stride 2; three decoder stages |
| Activations | ReLU, sigmoid |
| Reduction | Per-channel spatial mean for squeeze/excitation |
| Elementwise | Residual addition, broadcast channel-gate multiplication |
| Layout | Channel concatenation, fixed NCHW tensor shapes |
| Batch normalization | Execute as exported or fold offline with parity checks |

The exporter/plan builder needs constant/shape propagation, removal of
training-only auxiliary heads, weight packing, scratch-buffer lifetime
planning, and carefully verified convolution/normalization/activation fusion.
All can happen offline for this fixed model; a deployed general graph
optimizer is unnecessary. Fixed shapes do not eliminate the need to implement
stride, padding, dilation, channel grouping and boundary behavior correctly.

A runtime port does not retrain weights and needs no gradient engine or
training optimizer. If retraining is requested, the existing
[`train_v3ds.py`](../training/train_v3ds.py) uses AdamW with cosine learning-rate
scheduling, boundary-weighted BCE plus IoU losses, auxiliary supervision and
an optional teacher MSE term. These are training-side requirements, not part
of the deployed operator set.

## Dependency-refresh validation

Linux x86_64: `just verify-update` passed formatting, all-target/all-feature
Clippy, all-feature workspace tests, doctests, and core checks without default
features (including serde/zenlayout). All seven tract detectors were exercised
by their existing tests. One existing doctest remains ignored. Plugin-host
compilation passed; dynamic plugin inference was not exercised. The full
DUTS/crop evaluation and cross-version tensor-bit parity were not measured.

All-feature i686 workspace tests and doctests also passed through cross
0.2.5, with the optional private plugin host excluded as in CI. No test
expectations were changed.
