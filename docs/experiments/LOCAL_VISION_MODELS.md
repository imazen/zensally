# Local geometry models for crop evaluation

The subsequent [runtime study](../../benchmarks/runtime_models_2026-11-08.md)
executes all 11 downloaded models plus seven existing Rust adapters on 76
sources. Its timings, face-center diagnostics and native import failures are
separate from this original metadata-only inventory.

Inspected on 2026-11-08 using official Google model downloads, generated TFLite
schema metadata and retained SHA256 hashes. All 11 selected artifacts downloaded;
no inference, recall, latency, runtime RAM or composition-quality measurement was
performed. Bytes below are complete downloaded files, not installed SDK sizes.
The [evidence pointer](../../benchmarks/cached_evaluation_2026-11-08.pointer.md)
retains HTML, links, response headers, graph operators and task components.

| Model | File bytes | Actual graph input | Candidate role |
|---|---:|---|---|
| BlazeFace short-range f16 | 229,746 | 128x128x3 f32 | Near/selfie faces |
| BlazeFace full-range f16 | 1,083,786 | 192x192x3 f32 | Wider-range face comparison |
| BlazeFace full-range sparse f16 | 676,746 | 192x192x3 f32 | Smaller face artifact; speed unmeasured |
| EfficientDet-Lite0 int8 | 4,602,795 | 320x320x3 u8 | Objects/pets/product-category boxes |
| SSD MobileNetV2 f16 | 5,918,749 | 256x256x3 f32 | Alternative object baseline |
| Selfie Segmenter square f16 | 249,537 | 256x256x3 f32 | Person retention/boundary features |
| Selfie Segmenter landscape f16 | 250,177 | 144x256x3 f32 | Landscape person-mask comparison |
| Selfie Multiclass f32 | 16,371,837 | 256x256x3 f32 | Six-class person detail |
| Hair Segmenter f32 | 781,618 | 512x512x4 f32 | Optional hair boundary cue |
| Pose Landmarker Lite bundle | 5,777,746 | 224-square detector + 256-square pose | Soft joint-cut features |
| Face Landmarker bundle | 3,758,596 | 128-square detector + 256-square landmarks | Optional gaze/headroom cues |

Sources: [face detector](https://developers.google.com/edge/mediapipe/solutions/vision/face_detector),
[object detector](https://developers.google.com/edge/mediapipe/solutions/vision/object_detector),
[segmenters](https://developers.google.com/edge/mediapipe/solutions/vision/image_segmenter),
[pose](https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker),
[face landmarks](https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker).
The face documentation table says 128-square for full-range models; the fetched
full/sparse graphs are both 192-square. Graph metadata determines adapter shapes.

Existing tract adapters already embed MediaPipe-derived front BlazeFace
(248,139 gzip bytes, 433,256 raw ONNX) and Selfie Segmentation
(265,116 gzip bytes, 459,047 raw ONNX), alongside YuNet and UltraFace. Start with
these and compare full-range faces on distant, tiny, profile and group subjects.
No additional Rust dependency or model binary was committed by this inventory.

Operator implications: ordinary face graphs use conv/depthwise, activation,
add, reshape/pad/pool and resize/dequantize. Sparse full-range additionally uses
DENSIFY and DEPTH_TO_SPACE; smaller storage does not prove faster execution.
EfficientDet needs quantized conv/depthwise, resize, logistic and external box
decode/NMS: its raw heads are [1,19206,90] and [1,19206,4]. Existing ONNX adapters
cannot directly load these TFLite artifacts. Any conversion/lowering requires
verified preprocessing, operators and reference numerical/crop checks.

Selfie models include custom Convolution2DTransposeBias. Hair additionally uses
custom MaxPoolingWithArgmax2D/MaxUnpooling2D and has four input channels; its
preprocessing contract still needs verification. Pose and face task bundles
contain multiple graphs and auxiliary logic, not a single interchangeable file.
No optimizer is needed for frozen local inference; fitting the crop student
remains a separate training task.

Object models cover COCO categories, not arbitrary ecommerce SKUs or brand text.
Person masks do not segment arbitrary products. These signals help geometric
retention and cuts; they do not replace frozen artistic preferences.
[Detectron2](https://github.com/facebookresearch/detectron2) is a detection,
segmentation and keypoint framework rather than one mini model. A specific
instance-segmentation checkpoint could be an offline annotation baseline; it
was not downloaded or benchmarked here. Prefer compact geometry models for
shipping, with large annotation models kept outside the deployment dependency
graph. Independent face audits remain necessary for the never-cut-face goal.

## Scanpaths and composition features: research, 2026-11-08

The owner identifies pose landmarks plus faces and better general composition as
useful. The following are researched candidates, not executed models or measured
improvements to the crop student. No weights, new annotations, diagnostic arrays,
training change or runtime dependency are introduced by this note.

Viewer scanpaths and photographed-subject gaze are separate targets:

- [DeepGaze III](https://github.com/matthias-k/DeepGaze) predicts the next fixation
  distribution given an image and recent fixation history. Repeated sampling
  supplies possible scanpaths. Its
  [paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC9055565/) separates initial
  experimental fixation from subsequent free viewing and explicitly models
  center bias. It is a useful established baseline, not a runtime-size claim.
- [DeepGaze3.5-VL](https://github.com/Susmit-A/DeepGaze3.5-VL), linked by the
  original DeepGaze repository as its newer scanpath model, emits fixation
  coordinate sequences. The released implementation uses an InternVL3.5-8B
  base with free-viewing and visual-search adapters. The
  [2026 paper](https://arxiv.org/abs/2607.02083) also discusses task conditioning
  and fixation durations. Treat this as an offline teacher candidate; its
  deployment cost and crop-domain quality are unmeasured here.
- [Gazeformer](https://github.com/cvlab-stonybrook/Gazeformer) predicts
  goal-directed search scanpaths, with textual target conditioning. Its
  [paper](https://arxiv.org/abs/2303.15274) is relevant to product-search intent;
  free viewing and searching for a named object are different evaluation tasks.
- [Gaze-LLE](https://github.com/fkryan/gazelle) takes scene pixels and person head
  boxes to estimate where that person looks, with an in/out-of-frame variant.
  Its DINOv2 scene encoding is shared across people. This can suggest head-to-
  object or head-to-person relationships; it predicts the depicted person's
  attention, not the viewer's scanpath. A small decoder does not remove its
  pretrained visual-backbone cost. No latency is measured here.

Proposed composition features: visible joints and body-part boundary crossings;
headroom; room in the inferred looking/moving direction; hand-to-product and
person-to-person relationships; important text/product retention; group extent;
negative space; horizons and strong lines. Pose/head orientation is not calibrated
ocular gaze. Body cuts are soft, framing-dependent features (a deliberate torso
portrait is legal); supplied protected face bounds remain hard constraints.

A useful attention sequence representation is normalized fixation coordinates,
optional durations/confidence, and several alternative paths. Do not average
multimodal paths into one artificial trajectory. Early-fixation retention,
attention stolen by edge clutter, and retention of looked-at objects are proposed
crop features, not validated aesthetic metrics. Attention attraction alone does
not establish composition quality or protect faces/branding.

For a bounded future ablation, run an offline teacher on actual rendered crop
alternatives as well as source images, under fixed viewing/task assumptions.
Cropping changes the frame, context and center bias; clipping a source scanpath
is not equivalent to predicting attention on a crop. Cache compact coordinates,
keypoints, relations and scalar scores; do not save diagnostic maps by default.
Keep multiple acceptable crops and preference supervision as the composition
objective. Compare the current face/ranker baseline, added pose features, and
added gaze/scanpath summaries on fixed family-separated roles. Retain both
quality metrics and measured complete runtime; no deployment gain is assumed.
A shared-encoder student with auxiliary pose/attention heads now has a separate
[registered implementation and measured negative suitability results](../../benchmarks/composition_features_2026-11-08.md).
The corrected joint runs do not improve the frozen crop initializer; no
auxiliary teacher backbone is added to shipping inference.

## SAM follow-up access check, 2026-11-08

The current [official SAM 3 image checkpoint](https://huggingface.co/facebook/sam3/tree/3c879f39826c281e95690f02c7821c4de09afae7)
is manually gated. Public model metadata reports a 3,439,938,512-byte safetensors
file; a configuration download attempt with the available library
session returns `GatedRepoError`, HTTP 401. No weights are downloaded and no
sharing/terms agreement is submitted. This is an access blocker for that local
teacher probe, not an inference/quality result. The retained attempt is
`/mnt/v/output/zensally/composition-sam-screen/2026-11-08/access-check.json` on i265.

[SAM 3.1's release notes](https://github.com/facebookresearch/sam3/blob/main/RELEASE_SAM3p1.md)
describe Object Multiplex and multi-object video-tracking changes. Those results
are not used as static-photo crop-quality or runtime measurements here. Existing
local teachers/search work continues independently of this access check.
