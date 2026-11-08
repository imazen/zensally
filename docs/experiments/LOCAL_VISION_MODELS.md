# Local geometry models for crop evaluation

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
