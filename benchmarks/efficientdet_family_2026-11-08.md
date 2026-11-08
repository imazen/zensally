# EfficientDet-Lite1/Lite2 — 2026-11-08

Lite2 is a useful candidate for selective person detection; neither larger tier
is qualified as the default or as a complete face-protection detector. This
screen measures reference CPU execution and detection counts, not human
precision/recall or crop composition quality.

All five checkpoints process the same 101 owned-photo previews at confidence
0.3, maximum50 results, CPU. All505 calls finish. The existing MediaPipe Lite0
results match the earlier101-source screen exactly. The older TensorFlow
Android sample family and current MediaPipe assets have distinct weights;
differences cannot be attributed solely to backbone tier.

| Checkpoint | UINT8 square input | Weight bytes | Mean ms/image | Person-positive sources /101 | Person boxes |
|---|---:|---:|---:|---:|---:|
| MediaPipe Lite0 |320|4,602,795|13.33898|12|19|
| TensorFlow Lite0 |320|4,563,519|12.12009|10|19|
| TensorFlow Lite1 |384|6,077,535|22.01357|14|21|
| TensorFlow Lite2 |448|7,557,887|34.13932|18|34|
| MediaPipe Lite2 int8 v1 |448|7,515,971|36.62198|15|29|

Timing is zenbench's four exploratory paired blocks,24 process invocations
including A/A, on one i265 performance core (CPU2). Each process warms one
eight-image sweep, then measures20 sweeps of fixed preloaded owned RGB views.
Task preprocessing, inference and returned postprocessing are included;
model/input loading, image construction, hashing and IO are excluded. These
are means, not p95 or complete crop-pipeline latency. A/A changes−0.24526%.
Observed heavy-process counts span0–2 and cleanliness is not certified. The
packet fingerprint is
`efafde5fb410d7ab01d5f3e9c007924ff4df0290b83b24fb4544763d69526f48`.

Skiers11048 produce4/4/3/3/2 person boxes in the table's order. Ceiling/visitors
11214 produce5/6/5/8/5. Six earlier false-face/pose witness scenes produce no
person boxes under any checkpoint. More detections can include duplicates or
false positives, and fewer can include misses: counts and cross-model IoU are
not a human quality gate. No boxes are admitted as training labels or certified
complete face bounds. The corrected viewer exposes every source/checkpoint
and all object classes; browser checks decode all101 images and exercise40
source/model/class combinations with no exceptions.

## CPU and memory evidence

`sudo perf stat` measures800 calls per checkpoint with CPU2 affinity. Private
PID/mount namespaces avoid an unrelated stale NFS descriptor that blocked the
first profiler's process discovery. Only the task-owned blocked perf process
was manually terminated; the target had not started. Its rc137 is a manual
termination, not an OOM. Original partial results and interruption evidence
remain retained. All five isolated profile targets exit0.

| Checkpoint | Memory-bound slots % | Core-bound slots % | GNU-time target max RSS KiB |
|---|---:|---:|---:|
| MediaPipe Lite0 |10.0|38.7|255,600|
| TensorFlow Lite0 |11.2|34.6|261,228|
| TensorFlow Lite1 |9.2|36.4|310,140|
| TensorFlow Lite2 |8.6|37.4|301,884|
| MediaPipe Lite2 |8.2|39.7|314,200|

These are topdown pipeline-slot fractions for the whole target, including
startup; they do not directly measure DRAM latency or allocate wall time.
Core-bound slots exceed memory-bound slots for every tested checkpoint. Model
bytes are separate from Python/MediaPipe process RSS and are not Rust runtime
dependency size. No embedded Rust deployment of these checkpoints is measured.

run-heavy evidence on i265: full inference15s, peakRSS0.27GiB,
min-available24,697MiB, peak-load14.04; paired timing108s, peakRSS0.32GiB,
min-available24,248MiB, peak-load13.86; completed namespace profiles99s,
peakRSS0.30GiB, min-available23,993MiB, peak-load14.15.

Use Lite0 as the unchanged baseline while reviewing whether Lite2's additional
detections cover important missed subjects. A selective second pass is a
candidate, not a measured end-to-end cascade. No API spending, source-role
change, shipping detector switch or quality-model promotion occurs.

See the [registered protocol](../docs/experiments/EFFICIENTDET_LITE_FAMILY.md).
Official assets are linked by the
[TensorFlow sample download script](https://github.com/tensorflow/examples/blob/master/lite/examples/object_detection/android/app/download_models.gradle)
and the [MediaPipe object detector documentation](https://developers.google.com/edge/mediapipe/solutions/vision/object_detector).
The captured sample script has Git blob identity
`f97894603671f8611750157541a9b8eb97d630fd`.
