# EfficientDet-Lite family runtime/coverage screen

Owner requested Lite1 and Lite2 on2026-11-08 after the existing Lite0 results.
Use the same frozen101 owned-photo RGB previews and reference task thresholds
(confidence0.3,max50 results,CPU). Reuse existing Lite0 as a deployment baseline;
also acquire the official TensorFlow Android sample's Lite0/1/2 metadata assets
so the older Lite1 checkpoint is accompanied by its own published family.
Include the current MediaPipe Lite2 int8 version1 asset separately, preserving
its checkpoint identity rather than treating it as identical to the older one.

Pin the source download-models Gradle blob, asset URLs, byte sizes and SHA256.
Do not assume newer MediaPipe and older TensorFlow artifacts are identical or
attribute checkpoint/training/quantization differences solely to backbone size.
Record actual tensor/quantization metadata when available. Bound each download
to64MiB and all model artifacts to256MiB; never overwrite cached weights.

Before full inference, probe both original people witnesses11048/11214. Then
evaluate every101 fixed image with owned RGB, preserving all object classes,
scores and boxes. Compare person-positive images, counts, detector agreement
and those two witnesses. Counts/IoU between detectors are not human recall or
precision. No selected detector boxes become complete face annotations or
training targets automatically. The144 newly acquired Unsplash sources stay
a separate screen until owned preprocessing is qualified.

Measure CPU latency with the existing zenbench admission/pair protocol on fixed
preloaded RGB, preserving A/A and background-load evidence. Use sudo perf for
cycles/instructions and available memory-stall counters; record GNU-time max
RSS and run-heavy peak RSS. Compare actual model bytes separately from process
memory. Bound evaluation/profile jobs to4GiB host memory and two CPU threads.
Do not infer a missing latency, quality score or memory figure from another
model. No API spending or shipping model/policy change belongs to this study.
