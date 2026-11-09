# Photographic features for weak product extents — 2026-11-09

The already selected photographic student's frozen features do not beat the
geometry or shuffled-image controls on these noisy synthetic targets. Improved
crop ranking therefore does not establish useful object-bound prediction.

All nine registered MSE fits finish under the unchanged 48-fit/eight-development
family split and cached Florence unions. They reuse the same head, seeds,
normalization, geometry and epoch-zero-inclusive selection settings as the
[earlier feature comparison](florence_extents_2026-11-09.md). Only the frozen
visual descriptors change, extracted from the existing pre-TEST-selected
seed911/epoch64 checkpoint. No weights in that photographic model are changed.

| Input | Development mean IoU | Edge MSE |
|---|---:|---:|
| Photographic image features |0.313506|0.032327|
| Geometry only |0.317706|0.037000|
| Shuffled image features |0.322656|0.036529|
| Fit-mean box |0.304060|0.043919|

The paired eight-family image-minus-geometry IoU difference is−0.004200,
95% family-bootstrap percentile interval[−0.056589,+0.051801]; versus shuffled
images it is−0.009150,[−0.065941,+0.037928]. All geometry-only histories and
selected predictions match the previous study exactly. These are reused weak
development targets, not independent artistic or product-completeness labels.
The larger FocalNet result remains a different architecture/pretraining
comparison; these controls do not identify parameter count alone as the cause.

The first extraction guard hashed CHW float bytes against a manifest hash
defined over its original HWC u8 canvas. It failed before model inference or
artifact creation. The corrected guard reconstructs and hashes the original
u8 canvas and separately requires its unit-float CHW reconstruction to match
exactly. Every individual source/tensor identity now passes; the failed log is
retained. No source image, cached hash or established expectation is changed.

run-heavy on i265: descriptor extraction2s, peak RSS1.14GiB, minimum available
24048MiB, peak load6.37; fitting9s/1.43GiB/22989MiB/6.52. Extraction CUDA peak
allocated140,764,160 bytes/reserved182,452,224; fitting18,085,888/23,068,672.
These are bounded whole-stage costs, not deployment latency or GPU saturation.
No new teacher calls, gradients in the encoder, paid API usage, TEST inference,
diagnostic maps, complete-face claim or promoted model.

See [pre-registered protocol](../docs/experiments/PHOTO_STUDENT_EXTENT_TRANSFER.md).
