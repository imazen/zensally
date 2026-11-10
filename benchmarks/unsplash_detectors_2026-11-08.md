# EfficientDet on144 Unsplash photos — 2026-11-08

Larger tiers recover some people and miss others. None qualifies as complete
face coverage or establishes a default replacement. All five pinned checkpoints
finish the same144-image CPU screen (720 inference calls), with identical
owned RGB per image, confidence0.3 and maximum50 results.

| Checkpoint | Person-positive sources /144 | Person boxes | Added positive sources versus existing Lite0 | Lost positive sources |
|---|---:|---:|---:|---:|
| MediaPipe Lite0 |35|75|0|0|
| TensorFlow Lite0 |37|84|3|1|
| TensorFlow Lite1 |38|80|5|2|
| TensorFlow Lite2 |39|88|6|2|
| MediaPipe Lite2 int8 v1 |38|86|6|3|

These are detector counts, not human precision/recall. Counts do not tell us
whether a newly positive image contains a real person or whether all people
in it are covered. The working previews are strict-sRGB/Lanczos3 with1280 long
edge; the earlier101-source study used768 previews. Its paired timing is
separate and is not extrapolated into a new end-to-end latency claim.

Four visual witnesses are selected by the registered posthoc rule. At
`73Fjvy6Z9w0` (aircraft above an outdoor restaurant), existing Lite0 boxes one
person and TensorFlow Lite2 six. The extra boxes visibly align with additional
patrons, while several visible people still lack boxes. At`V-45oWTe9k4`
(crowded crossing), Lite0 returns14, Lite1 15 and TensorFlow Lite2 10; the
Lite0/Lite2 overlay comparison shows additional visible people left uncovered
by Lite2. These are assistant observations, not exhaustive face annotations.
The other two review witnesses are`KStSiM1UvPw` and`O3heYcghDfA`, chosen for
baseline-negative/other-positive disagreement. No observations become labels.

All144 gallery images decode at their expected dimensions and40 checkpoint/
class-toggle cases pass without browser exceptions. Four screenshots occupy
3,538,252 bytes, below the registered8MiB budget. run-heavy inference takes26s,
peakRSS0.29GiB, min-available21,586MiB, peak-load13.45. Browser verification
takes4s, peakRSS0.21GiB, min-available24,286MiB, peak-load13.12. These are
evaluation costs, not per-image runtime timing or GPU measurements.

Keep Lite0 as the baseline and investigate a selective larger-model pass on
distant/small people and disagreement cases. Its selection policy, latency and
crop benefit remain unmeasured; a clean Lite0 result cannot certify absence.
No model is promoted, no face guarantee weakened, no source roles assigned,
no gradients admitted and no paid model API called.

See [protocol](../docs/experiments/UNSPLASH_DETECTORS.md),
[owned preprocessing](unsplash_preparation_2026-11-08.md) and
[original weights/paired CPU costs](efficientdet_family_2026-11-08.md).
