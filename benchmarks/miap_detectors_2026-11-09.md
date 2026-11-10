# EfficientDet-Lite annotated people comparison — 2026-11-09

Lite2 improves annotated person recall on the broader development set, but still
misses most small people. No detector is promoted and no face guard is certified.
The five checkpoints, confidence0.3 and maximum50 results are fixed before
scoring; source masks are frozen independently of predictions. See the
[protocol](../docs/experiments/MIAP_DETECTOR_SCREEN.md) and
[immutable evidence](miap_content_screen_2026-11-09.pointer.md).

All five models run on all384 owned RGB previews: 1,920 calls, zero inference
failures. The primary subset has91 independently screened development photos,
491 non-group person targets and31 additional group records. Greedy descending
confidence matching requires IoU≥0.5 against the best unmatched individual box.
Predicted coordinates are not clipped. Groups are reported separately, not
treated as individuals. Four behavioral matching regressions pass.

| Checkpoint | Weight bytes | Matched /491 | Recall | Small matched /304 |
|---|---:|---:|---:|---:|
| MediaPipe int8 Lite0 | 4,602,795 | 181 | 36.86% | 60 |
| TensorFlow Lite0 | 4,563,519 | 190 | 38.70% | 67 |
| TensorFlow Lite1 | 6,077,535 | 198 | 40.33% | 74 |
| TensorFlow Lite2 | 7,557,887 | 227 | 46.23% | 90 |
| MediaPipe int8 Lite2 | 7,515,971 | 216 | 43.99% | 89 |

Small means normalized box area below0.02; medium below0.2; otherwise large.
TensorFlow Lite2 matches102/140 medium and35/47 large targets. Its overall gain
over current MediaPipe Lite0 is9.37 percentage points, with paired95% interval
6.33–12.22 points. Against TensorFlow Lite0 its gain is7.54 points, interval
4.49–10.86. Lite1 versus TensorFlow Lite0 is1.63 points, interval−1.67–5.68.
Intervals use5,000 paired source-component bootstrap draws, seed31. Each selected
source is a distinct component. Full per-source matches and all fit/dev, size,
occlusion and truncation strata remain retained.

This is recall against available annotations, which can be incomplete. It is not
official AP, a precision measurement, complete-person recall or face recall.
Lite2 still misses264/491 targets, including214/304 small targets. Neither those
numbers nor detector confidence authorize crops through unobserved faces.

Earlier [one-core i265 timings](efficientdet_family_2026-11-08.md) are separate:
MediaPipe Lite0 13.33898 ms, TensorFlow Lite1 22.01357 ms and TensorFlow Lite2
34.13932 ms. They include these model weights but are Python/TFLite reference
runs, not measured Rust dependency or binary overhead. Earlier sudo-perf evidence
attributes more slots to core work than memory stalls for both Lite1 and Lite2;
the original report retains counters and limits. No fresh latency claim follows
from this quality evaluation.

Reproduction: `just composition-prepare-miap-detectors`,
`just composition-evaluate-efficientdet`, independent content-mask freeze, then
`just composition-score-miap-detectors`. Producers are9a12ffa0 andfb006037 with
the unchanged runtime probe. Inference takes56 seconds under1 GiB/two threads;
run-heavy reports peak RSS **0.27 GiB**, minimum available RAM24,335 MiB and peak
load2.05. This is full evaluation cost, not per-image latency. The scoring run
reports peak RSS0.07 GiB. No training, TEST inference or paid model calls occur.

Lite2 is a useful stronger offline localization reference for the next controlled
composition study. A runtime switch needs measured crop benefit and a dependency
budget; larger weights and better person recall alone do not establish either.
