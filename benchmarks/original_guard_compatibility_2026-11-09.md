# Original-panel guard compatibility

The detector union is unsuitable as an unconditional artistic-crop constraint.
False face detections on a watch, honey packaging and mushrooms visibly remove
useful subject content. All guarded policies also lose full-panel quality
coverage. Keep the student and guards experimental; no model is promoted.

Follow the [registered protocol](../docs/experiments/OLD_GUARD_COMPATIBILITY.md)
on only the original25 DEV sources/21 families. Preserve94 FIT sources, source
hashes, roles, original candidates, targets and closed TEST. The same local
native UltraFace/YuNet and four MediaPipe assets evaluate these owned working
previews. Four-tile YuNet reuses the existing extraction, mapping and NMS rules.
No paid calls, fitting, new artistic labels or importance planes.

All250 local evaluations complete:50 full-view native,100 tile native and100
MediaPipe calls. Face geometry has175 source/model rows. Only two poses are
returned, both passing the unchanged head estimate. The shared tile mapping
reproduces all720 proposals in the earlier frozen real72 packet exactly.

## Quality and coverage

Original essential scope:448 valid requests. Full ORT gather encoder/query
execution reproduces every one of the550 raw selected-student decisions from
the preserved full CUDA gate, including two requests without ranking targets.
The raw essential regret remains0.244554802775383, below the unchanged original
C002 baseline0.24700142443180084. Lower regret is better.

| Policy | Offered /448 | Contain | Conditional regret | Raw regret on exactly those offered requests |
|---|---:|---:|---:|---:|
| Raw student |448|0|0.244554802775383|0.244554802775383|
| Faces: reject partial intersections |433|15|0.34987735748291016|0.2528929114341736|
| Faces: also retain one region |167|281|0.5624096393585205|0.2388658970594406|
| Faces + pose: reject partial intersections |433|15|0.3552370071411133|0.2528929114341736|
| Faces + pose: also retain one region |184|264|0.5418511033058167|0.23621101677417755|

All regions use the prior25-percent padding plus two pixels per edge. The
partial-only policy is unguarded on sources without detections; the retain-one
policy contains those sources. Pose can create new crops in the latter case
by supplying a head on a source without face detections. Whole heads may be
excluded; partial intersections with supplied boxes are rejected.

Conditional scores preserve the original per-source query means and inverse
family weights on the explicitly available source/query population. Strict
face and face+pose policies contribute only11/12 sources and9/10 families,
respectively; their scores cannot stand in for the25-source baseline. The
partial-only policies retain at least one request from every original source.
Coverage is reported separately with normalized floating-point weight mass;
the original float32 quality weights are unchanged.

Containment has no original ranking target. Its regret is unknown, never zero.
Every incomplete policy therefore has a null full-panel regret and cannot prove
the unchanged full-panel nonregression gate. Conditional regressions are also
large, even compared with raw on precisely the same offered requests.
This is reused weak supervision, not a new human artistic-quality score.

## Actual-pixel failures

Select the three largest essential conditional-regret increases, with distinct
source families.15 panes have12 distinct PNG hashes; every distinct image is
inspected directly. Owned cropping/resizing/PNG encoding and source copies
pass independent exact RGB readback. Source, raw, partial face guard, partial
face+pose guard and containment are displayed. No labels are admitted.

- [Watch](https://unsplash.com/photos/hbTKIbuMmBI),1:3 Maximal:
  short-range BlazeFace detects a face on the watch display. Its padded box
  rejects the subject crop; filtering selects nearly empty white space with
  only a narrow product fragment at the right. Weak regret rises by
  3.949056625366211. The raw crop also trims product sides; it is not certified
  as a catalogue crop, but the filtered output is visibly unsuitable.
- [Honey products](https://unsplash.com/photos/KnPzm5JgDPE),2:1 Minimal:
  tiled YuNet detects faces on packaging. The raw crop contains product/label
  content, while filtering moves to lower props and excludes the central
  product's label. Weak regret rises by2.870537042617798. Raw framing is not
  asserted to be excellent; the direction of lost subject content is visible.
- [Mushrooms](https://unsplash.com/photos/4iDTPgoZQ94),9:16 Minimal:
  full/tiled YuNet detects a face on the nonhuman scene. Filtering moves the
  main cluster to the right edge and clips its caps rather than retaining the
  centered cluster. Weak regret rises by1.7181072235107422.

The face+pose partial policy has identical pixels to the face partial policy
in these three witnesses. These are clear local false positives on the reviewed
sources, not a corpus-wide detector precision measurement. Do not add source-ID
exceptions or tune away the failures on this DEV set.

## Verification and resources

The first quality attempt stops before output because original banks contain
zero rectangles in ineligible tensor slots. The caller now evaluates geometry
only where existing eligibility says a candidate exists. The geometry validator,
bank, eligibility and thresholds are unchanged. Retain the failed log and
pre-correction source revision. A later coverage-only correction normalizes
floating-point weight mass; earlier successful quality data is retained under
`quality-before-coverage-normalization.bak`. No quality score or choice changes.

i265, two threads, run-heavy3GiB cap except rendering2GiB:

| Tool | Reported duration | Peak RSS | Minimum available RAM |
|---|---:|---:|---:|
| Prepare25 originals/100 tiles |1s|0.61GiB|23100MiB|
| Native detectors |3s|0.16GiB|22659MiB|
| MediaPipe tasks |3s|0.23GiB|22659MiB|
| Quality evaluation, before coverage-only correction |2s|0.68GiB|22575MiB|
| Witness rendering |1s|0.05GiB|23041MiB|

Independent native and MediaPipe jobs run concurrently within their caps.
These are evaluation-tool resource observations, not paired runtime latency.
The raw neural numerical replay is exact. Existing tile, pose and subject-policy
checks pass; all five original quality files remain byte-identical to
9f12692417d774653ed676d236e0f89eaafa7385.

Output:`/mnt/v/output/zensally/original-guard-compatibility/2026-11-09/`.
Keep detector outputs, exact tile pixels, input/source/model identities, all
valid/invalid query rows, explicit missing scores, source links and visible
failures. The retained packet budget is192MiB, including126056448 tile bytes.

The next useful test is independent subject/context corroboration for a proposed
face, with missed-face uncertainty kept explicit. Detector union or extra padding
alone cannot establish face protection while preserving artistic composition.
