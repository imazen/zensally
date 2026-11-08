# EfficientDet screen on the acquired Unsplash cohort

Registered2026-11-08 after all144 owned1280-long-edge previews and independent
PNG readbacks complete. Keep every source screening-only, with photographer
groups provisional and no fit/dev/test assignment or new training target.

Evaluate the same five pinned EfficientDet checkpoints as the101-source family
screen, on identical owned RGB per image, CPU, confidence0.3 and maximum50
results. Reuse immutable weights and the same MediaPipe Task runner. Preserve
all boxes/classes/scores and failures. No diagnostic output planes or paid API
calls. Bound the job to4GiB/two threads. This is another cohort/preprocessing
contract (1280 rather than768 previews); do not merge it into the old paired
timing or claim cross-cohort latency equivalence.

Report class/person counts, positive sources and cross-detector disagreements.
No unlabeled-image count is recall or precision. Select four posthoc visual
witnesses deterministically: the largest person-count disagreement, the most
multi-person output, a Lite0-negative/other-positive source when available,
then remaining sources in ID order. The same source may satisfy multiple
criteria; deduplicate before filling. Retain four bounded screenshots of the
first two witnesses under original Lite0/TensorFlow Lite2, total8MiB maximum.
These assistant observations are neither exhaustive annotations nor labels.

The viewer must decode144 actual images and exercise all five checkpoints,
four witnesses and person/all-class toggles. Default101-source viewer checks
remain unchanged unless the caller explicitly requests144. No shipping model,
threshold, face-protection contract or composition model is changed.
