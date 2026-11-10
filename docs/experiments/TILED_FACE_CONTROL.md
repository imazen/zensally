# Four-tile native face control

Registered before inference: reuse the fixed72 real sources,48 FIT/24 DEV,
native YuNet binary and cached full-source outputs. Four corner windows cover
ceil(3/5 width) × ceil(3/5 height) each, overlapping on both axes. Copy exact
owned RGB rows without resampling. The retained probe performs its original
640-square input preprocessing.288 new calls, no paid API calls or TEST pixels.

Map output boxes back through each integer source window, then combine with
cached full-source proposals by confidence-ordered IoU0.3 NMS. Compare keeping
all tile outputs against rejecting boxes within two source pixels of an internal
tile edge before NMS. Source-edge detections remain eligible. Preserve all raw
predictions and both controls. These are region proposals, not face guarantees.

Measure nomination-center overlap by role against the same incomplete Sol
head annotations. Apply actual crop constraints to the same fixed hybrid heads;
report crop offers, nominated-head cuts and weak reference matching together.
Higher proposal counts alone do not establish useful coverage. Do not infer
per-image runtime latency from this batched audit; runtime qualification follows
only a suitable quality result. Existing face/quality/numerical gates stay fixed.

Use at most2GiB/two threads per job with run-heavy; retain logs, exact source and
binary identities, mapped boxes and bounded artifact storage. No maps or new
image labels. No checkpoint training or promotion is authorized by this control.
