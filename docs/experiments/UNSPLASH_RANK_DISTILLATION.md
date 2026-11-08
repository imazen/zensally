# Local relative crop-rank distillation on acquired photos

Register before new composition-teacher scores or student fitting. The frozen
provider registry SHA256 is
`6f8d318cc870248dcc2400a235fcbaa5f7f0edf47d575d965bf5c72b5684d2f6`.
It has94 fit sources/81 components,25 development/21 components and25 TEST/22
components. Existing canonical VALIDATE/TEST images remain closed. New TEST
composition inference is excluded from this cache and all fitting/selection.
Detector-positive people coverage is24 fit/three development/eight TEST sources;
the small development people subset limits claims. Do not reroll roles.

This is weak local relative ranking, not new human acceptability grades or a
catalogue/hero/detail rubric. Purpose remains explicitly unspecified for these
labels. No scalar score becomes a complete subject/face annotation. The
shipping face-review guard and earlier numerical/crop gates remain unchanged.

Use the pinned FocalNet human-ranking ONNX model and already qualified cached
analysis/query split. Render119 fit/development images from owned1280 previews
to256 letterboxes with the same pinned owned resizer, recording input hashes.
Student input is RGB0–1 with zero padding; teacher input applies ImageNet
normalization only inside content. Preserve original native dimensions for
the exact-ratio integer candidate boxes and normalized full-source coordinates.

Generate the existing constrained_bank scales1/.9/.8/.7/.65 and five origins
per free axis, without inventing complete head constraints. Keep all eleven
existing ratios and both Minimal/Maximal area masks. Minimal uses the largest
native exact-ratio extent; Maximal uses smaller scales, with the unchanged
largest-extent fallback if none exist. No importance-retention filter is added:
the target is the teacher's relative rank, not a claim to replicate its full
solver policy. Retain raw128-slot scores, valid masks and original pixel boxes.
No diagnostic importance planes are saved.

Before full caching, choose three fit and three development witnesses by native
aspect nearest0.5/1/2, with distinct families. Require cached/full ONNX score bits
to match on all eleven ratios. All score arrays must be finite, boxes valid and
source/pixel/role identities exact. A source/query with fewer than two eligible
candidates or eligible-score standard deviation below1e-4 has no ranking target;
retain its unknown status and count rather than inventing preferences.

Normalize teacher scores within each eligible source/query to mean zero/unit
standard deviation. Retain up to32 deterministic higher/lower candidate pairs,
with teacher-zscore separation at least0.25; insufficient pairs stay masked.
Only fit targets supply gradients. Development targets select models. The
new TEST teacher and targets stay outside this cache.

## Matched student controls

Reuse RegionCropRanker's owned w24 encoder and epoch11 checkpoint, replacing
only its final two-output linear layer with one scalar rank output initialized
from the previous rank row. No acceptability output or inactive-head update.
Keep this as a separate research model, with no shipping API/export promotion.

Run frozen-image, trainable-image, geometry-only and permuted-image controls for
seeds31/137/911. Geometry-only uses zero feature maps and never advances encoder
weights/state. Permute images within role and exact256-letterbox content shape,
with seed31; keep target geometry fixed and report actual cross-family moves.

FP32/TF32-off, source batches16, AdamW lr0.001/weight decay0.01, gradient cap5.
Run220 epochs, one cyclic ratio/mode query per fit source per epoch, using a
fixed source-ID hash phase; each condition is visited ten times. Family weights
give each fit family equal total contribution. The source-query objective is
strong-pair logistic loss plus0.25 masked score MSE. Epoch zero is included.

Select minimum family-mean development teacher regret (best eligible teacher
score minus the teacher score of the student's choice, in teacher standard
deviations), then higher strong-pair accuracy, then earlier epoch. Lower training
loss alone does not establish better crops. Report exact top-choice agreement,
regret, strong-pair accuracy and rectangle motion, all seeds and essential versus
secondary ratios. Reused development families make intervals exploratory.

Retain scalar histories, predictions and every improved model-only state;
full optimizer/RNG checkpoints every25 epochs and at completion. Bound state
artifacts to4GiB, input/target cache to256MiB, host to4GiB/two threads and CUDA
to35% device memory. Measure actual resources and profile a real fitting path
before accelerating it; no saturation claim from small fixtures. No paid API.
