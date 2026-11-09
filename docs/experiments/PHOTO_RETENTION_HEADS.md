# Candidate-local photographic retention controls

Register before feature caching or fitting. Reuse the frozen photographic rank
encoder `c002a3a796868bcd9c4c291207e3ab5d636760aa3f92249d9bf9fd58d9cde0df`
and original119-source fit/development rank cache
`2583944f230150f675dfc9dcc94ee4e1cbb8d63e6b15aeb33eab02ee97bf6031`.
Use the reviewed generic-people substitution known-box cache
`101110157703882de1020b24df012d33f7a9adfd7b02cfb5c2f5588366c7918d`
from the frozen photo-subject packet. No TEST or supplementary-development source
supplies gradients or selects heads in this study.

Admit only the72 fitting sources' returned boxes as **noisy known-box** targets.
The15 development sources provide diagnostics/selection. Other32 sources remain
unknown, not negative. This admission does not assert subject completeness,
artistic quality or face safety. Preserve the143 reference boxes and documented
misses/prop failures; do not replace the shipping face guard with these targets.

Predict full containment and minimum returned-box area retention for each crop
candidate, instead of predicting one global subject rectangle. Reuse the existing
RegionCropRanker's22 pooled regions at both feature levels, source/candidate
geometry and region areas. Retention is a property of the candidate, so its input
mode is fixed to zero. No teacher box/text input is required at inference.
Retain the original scorer separately for composition ordering. It is frozen.

Cache image and shuffled-image candidate descriptors, keeping the **actual**
source crop geometry/content. Reuse the old within-role/exact-letterbox donor
mapping; do not simply permute already-pooled candidate vectors. Geometry-only
zeros only the visual descriptors. All three controls see identical geometric
inputs and fixed labels. Qualify cached descriptors against the original scorer
and require unchanged3-map-pixel selected-crop agreement on existing development
predictions before fitting. No numerical tolerance is relaxed.

For each of image/geometry-only/permuted-image, fit seeds31/137/911,110 epochs
including epoch-zero evaluation, one cyclic ratio per source/epoch, source-ID
hash phase modulo11. Every source/ratio is visited ten times. Source batches16,
equal family contribution, fixed partial-batch denominator16. AdamW head-only
lr0.001, decay0.01, gradient cap5, FP32/TF32-off. The frozen encoder never updates
BN/optimizer state. Initialize the first96-unit layer from the frozen composition
scorer; initialize a new two-output layer identically across matched controls.

The per-query loss averages positive/negative containment BCE equally when both
exist, uses ordinary BCE for a single-class query, plus0.5 mean squared error of
sigmoid minimum-retention fraction. Invalid padded candidates are masked. Both
output rows participate in every update; no inactive AdamW head is advanced.
Select minimum family-mean development containment Brier score, then earlier
epoch. Report all seeds, fraction MAE and class balance, not just selected loss.

For research decisions only, retain eligible candidates with containment
probability at least0.5, then rank them by the original frozen student. Empty
sets abstain for containment. Report offered crops, known-box violations,
recovered legal opportunities and unnecessary abstentions with denominators.
No threshold tuning on development; report essential ratios separately from6:1/
1:6. Retain exact integer infeasibility/coarse-bank/fallback statuses from the
reference audit. A classifier's confident prediction is not a face certificate.

Bound descriptor cache to1GiB, fit state to2GiB, host to4GiB/two threads and
CUDA35%. Save full selected/latest head optimizer/RNG states and scalar histories,
record live progress and actual resources. This first cached-head suitability
test makes no GPU-saturation or deployment-speed claim. New inference/export
optimization depends on a controlled quality gain. No paid API or diagnostic maps.
