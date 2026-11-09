# Crop-to-proposed-region risk student

Register before fitting. Hard body constraints lose44/53 accepted DEV endpoints;
measured face/pose fallback still cuts26–31/127. Test a learned soft risk signal
from explicit crop-to-region relations instead of unconditional vetoes or global
source-box compression. This remains a small weak-label suitability screen on
reused DEV, not independent artistic-quality certification.

Reuse frozen classifier packet07c10b48 and its full verified inputs through
person_risk_inputs.load:72 new sources48 FIT/24 DEV,25 original DEV, fixed
common endpoint4b801ad4/additive pair05f41e08, original query masks and candidate
banks. Reuse all existing face/pose and Lite2 outputs. No new inference/API calls,
labels, TEST or image processing. Region inputs never use nominated heads,
selected subjects or crop windows; candidate-to-region arithmetic is an input.

Four groups: union of seven face models, measured pose-derived head bounds,
padded upper-person view, padded full-person view. Each supplies eight values:
log1p region count; fractions contained/cut/excluded; mean intersected fraction
of region area; maximum intersected fraction among cuts; maximum source-relative
area among cut and contained regions. Geometry is evaluated only on eligible
native candidate boxes; nondetections are measured empty groups with zero values.
This gives32 features without capping region count or discarding overlaps.

Four equal hidden-width32 ReLU cut heads: geometry8+relations32; hidden96+
relations32; hidden96+relations32+existing attention32; matched hidden96+
permuted relations32+existing permuted attention32. Region donors use a
one-step ID-sorted rotation within FIT/DEV; original DEV rotates separately.
Donor boxes map outward into recipient native pixels. Existing auxiliary
permutation remains unchanged; the two donor maps need not match. Record both
feature identities; this control destroys alignment, not all correlations.

Keep the old32-epoch training protocol, seeds31/137/911, batch8/all22 queries,
known class/source/family balancing, FIT-eligible mean/std0.001, AdamW0.001,
decay0.0001, clip5, CPU FP32/two threads. Unknown candidates supply zero gradient.
DEV family-balanced known-class logistic loss selects epoch, earliest tie;
no crop metric selects checkpoints. Retain every epoch, optimizer and RNG.

For soft routing reuse strengths0,.25,.5,1,2,4,8 and existing FIT-only choice:
minimize family cut rate subject to no FIT reference-match or head-retention
loss versus zero; prefer match, retention, then smaller strength. Apply only
on sources with cached Lite2 people. Full offer coverage is required by soft
routing. Report all grid points, all12 selected policies,139 accepted essential
DEV endpoints,396 all-head-known requests, contain/unknown/secondary separately,
and original full448-request regret with unchanged baseline0.24700142443180084.
No original or new DEV crop metrics select strength. Zero must replay original
regret0.2406301498413086. Hard0.5 routing is a secondary abstention diagnostic.

Numerical checks use CPU reference, compact fused normalization and CUDA FP32
on cached feature inputs, comparing actual selected pairs and original three-
grid edge gate, including offer changes as failures. No encoder/teacher runtime
or feature GPU parity claim. If quality remains unsafe, no runtime export or
serving benchmark. Bounded actual-pixel witnesses are required before promotion.
Run-heavy4GiB/two threads; training CPU, CUDA qualification20-percent allowance.
64MiB artifact budget; deterministic feature arrays are hashed and recomputable
from source proposals and frozen candidates rather than duplicated on disk.

Actual-pixel witness control is hidden-regions-attention/seed137, fixed before
routing outcomes. Show a common-offer repair and a remaining/new cut; selection
of diagnostic images does not select the model or policy.
