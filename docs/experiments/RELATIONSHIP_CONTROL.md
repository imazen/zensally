# Source-head relationship control

Register before caching/fitting. Reuse the fixed72 real sources/48FIT24DEV,
their frozen gaze sidecars, donor assignments, hidden candidate values, weak
references, eligibility and C002 scorer. No API calls, new teacher inference,
TEST, role changes or nested positive-range admission.

Add16 bounded geometry features from at most16 local head proposals: head count,
center/whole-head retention, pair retention/separation, proximity-weighted pair
retention/separation, directed gaze-target-near-other-head retention/separation,
retained-group center/spans/edge margin, mean head area and pair distance.
Proximity weights use exp(-source distance squared/0.25 squared); directed
attention uses sender in-frame probability times exp(-target-to-other-center
distance squared/0.15 squared). These fixed descriptive scales are hypotheses,
not learned social relationships or proof that people interact. No medical,
demographic or identity attributes enter these features.

Keep the original16 pose features; compare baseline, pose, pose+relationships,
pose+source-permuted-relationships at equal scalar capacity. Donor relationships
are recomputed on recipient boxes; pose remains real in both relationship arms.
Use original seeds31/137/911,88 epochs/batch12, AdamW lr0.001 decay0.0001 clip5,
FP32 two CPU threads/4GiB cap and FIT-only column normalization. Freeze the
original base weights exactly; select essential DEV family head cuts, then
reference matching, then earliest epoch including zero. Unknown crops remain
unknown. All initial/selected/terminal states and histories are retained.

The12 fits test suitability only. Group/body retention and artistic choice can
conflict, and missing proposals are not absence of people. No promoted model,
face-safety claim, teacher-free transfer or serving cost follows from caching
relationships. Preserve the original failed transfer and attention panels.
