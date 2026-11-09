# Direct two-window scout — 2026-11-09

The first direct-head configuration fails suitability. It does not reliably beat
geometry-only window prediction and its raw proposals contain more nominated-head cuts. Keep it
as a controlled competitor; no production behavior, old-quality gate or source
role changes. Deployment and old-rank nonregression remain unmeasured.

The [registration](../docs/experiments/DIRECT_CROP_SCOUT.md) uses existing Sol
nominations on72 real families,48 FIT/24 DEV. The frozen C002 encoder produces
2048 source-grid features; the entire cache is751,735bytes. Its525 accepted
window queries and471 explicit contain recommendations give996 known offer
targets. Normalization uses FIT only;61 source descriptors move within fixed
role/content-shape permutation groups. No API calls or synthetic gradients.

The65,895-parameter head predicts two bounded origins/scales and an offer logit.
Minimal has maximum native integer-aspect extent. Maximal predicts a positive
bounded integer extent. Two nominated alternatives use permutation-invariant
assignment; one supervises both slots. No left/right spread or diversity is
forced. Unknown compositions remain unknown. Only editorial purpose is trained.

All nine image/geometry/permuted controls across seeds31/137/911 complete108
epochs with source-balanced coordinate L1 plus0.1 offer BCE, AdamW1e-3,
weight decay1e-4, clipping5, twelve sources/all queries, FP32/TF32 off.
Warmed-optimizer regression tests require inactive coordinate/scale parameters
and their AdamW steps to remain unchanged. DEV selection uses best-of-two
maximum-edge error, then offer Brier, then earliest epoch. It does not assert
the original rank nonregression prerequisite or promote this different model.

Essential accepted-reference metrics cover17 DEV families/139 requests, averaged
over three selected seeds and families. Edge error uses the best predicted/
nominated pair, measured in source-normalized128-grid pixels. Best-of-two is
not interchangeable with a ranker's single selected crop.

| Arm | Best2 edge error | Best2 match | Top1 match | Top1 nominated-head cut | Either cut |
|---|---:|---:|---:|---:|---:|
| Image |15.8475|17.7461%|13.0604%|44.4009%|56.8271%|
| Geometry only |16.4243|15.1286%|13.8214%|43.2088%|46.0815%|
| Permuted images |17.2847|14.7582%|12.8799%|45.0693%|54.4672%|

Image versus geometry: edge error -0.5767, paired95% family-bootstrap
CI[-2.7897,+1.3596]; best2 match +2.6175 percentage points,
CI[-1.7593,+6.2123]. Either-proposal cuts increase10.7456 points,
CI[+2.8429,+19.1474]. Image offer Brier0.336812 is worse than geometry0.225091
on the separate224 known essential offer requests. The finite annotation set
is incomplete and unreviewed; these failures do not reject all direct architectures.

All nine selected native replays are exact. Across72 images×22 requests×two
alternatives×nine models, all28,512 native rectangles are identical between CPU
and GPU, within source bounds and exactly at the target integer aspect ratio.
The unchanged three128-grid-pixel edge gate is retained. This establishes
numerical geometry only; it does not establish artistic or face quality.

The first launch stopped at epoch zero before any optimizer step because a
mutable progress file used an exclusive writer. It remains retained. Correcting
progress handling reproduces the initial native outputs and offer logits exactly,
then finishes the unchanged registered fit. Histories, selected and terminal
optimizer/RNG states and original failure evidence are preserved.

Nine-fit job:14s, run-heavy peak RSS1.62GiB under4GiB/two-thread cap. Source-cache
construction:1s,1.19GiB. Numerical/family analysis:1s,1.09GiB. These bounded
observations make no GPU-saturation or deployed-speed claim. Improving the
quality/constraint interaction precedes runtime acceleration of this configuration.
[Verified evidence](direct_crop_scout_2026-11-09.pointer.md).

## Native visual review

A separate eight-source viewer shows the two raw direct alternatives beside
the fixed hybrid ranker and unreviewed Sol proposals. The selected direct image
head is seed911/epoch99, chosen by the registered DEV edge-error rule. Its full
DEV native pixels and binary offer choices reproduce on CPU before rendering.
All704 crop/clipping/no-stretch browser checks pass. Ten screenshots totaling
30,527,470bytes are retained; four are visually inspected:3:1 Maximal,1:3 Minimal,
square Maximal and1:6 Maximal.

Both raw singer3:1 alternatives cut through the face; its offer head recommends
contain there. Some square/tall classroom and petition alternatives also cut
faces, while narrow crops can lose subjects. The singer's usable1:3 framing is
also rejected by the offer head, illustrating unnecessary containment. These
are visual research observations, not new training labels or a claim that every
unsafe raw coordinate is actually offered to the user. The windmill source has
an invalid Sol annotation and remains outside accepted-reference metrics.
