# Registered person-relationship transfer scout — 2026-11-09

Question: does broader human person-region supervision improve the existing
aspect-conditioned photographic crop ranker? Better body localization alone
does not qualify a composition model. No new paid teacher calls or TEST access.

Inputs are the frozen384-source acquisition and independent source mask from
[content screening](../../benchmarks/miap_content_screen_2026-11-09.md):275 fit/
91 development sources for person auxiliary tasks, one per source component.
All four older ranker quality modules and its evaluation script remain byte
unchanged. Existing119-source local-rank cache has94 fit/25 development sources
and21 development families; its TEST sibling stays closed. Initializer is the
previously selected C002 photographic model; epoch zero is always a candidate.
This is continued development, not an independent final test.

Use the same owned aspect-preserving256-square letterbox preprocessing as the
original rank cache, with explicit source content bounds. Cache uint8 inputs;
convert source batches on device. Target ten normalized statistics: union xyxy
of all annotated person/group regions, mean individual-box center xy, twice
the population standard deviation of those centers xy, log1p individual-record
count/log1025 and log1p group-record count/log1025. Group-only sources mask the
four individual center/spread dimensions. Counts are annotation records, not
true crowd cardinality. Demographic attributes and detector outputs are unused.
No heatmap objective, face target or artistic label is invented.

Four arms share seeds31/137/911,44 epochs,16-source batches and FP32 with TF32
off: rank-only, joint-image, joint-permuted and joint-geometry. The auxiliary
head uses4×4 average pools from both existing16-channel maps plus content bounds
and log source aspect, a64-unit ReLU hidden layer and ten bounded outputs. Union
coordinates are ordered. The head is discarded for serving, adding no runtime
operator or weight requirement. Existing rank architecture remains unchanged.

Each epoch first trains the original six rank batches using its fixed query
phases, then all fitting auxiliary batches in each joint arm. The44 epochs cover
each fitting source/query twice. Existing family balancing, pair logistic plus
0.25 masked score-MSE and candidate eligibility remain. Aux loss is mean masked
L1 over ten statistics plus0.2 generalized union-IoU loss, weighted0.5. AdamW
uses encoder lr0.0001, both heads0.001, decay0.01 and gradient norm cap5.
BatchNorm running statistics stay frozen in every arm; affine parameters can
train. Task-specific forward paths and set_to_none gradients must preserve
inactive head weights and warmed optimizer state.

Permuted auxiliary images use a fixed cross-source cyclic permutation within
role and five aspect bins (<0.5,<0.9,<1.1,<2,otherwise), seed31. Targets, content
bounds and source aspects stay attached to the recipient. Report unmoved cases
and aspect-bin limitations. Geometry auxiliary input zeros both feature maps
and never runs the image encoder. All arms retain true images for crop ranking.

Select each checkpoint by minimum family-mean essential-ratio development
teacher regret (1:3 through3:1), then higher pair agreement, then earlier epoch.
Record extreme1:6/6:1 separately. Compare seed means by paired21-family bootstrap
over source/query-aggregated values,5,000 draws, seed31. Useful transfer requires
joint-image regret to beat rank-only and permuted controls with95% intervals
below zero; auxiliary overlap alone cannot qualify. Do not loosen any original
quality, face or3-map-pixel arithmetic gate. A positive scout still needs export
numerical qualification and new independent artistic evidence before promotion.

Bounds:4 GiB host,0.35 CUDA allocator fraction, two host threads,16-source batch,
1 GiB cache allowance and2 GiB retained fit artifacts. Save every selected/final
state, optimizer/RNG/visit records, predictions and per-epoch observations.
Checkpoints and progress must name actual PID and completed cells. Preserve
failed attempts. Same-checkpoint native replay must retain decisions before
any runtime comparison. No claim of full GPU utilization follows from this scout.
