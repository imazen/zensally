# Primary-subject extent suitability

Registered 2026-11-08 before full-corpus localizer inference or extent-head fits.
The measured failure is semantic: a face-preserving crop can discard the bag,
garment, espresso machine or pet being featured. Eight approximate agent region
witnesses demonstrate that failure; they are not human ratings or certified face
coverage. Their conservative whole-product profile is distinct from hero/detail
uses. No shipping policy or canonical holdout assignment changes.

## Local supervision

Use pinned Grounding DINO tiny, revision
`a2bb814dd30d776dcf7e30523b00659f4f141c71`, as an offline teacher. A scout with
explicit product/story queries completed eight scenes. The successor freezes one
primary-object query for each of the 56 synthetic sources before inference.
Use FP32/TF32-off, Torch reference deformable kernels, threshold 0.3 and text
threshold 0.25, owned aspect-preserving Lanczos RGB with long axis 1024, and no
processor resize. Retain raw and visible clipped boxes, scores, labels, source
and preprocessing identities. Persist no diagnostic maps.

The target is the union of returned primary-object boxes, with explicit
uncertainty: no detection is missing supervision, not a negative/no-object label.
These text-conditioned weak targets do not establish runtime text understanding,
subject completeness, artistic quality or face safety. The runtime student will
not receive the teacher's query. False positives and source-specific query
choices limit this pilot. The eight scout scenes include development examples;
all resulting development comparisons are selection diagnostics.

Reuse the existing geometry-only source roles: 48 fitting sources in 27 families,
eight development sources in eight families. Shared fictional brands/scenarios
never cross those roles. This phase separately admits only primary-extent weak
targets; it adds no crop acceptability judgments and does not revise the earlier
annotation-only or geometry experiments. No paid model API is used.

## Frozen-encoder head gate

Freeze the same epoch-11 crop encoder and projections. Concatenate 4×4 adaptive
average pools of its stride-8/16 projected features (512 floats), plus four
aspect/content geometry scalars. Cache bounded descriptors and teacher extents.
Neural feature vectors are necessary training inputs, not saved diagnostic maps.

Compare a fit-mean box, an aspect-only MLP, an image MLP, and an image-feature
permutation control. All trained MLPs have the same 516→64→6 parameterization.
Two three-way softmax partitions decode left/width/right and top/height/bottom,
producing a normalized box with a fixed 0.0001 minimum normalized span;
only the width/height partition receives this floor, so margins may reach zero. Zero visual descriptors in the aspect arm; permute
visual descriptors within source role and fixed aspect bins (<0.5, <0.9, <1.1,
<2, otherwise), retaining each source's geometry scalars and labels. Report the
number of actually moved sources and singleton groups.

Seeds 31/137/911, 200 epochs, FP32, AdamW lr 0.001, weight decay 0.01, batch 16,
source-balanced mean squared error on four edges, clipping norm 5. Select on
minimum weak development edge MSE, including epoch zero. Report every seed,
mean IoU/edge error, paired source-family summaries and the fit-mean control.
Save immutable predictions/checkpoints at improvements and every 25 epochs.
Mask missing teacher sources explicitly and report coverage. No canonical
VALIDATE/TEST image participates in caching, fitting or selection.

Only an image-dependent gain over the aspect and permutation controls supports
further testing. Agreement with one localizer is not independent crop quality.
Any downstream extent feature remains soft unless the caller supplies reviewed
protection regions; it cannot set the existing face-coverage flag.

## Registered loss and encoder follow-up

The frozen-encoder MSE study completes all nine fits: image-head mean development
IoU is 0.31281736, versus 0.36907730 for the aspect head and 0.37582982 for the
fit-mean box. It does not establish useful primary-object localization.
Retain those results and test the objective separately: 5× mean edge L1 plus
2× generalized-IoU loss, with the same nine matched frozen-head runs, roles,
initialization, seeds, 200 epochs and optimizer settings. Select on this explicit
new objective and report all earlier edge/IoU metrics too. This is a development
follow-up, not an independent confirmation or a replacement of old results.

Then compare six end-to-end runs (image and within-role/aspect-permuted image,
three seeds) under the same L1/GIoU objective. Reuse the original encoder and
projections, remove its unused old scoring MLP and attach the same extent head.
BatchNorm running statistics remain frozen. Use FP32/TF32-off, batch 16, 200
epochs, AdamW head lr 0.001 / encoder lr 0.0001, weight decay 0.01, clipping 5.
Keep the initial fit-only descriptor normalization fixed. Report initial
native-vs-cached descriptor/head differences; no bit-parity assumption is made
across batch/layout conditions. Decode partition geometry in FP32 even in any
later mixed-precision training trial. Image-extent predictions remain soft;
all existing deployment face-coverage guards stay closed for unreviewed data.

The objective-only and end-to-end follow-ups also fail the image-dependence
controls. A separate frozen-feature check therefore reuses the already pinned
FocalNet encoder. Preserve 512 visual scalars: 4×4 pools of its 16-channel rank
projection, 2×2 pools of the 48-channel fused encoder features, 48 global means,
and 16 zero padding scalars. Use owned RGB with ImageNet normalization on content
and zero normalized padding, as in the retained FocalNet probe. This differs from
its upstream bilinear resize. Keep the same four geometry scalars, 516→64→6
heads, roles, permutations, seeds and 200-epoch L1/GIoU protocol. No new source
or target label is admitted. This is a representation suitability diagnostic;
the larger encoder's capacity/pretraining differs, and the reused development
set cannot provide independent confirmation.
