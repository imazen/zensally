# Matched encoder-training suitability check

Register before new gradients. The frozen Florence-target comparisons finish
27 fits: small epoch 11 and photographic-student descriptors fail IoU controls;
larger frozen FocalNet descriptors have a reused-development localization
signal. These models differ in pretraining and architecture, so parameter count
alone is not an identified cause. Test whether explicit object supervision can
improve the small encoder before optimizing a new training path.

Keep the same 56 synthetic sources, authored primary queries, 48-fit/27-family
and eight-development/eight-family roles. Copy the frozen Florence cache
SHA256 `f2d6987d8d2290ae51ad43b5bf46d4ea99cf3c3c156e0c384b24ca2c5ba0944f`
and its original role/aspect permutation into a fresh namespace. Never mutate
the frozen teacher/fitting packets. Original auxiliary image cache SHA256 is
`463e4884be5ada2128d08b9a755cf836d23e116059e6716390d64c0712b768a3`;
small encoder initializer SHA256 is
`910219ef1af5eecf4530b675fecddba04798b62e323def2b781de8fda5354f71`.
No new teacher calls, changed targets, human grades or TEST inference.

First run the owned frozen-head fitter with its existing L1/GIoU objective,
image/geometry/permuted arms and seeds31/137/911: nine fits, 200 epochs,
batch 16, FP32/TF32-off, AdamW 0.001/weight decay 0.01, clip 5. Select minimum weak
development objective including epoch zero; retain the mean-box control.
Then run the existing end-to-end image-extent fitter: image and within-role/
aspect-permuted arms, the same three seeds and 200 epochs, same loss/head
learning rate, encoder learning rate 0.0001. BatchNorm running statistics stay
frozen while affine parameters train. This is 15 fits, not a comparison between
different loss objectives. Original frozen MSE fits remain a separate result.

Only fit sources supply gradients; selection uses the eight reused development
families. Preserve optimizer/RNG checkpoints and the existing initial cached/
native numerical gate of at most three source-normalized128-grid pixels per
edge. The ordered softmax head must produce finite, nonempty, in-bounds boxes.
Report all seed means and paired family bootstrap intervals against the matched
frozen image head, geometry control and permuted end-to-end control. Do not
select a deployment model from these targets or claim face/product completeness.

Use run-heavy with 4 GiB host cap/two threads and at most35% CUDA allocator memory.
Monitor real PID and per-run epoch progress; record actual host RSS, allocator
allocated/reserved peaks and artifact size. Keep artifacts below4 GiB. Existing
protected data and all earlier expectations remain unchanged. Training-path
acceleration follows only if this objective shows useful measured suitability.
