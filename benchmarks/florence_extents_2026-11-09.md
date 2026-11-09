# Phrase-target extent fitting — 2026-11-09

The small frozen encoder fails its controls. Larger frozen FocalNet descriptors
give a positive weak localization signal on eight reused development families;
this is not artistic crop quality, complete product coverage or face safety.

All56 existing synthetic sources retain their48-fit/27-family and eight-
development/eight-family assignments and original primary-object queries.
Source IDs, native RGB SHA256 and roles join exactly across the old admission,
runtime and query manifests. All56 local Florence phrase-grounding calls finish
with EOS, no parser/truncation failure and101 valid native boxes. Their unions
are new noisy primary-extent targets, admitted only for fitting sources; a
missing/invalid response would be unknown. No old composition grade is admitted.

Both descriptor variants reuse frozen caches bit for bit. Targets, geometry,
roles and MSE training settings match. Each variant finishes nine fits: image,
geometry-only and within-role/aspect shuffled-image arms, seeds31/137/911,
200 epochs, batch16, FP32/TF32-off, AdamW0.001/weight decay0.01 and gradient
clip5. Selected epochs minimize development edge MSE, including epoch zero.
Weighting is per source, not per family. No end-to-end gradients occur.

| Frozen descriptor | Image mean IoU | Geometry mean IoU | Shuffled-image mean IoU | Mean-box IoU |
|---|---:|---:|---:|---:|
| Small epoch11 encoder |0.229205|0.317706|0.279763|0.304060|
| FocalNet |0.370635|0.317706|0.255537|0.304060|

Image/geometry edge MSE is0.037389/0.037000 for the small encoder and
0.025985/0.037000 for FocalNet. Every geometry-only history and selected
prediction matches exactly between studies. Across eight development families,
averaging seeds before a5000-draw paired family bootstrap, FocalNet image minus
geometry IoU is+0.052929,95% percentile interval[+0.026614,+0.082938]; versus
shuffled image it is+0.115098,[+0.006378,+0.248240]. These are reused selection
data, not independent confirmation or a deployment qualification.

The source-bound review retains every teacher entity and all six selected
development image heads rather than choosing a seed visually. Its first fixed
view, the coffee-machine scene, shows a plausible teacher machine box but a
FocalNet student box shifted left across the worker and missing much of the
machine. The aggregate weak gain therefore does not establish whole-product
preservation. The head has no text-query input; authored queries are supervision
only. Faces and crop composition need separate constraints and evaluation.

run-heavy on i265: teacher56 stage10s, peak RSS2.02GiB, minimum available
22438MiB, peak load4.78; small nine-fit stage8s/1.44GiB/21002MiB/3.64;
FocalNet nine-fit stage9s/1.44GiB/18855MiB/3.94. Each fitting stage's CUDA peak
allocated is18,085,888 bytes, reserved23,068,672 bytes. These small cached-head
fits do not saturate the GPU. Browser verification decodes all56 native images,
checks101 entities and48 saved predictions over336 controls, with zero failures
or exceptions; run-heavy2s/0.20GiB/23512MiB/3.95. No speedup claim is made.

See [pre-registered protocol](../docs/experiments/FLORENCE_EXTENT_FIT.md),
[region-teacher limitations](florence_regions_2026-11-09.md) and
[older distinct-target study](primary_extents_2026-11-08.md). Earlier results
are not overwritten or compared as if their different weak targets were gold.
No new TEST inference, paid API calls, diagnostic maps or model promotion.
