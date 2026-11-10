# Frozen photographic student transfer check

Register before extracting descriptors or fitting. The Florence target study
has a negative small-epoch11 result and a positive larger-FocalNet result on
eight reused development families. Encoder architecture and pretraining both
differ, so this does not identify model size as the cause. Test whether the
already selected, faster photographic student's frozen features are useful.

Pin the pre-TEST-selected trainable-image seed911/epoch64 checkpoint SHA256
`c002a3a796868bcd9c4c291207e3ab5d636760aa3f92249d9bf9fd58d9cde0df`.
Copy the frozen Florence targets/cache SHA256
`f2d6987d8d2290ae51ad43b5bf46d4ea99cf3c3c156e0c384b24ca2c5ba0944f`
and re-extract only its512 visual descriptors from the original56 auxiliary
image tensors. Bind every ID, role, family, RGB and tensor SHA256. Use identical
4x4 average pooling of both16-channel feature maps and FP32/TF32-off evaluation.
Do not change the preselected student, its source code or rank head. No crop
ranking development/TEST reevaluation or new teacher inference occurs.

Keep48 fit/eight development sources, targets/geometry and all nine frozen-head
MSE settings from the Florence study. Geometry-only histories/predictions must
match that study exactly. Use two threads, at most35% GPU allocator memory and
4GiB host bounds. Record source and encoder hashes, all controls and family
bootstrap uncertainty. This remains a reused synthetic development diagnostic,
with no human artistic-quality or complete-face claim and no model promotion.
