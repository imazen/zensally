# Extent augmentation control — 2026-11-09

Content-preserving reflection and mild color variation do not establish useful
image-dependent extent generalization. All six registered encoder-training runs
finish, retaining the same targets, roles, seeds, loss and optimizer settings.
Every epoch-zero prediction matches its unaugmented baseline exactly.

| Input | Selected development IoU | Objective |
|---|---:|---:|
| Augmented image |0.330665|2.191371|
| Augmented shuffled image |0.364638|2.171090|
| Unaugmented image |0.301438|2.270559|
| Unaugmented shuffled image |0.369064|2.142471|
| Geometry-only baseline |0.340012|2.242086|

Augmented image minus unaugmented image IoU is+0.029227, paired eight-family
bootstrap95% percentile interval[−0.028174,+0.092577]. Versus augmented shuffled
images it is−0.033973,[−0.104812,+0.028619]; versus geometry−0.009347,
[−0.084989,+0.059906]. All are reused development selection comparisons.
The intervention lowers final fit IoU from 0.910905 to0.861459, but augmented
final development IoU is0.236917. Overfitting remains. No model is promoted.

The flip operates within each actual content rectangle; it preserves asymmetric
letterbox padding and transforms its target in source coordinates. Brightness
and contrast use independent per-image[0.8,1.2] draws, clamp content and retain
zero padding. Development images are unaugmented. Independent asymmetric-padding
and strided-pixel tests pass, as do the four existing extent/gradient tests.
All six initial numerical gates pass at maximum 0.000015259 map pixels; all
selected development predictions replay exactly. Twelve selected/final native
checkpoint audits produce valid boxes on every fit/development source.

run-heavy on i265: six fits34 s, peak RSS 1.92 GiB, minimum available 22373 MiB,
peak load 2.93. GNU time maximum RSS 2,011,200 KiB. CUDA peak allocated492,968,448
bytes/reserved616,562,688. Audit3 s/1.52 GiB/22466 MiB/2.68, CUDA191,976,448/
268,435,456 bytes. Both use4 GiB/two threads; resources are whole-stage costs,
not deployment latency or a GPU saturation claim. No paid API calls, source
role changes, new images, TEST inference, diagnostic maps or face certification.

See [protocol](../docs/experiments/EXTENT_AUGMENTATION.md) and
[matched unaugmented study](florence_extent_finetune_2026-11-09.md).
