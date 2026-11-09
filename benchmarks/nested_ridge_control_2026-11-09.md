# Low-capacity nested-pair control

Reducing capacity and selecting regularization entirely within FIT does not
produce a suitable crop model. Measured attention makes one more accepted DEV
match than geometry, while53 nominated-head cuts remain. Paired edge-error
bootstrap intervals cross zero. No runtime export or model promotion follows.

The [registration](../docs/experiments/NESTED_RIDGE_CONTROL.md) keeps the same
99 finite weak FIT pairs on29 families and36 DEV pairs on14 families. Each
family gets equal total training weight. Three deterministic models predict
five bounded geometry values; all keep the same maximum-area outer and strict
inner. Geometry has five polynomial columns. Image and measured-attention
models each add eight PCA components and their aspect interactions, for29
columns. Image features come from frozen C002 source grids; attention uses the
same pooled32 pose/gaze/scanpath summary as the neural scout.

Every FIT family is held out once. Descriptor normalization, weighted PCA and
regression normalization exclude that family. All nine lambdas1e-4 through1e4
are retained per fold; DEV never selects lambda. One of29 families has only
secondary targets, so primary hyperparameter selection uses28 families/97
essential requests, while all29 folds and secondary scores remain recorded.
The first preflight incorrectly assumed every labeled family had an essential
request and failed before creating an output directory. The corrected reader
retains every target and fold; the original failure log remains preserved.

## Essential DEV results

Pair error covers36 requests/14 families; endpoint counts cover139 accepted
requests/17 families and reuse the original subject/face/proximity audits.

| Model | FIT-selected lambda | Paired edge error | Accepted matches | Head cuts | Heads retained |
|---|---:|---:|---:|---:|---:|
|Geometry|10000|22.873441860438835|14|55|95|
|Image PCA8|10|22.695568284542787|13|57|93|
|Measured attention PCA8|10|22.5913332358838|15|53|94|

Paired family bootstrap,2000 repeats/seed31 on the same reused DEV panel:

| Compared with geometry | Mean paired edge-error difference |95% interval |
|---|---:|---|
|Image|−0.17787357589604508|[−0.4102458169858901,+0.07011905974690219]|
|Measured attention|−0.282108624555034|[−0.6695002594912856,+0.12776252812674954]|

For measured attention, the family-balanced head-cut difference is
−0.016339869281045753, interval[−0.04341736694677871,+0.006069094304388421].
Reference matching difference is+0.0065359477124183, interval
[0,+0.0196078431372549]. These descriptive differences do not establish an
independent artistic-quality or face-safety improvement. Unknown frames are not
labelled bad merely because they miss a finite reference.

All models always emit432 essential DEV endpoints, including85 explicit
teacher-contain requests. They have no contain head. Secondary accepted DEV
coverage remains only three endpoints; every model matches zero and cuts one.
Sparse weak supervision and this geometry/feature representation remain
limitations; this study does not establish the best possible capacity or teacher.

## Numerical and resource evidence

All135 retained FIT/DEV native target pairs round-trip through bounded fractional
parameters exactly. Weighted dual ridge fits satisfy independently checked
primal normal equations on the unit fixture. Final FP64 reference versus CPU
FP32 and CUDA FP32 yields9504 rectangle comparisons, zero changed rectangles
and zero violations of the unchanged three-grid numerical gate. These tests use
cached descriptors, not the full image encoder or teacher pipeline.

Run-heavy on i265, two CPU threads under4GiB hard cap:1s, peak RSS1.15GiB,
minimum available23131MiB, peak load1.29. Family analysis1s/0.60GiB, minimum
23121MiB under2GiB/one thread. These are research-run resources, not deployed
latency or memory. The measured-attention model still needs attention inputs;
the image control does not require those teachers but remains unsuitable.

No API spending, fresh teacher inference, new labels, gradients through the
frozen encoder, TEST inference or changes to existing quality expectations.
All original five evaluator files remain unchanged. Local on i265:
`/mnt/v/output/zensally/nested-ridge-control/2026-11-09/`.
NAS: `/mnt/tower/output/zensally/nested-ridge-control/2026-11-09/`.
The16MiB packet retains all fold/lambda scores, models, predictions, numerical
results, family uncertainty, input identities and the initial failure log.
Frozen17-member packet,2836263 bytes before manifest, with independent complete
NAS membership/hash verification. Manifest SHA256
`318f88d1d3259433ead7d424f082a5ec63cb9908d1578e242968449286f910d5`.
