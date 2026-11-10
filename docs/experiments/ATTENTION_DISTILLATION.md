# Teacher-free attention distillation control

Register six linear feature probes before fitting: real versus fixed
source-permuted attention targets, seeds31/137/911. Reuse the frozen72-photo
48FIT/24DEV hidden cache and matched-seed full residual teachers. Keep C002
base weights frozen. No new API calls, TEST inference or nested-range labels.

Predict32 normalized pose/gaze/path features from96 cached image/ranker hidden
values. FIT-only eligible-candidate population normalization, std floor0.001.
Source-balanced eligible-candidate mean MSE, AdamW lr0.001 decay0.0001 clip5,
88 epochs/batch12/two CPU threads/4GiB hard cap. Select minimum DEV feature MSE
including epoch zero, earliest tie; crop metrics are observations, not selection.
The original teacher's FIT feature normalization is shared by both target arms.

Retain initial/selected/terminal model, optimizer, RNG, normalization, predictions
and all epoch metrics. This linear probe is a suitability control, not a full
relationship network or production student. Its composition with the fixed
teacher scalar can be fused algebraically into one96-input scalar layer.
Require measured numerical and crop-choice comparison before using that fusion;
do not infer identical floating-point arithmetic from the algebra. Disabling
the whole residual still uses the original scorer, but enabled fused students
must separately pass the unchanged old quality gate. Full quality, runtime and
artistic suitability remain unproven until measured.

All six probes subsequently complete. The [measured results](../../benchmarks/attention_distillation_2026-11-09.md)
find that all enabled fused students fail the original old quality gate, with
no crop-quality advantage for real versus permuted targets. CPU fusion changes
no selected crop across the fixed panel. Keep this negative suitability result;
no weight or production model is promoted.
