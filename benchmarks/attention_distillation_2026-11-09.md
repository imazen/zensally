# Linear teacher-free attention distillation

Six bounded linear probes complete, three seeds each for real and fixed
source-permuted attention targets. All six fail the unchanged original old
quality gate with their distilled residual enabled. No model is promoted.
The real probes do not improve the new weak crop metrics versus the permuted
control. This rules out promoting this simplified linear transfer; it does
not rule out attention supervision with a different representation or objective.

| Selected student | Essential nominated-head cut rate | Reference match |
|---|---:|---:|
| Real seed31/137/911 |0.364774 each|0.127103 each|
| Permuted seed31/137 |0.361294 each|0.131305 each|
| Permuted seed911 |0.361294|0.125422|

These are family means on the same17 DEV families/139 essential requests.
The full residual teacher previously matches0.141529; the original frozen
baseline matches0.111137. DEV selection uses feature MSE, not crop metrics.
Old essential teacher regrets for real seeds31/137/911 are0.321464/0.325675/
0.317997; permuted seeds0.322108/0.322995/0.319064. Every value exceeds the
unchanged C002 baseline0.24700142443180084. Disabling the entire residual keeps
the original base, but does not establish quality for an enabled student.

The probe predicts32 normalized attention features from96 cached hidden values.
It composes with a fixed residual teacher scalar and fuses algebraically into
one96-input scalar layer. Measured CPU fusion differs by at most
1.1920928955078125e-6 score units, with zero eligible crop-choice changes and
zero edge movement for each student's1,584 requests across72 photos/22 queries.
The comparison passes the existing three-grid numerical gate. This is not
Rust export parity, a complete runtime benchmark or an artistic-quality gate.
No teacher inference is needed for this fused representation; that observation
does not justify deploying the failed weights.

All88 epochs plus zero are retained with initial/selected/terminal model,
optimizer/RNG, normalization and DEV score/choice states. FIT-only hidden
normalization and teacher FIT feature normalization prevent DEV statistics
from entering training. Source-balanced eligible-candidate feature MSE,
AdamW lr0.001/decay0.0001/clip5, batch12. Source roles/families, TEST closure,
candidate banks, nested excellent admission and quality thresholds stay fixed.
Training uses CPU FP32/two threads with4GiB cap; no API calls or extra teachers.

Training takes30s/run-heavy peak RSS1.10GiB, min available22293MiB. The numerical
and old quality checks have their full run-heavy logs retained separately.
These are bounded research jobs, not saturated training or inference timings.
Reproduce with `just composition-distill-attention-linear`,
`just composition-audit-attention-fusion`, and
`just composition-evaluate-attention-fallback ... --fused-students`.
Evidence roots under `/mnt/v/output/zensally/` are `attention-linear-distillation`,
`attention-linear-fusion` and `attention-linear-oldgate`, each dated2026-11-09.
The registered [protocol](../docs/experiments/ATTENTION_DISTILLATION.md) remains
separate from these negative results.
