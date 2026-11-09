# Matched relationship feature students

Register12 fits: linear versus96->32 ReLU->32 probes, real versus fixed donor
relationship targets, seeds31/137/911. Use the frozen72-photo hidden cache,
pose16+relationship16 targets and matched-seed pose-relations residual teachers.
Pose remains real in both target arms; no additional teachers or API calls.

Keep the original scorer/encoder frozen. Initialize probe output to zero, so
initial scores reproduce the original base. FIT-only input normalization and
teacher FIT target normalization; source-balanced eligible-candidate feature MSE.
Reuse88 epochs, batch12, AdamW lr0.001 decay0.0001 clip5 FP32/two CPU threads,
4GiB cap per job. Linear and nonlinear jobs may run concurrently with these
explicit budgets. Select minimum DEV feature MSE including zero/earliest tie;
do not use crop metrics to select these probes. Retain every epoch and
initial/selected/terminal model/optimizer/RNG/predictions/normalization states.

Evaluate the enabled selected students on the unchanged old quality gate.
Nonlinear deployment retains its direct base skip plus learned ReLU residual;
it cannot be fused into one linear layer. Record actual arithmetic comparisons
before any compact export. No serving latency is inferred from layer sizes.
TEST remains closed, original quality gates remain fixed, no nested ranges are
admitted and no model promotion follows without artistic/face validation.
