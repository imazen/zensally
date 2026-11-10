# Matched linear and nonlinear relationship students

All12 registered students complete. Increasing probe capacity does not yield
a convincing real-target advantage over the matched permuted control, and all
enabled students fail the unchanged old quality gate. No model is promoted.

| Probe / target | Essential head cuts | Reference matching |
|---|---:|---:|
| Linear / real |0.377252|0.125422|
| Linear / permuted relationships |0.373967|0.125422|
| Nonlinear / real |0.351242|0.129282|
| Nonlinear / permuted relationships |0.356485|0.129904|

Values average three seeds' family means over the fixed17 DEV families/139
essential requests. The nonlinear real-target advantage in cuts is small and
matching does not exceed its permuted control. These are selected-DEV weak
metrics, not independent significance, human composition ratings or face safety.

Old essential teacher regret is0.24700142443180084 for C002. Enabled linear-real
seeds score0.303711/0.302333/0.301380; nonlinear-real0.331072/0.334165/0.334285.
The corresponding permuted students also all regress. Keep every failure and
the original gate; do not select a checkpoint using a relaxed expectation.
The image encoder and direct base weights remain frozen, but an enabled
residual can still change old choices. An exact disabled fallback does not
prove enabled quality.

Both probes predict16 pose plus16 relationship features using FIT-only input
normalization and matched-seed real teacher target normalization. Pose stays
real in both target arms. The nonlinear probe is96->32 ReLU->32; both output
layers initialize to zero. AdamW lr0.001/decay0.0001/clip5,88 epochs, batch12,
source-balanced eligible-candidate feature MSE. Minimum DEV feature MSE selects
including zero/earliest tie. No crop metric selects these probes. Original
48FIT/24DEV roles, reference/candidate identities and TEST closure are fixed.

Linear algebraic fusion and nonlinear compaction both measure zero eligible
crop-choice changes for every student's1,584 fixed-panel requests. Maximum
absolute score difference is9.5367431640625e-7 for each architecture's panel;
edge movement is zero. Nonlinear compaction folds normalization into its first
affine layer and the teacher scalar into its last affine layer while retaining
the original base skip. It remains nonlinear; it cannot become a single scalar
linear layer. Tests cover zero initial residual, unchanged base after training
and compact arithmetic with nonzero normalization. This is CPU arithmetic
validation, not Rust export parity or a serving-time measurement.

The two fit jobs run concurrently with separate4GiB/two-thread caps. Linear
takes41s/run-heavy peak RSS1.15GiB; nonlinear46s/1.18GiB. Both report minimum
available21236MiB. Arithmetic checks take1s with peaks0.79/0.78GiB under2GiB;
old quality checks take2s/1.55GiB under4GiB. These small cached jobs do not
establish saturated training throughput or product runtime costs.

The next missing control is old FIT rehearsal during attention training,
rather than post-hoc old DEV evaluation alone. A separate verified cache now
retains the original119 sources'22x128x96 hidden values and unchanged rank
labels/masks,94FIT/25DEV. Every captured CUDA score replays exactly through the
original scalar layer. Cache SHA256:
`41777134194b707c39d8362554afd4adb003e073ca1b31be6554390cbfd7a3af`;
size133,304,823 bytes. Preparation takes2s/peak RSS1.52GiB under4GiB/two threads.
This prepares rehearsal; no rehearsal fit or quality improvement is claimed.

Evidence: `/mnt/v/output/zensally/relationship-students/2026-11-09/` and
`/mnt/v/output/zensally/old-attention-replay/2026-11-09/`.
No API calls, new teacher inference, nested-range admission, TEST inference or
model promotion. See the [registered protocol](../docs/experiments/RELATIONSHIP_DISTILLATION.md).
