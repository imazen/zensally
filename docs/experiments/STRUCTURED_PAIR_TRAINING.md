# Structured nested pair fitting

Registered after frozen-bank selection and target-only oracle diagnostics.
No new teacher labels, paid calls, TEST access or existing gate changes.
Keep94 original FIT/25 DEV and48 newer FIT/24 DEV sources and family roles fixed.
Start every arm from contrastive-rehearsal seed31 checkpoint4b801ad4; freeze
the encoder, first scorer and base scalar. Train only its96→32 ReLU→32 probe.

Compare replay, independent-top and paired-top with seeds31/137/911,
32 epochs plus epoch zero, AdamW learning rate0.0003, decay0.0001, clip5.
CPU FP32, two threads,4GiB cap. Every batch has8 original and8 newer FIT
sources. A source-hash phase plus epoch modulo11 picks a ratio; train both
endpoint modes. Original family-weighted rank loss and0.25 newer acceptable-set
loss apply to every arm. Independent-top adds0.05 summed endpoint top-choice
cross entropy; paired-top adds0.05 feasible pair cross entropy. Both added terms
use only FIT requests with both endpoint targets valid. Pair labels maximize
the original frozen target sum over strict nested pairs; independent labels
maximize each endpoint separately. These remain weak ranking targets.

Feasible pairs use existing integer native candidate geometry and eligibility.
Pair sums use binary64 values of FP32 scores, ascending outer/inner ties.
No head/pose/teacher geometry filters or candidate changes enter selection.
Select among epochs whose raw AND joint-pair original essential DEV regrets
are each at most the unchanged original baseline0.24700142443180084; minimize
paired regret, raw regret, then epoch. Retain the best failing epoch separately
if no epoch qualifies, explicitly ineligible. Newer DEV metrics are diagnostic,
never a selection key. Preserve every epoch, optimizer and RNG state.

Qualifying candidates still require full neural numerical replay and original
full-model CUDA quality validation before any promotion. Nesting alone cannot
establish face safety or artistic composition. Artifact cap256MiB.
