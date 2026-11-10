# Pose, gaze and attention-flow experiment

Registered 2026-11-08 before feature fitting. Goal: improve aspect-conditioned
crop composition with reusable local supervision and bounded serving cost.
No paid API calls are planned. Canonical holdout roles and face guards stay fixed.

## Corrected inactive-head handling

The first nine end-to-end geometry runs used concatenated crop/auxiliary outputs
during both tasks. Autograd materialized zero gradients for the inactive head;
AdamW consequently advanced its existing momentum and decay during the other
task. Those original measurements remain retained, but their baseline comparison
is confounded and does not establish the causal benefit or cost of geometry.

Repeat the unchanged arms/seeds/40-epoch protocol in `joint-fits-task-isolated`,
computing only the active head during each fitting task. Inactive parameters
must have `grad=None`, retain their exact values and keep optimizer state intact.
Combined outputs remain available during evaluation. Do not alter source roles,
thresholds, labels or the original packet. The original training-cost study
measures the combined-output auxiliary graph; retain it under that explicit scope
and compare the task-isolated workload separately.

## First suitability gate

Use the existing frozen 64 TRAIN fit and 16 TRAIN development families. They
are a limited weak-label pilot with few people photos; no human-quality claim
or promotion can follow from this gate alone. The 56 generated photos are
annotation-only diagnostics here, not newly admitted composition training data.
Do not open canonical VALIDATE/TEST during feature/architecture selection.

Extract MediaPipe pose and face geometry, Gaze-LLE ViT-B in/out gaze targets and
DeepGaze III scanpaths locally. Pin source commits, checkpoint hashes, input RGB
and preprocessing identities. Persist compact landmarks, coordinates, confidence
and candidate scalars; no diagnostic maps. Models may compute internal arrays
needed for inference. No complete face-coverage assertion comes from a teacher.

Freeze the previously selected encoder/projections at epoch 11. Cache the exact
existing candidate descriptors and compare five matched rank/acceptability
heads: no added features, pose, pose+subject-gaze, pose+gaze+viewer-scanpath, and a
source-permuted feature control. Added inputs initially have zero weights so
all heads reproduce the frozen initial outputs. Existing weights and numerical
contracts remain intact; extensions live in separate experimental wrappers.

Predeclare seeds 31/137/911, 60 epochs, AdamW lr 0.0001, weight decay 0.01,
source-balanced loss, batch 16, clipping norm 5. Choose epochs using existing
TRAIN-development loss; report all arms/seeds, paired source-family summaries
and uncertainty. Keep the existing acceptance threshold and contain policy.
Do not promote a feature based on a single seed or teacher self-agreement.
Raw source-frame attention is a feature proxy, not a composition grade for an
unseen crop. Any subsequent direct crop-attention experiment uses actual
rendered candidates with a separate retained protocol and viewing assumptions.

## Next gates

If geometry improves held-out development rankings, test end-to-end student
training with auxiliary pose/attention heads and matched baselines. Measure
complete serving cost against the existing model pairs, with numerical checks
and sudo perf outside uninstrumented timing. Auxiliary teacher heads do not
become runtime dependencies by assumption. Keep fit/device caches explicitly
bounded and retain resumable checkpoints and source identities.

The photographic successor still needs composition preferences and independent
face audits. Research local aesthetic-crop/reward teachers for suitability;
teacher outputs are weak supervision, never independent human evaluation.
New sources require explicit recorded family/role admission before gradients.

## Registered auxiliary-distillation successor (2026-11-08)

The first gate did not improve development loss. A separate successor admits
the 56 previously reviewed synthetic photographs **only for local geometry
distillation**. It does not admit their composition preferences, make them
human rated, or amend the earlier annotation-only experiment. No PhotoFramer
rating supplies a gradient: its initial six-pair comparison is insufficient.

Group every shared fictional brand into one family and otherwise retain the
authored scenario family, including the ceramics aspect variants. Assign an
auxiliary development role when the first eight hex digits of SHA256 of
`geometry-distill-v1:<family>` modulo four equal zero; all others fit.
This gives 48 fitting sources in 27 families and eight development sources in
eight families. This synthetic development set has no branded commerce; report
that limitation rather than changing assignments after observing results.
Original canonical VALIDATE/TEST roles remain unopened and unchanged.

Use the same aspect-preserving 256-square owned-resize input and valid rectangle
as the original network. Offer 110 candidate boxes: eleven ratios, two crop
amounts (largest fitting rectangle and 75% linear size), five evenly spaced
positions along the available translation. Targets are the existing 32 compact
pose/gaze/scanpath features; pose and subject-gaze losses are masked when their
teacher has no detection. Missing detection does not certify absence.

Compare continued end-to-end crop fitting, crop fitting plus auxiliary geometry,
and a within-role source-permuted auxiliary control. All arms have identical
parameter counts; the auxiliary head is separate from the original two logits
and is removable for serving. Initialize every arm from the same retained
epoch-11 checkpoint. Seeds 31/137/911, 40 epochs, AdamW lr 0.00003, weight decay
0.01, clipping norm 5, batch 16, FP32 with TF32 disabled, auxiliary loss weight
0.1. Freeze BatchNorm running statistics in all arms (affine parameters remain
trainable), avoiding a different unlabeled-statistics adaptation between arms.
Each epoch runs the same four original TRAIN crop batches followed by three
auxiliary batches; the baseline performs the auxiliary forward pass without an
update. Select on the same 16 original TRAIN-development families, including
the frozen epoch-zero model as an explicit candidate for this new experiment.
Report geometry errors separately on the eight auxiliary development families.
Save immutable epoch predictions/checkpoints and retain failures. This tests
representation suitability, not human composition quality or safe deployment.
