# Pose, gaze and attention-flow experiment

Registered 2026-11-08 before feature fitting. Goal: improve aspect-conditioned
crop composition with reusable local supervision and bounded serving cost.
No paid API calls are planned. Canonical holdout roles and face guards stay fixed.

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
