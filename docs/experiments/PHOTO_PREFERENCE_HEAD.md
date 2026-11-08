# Photographic preference-head cohort — 2026-11-08

Registered before this cohort is fitted. The earlier mixed-corpus preference
head improves aggregate weak development metrics, but its photographic subset
does not improve pair accuracy. Documents and other excluded content must not
stand in for the requested photographic composition task.

Reuse the frozen 80-source descriptor cache and its unchanged source roles.
Select only the existing `people-commerce-v1` photographic screening categories:
general, interiors, nature, food and people. Metadata selects the cohort before
its new fits; it does not certify visual contents or photographic quality.

Fit sources (15): 11018, 11024, 11042, 11206, 11232, 11248, 11412, 11422,
11428, 11606, 11608, 11630, 12000, 12004, 12018.

TRAIN-development sources (5): 11068, 11220, 11460, 11638, 12008.

No family changes roles. All other cached sources are excluded from gradients
and checkpoint selection in this cohort. Canonical VALIDATE/TEST stay closed.
There is only one people-category development source, so this cannot establish
performance on group scenes, distant people, ecommerce or the wider target
distribution. No local teacher or generated-image labels are added.

Keep the [original head architecture, three arms, seeds, optimizer, loss and
FP32 contract](FOCAL_PREFERENCE_HEAD.md). Use 240 epochs with batch bound 16:
each epoch has one 15-source update, matching the original study's 240 optimizer
updates. This changes the cohort, source reuse and batch shape together; it is
not an isolated causal test of filtering. Include epoch zero in selection and
retain every checkpoint. Select by the five-family TRAIN-development loss,
without changing any quality/face threshold or earlier study gate.

Recreate role-local source permutations within the 15/5 subsets. Estimate the
acceptance prior only from the 15 fit sources. Keep every source's old candidate
bank, rendered labels, pair weights and exclusions unchanged. Report all seeds,
initial/selected losses and ranks, and paired source-family uncertainty with
the explicit five-family/selection-reuse limitation. Do not promote a network
or claim human artistic quality from this cohort.
