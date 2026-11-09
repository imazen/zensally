# Candidate head-cut classifier suitability

Registered after the full weak-head coverage audit, without changing any
existing labels, roles, frozen quality code or thresholds. Use accepted-positive
candidate masks as non-cut labels and existing explicit nominated-head cut masks
as cut labels. Every other candidate is unknown and supplies zero gradient.
Known class means balance each source; family weights balance sources. Source
coverage remains incomplete and no classifier output certifies face safety.

Keep48 FIT/24 DEV, original common endpoint student4b801ad4 and all candidate
banks fixed. TEST closed and no API spending. Compare geometry8, hidden96,
hidden+pose112, hidden+attention128 and matching permuted-attention128 inputs.
The latter three consume the already measured pose/gaze/scanpath features;
permutation donors and within-role assignments remain unchanged. Fit eligible
candidates only determine input means/std floors0.001. All heads are input→32
ReLU→1. Seeds31/137/911,32 epochs plus zero, batches8 sources/all22 queries,
AdamW0.001, decay0.0001, clip5, FP32 CPU two threads under4GiB.

Select minimum DEV family-balanced known-class logistic loss, earliest tie.
Crop cut/match/availability metrics do not select checkpoints. Retain every
epoch's weights, optimizer/RNG and scalar metrics; selected logits are retained
separately.64MiB output cap. Geometry and permutation controls distinguish image
signal and aligned auxiliary features from crop-position statistics.

For descriptive routing, admit candidates only at predicted cut probability
below0.5, then maximize the fixed common endpoint sum over existing strict
nested pairs. If none survives, both crops are unavailable; report this as
abstention, never as successful or zero-regret cropping. Report accepted-reference,
explicit-contain and unknown-composition cohorts separately, including lost
offers and known-head cuts. The probability threshold is fixed before fitting.

Original25 DEV ranking/coverage must subsequently be checked for heads whose
inputs exist there. Unmeasured auxiliary features must not be fabricated as zero.
Numerical replay, visible crops and runtime costs remain required before any
promotion; classification loss alone is a suitability diagnostic.

Before routing results, register a second descriptive score policy using the
previously qualified additive-linear seed137 correction. Keep its weights fixed;
apply exactly the same0.5 classifier threshold and pair abstention rule. This
separates face-risk filtering from the already known original quality failure
of uncorrected joint sums. Neither routing policy selects classifier checkpoints.

After cached routing shows auxiliary-feature signal but missing old inputs,
extend the unchanged local Gaze-LLE/DeepGaze III input/inference pipeline to the
original25 DEV sources. Reuse existing pinned original pose and tiled-face
results; verify the full226-member teacher subtree. Keep all existing weights,
seeds, preprocessing, head padding/cap, four six-fixation scanpaths, feature
definitions and source/role identities. No fitting or API calls. Persist only
compact teacher outputs, owned resized RGB and32-feature tensors,96MiB cap.
Compute auxiliary features only for eligible original candidate geometry;
unused slots are explicit masked placeholders. Missing source detections remain
measured empty predictions, not fabricated teacher results. A separately recorded
DEV-only permutation follows the same content-shape/hash rotation rule. This
allows full old-gate measurement without replacing absent teachers by zeros.
