# Natural-frequency retention loss ablation

Register before new fitting. Preserve the balanced-BCE study and its15-family
supplementary evaluation unchanged. Copy the exact qualified feature cache
`2b36c1215840dbc889481f6716eaddb5ba43d4b5af6fa434e3d1be6e91a9a3f1`
into a fresh study root. No new pixels, teacher queries, source roles or labels.

Change exactly one training axis: ordinary candidate-frequency BCE replaces
the equally weighted positive/negative class means within each source/query.
Keep the0.5 fraction-sigmoid MSE term, original first-layer initialization,
seeds31/137/911, all three controls,110 epochs, source/query visits, family
weights, optimizer,0.5 offer threshold and original14-family development Brier
selection. The fifteen supplementary families do not select checkpoints or
thresholds. The default fitter still implements the original balanced loss.

A mathematical constant-prediction control checks the mechanism before neural
fitting: with one positive among twenty valid candidates, balanced BCE has zero
constant-logit derivative at probability0.5; ordinary BCE has zero derivative
at0.05. Invalid padded positives have zero gradient and do not alter the prior.
This explains a possible interpretation mismatch of the old0.5 threshold, not
the cause of every observed crop failure or a guarantee of calibration.

Compare all nine selected models on the original development set, preserving
exact replay and source-family uncertainty. Freeze their identities before
subsequent supplementary evaluation; retain the old seed31/exact-shape null
recipe for fitting and the already frozen metadata mapping for supplementary
comparison. Measure offered-crop violations, coverage and clipping severity as
well as per-candidate Brier. Do not claim improved crops from Brier alone.
Complete-face and artistic labels remain missing; no model is promoted.

Use4GiB/two-thread host caps, CUDA35%,1GiB cache and2GiB fit-state bounds.
Preserve optimizer/RNG states, full scalar histories and all source identities.
No new paid API calls, TEST inference, metric maps or arithmetic/face gate change.
