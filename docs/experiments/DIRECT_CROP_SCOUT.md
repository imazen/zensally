# Direct two-alternative crop-head registration

Test direct aspect-conditioned windows as a controlled competitor to bank
ranking. Use the same72 real photographs/48 FIT and24 DEV families, cached
Sol windows, and frozen original C002 encoder. No API calls, TEST inference,
synthetic gradients, production changes or implied promotion.

Pool the two projected16-channel encoder levels into an8x8 grid over valid
source content, using the existing fractional area integrator. Concatenate
2048 values with log source aspect, log requested aspect and Minimal/Maximal.
A32-unit ReLU trunk predicts two origins, two scales and a crop-offer logit.
Keep pose/depth/gaze outside the runtime until a controlled gain justifies them.
The data supervises editorial use only; no catalogue/hero/detail claim follows.

Decode geometry separately: Minimal always uses the largest native integer
rectangle of the exact requested ratio; Maximal predicts a bounded positive
integer multiple of that ratio. Origins stay within source bounds. Do not force
diversity or leftmost/rightmost crops. One nominated alternative supervises both
slots; two use a permutation-invariant assignment. Unlisted compositions remain
unknown. Explicit teacher containment is a weak offer target, not a face guarantee.

Fit image, geometry-only and within-role/content-shape permuted-image controls
across seeds31/137/911. Normalize features on FIT only. AdamW lr1e-3,
weight decay1e-4, clip5, FP32/TF32 off,108epochs, twelve source images per
batch and all queries. Coordinate L1 plus0.1 offer BCE. Inactive origin/scale
heads have absent gradients so AdamW cannot advance their state.

Select minimum DEV source-family-balanced nominated-window edge error, then
offer Brier and earliest epoch. Report raw top1/best-of-two matching within the
unchanged three128-grid-pixel edge tolerance, head cuts, offer/contain errors,
mode/ratio results and paired family intervals against controls. Retain exact
selected native replays, terminal optimizer/RNG state and bounded resource logs.

This is a suitability scout: it does not yet measure the existing old-rank
nonregression gate, actual-face coverage, artistic gold or deployment cost.
Those remain required before any promotion. Passing direct geometry tests or
matching a finite weak proposal is not sufficient for the intended product.
