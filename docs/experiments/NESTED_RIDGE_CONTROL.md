# Low-capacity nested-pair control

Registered after continuous paired-head packet52e5852f. The136677-parameter
image heads fit29 source families better than geometry but generalize worse.
Before more optimizer steps, test deterministic weighted ridge regression on
the same99 existing FIT pairs. Do not add labels or use DEV for hyperparameters.

Three controls: geometry polynomial; geometry plus eight principal components
of raw frozen source-grid image descriptors; geometry plus eight principal
components of the measured query-conditioned32-value attention summary. All
feature-bearing heads use the same29 design columns: five source/target
log-aspect quadratic terms, eight PCs, PC×target log-aspect and PC×its square.
Geometry uses the first five. Predict five bounded fractional geometry values
for the exact same maximal-outer/strict-inner decoder. No contain head.

Every labeled source has total weight one before global normalization. Choose
lambda from1e-4 through1e4 in powers of ten by leave-one-FIT-family-out mean
essential max-edge error. Fit raw descriptor normalization, PCA and regression
normalization anew inside each fold, excluding the held family. The held family
supplies no fitted feature statistics or labels. Tie prefers larger lambda.
Secondary FIT points may train a fold but do not select lambda. Fit final
transforms on all99 FIT pairs only after selection. No stochastic seed fitting.

Report every fold/lambda, selected lambda, actual audited DEV match/head-cut
counts and selected-pair coordinate error. Compare against the registered
geometry control. Do not infer excellent crop quality from coordinate agreement.
Use the same source roles, cached positive/reference audits and three-grid
numerical gate; TEST and original fixed-bank quality files remain unchanged.
Check FP64 reference against CPU/GPU FP32 native outputs on all72 sources.
No model promotion, API calls, fresh teacher inference or synthetic sources.
Run-heavy4GiB/two CPU threads,20 percent GPU memory,16MiB artifact budget.
