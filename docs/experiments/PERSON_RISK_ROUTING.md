# Person-conditional cut risk and soft pair ranking

Registered after all15 hard cut classifiers lose original coverage. Keep their
selected weights, all input caches, roles, masks and the qualified additive-linear
seed137 pair scorer fixed. No new training labels, teacher calls, API spending,
TEST access or quality threshold changes. Compare source applicability everywhere
versus only where cached EfficientDet-Lite2 returns a person at its existing0.3
threshold/max50 results. Exact source RGB/size and frozen detector identities
must match. Absence of detection is not a claim that no person/face exists.

Compare hard0.5 candidate filtering with soft risk ranking. Soft pair score is
the fixed corrected score minus lambda times the sum of its two candidate cut
probabilities. FP32 probabilities are converted to binary64 before pair sums
and penalties; ascending index ties. Lambda grid0,0.25,0.5,1,2,4,8. Soft ranking
keeps the original feasible-pair bank and does not introduce abstention. Hard
filtering preserves explicit paired unavailability. With inapplicable source,
both policies must reproduce unfiltered pair choices exactly.

Select lambda separately per head/applicability using newer FIT essential
accepted-reference family metrics only. Require reference-match and nominated-
head retention rates at least their lambda-zero values; then minimize head-cut
rate, maximize match/retention, and prefer lower lambda. Lambda zero always
qualifies. Original DEV and newer DEV never select lambda. Retain every grid
point, including failures, and explicitly mark zero-strength selections.
Nominated-head retention is an aggregate FIT selection constraint, not a
universal requirement to retain every person or an artistic-quality label.

Report accepted-reference, contain and unknown-composition DEV diagnostics,
secondary ratios and full original448-endpoint quality/coverage under the same
gate. Preserve per-request decisions for selected lambdas and hard controls.
This evaluates whether domain applicability and continuous risk ranking avoid
unnecessary abstention; it does not make remaining face cuts acceptable.
Two CPU threads under4GiB,32MiB artifact cap. Numerical/visible review and
runtime costs remain prerequisites for any promotion.

The [measured result](../../benchmarks/person_risk_routing_2026-11-09.md) finds
full offers and original gate passes for all three aligned attention policies,
with persistent and newly introduced face cuts. No model is promoted.
