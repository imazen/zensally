# Frozen-bank nested pair selection

Compare the selected contrastive seed31 model4b801ad4 on original119 and real72
FIT/DEV sources, all11 ratios. Keep source/family roles, weights, scores,
eligibility and native candidate geometry fixed. No TEST, new training or labels.

Three policies: independent original choices; outer-first keeps raw Minimal
and selects the highest-scoring strictly smaller Maximal candidate inside it;
joint-sum maximizes the sum of Minimal/Maximal scores over such nested pairs.
Pair sums use binary64 values of the binary32 model scores, with ascending
candidate-index ties. No head/pose/teacher label filters enter selection.

If no strict inner candidate exists, keep raw Minimal and mark Maximal unavailable.
Do not substitute an equal crop, fabricate a smaller box or score missing output
as zero regret. Report strict-pair availability separately from endpoint coverage.
Every offered structured pair must have matching exact native aspect, valid
bounds, maximum-area outer Minimal and a strictly smaller contained inner.

Evaluate original weak target regret and coverage on the unchanged25 DEV panel,
and separate head-cut/reference-match diagnostics on real24 DEV. Preserve per-query
availability: known endpoints need not come in labeled pairs. Incomplete coverage
has no full-panel quality claim. Record raw numerical replay against frozen gate
decisions before interpreting quality differences.

Use the existing four-FIT-family review as bounded actual-pixel witnesses,
particularly the performer gesture lost under joint pairing. Evaluate geometry
and artistic consequences separately; no complete-face or artistic guarantee.
Keep all prior failures, unchanged quality thresholds and zero paid API spending.
