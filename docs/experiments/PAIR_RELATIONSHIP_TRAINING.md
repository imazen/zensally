# Fixed endpoint, learned pair relationship correction

Registered after all structured endpoint fits fail the original quality gates.
Freeze the common4b801ad4 endpoint student and all embeddings. Keep original
94 FIT/25 DEV and newer48 FIT/24 DEV roles, all native banks and targets fixed.
TEST closed; no API calls or new labels. Raw endpoint outputs remain identical
by construction, which must be checked numerically.

Train additive-linear, pair-real and pair-permuted controls, seeds31/137/911,
32 epochs plus zero. Batch8 original FIT sources, source SHA phase plus epoch
modulo11 ratios, AdamW0.001, decay0.0001, clip5, two CPU threads,4GiB cap.
Only requests with both old endpoint targets valid supply gradients. Supervise
the distribution of strict pair target sums with temperature0.5 listwise KL.
Normalize features using the unchanged endpoint student's FIT-derived buffers.

Additive-linear adds separate96→1 endpoint projections initialized zero.
Pair-real projects each endpoint96→8, then adds their scaled dot product to
the fixed endpoint sum. Inner projection initializes zero; outer is random.
Pair-permuted has the same architecture but cyclically deranges inner embeddings
among eligible candidates for each request; placeholders never move. All arms
initially replay the fixed binary64 pair sum exactly. Correction matmul is FP32,
cast to binary64 before addition. Ascending outer/inner index ties.

Select old essential paired regret only among epochs passing the unchanged
0.24700142443180084 limit; raw output must remain identical to the common model.
Retain the best failing epoch as explicitly ineligible if none passes. Newer
DEV pair head-cut/reference diagnostics do not select checkpoints. Save every
epoch state and optimizer/RNG;64MiB artifact cap. These are weak target fits,
not excellent interval or complete-face certification. Qualifying output still
needs full neural numerical/quality checks, visible review and measured runtime
before promotion. No compiled model export is authorized by this experiment.

The [measured result](../../benchmarks/pair_relationship_training_2026-11-09.md)
finds six gate passes but no demonstrated relationship-feature benefit or face
safety. Full neural replay is exact; no model is promoted.
