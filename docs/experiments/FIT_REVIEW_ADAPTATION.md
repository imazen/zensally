# Weak FIT endpoint-preference adaptation

Register a research pilot using six Codex visual endpoint comparisons from
three existing FIT families in the [nested-pair review](FIT_PAIR_REVIEW.md).
Judgments are weak, from the coordinating model, not independent human ratings.
Keep purpose-sensitive observations and the ambiguous/crowd holds. No excellent
interval, continuous crop label, source-role move or old-label admission changes.

Start every arm from the exact selected contrastive seed31 checkpoint4b801ad4.
Keep C002 encoder, first scorer and base scalar frozen. Train only the existing
96→32 ReLU→32 normalized residual probe, with unchanged mean/std and fixed
teacher projection. No runtime pose/gaze/relationship input is added.

Matched arms: original+new weak-set replay; the same replay plus reviewed
endpoint preference loss; and replay plus reversed endpoint preferences as a
negative control. Seeds31/137/911,16 epochs plus zero, AdamW learning rate0.0003,
decay0.0001, clip5, FP32 CPU/two threads/4GiB cap.

Reuse the original94 FIT/25 DEV hidden cache and real48 FIT/24 DEV cache.
Batch8 old plus8 new, phase one query per source/epoch. Preserve original
family-weighted rank objective plus0.25 new acceptable-set loss. Review arms add
0.05 times weighted mean softplus of dispreferred-minus-preferred score; reverse
the direction only in the registered negative control. The six labels retain
their explicit weights and candidate identities. No synthetic images or labels.

Select among epochs whose original essential CPU regret does not exceed the
common checkpoint's own epoch-zero regret. Then minimize the FIT reviewed
preference loss, old regret and epoch number. New DEV weak metrics are diagnostic,
not a selection key. Keep every epoch state, optimizer/RNG and all choices.
Require the unchanged original full-model CUDA quality gate before any success
claim. Fitting six seen comparisons is not held-out artistic improvement.

No API calls, new teacher inference or TEST access. Keep model experimental;
numerical serving/runtime cost and broader artistic quality remain separate gates.
