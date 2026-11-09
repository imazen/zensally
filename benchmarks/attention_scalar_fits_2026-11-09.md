# Frozen-encoder attention scalar ablations

All15 registered fits complete: five arms, seeds31/137/911,88 epochs plus
epoch zero. They use the prior frozen Sol weak references and candidate bank,
not the nested pilot ranges, which still have zero excellent admissions. The
image encoder and first scorer layer stay frozen;129 scalar parameters include
masked inactive weights. Source roles, candidates and old quality files remain
unchanged. No API calls, TEST inference or model promotion.

Essential DEV results average each seed within family before averaging the17
families/139 requests. Lower nominated-head cuts and higher reference matching
are preferred. Selection uses these same DEV labels, so results are descriptive
and selection-biased, not an independent quality test.

| Arm | Nominated-head cuts | Reference matching |
|---|---:|---:|
| Retrained scalar baseline |0.337009|0.141949|
| Pose |0.305008|0.176339|
| Pose + gaze |0.304261|0.190671|
| Pose + gaze + scanpaths |0.306129|0.190485|
| Source-permuted full features |0.309273|0.149334|

All head-cut difference intervals versus baseline include zero; the permuted
control improves nearly as much. Full-feature reference matching improves by
0.048536 with paired family bootstrap95-percentile interval[0.004475,0.092310].
This is2000 resamples, fixed seed31, with seeds averaged before resampling.
It is not an independent significance claim. Scanpaths add no visible matching
gain over pose+gaze on this panel; a direct paired comparison is still needed.
Old-cohort quality and teacher-free transfer have not been evaluated for these
heads. The remaining head-cut rates preclude a face-safety claim.

FIT-only population normalization uses eligible candidates and std floor0.001;
the permuted arm gets its own FIT normalization. Loss averages contrastive
queries within source, then sources within batch12. All22 queries are included;
no-positive or no-negative queries give no gradient. AdamW lr0.001, decay0.0001,
clipping5, FP32 CPU/two threads. Selection is essential family head-cut rate,
then reference match, then earliest epoch including zero. Unknown crops stay
unknown. Initial/selected/terminal checkpoint states retain model, optimizer,
RNG, normalization and DEV scores/choices; every epoch's metrics are retained.

The original CPU launch stopped before training because exact comparison to
CUDA-cached scores failed. That packet/log remains retained. The corrected run
requires every arm to exactly replay a common CPU linear-layer initializer and
records the CUDA comparison separately: maximum absolute difference
4.76837158203125e-7, zero eligible-choice changes across72x22 queries. No runtime
numerical gate changes. All15 arm initializations pass exact CPU replay and
inactive weights remain exactly zero after every optimizer step.

Successful run-heavy:22s, peak RSS1.07GiB, min available22356MiB,4GiB hard cap,
two threads. Analysis:0s, peak RSS0.59GiB,2GiB cap. These are small cached fits,
not inference timings or CPU/GPU saturation claims. Evidence lives at
`/mnt/v/output/zensally/attention-scalar-fits/2026-11-09-cpu/`.
Reproduce using `just composition-fit-attention-scalar PYTHON CACHE OUTPUT`
and `just composition-analyze-attention-scalar PYTHON ROOT`.
