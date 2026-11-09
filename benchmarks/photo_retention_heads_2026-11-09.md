# Candidate-local photographic retention — 2026-11-09

The visual retention head reduces average weak-reference error, but its
classification gain is uncertain across families and its crop offers still
violate known boxes. It is not qualified as a subject/face guard. This remains a
research classifier; the original composition model and face gates are unchanged.

All nine registered fits complete: image/geometry/permuted controls, three
seeds each,110 epochs,73,442 trained head parameters. The frozen photographic
encoder supplies candidate-local region descriptors, not an inferred global
subject box. Its first scorer layer initializes the new head. The existing
composition ordering remains frozen and separate. The
[protocol](../docs/experiments/PHOTO_RETENTION_HEADS.md) fixes the loss,0.5 offer
threshold and selection by development Brier before fitting.

Only72 fitting sources/61 families supply gradients from noisy known-box
targets. Fifteen development sources/14 families select heads. Targets are the
[generic-people substitution](photo_subject_retention_2026-11-09.md), including
its documented misses. No complete subject, face or artistic annotations exist.
The other32 original sources remain unknown for these targets. Permuted controls
move visual inputs within role/exact letterbox shape, while **actual source crop
geometry** remains fixed. Geometry-only zeros704 visual descriptor values.

| Head, three-seed mean | Containment Brier, lower better | Retention-fraction MAE |
|---|---:|---:|
| Image |0.096047|0.187841|
| Geometry-only |0.103605|0.212173|
| Permuted image |0.105718|0.199289|

Selected image epochs are12/15/13; geometry103/110/110; permuted20/21/21.
Across14 development families, averaging source/seed diagnostics within family
before5,000 paired bootstrap draws, image-minus-geometry Brier is−0.007558,
95% percentile interval[−0.016771,+0.003819]; versus permuted it is−0.009671
[−0.027461,+0.010259]. Neither excludes zero. Fraction MAE minus geometry is
−0.024333 [−0.042431,−0.005394], but versus permuted is−0.011449
[−0.033090,+0.010359]. These are reused selection families, not confirmation.

There are330 development requests and171 known-box opportunities in the fixed
candidate/mode bank. Probability at least0.5 filters candidates before the
original student's composition ordering; an empty set abstains. Per-seed counts:

| Image seed | Offered crops | Known-box violations | Successful known-box crops | Unnecessary abstentions |
|---|---:|---:|---:|---:|
|31|188|53|135|25|
|137|176|44|132|28|
|911|170|40|130|30|

Geometry controls offer124–132 crops, with26–28 violations; shuffled controls
offer175–195, with48–65 violations. Lower error/greater crop coverage does not
establish safe offers. An always-contain policy would have zero crop violations
and zero crop coverage. These are returned-box diagnostics, not full-person or
face recall. The330 requests include one without an informative old ranking
target, so they differ from the329-query original-rank audit. Source/group
coverage gaps and known geometric/coarse-bank/fallback failures remain explicit.

All nine selected heads replay saved development metrics and decisions exactly.
The pooled-feature cache also reproduces all original scorer bits and all329
informative original development crop choices, with zero edge movement. Every
fitting source/ratio receives ten visits; development receives none. No encoder
or BN state is trained. Full head optimizer/RNG checkpoints and scalar histories
remain retained; all three controls use the same initial hidden layer and seeds.

On i265, run-heavy reports feature caching3s, peakRSS2.85GiB,
min-available22,228MiB, peak-load1.89; fitting13s/2.12GiB/22,865MiB/1.83;
selected-state audit2s/1.81GiB/23,682MiB/1.61. All use4GiB/two-thread host caps.
GNU-time maxRSS for cache/fitting is2,984,612/2,226,892KiB. The feature cache
occupies751,084,757 bytes, below1GiB. CUDA allocated/reserved peaks are
229,106,688/266,338,304 for caching;896,954,880/929,038,336 for fitting;
887,731,200/926,941,184 for audit. These cached-head resource observations are
not deployment timings, GPU occupancy or training-saturation measurements.

All nine selected state identities are retained for a subsequent fixed-model
supplementary-development evaluation before further tuning. This study opens
no TEST or supplementary-development labels, adds no paid API spending or
diagnostic maps, and promotes no quality model. Any later joint end-to-end or
purpose-aware fitting needs its own controls; its success is not implied here.
See the [artifact pointer](photo_retention_heads_2026-11-09.pointer.md).
