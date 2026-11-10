# Frozen retention models on people scenes — 2026-11-09

Candidate-local visual features improve weak-reference classification on this
cohort, but the resulting crop offers still cut through people. The clipping is
substantial, not a low-bit or three-map-pixel numerical issue. The models remain
unqualified as region/face guards.

All nine earlier selections are frozen before this evaluation: three image,
geometry and shuffled-image seeds. No epoch or0.5 offer threshold changes. The
registered15-family supplementary-development cohort contains14 people scenes
and one wildlife scene. All15 existing local queries/23 boxes are reused with
zero new teacher calls. These sources were previously inspected for other
studies; they are not unseen final TEST or human artistic annotations.

| Three-seed mean | Containment Brier | Retention-fraction MAE |
|---|---:|---:|
| Image |0.067470|0.201820|
| Geometry-only |0.078119|0.188013|
| Permuted image |0.086275|0.210799|

Image-minus-geometry Brier is−0.010649,15-family95% bootstrap percentile interval
[−0.020150,−0.000735]; versus permuted is−0.018805 [−0.036196,−0.003563].
These5000-draw, seed31 intervals favor visual classification on the incomplete
weak boxes. Fraction MAE does not improve over geometry; both fraction-error
comparison intervals include zero. A better per-candidate score is insufficient
evidence for the resulting selected crop.

There are330 requests/seed, with114 opportunities in the fixed bank. Of270
essential-ratio requests,108 have an opportunity; of60 secondary requests, six
do. Counts for the fixed0.5 offer policy:

| Image seed | Offered | Known-box violations | Successful crops | Unnecessary abstentions |
|---|---:|---:|---:|---:|
|31|169|117|52|36|
|137|158|104|54|35|
|911|161|112|49|39|

Essential ratios alone have102–111 violations per image seed. Geometry controls
have87–89 violations among129–133 offers; shuffled controls134–146 among180–193.
These denominators expose coverage/error tradeoffs rather than implying that
greater abstention is better composition. Always containing the source would
have no crop violations and no offered crops.

Independent native-integer geometry agrees with every stored selected-crop
containment/fraction result. Image-seed violations have median maximum lost
reference-edge distances20.75–21.15 on the normalized128 grid; only5–6 violations
per seed lie within three grid pixels, and16–23 crops retain zero reference area.
The three-pixel figures are diagnostics only; no containment/face gate is relaxed.
In fixed rendered witnesses,12016's1:3 crop cuts through the close profile's nose
and mouth, and11048's1:3 crop removes most of the foreground winter-sport
participant. All four fixed3:1 witnesses abstain; the monkey1:3 witness also
abstains. The classifier is not promoted despite its favorable Brier interval.

The source-bound viewer exposes all15 sources, nine controls,11 ratios and both
amounts. All2,970 view combinations pass source identity, native crop, aspect,
overlay and explicit SVG clip checks. Eight screenshots total3,757,481 bytes,
below8MiB; all are visually inspected. It uses owned working previews with exact
native crop coordinates, not newly rendered training images or metric maps.

The new null mapping is bound from metadata before predictions: SHA256 order
then cycle within exact letterbox shape, moving14 sources; one singleton stays
fixed. This differs from the original fitting permutation and is reported as a
supplementary control. Original scorer bits replay exactly from new pooled
descriptors. Every selected checkpoint/source/reference identity remains fixed.

On i265, run-heavy evaluation takes2s, peakRSS1.51GiB, minimum available24,597MiB,
peak-load1.57 under4GiB/two threads. GNU-time maxRSS1,580,992KiB. CUDA peak
allocated/reserved413,515,264/473,956,352 bytes. Browser verification takes1s,
peakRSS0.20GiB, minimum available24,504MiB, peak-load1.47. These are resource
observations, not controlled deployment timing or occupancy claims.

One next hypothesis is calibration of the training loss: the current head uses
equal positive/negative BCE weight per query, even when positive candidates are
rare, while offers threshold its sigmoid at0.5. A constant-score analytic control
and matched unweighted-BCE fits can test that hypothesis without changing any
existing threshold or frozen model. This does not yet establish the cause of
the clipping failures. Artist/face qualification and broader people fitting
coverage remain missing. No new gradients, TEST inference, paid API calls or
model promotion occur in this evaluation.

See the [protocol](../docs/experiments/RETENTION_SUPPLEMENTARY.md) and
[artifact pointer](retention_supplementary_2026-11-09.pointer.md).
