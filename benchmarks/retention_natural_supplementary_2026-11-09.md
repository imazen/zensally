# Ordinary-BCE retention on supplementary people — 2026-11-09

Ordinary BCE reduces known-box violations mainly by offering fewer crops; it
does not repair the face-cutting failure. The close-profile1:3 witness still
cuts through the face. Neither retention classifier is promoted as a guard.

All nine ordinary-BCE heads are selected only on the original14 development
families and frozen before this evaluation. The15 supplementary families,
23 cached boxes, fourteen moved null inputs, native candidate bank, original
composition scores,0.5 threshold and numerical/face gates are identical to the
[balanced-loss comparison](retention_supplementary_2026-11-09.md). There are no
new gradients, teacher calls, source assignments or paid model API calls.

| Three-seed mean | Ordinary Brier | Balanced Brier | Ordinary fraction MAE |
|---|---:|---:|---:|
| Image |0.050283|0.067470|0.201970|
| Geometry |0.048668|0.078119|0.196428|
| Permuted |0.060293|0.086275|0.207661|

Ordinary-minus-balanced image Brier is−0.017188,15-family95% bootstrap interval
[−0.028155,−0.006973]. However, the ordinary image model no longer beats its
geometry control: image-minus-geometry is+0.001615 [−0.007377,+0.011705]; versus
permuted is−0.010011 [−0.023736,+0.003810]. Fraction-error intervals include zero.
This is previously inspected supplementary development, not unseen confirmation
or human artistic quality. Better Brier alone does not qualify offered crops.

Each seed has330 requests and114 available bank opportunities:

| Image seed | Offered | Known-box violations | Successful crops | Unnecessary abstentions |
|---|---:|---:|---:|---:|
|31|103|64|39|56|
|137|85|50|35|63|
|911|64|40|24|76|

Balanced image models offer158–169 crops with104–117 violations and49–54
successes. Ordinary geometry offers34–39 with13–17 violations, and ordinary
shuffled images94–115 with63–80 violations. Essential ratios alone still have
40–63 violations per image seed. The changed class weighting therefore produces
a different error/coverage tradeoff rather than a face-preserving policy.

Image violations have median lost-edge distances14.90–17.49 on the normalized
128 grid. Only4–5 violating crops per seed lie within three grid pixels;2–6
retain zero reference area. These are diagnostics, not a relaxed gate. The
fixed seed31 viewer now abstains on the hat portrait and winter-sport1:3 cases,
but still offers a partial-face1:3 crop for12016, with47.94-grid-pixel maximum
reference-edge loss. All four fixed3:1 cases and the monkey1:3 case abstain.
The returned boxes also remain incomplete; no claim of full-face recall exists.

All original scorer bits replay exactly from fresh pooled descriptors. The
viewer decodes15 owned source previews and checks all2,970 source/model/ratio/
amount combinations, including explicit crop clips and native integer geometry.
Eight inspected screenshots total3,671,938 bytes below8MiB. The viewer identifies
ordinary versus balanced loss explicitly; all native containment/fraction
receipts agree independently with stored selected-crop values.

run-heavy on i265: evaluation2s, peakRSS1.51GiB, minimum available23,672MiB,
peak-load3.78 under4GiB/two threads; CUDA allocated/reserved peaks
413,515,264/473,956,352 bytes. Review building1s, peakRSS0.62GiB,
minimum available24,703MiB, peak-load2.89. Browser checking1s/0.20GiB/
24,526MiB/2.89. These are resource observations, not controlled runtime or
training-saturation measurements. Source roles and all face/numerical gates
stay fixed. The next useful objective should evaluate the **selected query
action**, including abstention, rather than treating per-candidate classification
loss as evidence of safe composition. No model is promoted.

See the [loss protocol](../docs/experiments/RETENTION_LOSS_CONTROL.md),
[fitting comparison](retention_natural_loss_2026-11-09.md) and
[artifact pointer](retention_natural_supplementary_2026-11-09.pointer.md).
