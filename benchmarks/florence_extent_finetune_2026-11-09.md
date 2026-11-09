# Encoder training with cached object targets — 2026-11-09

The small encoder learns the fit boxes but does not generalize usefully to the
eight development families. All 15 registered fits finish: nine frozen-head
L1/GIoU controls and six encoder-training image/permuted runs. Targets, roles,
seeds, head, objective and source permutation remain matched. The encoder uses
learning rate 0.0001; the head 0.001. No teacher is rerun or model promoted.

| Model/input | Selected development IoU | Selected objective |
|---|---:|---:|
| Frozen image features |0.253200|2.445554|
| Geometry only |0.340012|2.242086|
| Frozen shuffled features |0.343473|2.250881|
| Trained image encoder |0.301438|2.270559|
| Trained shuffled encoder |0.369064|2.142471|

The trained-image gain over frozen-image is+0.048238 IoU, paired eight-family
bootstrap95% percentile interval[−0.015813,+0.133582]. Against geometry the
delta is−0.038574,[−0.117460,+0.033405]; against trained shuffled images it is
−0.067626,[−0.133438,−0.004840]. These are reused development selection data,
not independent confirmation. The primary-object query is supervision only;
neither model is text conditioned or trained to judge artistic composition.

The additional audit evaluates selected and final checkpoints on both explicit
roles without new gradients. The frozen-image head reaches fit IoU 0.900528 at
epoch 200 but development 0.230108. The trained-image encoder reaches fit 0.910905
and development 0.261546; its selected epochs6/114/3 give fit 0.635470 and
development 0.301438. Shuffled models also fit their training targets closely.
This is direct evidence of overfitting, rather than inability to optimize the
observed targets. More steps alone are not supported by this experiment.

All six initial cached/native box checks pass the unchanged three-map-pixel
edge gate; largest difference 0.000015259 pixels. Every selected development
prediction replays bit for bit in the audit. All 30 selected/final checkpoint
audits produce finite, ordered, in-bounds boxes for48 fit/eight development
sources. Valid geometry is not complete object or face coverage.

run-heavy on i265 reports62 s for the complete15-fit stage, peak RSS 1.95 GiB,
minimum available 21801 MiB, peak load 5.29. GNU time max RSS is2,046,640 KiB;
encoder-training CUDA peak allocated497,382,400 bytes/reserved694,157,312.
The audit takes2 s/1.52 GiB/22931 MiB/3.82, with CUDA192,092,160/243,269,632 bytes.
These whole-stage costs are not deployment latency or GPU occupancy. Both
commands use a4 GiB host cap and two threads. Checkpoints and RNG state remain
retained. No API spending, diagnostic maps, new TEST inference or changed face
guard. See [registered protocol](../docs/experiments/FLORENCE_EXTENT_FINETUNE.md).
