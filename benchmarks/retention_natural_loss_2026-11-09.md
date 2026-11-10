# Natural-frequency containment loss — 2026-11-09

Ordinary BCE changes the crop-coverage/error tradeoff without establishing a
safe guard. All nine matched controls finish with the same frozen features,
initialization,72-fit/61-family and15-development/14-family roles, source/query
visits, selection rule and0.5 offer threshold. Only class weighting changes.
No reference, encoder, original composition score or old model is replaced.

The analytic gradient controls confirm that a constant predictor with one
positive among twenty valid candidates has its balanced-BCE optimum at0.5 and
ordinary-BCE optimum at0.05. Invalid padded positives contribute zero gradient.
All four tests pass. Their first attempt assumed FP64 gradient accuracy despite
the original loss's FP32 class targets; the corrected analytic tolerance derives
from FP32 machine epsilon. The original FP32 arithmetic and strict padding-gradient
test remain unchanged; both logs are retained. This validates a mechanism, not
the cause of every observed face cut or a claim of neural calibration.

| Natural-loss head, three-seed mean | Containment Brier | Fraction MAE |
|---|---:|---:|
| Image |0.109737|0.178761|
| Geometry |0.121084|0.216806|
| Permuted |0.112379|0.197065|

For comparison, balanced image Brier/MAE is0.096047/0.187841. Natural-minus-
balanced image Brier is+0.013690,14-family95% bootstrap interval
[−0.000877,+0.028330]; fraction MAE−0.009080 [−0.015324,−0.002607].
These reused development families do not establish an overall quality gain.
The image selections are epochs43/43/10, geometry109/109/108, permuted31/31/20.

At the same0.5 threshold on330 development requests with171 bank opportunities:

| Image seed | Offered crops | Known-box violations | Successful crops | Unnecessary abstentions |
|---|---:|---:|---:|---:|
|31|123|15|108|57|
|137|116|14|102|64|
|911|90|13|77|89|

Balanced images offer170–188 crops with40–53 violations and130–135 successes.
Natural geometry offers40–49 with1–2 violations, and natural permuted images
88–95 with14–17 violations. Lower violation counts partly reflect greater
abstention. Natural-minus-balanced image violation rate per request is−0.086580
[−0.123377,−0.050866], while successful-crop rate also falls−0.119048
[−0.166667,−0.075758]. No threshold is tuned or quality model promoted.

All nine selected metrics/decisions replay exactly. The751,084,757-byte feature
cache is copied byte for byte from the original study, not recomputed. Models
still train only73,442 head parameters. Supplementary people scenes do not select
these epochs; all nine identities are frozen before subsequent evaluation there.

run-heavy fitting on i265:12s, peakRSS2.12GiB, minimum available22,826MiB,
peak-load1.81 under4GiB/two threads; CUDA peak allocated/reserved
896,954,880/929,038,336 bytes. Analytic tests take1s, peakRSS0.62GiB,
minimum available24,713MiB, peak-load1.81 under1GiB/one thread. These are resource
observations, not timing comparisons, deployment latency or GPU-saturation claims.
Full fitting/audit resource logs and failed analytic attempt are retained.

The targets remain incomplete weak query boxes, not full subjects, complete
faces or artistic labels. No new teacher/API calls, TEST inference, source roles,
diagnostic maps or face/numerical gate changes occur. See the
[protocol](../docs/experiments/RETENTION_LOSS_CONTROL.md) and
[artifact pointer](retention_natural_loss_2026-11-09.pointer.md).
