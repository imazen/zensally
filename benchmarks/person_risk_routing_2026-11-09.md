# Person applicability and soft cut-risk ranking

Person-conditional soft risk preserves all original and newer endpoint offers.
All three aligned attention heads pass the original gate; one FIT-selected
run reduces newer accepted-reference weak cuts45→40 and raises matches16→23.
It repairs11 weak cuts but introduces6. Remaining visible face cuts prohibit
promotion; no continuous excellent crop interval is admitted.

The [registered protocol](../docs/experiments/PERSON_RISK_ROUTING.md) keeps all15
previous cut heads and the additive-linear seed137 pair scorer fixed. No new
training labels, model weights, teacher inference, API spending or TEST access.
Existing EfficientDet-Lite2 results use their unchanged0.3 confidence/max50
contract and exact RGB/size identity. It detects people on70/72 newer and3/25
original sources. No detection is not evidence of no person or complete face
coverage. All22 original sources without detections replay unfiltered choices
exactly under person applicability.

## Fixed policies and FIT-only selection

Hard filtering keeps the fixed0.5 probability threshold and explicit paired
abstention. Soft scoring subtracts lambda times the sum of endpoint cut risks
from the fixed corrected pair score; it changes neither candidate geometry nor
strict-pair feasibility. FP32 probabilities are promoted to binary64 before
pair sums/penalties; ascending index ties. Compare application everywhere and
only on sources with a person detection. Grid0/0.25/0.5/1/2/4/8 yields241 scalar
configurations including unfiltered baseline and hard controls.

Select lambda per head/applicability on newer FIT essential accepted references
only. Match and nominated-head-retention rates cannot decrease from lambda zero;
then minimize family cut rate, maximize match/retention, prefer lower lambda.
All30 selected strengths are nonzero. Neither original nor newer DEV chooses
lambda; classifier epochs were previously selected on the reused DEV labels.
Keep that distinction explicit when interpreting generalization.

## Complete offer comparison

Original essential panel:448 valid endpoints, unchanged regret gate
0.24700142443180084. Newer accepted-reference panel:139 endpoints/17 families.
Unfiltered corrected output scores0.2406301498413086 on the original panel and
offers139 newer endpoints, with45 weak cuts,16 matches and117 retaining a
nominated head. Every selected soft policy keeps all those offers.

| Person-conditional attention | Lambda from FIT | New weak cuts | Matches | Head retained | Original regret |
|---|---:|---:|---:|---:|---:|
| Seed31 |0.5|45|16|120|0.24162159860134125|
| Seed137 |8|40|23|126|0.2440626323223114|
| Seed911 |2|45|18|121|0.24334107339382172|

All three original gates pass. Matched permuted-attention policies select
0.5/0.5/1, cut46/46/46, match16/16/15 and retain117/117/119 heads, with full
offers and all original gates passing. Among all15 heads,13 person-conditional
soft policies pass the original gate versus6 applied everywhere. This is not
13 new independently trained models or13 face-safe policies.

Person-conditional hard filtering recovers original coverage to396–434/448
across heads, but none of15 regains full coverage. Aligned hard heads offer
422/426/420 original endpoints and99/98/103 newer accepted endpoints, cutting
20/21/21. Soft risk avoids that abstention cost while retaining substantial
unsafe outputs. Neither policy solves the never-cut-a-face requirement.

On all264 newer DEV ratio pairs, selected aligned soft policies change53/203/139
pairs. Median inner/outer native area ratio remains0.80718994140625, p95 and
maximum0.81; the apparent gains do not come from making endpoints equal.
These bank-scale observations do not establish excellent Minimal/Maximal framing.
Secondary ratios and contain/unknown-composition cohorts remain separately
reported in the packet; they do not become admitted crop labels.

## Family uncertainty

Three matched seeds averaged before2000 family resamples/seed31,139 common
endpoints and17 families. Aligned attention minus unfiltered baseline:

| Family metric difference | Mean | Paired95% percentile interval |
|---|---:|---|
| Weak cuts |−0.011585038055626289|[−0.04068199502023031,+0.011437908496732025]|
| Reference matches |+0.021226268285091814|[+0.0035792094615624027,+0.04357298474945533]|
| Nominated-head retention |+0.032034632034632034|[+0.007378405907817673,+0.062388591800356503]|

Against permuted attention, match difference+0.02262682850918145 has interval
[+0.004139433551198257,+0.045253657018362904]; head retention difference
+0.02923351158645276 has interval[+0.007002801120448178,+0.056869747899159634].
The cut difference interval still includes zero. This is exploratory evidence
on a repeatedly consulted DEV panel, not independent TEST confirmation, human
artistic ratings or separate attribution to gaze versus scanpaths.

## Full network replay and actual pixels

Full fixed encoder/query, endpoint, pair and cut-risk inference on25 original
plus24 newer cached owned DEV image tensors produces1078 source-query
predictions. Compare all30 FIT-selected soft policies and30 hard controls through
compact CPU heads and full CUDA network inference:240 comparison rows, zero
endpoint changes, zero availability changes, zero edge movement and zero
violations of the unchanged three-grid numerical gate. Measured auxiliary
features and cached detector applicability are reused; preprocessing and
teacher/detector inference are outside this numerical check.

Three deliberately selected visual requests contain21 PNG panes/15 distinct
hashes; all15 directly inspected. Owned crop/resize/PNG output passes independent
exact RGB readback. HTTP readback verifies21 PNGs and4964510 image bytes.

- [Classroom6f66b42a0250bed6](https://www.flickr.com/photos/deb_and_dave/1243726188),1:1: soft risk shifts left and restores the
  foreground woman in Minimal. Maximal still clips the background woman at the
  right. Endpoint-level repair is not safe pairing.
- [Waterfall48cd9b5fa18a01d8](https://www.flickr.com/photos/bnphotos400/7591424414),1:3: soft risk shifts toward the waterfall/right-side
  person and introduces a cut head fragment at the left of Minimal; its inner
  excludes that head. This is one of six introduced accepted-panel weak cuts.
- [WindowtDCuhcDEgOk](https://unsplash.com/photos/tDCuhcDEgOk),3:1: unconditional risk shifts from the upper-window detail
  to a darker lower-blinds strip. Person applicability restores the unfiltered
  result exactly. The gain is teacher-score recovery, not an excellent-crop label.

http://dev.lan:3300/zensally/person-risk-routing/2026-11-09/witnesses/index.html

## Resources and retained data

i265, run-heavy/two CPU threads under4GiB; family analysis one thread/2GiB:

| Run | Reported duration | Peak RSS | Minimum available RAM |
|---|---:|---:|---:|
| Risk grid and FIT selection |32s|1.35GiB|22062MiB|
| Full network numerical qualification |13s|1.89GiB|21451MiB|
| Visual witnesses |2s|1.35GiB|22157MiB|

These are experiment resource observations, not serving latency or memory.
Gaze/scanpath/pose teachers and person inference have not disappeared from the
feature-enabled pipeline; teacher-free student transfer and serving costs
remain required. No compiled runtime or model is promoted.

Frozen75-member packet,19832890 bytes before identity/HTTP check/manifest,
32MiB budget. Manifest SHA256
`8e2456a64197e835ad58a366eb3fd8008ee11b0e629d4c18e7dfd9a4cff44ac2`.
On i265: `/mnt/v/output/zensally/person-risk-routing/2026-11-09/`.
NAS: `/mnt/tower/output/zensally/person-risk-routing/2026-11-09/`.
Complete membership/hash readback verifies the mirror separately. Every grid
point, FIT selection, source applicability, chosen decisions, numerical row,
source link and actual-pixel review remains retained.
