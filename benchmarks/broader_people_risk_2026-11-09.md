# Broader people crop-risk evaluation

No model is promoted. Measured attention improves the local composition proxy
on this broader development panel, but automatic head cuts and visible face/
composition failures remain. These controls still need detector/pose and, for
attention arms, gaze/scanpath teacher inputs. They are not a cheap image-only
runtime. Closing this research pass adds no API spending or TEST exposure.

## Scope and frozen inputs

The [registered panel](../docs/experiments/BROADER_PEOPLE_RISK_PANEL.md) selects
16 previously acquired OpenImages sources: two per eight count/scale cells,
from59 eligible remaining DEV sources. It excludes the recent72 risk families
and eight earlier crop-review families. Roles and source-family IDs remain
fixed. Prior preview and broad-rank development exposure means this is broader
DEV evidence, not untouched TEST. No fitting or selection uses these outcomes.

Twelve frozen risk heads use C002 hidden96, region relations32 and measured
attention32 where required. Full local face/pose inference completes160 calls;
gaze and scanpath complete16 each. No model downloads or paid calls. Measured
regions are face-positive13/16, pose-positive5/16 and person-positive16/16.

All288 essential endpoints are offered;282 have relative local teacher targets.
The head-known subset contains234 endpoints. Below, regret is the family-mean
local FocalNet relative composition proxy, lower is better. Head-cut flags
come from supplied automatic proposals also used as inputs: they are not
independent face truth. All64 secondary endpoints remain offered, scored
separately in the packet. Do not apply the old0.247 gate to this distribution.

|Frozen head-gated policy|Proxy regret|Supplied-head cuts /234|
|---|---:|---:|
|unfiltered|0.594591|105|
|geometry-regions-seed31|0.776339|55|
|geometry-regions-seed137|0.621426|54|
|geometry-regions-seed911|0.758242|50|
|hidden-regions-seed31|0.603269|47|
|hidden-regions-seed137|0.612593|50|
|hidden-regions-seed911|0.578732|52|
|hidden-regions-attention-seed31|0.523959|54|
|hidden-regions-attention-seed137|0.561194|53|
|hidden-regions-attention-seed911|0.556315|56|
|hidden-permuted-regions-attention-seed31|0.596186|106|
|hidden-permuted-regions-attention-seed137|0.526134|94|
|hidden-permuted-regions-attention-seed911|0.590031|107|

All three aligned attention seeds improve essential proxy regret relative to
baseline. The permuted seed137 also improves that proxy while retaining94
cut flags; proxy improvement alone cannot establish sound subject composition.

## Numerical checks and visual limits

All48 compact/CUDA routing comparisons cover176 pairs each, with zero changed
endpoints, offer changes or three-grid violations. This is cached head-level
qualification, not an end-to-end teacher/runtime qualification. Replaying the
old72-source attention contract checks11,668,992 eligible scalar feature values
bit for bit across real/permuted inputs and all32 columns. C002 last-linear
head replay on the new16 sources is exact.

The blind gallery has32 comparisons/128 crop panes plus16 source previews.
Only22/32 comparisons are sealed. Repeated user-reported chat crashes during
image inspection ended visual intake. No corresponding failed local tool call
was observed; the crash cause is unverified. Cases22–31 remain unsealed; the
identity key stays unopened and no aggregate per-model visual result is claimed.
Do not unblind a partial panel. These observations are assistant evaluation,
not independent human ratings, training labels or interval-interior acceptance.

Visible failures include face cuts on a speaker, partial people, sliced text,
and animal fragments. Some alternatives repair individual frames. Both systems
remain unsuitable for a no-face-cut or excellent-composition guarantee. No
new excellent nested crop interval is admitted from this inspection.

## Resources and reproduction

Observed on i265 through run-heavy, each local job separately capped:

|Job|Elapsed seconds|Peak RSS GiB|Memory cap GiB|
|---|---:|---:|---:|
|Native face calls|3|0.07|4|
|MediaPipe face/pose|3|0.22|4|
|C002 extraction|2|1.58|4|
|Gaze/scanpath|16|1.74|4|
|Attention feature replay|53|0.63|2|
|All12 frozen-head routing|3|0.99|4|
|Blind crop rendering|1|0.05|2|

These are research job observations, not per-image serving latency. Full
run-heavy logs retain peak RSS, minimum available memory and peak load. Source
producer commits016a5a72/671bba1d/eb41d1a/3b318eb2/de6b40ff. The first attention
cache attempt rejected unused padded zero boxes before output. Eligible-only
feature extraction repaired the caller; validator, feature definitions and
quality expectations are unchanged. Failed log retained.

Artifacts on i265: `/mnt/v/output/zensally/broader-people-risk/2026-11-09/`.
Plan, model/cache identities, full predictions, all summaries, logs, rendered
pixels and sealed reviews are retained. `SHA256SUMS.json` binds membership.
NAS mirror uses the same suffix under `/mnt/tower/output/zensally/` on dev.
Reproduction uses the composition-*broader-people* recipes in justfile; no
rerun is needed for this report-only closure.

## Decision

Retain the current serving behavior and these research checkpoints. Do not
ship the risk heads as an artistic crop model. A subsequent scoped training
phase would need reliable subject-choice and excellent nested-pair labels,
including negative face/animal/text boundary examples. That phase is not
started here. No new API spend, dependencies, training, public API or release.

Frozen packet:614 members,148715116 member bytes; manifest SHA256
`396aa138e6eabf9ae3f6b5249a8c38a01de7581741cb6059d377e0a1f73095b3`.
All614 NAS members independently hash-verified on dev. HTTP delivery checks
all144 PNGs plus index byte for byte against local files; no additional
visual inspection.
