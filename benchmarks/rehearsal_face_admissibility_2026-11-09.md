# Local face candidate admissibility

The fixed local face filter reduces some weak head cuts but cannot enforce
complete face protection or artistic framing. No model or filter is promoted.
No new paid API calls, fitting, teacher inference or TEST inference occurred.

Evaluate the existing139 essential DEV requests from17 source families.
Keep candidate boxes, eligibility, scores and all selected checkpoints fixed.
Report original C002 and replay-only/contrastive seeds31/137/911 without choosing
a new winning seed. Three existing local detectors, padding0/10 and two policies
are descriptive controls, not independently validated operating-point choices.

The filter rejects every candidate intersecting a supplied detection partially.
It allows whole detections to be excluded. The stricter policy also requires
one detection wholly inside the crop. If no candidate survives, return contain.
No detections means an unguarded crop under the first policy and contain under
the second. Detection is not a reviewed inventory of every face/head.

Candidate edges use the actual-pixel viewer's rounded native coordinates.
Detector boxes round outward. Padding reuses `padded_heads`:10 percent plus two
source pixels on each edge, clipped to the image. Reuse the existing
`face_relation` geometry; the reviewed-coverage requirement of shipping
`admissible` remains unchanged. This research audit does not satisfy that gate.

Family-balanced means over the three contrastive seeds:

| Policy | Weak head-cut rate over all requests | Accepted-reference match | Crop offered |
|---|---:|---:|---:|
| Raw ranker |0.3643550349432702|0.12157853040205979|1.0|
| Tiled YuNet, padding10, require one |0.26289222759810993|0.14806468041762158|0.9411764705882352|

The same filter on replay-only yields cut0.27681011798658856 and
match0.14446990917579153, versus raw0.4024276377217553 and0.11048298107121635.
These are reused DEV weak labels; no statistical or artistic-quality admission
follows. A contain decision contributes no cut and no reference match; do not
interpret fewer population cuts as an equal improvement in offered crops.

Seed31, padding10:

| Detector / policy | Offered /139 | Weak cuts among offered | Reference matches |
|---|---:|---:|---:|
| Raw |139|43|19|
| Single YuNet, reject partial |139|37|23|
| Single YuNet, require one |101|26|20|
| Tiled YuNet, reject partial |139|35|23|
| Tiled YuNet, require one |135|30|23|
| BlazeFace sparse, reject partial |139|43|19|
| BlazeFace sparse, require one |51|24|9|

The tiled strict filter repairs12 existing weak cuts while still offering crops,
introduces1 new cut, and removes2 cuts through containment:43-12+1-2=30.
Seeds137/911 each repair12, introduce3/2 and retain35/35 cuts among135 crops.
All displayed survivors obey the supplied local boxes; many separately nominated
heads remain cut. Detector coverage and nominated-head accuracy are both
incomplete. The strict BlazeFace result24/51 illustrates why abstention and
conditional failure rates must be reported together.

## Actual pixels

Nine panes, eight distinct PNG hashes, three deliberately selected source
families. All eight distinct images were inspected directly; owned crop/resize
and PNG encoding pass independent exact RGB readback. No labels are admitted.

- [Counter group](https://www.flickr.com/photos/projetoluzevida/15328959047),
  1:1 Maximal: the filtered crop is byte-identical to raw. The left person's
  head/upper body remain clipped at the left edge. Contain shows the omitted
  person and the wider scene. At least one retained detection does not protect
  an undetected neighboring head.
- [Baseball field](https://www.flickr.com/photos/mathplourde/4846668094),
  3:1 Minimal: a small vertical shift clears the weak head-cut flag, while
  foreground lower-body truncation and background body/head fragments remain.
  Head-box compatibility is insufficient for good composition.
- [Helmeted subject](https://www.flickr.com/photos/55179598@N02/5164778294),
  16:9 Maximal: filtering moves the crop upward, trimming more of the head at
  the bottom. The subject faces away; this is a weak nominated-head regression,
  not evidence of a newly sliced visible facial feature.

The viewer preserves source links and exact coordinates. No browser-layout claim
is made. The packet retains all policy predictions and checkpoint/detector/cache
identities. Output root:
`/mnt/v/output/zensally/rehearsal-face-admissibility/2026-11-09/`.
The first output is preserved separately as`2026-11-09-first.bak`.

## Verification and limits

Reproduce with `just composition-audit-rehearsal-faces` and
`just composition-render-face-candidate-witnesses`; exact arguments are retained
in the packet logs. i265 CPU/two threads, run-heavy3GiB cap: audit4s,
peak-RSS0.63GiB, minimum available22653MiB. Rendering under2GiB cap: reported0s,
peak-RSS0.05GiB, minimum available23137MiB. These are tool resource observations,
not runtime latency or deployment-memory measurements.

The existing eight subject-policy checks pass. Both new tools compile. All five
frozen quality files match9f12692417d774653ed676d236e0f89eaafa7385 byte for byte.
The original numerical and quality gates remain unchanged. Post-filter original
quality regret, full-neural gated-choice parity and filter runtime cost are not
measured here. None of these weak masks establish complete face recall.

Next work must address missed heads and whole-body framing, separate from
candidate scoring. A stronger local region source must be checked against the
retained misses before another student fit or deployment claim.
