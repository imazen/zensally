# Local face coverage and wider head protection

Wider protection fixes the retained counter-group witness, but still leaves
many weak head cuts. No model or guard is promoted. This extends the
[first candidate audit](rehearsal_face_admissibility_2026-11-09.md) without new
API calls, detector inference, fitting, TEST access or changed quality gates.

All139 essential DEV requests/17 families, original scores and selected
replay-only/contrastive checkpoints remain fixed. Screen seven existing face
outputs and two unions, not pose/body boxes. The two unions are tiled-interior
YuNet plus Face Landmarker, and all seven face outputs. Repeated detections
are retained; all must satisfy the same whole-or-excluded constraint.

## What causes the surviving cuts?

Frozen Sol nominated-head boxes reproduce every cached eligible negative bit
on this DEV cohort using native integer geometry. For tiled YuNet, padding10,
require one retained detection, seed31:

- 30 failing requests contain35 cut-head occurrences. These are repeated
  crop/head events, not35 independent people or source images.
- 7 cut-head occurrences have no overlap with any padded detection.19 overlap
  only retained detections,8 only excluded detections and1 both. Positive
  rectangle overlap is diagnostic, not a verified person association.
- 26 of30 failures have a candidate that contains at least one nominated head
  and cuts none;14 have an accepted positive candidate. These weak nominations
  are unavailable at runtime and are not used to select actual crops.
- 4 of139 requests have no candidate satisfying that weak whole-head rule.
  Enlarging the bank alone cannot repair the other26 current choices.

The frequent retained-face overlap motivates checking larger head extents.
The existing padded-head primitive already supports25 percent; register that
additional operating-point control after the diagnosis. Preserve padding0/10
outputs. This is exploratory DEV analysis, not validation of a chosen setting.

## Measured controls

Seed31, at least one detection retained; every other detection retained or
excluded, otherwise contain:

| Local inputs / padding | Crops offered /139 | Weak cuts among crops | Accepted-reference matches |
|---|---:|---:|---:|
| Tiled YuNet /10 |135|30|23|
| Tiled + Face Landmarker /10 |135|29|24|
| All seven face outputs /10 |135|28|23|
| Tiled YuNet /25 |133|23|23|
| Tiled + Face Landmarker /25 |133|23|23|
| All seven face outputs /25 |133|20|23|

Padding adds the registered percentage of detected width/height plus two source
pixels per edge, clipped to the source. Whole people may be excluded; no rule
requires all subjects to fit. No-detection inputs force containment in this table.

Across contrastive seeds31/137/911, all-face-union/padding25 produces weak
cuts20/24/25 among133 offered crops each. Family-balanced mean population
cut0.17938205585264408, reference match0.1611365758424582 and
offer0.9281045751633986. Tiled/padding10 yields0.26289222759810993,
0.14806468041762158 and0.9411764705882352 respectively. Population cut counts
include containment as no cut; use the offered denominators when judging safety.
No significance, complete-face or artistic-quality claim follows.

All-face/padding25 seed31 repairs22 raw weak cuts, introduces3, and removes4
through containment:43-22+3-4=20.17 remaining failures have a weak-safe bank
candidate;10 have an accepted positive candidate.22 cut-head occurrences remain:
7 with no overlapping detection,12 overlapping retained detections,3 overlapping
both retained and excluded detections. More local detectors and padding do not
close the head-extent/coverage problem.

Single-output padding10 controls further show coverage limits: sparse BlazeFace
24 cuts/51 crops, UltraFace23/54, short-range BlazeFace8/19, Face Landmarker8/19.
Do not compare the smaller cut totals without these much smaller denominators.
Full tiled and interior-tiled YuNet give the same seed31 counts30/135 here;
this does not establish their outputs are identical on every request.

## Actual-pixel followup

Render the same three witnesses with all-face/padding25. All nine panes use
owned crop/resize/PNG encoding and independent exact RGB readback. Eight hashes
match the previously inspected packet; the one new distinct image was inspected
directly, along with the retained helmet failure. All nine distinct PNGs in
this packet therefore have direct review evidence, including prior exact hashes.
No training labels or excellent intervals are admitted.

- [Counter group](https://www.flickr.com/photos/projetoluzevida/15328959047),
  1:1 Maximal: the new crop retains the two selected visible heads. The previously
  partly intersected left person's head now fits, and the farther-left head
  is excluded. This is a visible repair on this one deliberately selected scene.
- [Baseball field](https://www.flickr.com/photos/mathplourde/4846668094),
  3:1 Minimal: identical pixels to the earlier padding10 filtered view. The weak
  cut flag clears but body truncation and background fragments remain.
- [Helmeted subject](https://www.flickr.com/photos/55179598@N02/5164778294),
  16:9 Maximal: identical pixels to the earlier filtered failure. The rear-facing
  head is trimmed at the bottom; this is not a visible-face-cut observation.

## Reproduction and limits

Output:`/mnt/v/output/zensally/rehearsal-face-expanded/2026-11-09/`, with
padding25 in its own subdirectory. Tools:
`composition-audit-rehearsal-faces --expanded`,
`composition-diagnose-face-candidate-misses`, and the actual-pixel renderer's
explicit`--policy union-all-faces/pad25/require1`. Exact executable arguments,
source snapshots, source links, all rows and identities are retained in the packet.

i265, two CPU threads, run-heavy3GiB cap:

| Tool | Reported duration | Peak RSS | Minimum available RAM |
|---|---:|---:|---:|
| Expanded padding0/10 audit |14s|0.66GiB|22347MiB|
| Padding10 miss diagnosis |1s|0.66GiB|23125MiB|
| Expanded padding25 audit |7s|0.64GiB|22634MiB|
| Padding25 miss diagnosis |1s|0.63GiB|23018MiB|

These are tooling resource observations, not serving latency. Existing policy
checks pass and frozen quality sources remain unchanged. Post-filter original
quality regret, complete-neural gated-choice numerical parity and detector-union
runtime costs remain unmeasured. Keep the student and operating points experimental.

The useful next distinction is source-level head uncertainty versus body framing:
test existing pose cues on the retained counter/baseball/helmet failures before
another fit, and measure gated serving behavior separately from artistic quality.
