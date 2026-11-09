# Pose head constraints and body framing

Pose-derived head bounds add limited weak head protection while reducing crop
coverage and accepted-reference matching. They do not repair the retained body
framing failure. Keep pose as an experimental cue; no model or guard is promoted.

This [registered extension](../docs/experiments/POSE_COHORT_CONSTRAINTS.md) uses
the existing72 photographic FIT/DEV sources, cached full-source Pose Lite results,
and the unchanged head-bound function from the earlier two-image experiment.
All48 detected poses on41 sources pass its strict finite/visibility/presence
gate. The other31 sources have no pose. No new API calls, teacher inference,
training labels, fitting or TEST access occur.

Add pose head bounds alone, to tiled YuNet, or to all seven cached face outputs.
Retain original face boxes. Convert normalized landmarks to native coordinates,
then apply the unchanged head estimate and the same extra25-percent policy
padding used on faces. Missing poses are unknown; they are never negative labels.
The guard permits whole heads to be excluded when selecting a person.

All prior133 model/policy summaries from the padding25 face audit replay exactly.
Scores, candidates, selected seeds, roles and source families remain fixed.

## Weak labels and containment

Seed31 on139 essential DEV requests/17 families; require one region retained,
all other regions retained or excluded, otherwise contain:

| Regions | Offered crops | Weak head cuts among crops | Accepted-reference matches |
|---|---:|---:|---:|
| All faces |133|20|23|
| Pose head bounds only |85|22|13|
| Tiled YuNet + pose bounds |130|20|21|
| All faces + pose bounds |127|17|21|

Relative to all faces, adding pose bounds repairs3 cuts while still offering
crops, introduces3 and removes3 through containment:20-3+3-3=17. Six requests
newly contain, with no new crops. Thus the net cut-count reduction for this seed
comes from containment. Seeds137/911 each repair4, introduce1 and remove3 through
containment; six requests newly contain for each seed as well.

Across the three contrastive seeds, all-faces+pose has cuts17/18/19 among127
crops each. Family-balanced mean population cut0.14902951079421667,
reference match0.1284568372803667 and crop offer0.8643790849673203.
All-faces alone gives0.17938205585264408,0.1611365758424582 and
0.9281045751633986. Reused weak labels do not establish statistical or artistic
quality, and containment must not be hidden in a population safety rate.

## Body diagnostics

Reuse the first16 registered composition features without changing their
confidence-product/0.25 gate. This differs from the strict min0.5 head-bound
confidence gate. Exclude absent-pose sources rather than interpreting zeros
as good body framing. The resulting cohort is91 requests/11 source families
per seed, pooled across the three equal-sized seed panels and family-balanced.

Among offered crops:

| Diagnostic | All faces | All faces + pose |
|---|---:|---:|
| Requests per seed |89|83|
| Head-joint retention |0.8906010839861864|0.8892356623804204|
| Whole-pose retention |0.4968516332152696|0.4748720975993704|
| Torso endpoint cuts |0.07735362280816827|0.07890594254230618|
| Shin endpoint cuts |0.14207005116096025|0.15864817001180637|

The offered populations differ. The full91-request view separately includes
full-source geometry for containment; head retention rises but shin endpoint
cuts also rise. Neither view shows consistent body-framing improvement.
These estimated joint features are geometry diagnostics, not composition grades
or proof of complete person coverage. Cached normalized candidate geometry is
unchanged; these features do not measure the resizer's rounded pixel boundary.

## Actual pixels and neural replay

Nine panes from the same three deliberately selected DEV witnesses. Eight PNG
hashes match the previous reviewed packet; the new group crop was inspected
directly. Owned crop/resize/PNG encoding passes independent exact RGB readback.

- [Counter group](https://www.flickr.com/photos/projetoluzevida/15328959047),
  1:1 Maximal: pose bounds select a tighter crop of one person, excluding the
  other visible heads. The selected head fits. The previous all-face crop kept
  two selected people. Editorial purpose is needed to choose between them;
  no universal artistic preference is admitted.
- [Baseball field](https://www.flickr.com/photos/mathplourde/4846668094),
  3:1 Minimal: unchanged pixels from the earlier face-filtered view. Foreground
  lower-body truncation and background fragments remain.
- [Helmeted subject](https://www.flickr.com/photos/55179598@N02/5164778294),
  16:9 Maximal: no cached pose exists; pixels remain identical to the earlier
  rear-head failure. Absence of pose cannot protect that head.

Complete ORT gather encoder/query inference on the cached owned256x256 image
tensors reproduces all417 seed31 decisions:139 requests each for raw, all-face
and all-face+pose policies. Zero index changes, zero crop/contain changes,
zero edge movement and zero violations of the unchanged three-grid gate.
Reconstruct native guard boxes and verify admissible counts for every request.
Bind exact model weights, graph hashes, source roles/families and detector/pose
identities. This covers neural execution plus fixed guard decisions; it excludes
image preprocessing and detector inference and certifies no artistic quality.

## Resources and limits

i265, run-heavy/two threads,3GiB cap except rendering2GiB:

| Tool | Reported duration | Peak RSS | Minimum available RAM |
|---|---:|---:|---:|
| Pose candidate audit |11s|0.65GiB|22629MiB|
| Framing diagnostics |1s|0.64GiB|23090MiB|
| Actual-pixel rendering |0s|0.06GiB|23084MiB|
| Complete-neural guarded qualification |1s|0.63GiB|23077MiB|

These are tooling observations, not deployment-memory or paired latency claims.
The [earlier runtime screen](runtime_models_2026-11-08.md) separately measured
Pose Lite at12.111ms on its own six-image workload. No current combined detector,
pose and crop latency is measured; do not add isolated timing rows.

The existing pose-bound and composition-feature checks pass. Frozen quality
files remain byte-identical to9f12692417d774653ed676d236e0f89eaafa7385.
Post-guard original quality regret remains unmeasured. Face completeness,
body framing and per-request artistic purpose remain unresolved.

Output:`/mnt/v/output/zensally/rehearsal-pose-constraints/2026-11-09/`.
Exact commands, input identities, rejected/missing geometry, scalar diagnostics,
predictions, source links, reviewed pixels and full neural checks are retained.
Keep full pose at runtime optional until its incremental benefit and combined
cost justify it; cached weak head safety alone does not justify making it mandatory.
