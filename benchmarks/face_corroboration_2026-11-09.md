# Face corroboration suitability

Person context and detector-family agreement flag the three reviewed nonhuman
false-positive scenes, but neither is safe as a hard rejection rule. Retain
unsupported proposals as uncertain. No face box, crop decision, model score,
quality threshold or training label changes; no model is promoted.

The [registered diagnostic](../docs/experiments/FACE_CORROBORATION.md) uses only
existing cached outputs. Face and EfficientDet-Lite2 person inputs match exact
owned RGB hashes and dimensions for all72 real sources and25 original DEV
sources. Frozen member hashes, roles and families are verified. No new model
inference, paid API calls or TEST access occurs.

Treat full/tiled YuNet as one family, UltraFace as one and all MediaPipe
BlazeFace/Face Landmarker paths as one conservative face-stack family. These
are engineering groups, not statistically independent classifiers. In particular,
repeated YuNet views do not supply multiple independent votes.

Another-family support means IoU at least0.1;0.3/0.5 are sensitivity controls.
Person support means the proposal center lies inside a cached Lite2 person box,
using the existing confidence0.3/max50 outputs. `Both` and `either` combine those
signals. No operating point is selected for deployment.

## Annotation-scope coverage

67 of72 sources have valid cached weak Sol annotations:45 FIT with256 head
nominations,22 DEV with84. These nominations include backs, occlusions and broad
head regions. Counts are not independent people or human-rated face recall.

Primary: weak nominated head center inside any retained face proposal.
Secondary: proposed face center inside the weak head nomination. The secondary
direction is registered separately after witness inspection; preserve the primary
results. Neither direction is identity association or complete head containment.

| Rule, IoU0.1 | FIT primary /256 | FIT secondary /256 | DEV primary /84 | DEV secondary /84 |
|---|---:|---:|---:|---:|
| All proposals |166|180|39|41|
| Another family |87|90|18|21|
| Person context |114|128|31|39|
| Both |60|62|18|21|
| Either |141|156|31|39|

Person context would lose52 of180 FIT secondary associations and2 of41 DEV
associations already present in the proposal bank. Another-family agreement
would lose90/180 and20/41, respectively. Different cohorts and unreviewed labels
prevent a general precision/recall claim. Do not infer safety from the smaller
DEV context loss while ignoring FIT.

Whole nominated head containment by raw proposals is only1 FIT and4 DEV before
corroboration; after person context it is1 FIT and0 DEV. Raw face boxes and broad
head nominations target different extents. This is not the earlier padded crop
guard measurement and cannot establish protection of actual heads.

At IoU0.3, DEV primary counts are unchanged. At0.5, family/both counts drop to17;
person/either stay31. All operating points and original uncertain proposals remain
retained. Duplicate views contribute to proposal counts, not extra head denominators.

## Reviewed negative challenge

The original [false-positive witnesses](original_guard_compatibility_2026-11-09.md)
contain no visible human face in the reviewed sources. There are3 raw proposals
on the [mushrooms](https://unsplash.com/photos/4iDTPgoZQ94),2 on the
[watch](https://unsplash.com/photos/hbTKIbuMmBI),4 on the
[honey products](https://unsplash.com/photos/KnPzm5JgDPE). These are9 model/view
proposals, not9 distinct falsely detected faces.

Every proposal lacks both another-family and person support at all three
registered IoU controls. Corroboration therefore flags them uncertain. This does
not silently remove constraints or claim a repaired crop. The three deliberately
selected scenes are Codex visual observations, not a new human precision dataset.
No source-ID exception or training label is admitted.

## Four bounded loss witnesses

Copy exact owned source PNGs with independent exact RGB readback. All four source
images are inspected directly; no importance maps, overlays or new crop labels.
Selection uses the three largest primary person-context losses, then the largest
remaining other-family loss, with distinct source IDs.

- [Crowd source3b246ffc33263bc1](https://www.flickr.com/photos/technoevangelist/16445220703): primary6→0 with person context, but secondary
  associates only1 head even before filtering. One very large short-range
  BlazeFace box spans much of the crowd and covers multiple nominated centers;
  point coverage must not be presented as six correctly detected faces.
- [Counter source9b15ccd240edae57](https://www.flickr.com/photos/projetoluzevida/15328959047): secondary4→3 with person context. The lost
  small proposal corresponds geometrically to the distant right-edge head;
  visible foreground heads remain supported. This is a missed-context risk,
  not evidence that an unsupported proposal is nonhuman.
- [Surf sourcee1401a8ae8378803](https://www.flickr.com/photos/ltobrooklyn/1354623885): secondary2→1 with person context. A small head
  near the wave has a tiled YuNet proposal without cached person support.
  Both existing head associations disappear under another-family agreement.
- [Road sourcef4261b0e8a508c69](https://www.flickr.com/photos/anoopsworld/2638938714): secondary2→0 with another-family agreement,
  while person context retains both. The source visibly contains people,
  including a profile face; absence of another face-family vote cannot be a
  non-face label.

Source links for all97 source records are preserved in the packet, including
the four displayed photographic witnesses. No complete inventory is certified.

## Verification and resources

The second association run preserves every original primary summary field.
The final replay verifies actual frozen input members and source roles/families;
all five output JSON files match the secondary run byte for byte. All original
quality code remains unchanged. No cropped output or model arithmetic changes.

i265, run-heavy3GiB/two threads: primary0s, peak-RSS0.59GiB,
minimum available23091MiB; secondary0s/0.59GiB/23080MiB;
member-verified replay1s/0.59GiB/23075MiB. Source-witness rendering under2GiB:
0s/0.03GiB/23075MiB. These are evaluation-tool observations, not detector or
deployment runtime measurements.

Output:`/mnt/v/output/zensally/face-corroboration/2026-11-09/`.
Retain all primary/secondary/replayed signals, full proposal records, thresholds,
input identities, source links and bounded visual witnesses under32MiB.

These results support using corroboration as uncertainty/context information,
not a hard deletion rule. The next fitting work should add explicit FIT-only
comparative composition supervision for body, product and story framing, while
preserving DEV/TEST roles and the frozen old-quality gate. More head thresholds
alone do not supply an artistic target.
