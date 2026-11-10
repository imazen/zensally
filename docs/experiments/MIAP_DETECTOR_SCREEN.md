# Fixed detector suitability on annotated people

Before reading detector quality on the new corpus, freeze the five existing
EfficientDet checkpoint identities and identical384 acquired fit/development
inputs. Use the unchanged MediaPipe CPU task confidence0.3/maximum50 results.
Retain all1,920 inference outputs; do not time this as a runtime benchmark or
generate mask maps. Host bound1GiB/two threads; all source roles stay fixed.

Complete content/target observations independently of predictions before computing
the primary score. Primary is non-group annotated person recall on development
sources marked photographic/candidate-person-target and free of relationship
flags. This is recall against the supplied annotations, not complete people or
face recall. Group boxes are not individual targets; do not claim official AP or
precision without a separate group protocol. Match person predictions in descending
confidence to the highest-IoU unmatched individual box atIoU0.5. Report all sources
secondarily, role and normalized individual-area strata (<0.02/<0.2/otherwise),
occlusion/truncation flags and source counts. Quarantines/exclusions remain visible.

The manual source mask must not be chosen from model successes. Models were fixed
before acquisition. Existing TEST pixels stay closed and no teacher output is
admitted as a crop/face label. The earlier one-core timing/counter studies remain
separate; this experiment tests localization suitability on broader people data.
No shipping detector switch, artistic-quality or complete-face claim follows.

The independent content mask is frozen at SHA256
`e443d102d0069d89197c81e4c0032eff2335e4d9b85f01503a2248ed48f88f80`
before scoring. Normalize raw predictions by working width/height without clipping
their bounds to improve overlap. Pairwise recall differences use5000 seed31 source-
component bootstrap draws; one source per component makes that the family unit.
Retain every matched/unmatched individual annotation and all group counts. Tests
require one-to-one matching, exclusion of group targets, correct non-square axis
normalization and no artificial improvement from clipping out-of-image boxes.
