# Broader photographic crop ranks — 2026-11-09

Broader composition supervision improves local-teacher agreement on new people
photos, but end-to-end training regresses the old cohort. It is not a replacement
model. Whole selected faces still require independent constraints, as the rendered
witnesses show. See [registration](../docs/experiments/MIAP_BROAD_RANKS.md) and
[evidence pointer](miap_broad_ranks_2026-11-09.pointer.md).

The fixed local FocalNet teacher supplies366 screened source references:275 fit/
91 development,8,052 ratio/mode requests,7,931 informative conditions and208,314
strong preference pairs. All66 cached/full numerical witness arrays match bit
for bit before completing the cache. Source/photographer/license hyperlinks,
native integer boxes, finite raw scores and unknown conditions remain retained.
No person/detector region conditions these teacher scores. These are weak local
rankings with purpose unspecified, not human artistic or face labels. No TEST
source or paid model call enters this study.

The combined fit uses369 sources: earlier94 plus new275. Development stays
separate: earlier25 sources/21 families and new91 sources/91 families. Four arms
× three seeds use88 epochs and the original unchanged model/evaluation sources.
Every fitting source/query is visited four times. Every selected native replay
preserves all2,552 development source/query decisions and metrics,12/12.

Three-seed means on essential1:3..3:1 ratios:

| Arm | New regret ↓ | New pair agreement | Old regret ↓ | Selected epochs |
|---|---:|---:|---:|---|
| Fixed initializer | 0.585805 | 79.18% | 0.247001 | 0 |
| Frozen image | 0.538581 | 80.74% | 0.227625 | 15,12,9 |
| Trainable image | 0.446518 | 83.89% | 0.320942 | 59,36,56 |
| Geometry | 0.794031 | 74.45% | 0.708803 | 85,81,88 |
| Permuted image | 0.818155 | 71.95% | 0.326331 | 4,2,3 |

Trainable-image new-cohort regret improves by0.139288 versus the initializer,
paired95% family interval−0.196677..−0.082051. Against geometry its difference
is−0.347514 [−0.448142,−0.252387]; against permuted images−0.371638
[−0.463510,−0.282580]; against frozen images−0.092063
[−0.145072,−0.042308]. The registered new-photo agreement gate passes. Old
regret worsens from0.247001 to0.320942, so this is not overall qualification.
The frozen-encoder head has better point estimates on both cohorts; uncertainty
and composition-policy evaluation still matter. Reused development intervals
are exploratory. Source-family averaging occurs before5,000 paired bootstrap
draws, seed31; no source role or quality expectation is changed.

The selected review candidate is trainable-image seed31, weights SHA256
`1b7c683acec9ae8057f9c3fffdae5053809884ddbab18ab8fe48a2042398f31c`.
Eight development sources are fixed by count/scale cell and source hash
independently of detector and crop predictions. The viewer compares teacher,
initializer, candidate and geometry. Browser checks verify all704 crop views
across eleven ratios, exact native coordinates and clipping, with zero exceptions.
Ten full-page screenshots total25,259,014 bytes within96 MiB. All ten are visually
inspected in the active conversation; observations are machine estimates.

Visible gains include the street scene9b936a3ca0b95dd0 and canal scene62b9d14ab189fe04
at3:1 Minimal: the broader candidate keeps the people/paddlers where the earlier
model selects upper architecture/shoreline. Failures remain: selfie7c18d8978504e1d4
at3:1 cuts the main face, and some tight portrait/dancing crops intersect visible
faces. Teacher crops also show truncation. Full-source/square views are useful
context; neither teacher agreement nor face-box detector confidence approves a
crop. The owner's subsequent clarification permits choosing a whole person or
subgroup and excluding other people entirely; it does not license slicing a
visible face. The [subject-choice successor](../docs/experiments/SUBJECT_SELECTION_V4.md)
keeps this distinction explicit and preserves these original unconstrained results.

Cache producer8ffd5f30 takes11 seconds, run-heavy peak RSS **1.11 GiB**, minimum
available RAM23,604 MiB and peak load2.60. Fit producer1d1b9bc7 takes266 seconds
under4 GiB/two threads; peak RSS **2.38 GiB**, minimum available RAM21,282 MiB,
peak load5.26. CUDA peak allocated/reserved bytes888,235,008 /1,021,313,024 exclude
some driver accounting; another GPU task is present, and no saturation or isolated
speed benchmark is claimed. Exact commands/GNU time output remain retained.
The gallery build reports peak RSS0.85 GiB and browser check0.23 GiB.
