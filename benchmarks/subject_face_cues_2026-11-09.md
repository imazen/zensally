# Automatic face/pose constraints on the real72 panel

Local cues help avoid some nominated-head cuts but remain incomplete. None is
promoted as a complete face guard. All288 calls finish with fixed checkpoints,
owned RGB and unchanged48 FIT/24 DEV source families. No new labels, API calls
or TEST inference. No segmentation/metric maps are produced.

| Cue | Sources with proposals /72 | Proposals | Weight-file bytes |
|---|---:|---:|---:|
| Short BlazeFace |13|16|229,746|
| Sparse full-range BlazeFace |30|53|676,746|
| Pose Lite padded head estimate |41|48|5,777,746|
| Face Landmarker contour bounds |9|11|3,758,596|

These are model outputs, not person/face recall. Pose bounds reuse the existing
facial/shoulder confidence0.5 filter and padded head geometry; all48 outputs
pass that filter here. Face Landmarker is bounded at16 faces and Pose Lite at8
poses. Face detections keep confidence0.5. Normalized proposals are clipped
explicitly to source bounds; raw model outputs remain retained.

On67 structurally valid Sol sources, its340 nominated head centers include
backs/occlusions and incomplete/unreviewed geometry. One-to-one point-inside-box
matching finds16 centers for short BlazeFace,45 sparse full-range,36 pose heads
and11 landmark bounds. This measures annotation-scope overlap, not human face
recall, precision or whole-head containment. The fixed DEV subset has84 such
nominations on22 sources; matched counts are5/14/13/4 respectively.

## Actual crop constraint behavior

Apply each cue to the three already-fixed Terra-trained hybrid heads. A legal
crop contains at least one returned region and partially intersects none of
that cue's returned regions; otherwise return contain. Other people may be
wholly excluded. Do not set the shipping complete-coverage flag.

All-requested essential evaluation includes teacher containment,22 Sol DEV
families and308 requests. Means balance families and average three seeds.
The cut column is the fraction of all requests with a selected crop cutting
a Sol-nominated head; it is not the fraction conditional on offering a crop.

| Policy | Crop offered | Sol-nominated head cut | Reference match |
|---|---:|---:|---:|
| Raw learned choice |100%|37.9870%|7.9004%|
| Short BlazeFace constraint |13.9610%|8.7662%|1.6234%|
| Sparse full-range constraint |34.0909%|16.8831%|3.4632%|
| Pose padded-head constraint |49.6753%|15.4762%|6.3853%|
| Face Landmarker constraint |11.3636%|6.3853%|1.6234%|
| Supplied Sol-head constraint |90.5844%|0 by construction|9.8485%|

Local constraints abstain frequently and still cut unreturned nominated heads.
The supplied-head row is an offline geometric control with unreviewed ROI inputs,
not an automatic runtime result. A larger positive-source count does not establish
better composition. The unresolved problem is coverage and subject/constraint
selection; these cues remain candidates for bounded feature ablations.

All four local evaluations together take5s with run-heavy peak RSS0.24GiB
under a3GiB/two-thread cap. The fixed-model crop constraint audit takes2s,
peak RSS1.30GiB. These include initialization and IO, and are not per-image
deployment timings. Earlier [runtime profiles](runtime_models_2026-11-08.md)
remain separate observations on another cohort. No runtime-cost sum or new
latency claim is inferred.
