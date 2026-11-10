# Pinned semantic-feature pair suitability

Richer pretrained features do not establish a suitable crop model in this
screen. Global DINO features produce the same accepted-reference metrics as
geometry; spatial features change1472/1584 emitted endpoints but still match
only14/139 accepted essential DEV references. No student transfer, runtime
export or model promotion follows from this result.

## Inputs and fixed controls

The [registered screen](../docs/experiments/SEMANTIC_PAIR_FEATURES.md) reuses
the exact72 owned448-square gaze RGB inputs and226 pinned local teacher
members. The existing [gaze-teacher setup](../docs/experiments/REAL_ATTENTION_CONTROL.md)
contains the DINOv2 ViT-B/14 backbone. Its class token supplies768 global
features;4x4 average pooling supplies12288 spatial values. Gaze heads are
bypassed. All72 backbone inferences finish, with no download, API call or saved
diagnostic maps. Descriptor cache SHA256:
`dcab2e23a47cb57fa36b00c8592f7b0668d52a817f1a848882c5fbb61e8c68e1`.

Existing gaze inputs stretch the source to448-square. This screen does not
evaluate aspect-preserving DINO preprocessing or every semantic representation.
Source/family roles and RGB identities remain unchanged; no new crops or labels.

Both descriptors enter the same eight-PC/29-column weighted ridge control,
with per-fold normalization/PCA excluding each held FIT family. All29 folds
remain recorded;28 essential folds select lambda among the same nine values.
The99 FIT weak endpoint pairs and36 DEV paired references remain unchanged.
The prior geometry baseline is the frozen318f88d1 packet, not a retuned model.

## Essential DEV outcomes

| Model | FIT-selected lambda | Paired edge error | Accepted matches | Head cuts | Heads retained |
|---|---:|---:|---:|---:|---:|
|Frozen geometry|10000|22.873441860438835|14/139|55|95|
|DINO global PCA8|10000|22.873441860438835|14/139|55|95|
|DINO spatial PCA8|10|22.730732212569215|14/139|54|94|

Global changes15 of1584 complete FIT/DEV emitted endpoints, with no changes to
these accepted DEV metrics. Spatial changes1472 complete endpoints. Both still
emit all85 explicit teacher-contain essential DEV requests; there is no offer
head. Three accepted secondary endpoints are too few for an extreme-ratio
quality claim: both match zero, global cuts one and spatial cuts zero.

For spatial versus geometry, family-balanced paired error difference is
−0.14270964786961873, bootstrap95% interval
[−0.41990008859129135,+0.15118130176723418] across14 families/36 requests.
The nominated-head cut difference is−0.025210084033613446, interval
[−0.0742296918767507,+0.008403361344537815] across17 families/139 accepted
endpoints. Every accepted reference-match flag is unchanged for both models.
Bootstrap uses2000 repeats/seed31 on reused DEV, not independent TEST evidence.

This rejects these compressed descriptors with this small weak-pair fitter as
a deployment choice. It does not establish that DINO has no composition value,
that other spatial/subject-conditioned models fail, or that all crops outside
the incomplete reference set are artistically poor. Exact nested geometry is
not sufficient to learn the intended subject/story or protect every real face.

## Numerical and resource evidence

FP64 reference versus CPU/GPU FP32 produces6336 native rectangle comparisons,
zero changed rectangles and zero violations of the original three-grid edge
gate. Bounds, exact aspect, maximum-area outer and strict nesting remain valid.
These are descriptor/head checks, not a full encoder deployment comparison.

Run-heavy on i265 under4GiB/two CPU threads and20-percent GPU allowance:

| Stage | Duration | Peak RSS | Minimum available RAM | Peak load |
|---|---:|---:|---:|---:|
|72 backbone descriptors|5s|1.43GiB|22219MiB|1.60|
|Both FIT-family-selected models|1s|1.42GiB|23112MiB|1.37|
|Family analysis|1s|0.60GiB|23126MiB|1.44|

Extraction Torch CUDA peak allocated412032512 bytes, reserved463470592 bytes.
These are research-run resources, not isolated serving latency or memory.
The expensive teacher is not removed by fitting a small linear output head.

All descriptors, source/teacher hashes, fold/lambda scores, fitted models,
predictions, numerical rows and logs are retained under64MiB. No paid API
spending, synthetic data, intermediate labels, TEST access or changes to the
five original quality evaluator files. Local on i265:
`/mnt/v/output/zensally/semantic-pair-features/2026-11-09/`.
NAS: `/mnt/tower/output/zensally/semantic-pair-features/2026-11-09/`.
Frozen18-member packet,6694059 bytes before manifest, independently verified
with complete NAS membership/hash readback. Manifest SHA256
`e008340b13edabadda27371d0fe74bbf835c407533dfd1717887b07f82f00edc`.
