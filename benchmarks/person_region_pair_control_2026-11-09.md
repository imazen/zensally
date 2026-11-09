# Automatic full-person region pair control

Cached full-person boxes lower paired coordinate error nominally but do not
qualify a crop model. Both controls still cut55/139 nominated heads and emit
all85 essential DEV requests labeled contain. No runtime export or promotion.

## Fixed inputs and selection

[Registration](../docs/experiments/PERSON_REGION_PAIR_CONTROL.md) reuses
EfficientDet-Lite2 confidence0.3/max50 results. Exact source RGB and dimensions
bind72 sources,70 person-positive. Existing full-person projection adds10-percent
padding and clips to visible source bounds. The47-value descriptor has eight
area-sorted boxes and seven global values. All-person descriptors omit26 slots
in total; largest-person descriptors omit none. Zero detections do not certify
absence. No chosen-subject ID, annotation or crop window enters the descriptor.
Cache SHA256: cd400d66689630e1d6107064ee019b5a920310b6d22effa7edafc87c622244ea.

The same PCA8/29-column paired ridge uses99 FIT requests,29 family folds,
28 primary folds and nine lambdas. Held families are excluded from normalization
and PCA. DEV never selects lambda. Geometry remains frozen at318f88d1.
Matching-aspect outer Minimal is maximum-area; inner Maximal is strictly nested.
No excellent endpoint/intermediate labels, new inference, API spending or TEST.

| Model | FIT-selected lambda | Paired DEV edge error | Matches | Head cuts | Heads retained |
|---|---:|---:|---:|---:|---:|
|Frozen geometry|10000|22.873441860438835|14/139|55|95|
|All persons PCA8|1|21.424696855364157|16/139|55|95|
|Largest person PCA8|0.1|20.538950150365334|15/139|55|95|

All-person versus geometry paired error difference is−1.448745005074673,
family-bootstrap95% interval[−3.8932311370639656,+0.7322298000065931].
Largest-person difference is−2.3344917100734994,
interval[−5.819203345794734,+1.320811604070058]. Both cover14 families/36
requests. All eight family-balanced metric intervals cross zero; bootstrap
uses2000 repeats/seed31 on reused DEV, not independent TEST.
All-person changes1538/1584 emitted endpoints; largest-person changes1544.
Secondary DEV has only three accepted endpoints: neither matches any, with
2 and0 cuts respectively. Zero cuts can accompany zero retained heads.

## Actual-pixel failures and numerical evidence

All14 panes on two fresh witness sources were directly inspected. Playground
sourcea1d2f44a45a19cbd at3:1: geometry clips the child at the bottom; all-person
still truncates the body; largest-person recovers the child but changes the
inflatable structure framing. Work-site source1b8020d5a8d620a5 at9:16: geometry
and all-person omit the worker; largest-person introduces a partial helmet/head
at the right boundary. A weak cut repair is not artistic certification, and
moving toward a person without a safety constraint can create a new cut.

All6336 FP64 versus CPU/GPU FP32 native rectangle comparisons replay exactly,
with zero three-grid violations. Bounds, aspect, maximum-area outer and strict
nesting pass. These checks cover the cached-descriptor head, not detector
execution or a complete runtime deployment. PNG outputs use owned crop/resize/
encoding and independent exact readback.

Run-heavy on i265, two threads except analysis/gallery one thread:

| Stage | Duration | Peak RSS | Minimum available RAM | Peak load |
|---|---:|---:|---:|---:|
|Descriptor preparation,2GiB cap|1s|0.56GiB|23834MiB|2.35|
|Both fits,4GiB cap|1s|1.14GiB|25214MiB|1.64|
|Family analysis,2GiB cap|0s|0.61GiB|25211MiB|1.76|
|Gallery,2GiB cap|1s|0.60GiB|25212MiB|1.22|

These are research-run resource observations, not serving latency. Models,
fold scores, predictions, source identities, numerical results, analysis, logs
and14 PNG witnesses are retained under16MiB. Original evaluator files, roles
and quality gates remain unchanged. No model promotion.

Local on i265: `/mnt/v/output/zensally/person-region-pair-control/2026-11-09/`.
NAS: `/mnt/tower/output/zensally/person-region-pair-control/2026-11-09/`.

Frozen59-member packet,7162517 bytes before manifest, complete NAS membership
and SHA256 readback verified. Manifest SHA256:
`a16fbdd8f9dfbf24bf62c459621daf3a87c1a4b1bade79a4317ae44372b4190e`. Advertised gallery HTML and every PNG match local bytes over HTTP.
