# Source person-region pair control

Registered while the independent unquantized critic screen runs. Source-grid,
attention-summary and semantic PCA controls fail suitability; prior region
controls used face/pose boxes. Test cached full-person boxes as a different
automatic source geometry signal, not chosen-subject IDs or crop coordinates.

Reuse the pinned EfficientDet-Lite2 results and exact source identity checks at
confidence0.3/max50. Reuse the existing person_views full-person context
projection, with its registered10-percent padding and visible-source clipping.
Two47-value descriptors reuse region_context: all person context boxes, and only
the largest visible person context box with lexicographic coordinate tie break.
The descriptor retains eight slots plus global extent/centroid/count; record
omitted slots. Nondetections stay zero regions, not certified person absence.
No source annotation, selected subject ID, crop window, teacher decision or
target is used to create these descriptors.

Use the exact same eight-PC/29-column fractional paired ridge head,99 FIT weak
pair requests,29 family folds and nine lambdas as the frozen318f88d1 control.
Only28 essential folds choose lambda. Per-fold normalization/PCA exclude the
held family. DEV reports never select lambda. Compare to frozen geometry and
source-grid controls; no new endpoint or interval quality labels. The detector
already ran locally; no fresh detector, VLM or paid API calls are added here.

Report all139 accepted essential DEV endpoint matches/head cuts and36 paired
request errors, output coverage on explicit-contain requests, and the unchanged
FP64 versus CPU/GPU FP32 three-grid gate. These source-only person cues do not
certify all heads or resolve subject choice. No production API, export, TEST or
original quality-gate change. Dataset preparation2GiB/two CPU threads. Fitting
4GiB/two threads and20-percent GPU is admitted only after the large critic
process is verified terminal.16MiB artifact budget.
