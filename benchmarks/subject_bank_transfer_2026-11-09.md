# Fixed bank-head transfer to Sol — 2026-11-09

The candidate-bank gain carries over to cached Sol annotations, but raw crops
remain unsafe. All six Terra-trained frozen heads are fixed before this check;
no checkpoint is reselected, no API request or gradient is added, and TEST stays
closed. This is another annotation view of reused DEV photos, not an independent
human test. Model inputs are bit-identical between teacher panels within each bank.

Sol accepts525 weak-window requests across the real72 sources. Broad-Maximal
represents231 and hybrid331; their contrastive counts are133 and258. Original
coordinates, source families and48/24 FIT/DEV roles remain unchanged.

Accepted-essential evaluation uses17 DEV families/139 requests. Means average
the three fixed frozen heads, balanced by family.

| Bank | Initial match | Learned match | Initial nominated-head cut | Learned cut |
|---|---:|---:|---:|---:|
| Broad-Maximal |11.7066%|15.4906%|41.6773%|36.9001%|
| Hybrid |11.1137%|15.5140%|40.8395%|33.2909%|

Broad-Maximal learning: match +3.7841 percentage points, paired95% family
bootstrap CI[+0.8149,+7.5682]; cut -4.7772 points, CI[-10.2245,-0.1273].
Hybrid learning: match +4.4003 points, CI[+1.1204,+8.5331]; cut -7.5486 points,
CI[-14.1923,-1.3010]. Five thousand resamples, seed31, average seeds before
resampling. These intervals remain conditional on reused DEV data and unreviewed
finite windows. Nominated heads include backs/occlusions and do not certify faces.

The separate all-requested denominator includes teacher containment:22 Sol DEV
families/308 essential requests. Hybrid raw cuts change43.1818%→37.9870%.
A supplied-head-only constraint requires at least one whole nominated head and
no partially intersected nominated heads; it offers90.5844% of requests with
zero such cuts by construction. This may wholly exclude other people. It does
not establish actual detection coverage, alter shipping guards or promote a model.

Two cache jobs each take2s with run-heavy peak RSS0.90GiB. The fixed-head audit
takes2s,1.40GiB. The first audit stopped on NumPy integer scalar types before
producing results; the corrected caller converts to Python integers. The strict
geometry validator and coordinates remain unchanged; the failed log/source is
retained. Runtime costs of automatic localization remain separate/unmeasured.

[Bank fitting and visual failures](subject_bank_controls_2026-11-09.md).
[Direct competitor](direct_crop_scout_2026-11-09.md).
