# Cached attention supervision audit

The fixed72-photo panel has no eligible candidate marked both reference-positive
and explicit head-cut negative. Candidate support remains incomplete:132/364
essential FIT requests and57/139 DEV requests have valid accepted references
but no eligible candidate matching them. This is a bank-support limitation,
not evidence that every unmatched crop is artistically bad. Preserve the
unknown-candidate policy and original losses/thresholds.

| Essential ratios1:3 through3:1 | FIT | DEV |
|---|---:|---:|
| Sources |48|24|
| Queries with accepted references |364|139|
| Queries lacking an eligible reference-positive |132|57|
| Frozen C002 selected reference-positive |58|16|
| Frozen C002 selected explicit nominated-head cut |73|52|
| Queries with varying pose features |214|85|
| Queries with varying gaze features |317|127|
| Queries with varying scanpath features |353|132|

Feature variation considers only eligible candidates and valid-reference
queries. Source-level counts/confidence are constant within a query and cannot
change its ordering through the registered linear scalar extension. The14
varying pose columns, six gaze columns and eight scanpath columns include body
segment cuts, head/target retention, looking room and path transitions. These
are model predictions, not measured human attention or aesthetic benefit.

The audit loads the exact a2120976 attention cache, trains nothing, makes no API
calls and keeps TEST closed. All-ratio counts and per-block details remain in
`/mnt/v/output/zensally/attention-supervision-audit/2026-11-09/summary.json`.
The successful run-heavy invocation takes1s, peak RSS0.66GiB, minimum available
23056MiB, under a2GiB/two-thread cap. This is cache analysis, not serving cost.
An earlier attempt failed before writing output because its parent directory
did not exist; its log is retained separately. Parent creation is corrected.

Reproduce with `just composition-audit-attention-supervision PYTHON CACHE OUTPUT`.
The fifteen registered fits remain unstarted while the label-quality audit is
unresolved. No gate is weakened and no crop model is promoted.
