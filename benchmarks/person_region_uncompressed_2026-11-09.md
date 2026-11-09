# Uncompressed person-coordinate control

Keeping all47 automatic person descriptor values does not resolve suitability.
Largest-person paired DEV edge error is20.421618055285936, but54/139 accepted
essential endpoints still cut nominated heads. All-person has53 cuts. Neither
is promoted or exported.

The [registered comparison](../docs/experiments/PERSON_REGION_UNCOMPRESSED.md)
changes PCA8 to FIT-normalized identity components:146 design columns instead
of29. Source-only detections,99 FIT targets,29 folds/28 primary folds, all nine
lambdas, source weights,36 paired DEV requests and all quality gates stay fixed.
Held-family transforms never use the held family. No new labels, detector/VLM
inference, API calls or TEST. These remain weak-reference composition metrics.

| Representation | FIT-selected lambda | Paired DEV edge error | Matches | Head cuts | Heads retained |
|---|---:|---:|---:|---:|---:|
|Frozen geometry|10000|22.873441860438835|14/139|55|95|
|All persons PCA8|1|21.424696855364157|16/139|55|95|
|All persons47|10|22.608753194851875|16/139|53|95|
|Largest person PCA8|0.1|20.538950150365334|15/139|55|95|
|Largest person47|0.1|20.421618055285936|15/139|54|97|

Both raw models emit432 essential DEV endpoints, including all85 requests
labeled contain; there is no offer head. All-person changes1532/1584 complete
endpoints versus geometry; largest-person changes1543. Three secondary accepted
DEV endpoints yield no matches: all-person has zero cuts/three heads retained;
largest-person has zero cuts/zero retained. These tiny counts do not establish
extreme-ratio quality.

Family-balanced paired-error difference versus geometry is−0.2646886655869568,
95% interval[−1.6064790661307404,+1.1147266596245664] for all-person;
−2.4518238051528956,[−5.966202166813313,+1.290719356871166] for largest-person.
All eight family metric intervals cross zero. Bootstrap uses2000 repeats/seed31
on reused DEV, not independent TEST. Compression is not established as the
main failure cause; larger linear region features still do not learn safe
subject/story selection from these weak targets.

All6336 FP64 versus CPU/GPU FP32 native rectangle comparisons replay exactly,
with zero three-grid violations. Native bounds, aspect, maximum-area Minimal
and strict nested Maximal pass. This covers the cached head, not a full detector
or runtime. No new gallery was generated for these still-unsuitable variants;
actual-pixel conclusions remain limited to the preceding PCA8 witnesses.

Run-heavy i265: fitting under4GiB/two threads/20-percent GPU finishes1s,
peak RSS1.14GiB, minimum available25103MiB, peak load1.39. Family analysis under
2GiB/one thread finishes1s, peak RSS0.61GiB, minimum available25067MiB, load1.45.
These are research-run observations, not serving latency.

Local on i265: `/mnt/v/output/zensally/person-region-uncompressed/2026-11-09/`.
NAS: `/mnt/tower/output/zensally/person-region-uncompressed/2026-11-09/`.
Models, folds, predictions, precision rows and logs are retained under16MiB.

Frozen13-member packet,1834740 bytes before manifest; complete NAS membership
and hashes verified. Manifest SHA256:
`48be1aa863bdabaf26c7a17cc5eee4e69362c7983080ccc38e9bc1e0a269d299`.
