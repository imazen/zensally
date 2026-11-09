# Region-conditioned direct crop controls

All45 registered fits complete. None qualifies for promotion. Appending local
face/pose positions does not reliably improve finite weak-window matching over
the matched no-region input. This is a negative suitability result, not a
finding that pose has no compositional value.

The [registration](../docs/experiments/REGION_CONTEXT_SCOUT.md) fixes72 real
families,48 FIT/24 DEV, the C002 encoder,108 epochs and three seeds. Five
region inputs each have image, geometry-only and permuted-image controls.
All heads have67,399 parameters and the same47 added source descriptors.
FIT-only normalization and role/content-shape donor restrictions remain fixed.
No synthetic gradients, paid calls or TEST inference are added.

Means below balance17 DEV families with139 accepted essential queries, then
average seeds. Edge error uses source-normalized128-grid units. Nominated-head
cuts concern unreviewed Sol regions, including backs and occlusions.

| Input | Edge error | Best of two match | Either proposal cuts a nominated head |
|---|---:|---:|---:|
| No regions |16.7575|15.0150%|53.9780%|
| Local face |16.6568|15.9207%|55.3046%|
| Local pose |16.5463|15.1426%|52.2724%|
| Both |16.6650|16.7221%|53.0822%|
| Supplied Sol heads |16.4105|15.3698%|54.5623%|

Both local cues improve best2 matching by1.7071 percentage points, with paired
family bootstrap95% interval[-1.2544,5.2711]. Its edge-error change is-0.0925,
interval[-0.4117,0.2323]. All four matched-reference improvement intervals
include zero. The supplied-head descriptor is a bounded same-call input with89
omitted slots across sources; it is not complete localization or an ideal oracle.
Finite accepted windows are not an exhaustive list of acceptable compositions.

All45 selected native replays are exact. Repeated geometry controls are
bit-identical across variants for each seed. Across142,560 CPU/GPU rectangle
comparisons, maximum edge movement is0.1887894 grid pixels, below the unchanged
three-grid-pixel gate; not every native rectangle is bit-identical. The old
quality gate and exported deployment runtime remain unmeasured for these heads.

Compact checkpoint storage preserves epoch-zero, selected and terminal
optimizer/RNG states, histories and predictions. A detached CPU snapshot holds
only the current best state until the fit ends; default full checkpoint behavior
is unchanged. A warmed AdamW regression verifies snapshot independence.

Four independent variant jobs run concurrently under4GiB/two-thread caps;
run-heavy reports42–43s each and peak RSS1.60–1.62GiB, minimum available RAM
17,358MiB. The final oracle job takes13s, peak RSS1.63GiB. These include IO and
shared GPU scheduling; they establish neither isolated speed nor saturation.

Evidence: `/mnt/v/output/zensally/region-context-scout/2026-11-09/`.
Inspect the native summaries and complete logs for source/checkpoint identities.
The next useful distinction is explicit geometric placement versus learned
composition; [the placement controls](subject_origin_projection_2026-11-09.md)
keep those effects separate.
