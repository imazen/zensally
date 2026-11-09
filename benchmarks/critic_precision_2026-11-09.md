# Unquantized local critic qualification

The pinned PhotoFramer critic fails its registered suitability screen after
removing quantization. No new scores become labels and the wider panel is not
admitted under this protocol. This measures agreement with six cached weak
preferences, not human artistic correctness.

## Fixed comparison

[Registration](../docs/experiments/CRITIC_PRECISION_QUALIFICATION.md) pins
PhotoFramer revision75d7da9ffcbe794611eb69ff552bcf98de220cf4, asset manifest
aab4ad8d and original16-image plan51b78892. Four synthetic source probes and
six mixed-content crop pairs reuse identical owned RGB, prompt, grid, input
lengths, seed31, SDPA and512-token bound. The historical pair labels remain
training-ineligible. All weights are local; no downloads, API calls or TEST.

| Precision | Parsed | Truncated | Correct pair orders | Ties | Reversals | Margin at least0.05 |
|---|---:|---:|---:|---:|---:|---:|
|Unquantized BF16|16/16|0|4/6|1|1|3/6|
|NF4|16/16|0|3/6|2|1|2/6|
|NF4 with BF16 vision|16/16|0|3/6|2|1|2/6|

The unchanged qualification requires all six correctly ordered with at least
0.05 separation. BF16 changes three pair orders versus NF4 and one versus
NF4/BF16 vision. Maximum absolute rating changes are0.17 and0.13 respectively.
Quantization affects results but does not explain all qualification failures.

All18 calibration PNGs were inspected with whole-source context. The panel
includes architecture, flower, food, portrait, a fisheye scene and a planet
image; it is not a broad people/ecommerce evaluation. Several pairs differ
subtly and could both be usable. A weak-label mismatch alone does not prove an
artistic error. The critic scores crops individually without the original
source, target ratio, purpose or selected story as context.

## Memory repair and resources

Initial12GiB GPU/8GiB CPU resident-weight placement passes one probe, then
fails after four responses: a1.02GiB transient weight transfer exceeds the
80-percent runtime GPU cap. A separate10GiB GPU/10GiB CPU placement completes
all16 under the same16GiB host cap. CPU weight residency is not CPU execution;
placement hooks record the execution devices. No disk offload or4-bit parameters
are used in the reference. The four shared prefix responses, ratings and token
counts replay exactly after the resource repair. Failed artifacts are retained.

Run-heavy on i265, two CPU threads unless noted:

| Stage | Duration | Peak RSS | Minimum available RAM | Peak load |
|---|---:|---:|---:|---:|
|One-source probe|77s|6.69GiB|21940MiB|2.19|
|Initial expansion, CUDA OOM|190s|8.05GiB|22996MiB|2.52|
|Repaired complete16|1203s|12.57GiB|23508MiB|2.89|
|Comparison|3s|0.81GiB|24612MiB|1.64|
|Pixel gallery, one thread|1s|0.11GiB|23900MiB|2.42|

These are research-run resources, not isolated serving comparisons. Existing
16.6GB model weights remain external hash-bound references. Complete responses,
input identities, placements, failure record, comparisons, logs and bounded
witnesses are retained under40MiB. No student fitting, runtime export or quality
gate changes follow.

Local on i265: `/mnt/v/output/zensally/critic-precision/2026-11-09/`.
NAS: `/mnt/tower/output/zensally/critic-precision/2026-11-09/`.
