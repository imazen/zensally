# Ranked ROI geometry pilot

A separate deterministic crop path replaces learned crop scores with explicit
subject choices and exact integer geometry. Previous research remains unchanged
at review/fixed-w24-depth /18107fae. New code is on review/roi-ranked-geometry.
The earlier saliency placement fix is documented in
[w24 crop placement](w24_crop_placement_blur_2026-11-07.md); no claim is made that
an earlier learned composition model had passed artistic qualification.

## Behavior

Reuse cached EfficientDet-Lite2 object/person boxes and cached face/pose boxes.
Rank whole-object candidates by confidence times square-root source area, with
same-class IoU0.7 deduplication. A union of detected people is an alternative,
not a requirement. Head-context portrait-detail candidates are fallback options
and explicitly labeled; they do not represent whole-person retention.

For each ROI the solver searches largest to smallest exact-ratio size, with an
exact origin sweep containing that ROI and intersecting no supplied face box.
Placement preserves the subject's relative source position, not image-center
placement. A tighter child preserves the chosen ROI plus every face retained
by the outer crop, with a10-percent context margin clipped to the outer bounds.
Every intermediate nested rectangle then preserves those supplied faces.
When no distinct tight child fits, return a single frame explicitly. When no
subject fits, return contain/padding, never silently fall back to center-crop.
The tool returns alternatives so callers can explicitly choose another subject.

This protects supplied boxes only. Detector misses, false positives and semantic
subject ranking remain unqualified. The heuristic does not evaluate gaze,
sharpness, text/logo visibility, products outside the detector vocabulary or
artistic composition. There is no learned subject-ranker, Rust runtime export
or change to the crate's default serving path. No new dependency or paid call.

## Finite validation

Independent exhaustive pixel-grid origin enumeration matches the event sweep
on300 randomized small scenes. A separate100-scene enumeration verifies largest
feasible size and nested face membership. Selection tests cover choosing one
person while excluding another, impossible geometry and explicit single-frame
results. These prove the specified geometric behavior, not detector accuracy.
Run `just composition-roi-geometry-test`.

Existing16-source DEV panel, nine ratios1:1/4:3/3:4/16:9/9:16/2:1/1:2/3:1/1:3:

|Outcome|Requests|
|---|---:|
|Whole detected object/group selected|137|
|Portrait-detail fallback|3|
|Contain/pad required|4|
|Offered total|140/144|
|Strict nested pairs|97|
|Offered single frames|43|
|Selected requests cutting supplied faces|0/140|
|Center-crop requests cutting supplied faces|42/144|

The face-cut comparison includes four ROI abstentions. It is proposal geometry,
not independent real-face or human preference evidence. These are inherited DEV
sources already used in research; no TEST opened or roles reassigned.
Only21/140 outer crops are within1-percent of source center on both axes;
median maximum-axis center displacement is0.0967787020726574 of source size.
Noncentral placement alone does not imply artistic quality.

A UI-only regeneration retains identical numerical summaries; initial artifacts
remain in the adjacent2026-11-10-initial folder. Both local runs complete with
run-heavy peak RSS0.04GiB and elapsed0s at its integer-second resolution, capped
at2GiB/one thread. Final min available20400MiB/peak load1.91. These are local
research-job observations; no per-image latency claim follows.

Gallery shows source, plain center, previous learned Minimal and ROI Minimal/
Maximal; expand alternatives to see second/third choices. Global ratio selector
shows16 sources at a time. Exact source previews are clipped by SVG viewBox;
no third-party pixel generation. HTTP index plus all16 source PNGs match local
bytes. No gallery images were pulled into model context for visual judging.
Browser automation was unavailable locally; no browser-rendering claim is made.

## Use and artifacts

`just composition-roi-crop < request.json` reads width, height, ratio,
regions[{box,confidence,kind}] and faces[] from JSON stdin. Coordinates are
integer source-pixel left/top/right/bottom. Output contains a result per ranked
ROI, explicit unavailable/single-frame/nested-pair status and face exclusions.
It requires an upstream detector or caller-supplied ROIs.

Full pilot recipe: `just composition-roi-geometry-pilot PARENT OBJECTS OUTPUT`.
Parent is the frozen broader-people-risk/2026-11-09 packet, object input is
miap-detector-screen/2026-11-09/full/tf-efficientdet_lite2/results.json.

Artifacts checked on i265:
`/mnt/v/output/zensally/roi-ranked-geometry/2026-11-10/`.
Full predictions, source attribution inherited from the parent, source previews,
summary, identity and resource logs are retained. Earlier full research is saved;
this pilot adds a reviewable ROI-and-geometry implementation without retraining.

Frozen final packet:22 members; manifest SHA256
`81fa9cde89b9fa31dd82a8ef6c226bf34cb1f55d96911ea8ae411c80e5a9ae0e`.
All22 members independently verified on the NAS mirror on dev at
`/mnt/tower/output/zensally/roi-ranked-geometry/2026-11-10/`.
The19-member initial packet is also mirrored and independently verified.
JSON-stdin CLI smoke check selects a right-hand subject and excludes the
left-hand face. Original research branch18107fae remains remotely preserved
and its CI run42821391196 passed. New tooling is locally verified; no separate
CI result is claimed for the new branch.
