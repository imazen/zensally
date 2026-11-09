# FIT-only nested pairs and visual preference pilot

The constructed options satisfy the requested pair geometry: matching target
aspect, maximum-area outer Minimal, strictly smaller inner Maximal entirely
inside it. Visual review supplies six weak endpoint comparisons, not excellent
interval labels. A bounded adaptation fits lower preference loss but establishes
no artistic or head-safety improvement. No model is promoted or API spending added.

## Review data and geometry

Follow the [review protocol](../docs/experiments/FIT_PAIR_REVIEW.md). Hash-select
four existing FIT families, two with one weak head nomination, one with two,
one with three or more. These are nomination strata, not actual face counts.
Request3:1 and1:3 for each: eight requests, with no DEV/TEST role changes.

The original24 proposed pairs over-select similar framing. Retain them and add
a separate tighter variant with seven extra pairs whose inner area is36–64
percent of the outer area. The eighth request has no additional distinct pair
meeting that sampling condition. The original controls remain pixel-identical.
Both packets together contain55 proposed pair instances; these are correlated
options on eight requests, not55 independent examples.

All55 pairs pass exact native-integer aspect, source bounds, maximum outer area,
strict nesting and strictly smaller inner-area checks.150 PNG files contain55
distinct hashes; every distinct image is directly inspected. Owned resizing/
cropping/PNG encoding passes independent exact RGB readback. The eight raw
student Minimal/Maximal references are nested in only three requests. Pair
geometry is an additional structural problem, separate from artistic ranking.

The offline proposals use cached weak nominated heads to curate candidate
options. They do not establish complete face coverage or runtime availability.
No outside-region negative labels or acceptable continuous intervals are inferred.

Source links and scoped observations:

- [Performer8ad2a46a2ad698cf](https://www.flickr.com/photos/sfslim/15470003710): prefer more of the open-hand gesture in the wide
  frame and retain the hand in the tall outer frame. Several curated pair
  origins sacrifice that gesture despite preserving the nominated head.
- [Tunnel5e10be4e4138b512](https://www.flickr.com/photos/ianv/3441464737): a closer wide arch/silhouette view is a purpose-sensitive
  weak preference; in the tall view prefer more clearance around the complete
  silhouette over the tighter edge-constrained frame.
- [Bicycle scenea6b5c71678cc8c3f](https://www.flickr.com/photos/mattmurf/1745951744): for a person-in-context editorial story prefer
  the foreground person retained over lower crops showing only a body fragment.
  The tall request changes subject/story or crowds the head at an edge; hold
  positive preference labels for that request.
- [Crowd8e56e120b6c98afb](https://www.flickr.com/photos/amnesiaibiza/7675778352): incomplete head inventory and boundary fragments prevent
  positive labels for either ratio. Compatible weak head geometry is insufficient.

The packet retains original source URLs with every source. Six viewed endpoint
preferences across three FIT families are admitted only to this research pilot.
Reviewer is the same coordinating Codex visual model, not an independent human.
Weights0.5/0.75/1.0 express declared weak preference strength, not calibrated
confidence. No excellent interval, automatic pair-quality label or old-label
admission changes. Preferred crops are not certified absolutely good or face-safe.

## Matched adaptation

The [registered fitting protocol](../docs/experiments/FIT_REVIEW_ADAPTATION.md)
starts all arms from the exact selected contrastive seed31 checkpoint4b801ad4.
Keep encoder, first scorer, base scalar, input normalization and teacher projection
fixed. Train only the existing96→32 ReLU→32 residual probe.

Nine fits: replay, replay+review and replay+reversed-review; seeds31/137/911.
16 epochs plus zero; AdamW0.0003/decay0.0001/clip5; original94 FIT and real48 FIT
replay, batch8+8, existing rank objective plus0.25 acceptable-set loss. Review
arms add0.05 weighted preference softplus. Preserve every epoch, optimizer/RNG,
selected model and choices. The reviewed pair candidates were all unknown in
the old positive/negative masks, so no explicit old pair label is overridden.

Select only epochs whose old essential CPU regret does not exceed the common
checkpoint's own epoch-zero value. Then minimize FIT preference loss, old regret
and epoch number. New DEV metrics are diagnostic only; no TEST is opened.

| Arm | Selected epochs | Preference loss | Agreement on six seen pairs |
|---|---|---:|---:|
| Replay |0/0/0|1.2577033042907715|3/6 each|
| Review |9/9/10|mean0.9228475093841553|3/6 each|
| Reversed review |0/0/0|1.2577033042907715|3/6 each|

All six seed31 preference margins improve, but the three previously wrong
directions remain wrong. The new essential DEV family-balanced weak head-cut
rate worsens from0.3583481877599524 to0.3714200831847891; accepted-reference
matching moves from0.12425515660809777 to mean0.11996010525422289.
Do not call lower seen-pair softplus an artistic-quality improvement.

## Original quality and numerical checks

All nine selected fits pass the unchanged full-model CUDA original-quality gate.
Original C002 essential regret remains0.24700142443180084. Review seeds31/137/911
yield0.24189724028110504,0.2439175397157669 and0.24359741806983948.
Replay/reversed controls retain the initializer at0.2445547878742218.
The slight CPU/CUDA metric differences are recorded separately; no threshold
or frozen quality implementation changes.

Compact normalization/projection folding preserves every eligible crop choice
for nine models×1584 fixed FIT/DEV requests each: zero index changes, zero edge
movement, maximum absolute score difference1.1920928955078125e-6. The unchanged
three-grid numerical gate passes. This is CPU cached-head arithmetic, not new
Rust export parity or serving timing for these changed weights.

## Learnability check

After the nine fits, register a separate deliberate overfit probe: same
initializer, only the six weighted preference losses, AdamW0.003, max256 steps,
stop at the first6/6 agreement. It reaches6/6 in11 steps, loss0.32070040702819824.
But old same-platform regret rises from0.244554802775383 to1.060389757156372.
Thus these preferences are representable by the existing small head, while
unconstrained fitting damages the old ranking. No broader learning or model
promotion follows. No replay or gate is disabled in the nine matched experiments;
the separate probe is explicitly an overfit diagnostic.

## Resources and retained evidence

i265, run-heavy/two CPU threads except the one-thread learnability probe:

| Tool | Reported duration | Peak RSS | Minimum available RAM | Cap |
|---|---:|---:|---:|---:|
| Original pair rendering |2s|0.63GiB|23059MiB|3GiB|
| Tighter pair rendering |2s|0.63GiB|23071MiB|3GiB|
| Nine matched fits |10s|1.16GiB|21994MiB|4GiB|
| Original full CUDA gates |2s|1.55GiB|22667MiB|4GiB|
| Compact arithmetic |1s|0.81GiB|23027MiB|3GiB|
| Six-pair learnability |2s|1.14GiB|22890MiB|3GiB|

These are bounded tool/training observations, not deployment latency claims.
All five original quality files remain byte-identical to
9f12692417d774653ed676d236e0f89eaafa7385. No paid model API calls or new local
detector/teacher batch inference occur. The six new model judgments come from
the coordinating Codex visual review explicitly declared above.

Review root:`/mnt/v/output/zensally/fit-pair-review/2026-11-09/` under128MiB.
Fit root:`/mnt/v/output/zensally/visual-preference-adaptation/2026-11-09/`
under128MiB. Retain all source links, precise geometry and indices, viewed-image
hashes, weak judgments/holds, every fit epoch, full-model quality results,
arithmetic checks and failed-quality learnability model.

Next test outer-first nested selection against joint summed scores, preserving
the raw Minimal choice while measuring Maximal availability and quality. The
performer witness shows why improving a pair sum can sacrifice a useful gesture
in the outer crop. Keep that structural test separate from further weak-label fitting.
