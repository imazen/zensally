# Fixed endpoint, learned pair correction

Six of nine pair-correction fits pass the original nested quality gate while
keeping raw endpoint output unchanged. The simple additive control and the
permuted-inner control each pass two of three seeds, as does the real bilinear
head. This does not establish a benefit from meaningful pair relationships.
Visible face failures remain; no crop model or runtime policy is promoted.

The [registered protocol](../docs/experiments/PAIR_RELATIONSHIP_TRAINING.md)
keeps the4b801ad4 endpoint model, cached embeddings, native banks and source roles
fixed.94 original FIT/25 DEV,48 newer FIT/24 DEV; TEST stays closed. Three arms
times seeds31/137/911,32 epochs plus zero. Only existing FIT requests with both
endpoint targets valid supply pair gradients. Temperature0.5 listwise KL learns
the frozen pair target-sum distribution, using AdamW0.001 and clip5. No API calls,
new teacher judgments, face annotation admission or excellent interval labels.

Additive-linear uses separate96→1 corrections. Pair-real uses96→8 projections
and their scaled dot product. Pair-permuted deranges inner embeddings within
eligible candidates, with the same architecture and optimizer. Corrections
start at zero. FP32 correction scores are added to the fixed endpoint sums in
binary64, with ascending candidate ties. Raw endpoint scoring is never changed.

## Original quality and newer weak diagnostics

Original essential quality covers448 valid endpoints on25 DEV sources. The
unchanged gate is0.24700142443180084. Every corrected checkpoint retains raw
regret0.244554802775383. Initial nested regret is0.2601603865623474.

| Arm | Seed31 nested regret | Seed137 | Seed911 | Passing seeds |
|---|---:|---:|---:|---:|
| Additive-linear |0.2400963306427002|0.2406301498413086|0.24977056682109833|2/3|
| Real pair |0.2601603865623474|0.24007758498191833|0.24680061638355255|2/3|
| Permuted inner |0.2601603865623474|0.24295905232429504|0.24570882320404053|2/3|

All passing epochs are epoch1; all32 later epochs and optimizer/RNG states are
retained. Passing selection uses only original DEV regret, with newer DEV
diagnostic only. Three failing runs keep explicitly named best-ineligible states.
This remains a repeatedly consulted development panel, not independent TEST
evidence or a human artistic-quality measure.

Newer essential weak annotations cover139 endpoints/17 families. Common joint
selection has45 weak-head cuts and16 reference matches. Chosen corrections have
45–46 cuts and16 matches. Real seed137 repairs no weak-head flags and introduces
none; real seed911 introduces one and repairs none. Additive seed31 repairs one
but introduces two. The old-panel gain does not establish improved people framing.

## Full neural inference and actual pixels

Recompute frozen encoder and candidate embeddings from the owned cached image
tensors for25 original and24 newer DEV sources. All nine heads reproduce all
275 original and264 newer pair decisions exactly: zero endpoint index changes,
zero edge movement, zero three-grid numerical gate violations. All six original
quality passes and every newer weak diagnostic reproduce. This covers the image
network, endpoint and pair heads, excluding preprocessing and detector inference.

Four rendered request instances contain three unique requests; the fixed
counter-group regression duplicates the selected weak repair request. Do not
count these as four independent examples.44 PNG panes contain15 distinct hashes;
all15 were directly inspected. Owned resize/PNG output passes independent exact
RGB readback. HTTP readback verifies all44 PNGs and11141027 image bytes.

- [Child and inflatable play area](https://www.flickr.com/photos/mpeinadopa/3941433819),3:4: common Minimal leaves the child
  as a partial figure at the left. Real seed911 shifts further and adds the weak
  head-cut flag. Additive seed31 excludes the child entirely; no artistic
  preference or positive training label is inferred from that subject change.
- [Counter group](https://www.flickr.com/photos/projetoluzevida/15328959047),1:1: common nested output still cuts a person at the left.
  Additive seed31 shifts toward the elder; its inner crop still clips the middle
  person's face at the left boundary. A nominated-head repair does not certify
  complete face safety. Real seed137 retains the earlier failure.
- [Moon and boat](https://unsplash.com/photos/FFPu5QDn40Y),1:3: common outer favors moon/reflection and leaves a boat
  fragment at the right; corrected frames favor boat/mountains and exclude the
  moon. The large target-score gain is a story change, not an excellent-pair label.

Gallery:

http://dev.lan:3300/zensally/pair-relationship-training/2026-11-09/witnesses/index.html

The first renderer asked for a real-seed137 head repair, which does not exist,
and failed before writing output. Its log is retained. The corrected selection
uses the actually measured additive-seed31 repair; coordinates, annotations,
quality expectations and earlier artifacts remain unchanged.

## Resources and evidence

i265, run-heavy, two CPU threads,4GiB hard cap; rendering3GiB:

| Run | Reported duration | Peak RSS | Minimum available RAM |
|---|---:|---:|---:|
| Nine fits |32s|1.57GiB|21946MiB|
| Full neural qualification |2s|1.63GiB|23085MiB|
| Corrected witnesses |1s|0.75GiB|23077MiB|

These are tool observations, not serving latency or deployment memory.
No Rust export, runtime benchmark or promotion follows from this experiment.

Frozen462-member packet,37774593 bytes before identity/HTTP check/manifest,
64MiB budget. Manifest SHA256
`698972129157fa6ffce1c748f94bacac12a42b703ddea7c300b5445743b93020`.
On i265: `/mnt/v/output/zensally/pair-relationship-training/2026-11-09/`.
NAS: `/mnt/tower/output/zensally/pair-relationship-training/2026-11-09/`.
Every epoch, optimizer/RNG state, full neural comparisons, events, source links,
rendered pixels, resource logs and direct-review observations are retained.
