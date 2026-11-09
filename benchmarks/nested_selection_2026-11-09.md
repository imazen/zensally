# Frozen-bank nested output selection

Nesting can be enforced over the existing banks, but all three learned nested
policies fail the unchanged original quality gate. Outer-first preserves raw
Minimal, yet can clip a head or lose useful subject content in Maximal. Keep
these selectors experimental; no model or runtime policy is promoted.

The [registered study](../docs/experiments/NESTED_SELECTION.md) keeps checkpoint
4b801ad4, original119 and real72 FIT/DEV sources, all11 ratios, scores,
eligibility and integer-native candidates fixed. No head or teacher labels
enter selection. All raw original DEV choices replay the preserved full-model
gate exactly. No fitting, new labels, paid API calls or TEST access.

Independent picks the two original mode winners. Outer-first keeps raw Minimal
and selects the best strictly smaller Maximal inside it. Joint-sum selects the
best score sum over all strict pairs. The separately registered Maximal-first
control keeps raw Maximal when a containing Minimal exists. Missing strict
pairs retain raw Minimal and mark Maximal unavailable; no equal-crop fallback
or fabricated quality score is used. Pair sums use binary64 addition of the
binary32 model scores, with ascending index ties.

## Availability and quality

Essential ratios3:1 through1:3, including the intervening registered ratios:

| Original25 DEV /225 pairs | Strict pairs | Minimal changes | Maximal changes |
|---|---:|---:|---:|
| Independent |164|0|0|
| Outer-first |225|0|61|
| Joint-sum |225|22|41|

The25-source quality panel has448 valid essential endpoint targets. Every
policy offers all448 endpoints here, so these are complete comparable scores:

| Policy | Original essential regret | Pass unchanged0.24700142443180084 gate |
|---|---:|---|
| Independent |0.244554802775383|yes|
| Outer-first |0.25255903601646423|no|
| Joint-sum |0.2601603865623474|no|
| Maximal-first |0.25999927520751953|no|

Lower is better. Do not loosen the gate because the geometry is correct.
The symmetric extra-control rerun preserves every original three-policy summary.

| New24 DEV /216 essential pair requests | Strict pairs offered | Minimal changes | Maximal changes, including missing |
|---|---:|---:|---:|
| Independent |78|0|0|
| Outer-first |185|0|138|
| Joint-sum |216|112|65|

Thus31 raw Minimal origins have no strictly nested child in the current Maximal
bank. Joint selection recovers availability by moving Minimal, which can sacrifice
useful context or gesture. This is a candidate-layout/ranking distinction, not
proof that no valid native inner crop could be generated geometrically.

Separate new DEV weak annotations cover139 essential endpoints/17 families:

| Policy | Offered /139 | Weak cuts | Reference matches |
|---|---:|---:|---:|
| Independent |139|43|19|
| Outer-first |135|48|19|
| Joint-sum |139|45|16|
| Maximal-first |129|39|19|

The smaller Maximal-first cut count comes with10 unavailable annotated endpoints.
It cannot be interpreted as an equal safety improvement. No policy establishes
complete face coverage or artistic suitability; unknown endpoints remain unknown.

## Full neural numerical replay

Full ORT gather encoder/query inference on191 cached owned image tensors runs
4202 neural queries and8404 pair decisions across all four policies. Zero outer
or inner index changes, zero availability changes, zero edge movement, zero
violations of the unchanged three-grid numerical gate. Every offered structured
pair remains strictly nested in serving replay. This covers encoder/query and
native-bank selection, excluding preprocessing, detectors and artistic safety.

## Four actual-pixel witnesses

34 panes,21 distinct PNG hashes, all21 inspected directly. Two FIT performer
requests and two deliberately selected DEV regressions are diagnostic examples,
not a representative quality sample. Owned crop/resize/PNG encoding and copied
sources pass independent exact RGB readback. Unavailable Maximal has text only;
it is not displayed as a full-source crop or containment result.

- [Performer8ad2a46a2ad698cf](https://www.flickr.com/photos/sfslim/15470003710),1:3: outer-first preserves the raw outer's open hand
  but has no child. Joint/Maximal-first move the outer and clip more of that
  gesture. Strict nesting alone cannot choose an editorial story.
- Same performer,3:1: outer-first selects a tighter inner that loses more of the
  hand at the bottom than the independent Maximal. This is the already retained
  weak comparison; no new training label is admitted.
- [Moon/boat sourceFFPu5QDn40Y](https://unsplash.com/photos/FFPu5QDn40Y),1:3: raw Minimal emphasizes moon/reflection while
  raw Maximal emphasizes the boat. Outer-first forces Maximal into the moon-side
  outer and loses the boat. Maximal-first moves the outer toward the boat.
  This request has the largest old essential Maximal weak-regret increase under
  outer-first. Primary-subject consistency matters as well as containment.
- [Forest sourceaf8912bcf7fae386](https://www.flickr.com/photos/sebilden/21038684695),1:3: raw endpoints retain the standing person;
  outer-first clips the head at the left edge. The full neural runtime reproduces
  this exact failure. Joint pairing moves the outer and retains more person
  context, but its broader quality gate still fails.

Source hyperlinks are retained in every witness record and viewer.

## Frozen-target oracle

After the learned-policy failures, measure an explicitly non-runtime oracle
that maximizes frozen endpoint target sums over strict pairs. Invalid target
modes contribute zero only to its diagnostic sum and remain excluded from quality
aggregation. It does not create new labels or alter the existing target masks.

The original bank supports strict pairs for all225 essential DEV requests.
Target-only nested selection gives essential regret0.00809677317738533 on448
valid endpoints. FIT essential regret is0.0092582106590271 on1682 valid endpoints.
There are836 FIT and223 DEV essential pairs with both targets valid; paired
training must not invent a target for the remaining one-mode-labeled requests.

This oracle uses held-out target values only as a diagnostic, not inference or
learned output. It does not prove generalization, face safety or an exact global
optimum under every floating point aggregation order. It shows substantially
better target-scored nested pairs exist without relaxing the quality threshold,
which motivates testing a structured pair-training objective.

## Resources and limits

i265, run-heavy/two threads,4GiB cap except2GiB rendering and3GiB oracle:

| Tool | Reported duration | Peak RSS | Minimum available RAM |
|---|---:|---:|---:|
| Three-policy cached audit |3s|1.14GiB|21827MiB|
| Full neural replay |11s|0.68GiB|22432MiB|
| Actual-pixel rendering |0s|0.07GiB|23069MiB|
| Target oracle |1s|0.61GiB|23028MiB|

These are evaluation-tool observations, not paired runtime latency or deployment
memory claims. Small-bank exhaustive selector tests pass, including index ties,
explicit absence and masked zero placeholders. All five original quality files
remain byte-identical to9f12692417d774653ed676d236e0f89eaafa7385.

Output:`/mnt/v/output/zensally/nested-selection/2026-11-09/`. Retain original and
extra-control summaries, every endpoint/pair, full-neural replay, oracle targets,
source links and reviewed failures. No model or selector is promoted.
