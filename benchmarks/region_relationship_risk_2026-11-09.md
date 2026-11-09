# Learned crop-to-region risk suitability

Explicit region relationships improve weak nominated-head cut ranking while
preserving offers, but do not establish a safe artistic crop model. All nine
aligned person-only policies fail original composition compatibility. A
separate measured-head applicability rule passes all12 original gates, but
it can alter only one of25 original sources. It is weak evidence for general
people composition. All aligned policies still cut26–34/139 accepted DEV
endpoints. No model promotion, image-only transfer or runtime export.

## Frozen data and training

[Registered study](../docs/experiments/REGION_RELATIONSHIP_RISK.md) reuses
48 FIT/24 DEV newer sources and25 original DEV, frozen candidate banks,
common endpoint4b801ad4/additive pair05f41e08 and existing known positive/cut
masks. Other candidates stay unknown. Source/family roles and TEST closure
remain fixed. No new teacher inference, paid calls, labels or input images.

Four source-only region groups are cached seven-model face union, measured
pose heads, padded upper-person and full-person views. Eight relations per
group encode count, containment, cuts, exclusion, intersection severity and
source-relative size. Arithmetic operates only on eligible native candidates;
all regions remain except exact duplicates. Missing groups are measured empty.
No nominated head, subject choice or reference box creates these input regions.

Four input→32 ReLU→1 heads have40/128/160/160 inputs and
1345/4161/5185/5185 parameters. They compare geometry+relations, hidden+relations,
hidden+relations+attention and matched permuted relations/attention. Region
donors rotate by source ID within FIT/DEV; original DEV rotates independently.
Donor coordinates map outward with integer arithmetic. Existing auxiliary
permutations stay unchanged and can use different donors. Both feature arrays
are hash-bound and recomputed exactly for evaluation without duplicating them
on disk. These heads still need measured detector/pose inputs at inference;
the attention arms additionally need measured pose/gaze/scanpath summaries.
They are not image-only distilled students or a cheap complete runtime.

Twelve fits keep32 epochs plus zero, seeds31/137/911, batch8/all22 queries,
known-class/source/family balancing, FIT-eligible normalization/std floor0.001,
AdamW0.001/decay0.0001/clip5, CPU FP32/two threads. DEV known-class logistic
loss selects epoch, earliest tie. Every epoch, optimizer and RNG is retained.
No crop quality or availability metric selects a checkpoint.

## Actual crop outcomes

Fixed unfiltered parent:139/139 accepted essential DEV offers,45 cuts,
16 matches and117 retained heads. All-head-known essential DEV has165/396
cuts. Original448-request regret is0.2406301498413086, below unchanged
0.24700142443180084 threshold.

Soft strengths0,.25,.5,1,2,4,8 use the existing FIT-only rule: minimize family
cuts without losing FIT match/retention versus zero, then match, retention and
smaller strength. Every grid point is retained. No old/new DEV crop metric
selects strength. All soft policies keep full139 accepted DEV and448 original
offers, and still emit all85 explicit-contain DEV requests.

|Model|Epoch|DEV classification loss|Person-only cuts|Head-gated cuts|Head-gated matches|All-head-known cuts|Original person-only regret|Original head-gated regret|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|geometry-regions-seed31|31|0.37068015336990356|26|26|21|107/396|0.2495829463005066|0.24106267094612122|
|geometry-regions-seed137|32|0.38427048921585083|31|31|19|122/396|0.24993380904197693|0.24106267094612122|
|geometry-regions-seed911|32|0.38430124521255493|27|27|21|103/396|0.24963879585266113|0.24106267094612122|
|hidden-regions-seed31|18|0.37086057662963867|32|32|21|121/396|0.2653956711292267|0.2405223399400711|
|hidden-regions-seed137|20|0.39484450221061707|33|33|21|133/396|0.2591230869293213|0.2406301498413086|
|hidden-regions-seed911|18|0.4108850657939911|34|34|20|123/396|0.2561260759830475|0.2406301498413086|
|hidden-regions-attention-seed31|18|0.40718716382980347|30|30|25|117/396|0.2523447573184967|0.24054767191410065|
|hidden-regions-attention-seed137|19|0.3942192792892456|29|30|24|120/396|0.2555060088634491|0.2406301498413086|
|hidden-regions-attention-seed911|14|0.40434783697128296|31|31|25|124/396|0.24987024068832397|0.2405223399400711|
|hidden-permuted-regions-attention-seed31|12|0.5067715048789978|46|46|16|167/396|0.24126291275024414|0.24054767191410065|
|hidden-permuted-regions-attention-seed137|18|0.49449992179870605|48|48|16|169/396|0.2506612539291382|0.24027128517627716|
|hidden-permuted-regions-attention-seed911|11|0.5029048919677734|46|46|15|169/396|0.24221327900886536|0.24063459038734436|

Nine aligned person-only policies fail the original gate; two of three
permuted controls pass it while failing to improve head cuts. Aligned inputs
have lower DEV classification loss than permutation, but adding the existing
attention summary is not consistently better than region relations alone.
This shows signal in these features, not a reliable global quality improvement.

### Applicability mechanism and limits

The original wallet sourceMqOGVcLcJ3E has person boxes but no face/pose-head
proposals. Body-risk alone changes product framing. A separately registered
mechanism test requires both Lite2 person and a measured face or pose head.
It preserves all original person-only results, weights, inputs and FIT-only
strength selection. This diagnosis uses reused DEV pixels; it is not an
independent validation set or a source-specific override.

Person applicability is70/72 newer sources and3/25 original. Measured-head
applicability reduces this to69/72 and1/25. Every inactive source retains
baseline choices exactly, including the wallet. All12 gated policies pass the
original gate, but this old panel scarcely tests changed people crops. The
primary hidden-regions-attention/seed137 still has30/139 cuts,24 matches and
134 heads retained. It repairs17 and introduces two cuts versus the parent.
Its broader essential cohort has120/396 cuts, versus165 before; secondary
has37 cuts. No face or artistic certification follows.

Hard0.5 routing is retained separately as an abstention diagnostic. In
person-only aligned models it offers112–119/139 accepted endpoints with11–19
cuts. These reductions include withdrawals and do not replace soft-policy
coverage requirements. Unknown/contain/FIT/secondary rows remain separate.

## Pixels and numerical behavior

All10 PNG panes were inspected. The fixed seed137/new3:4 cave crop admits a
partial right-hand person's head/body while shifting to the foreground person.
Faces mostly turn away; a weak head-cut flag is not a human face judgment, but
framing still fragments people. The original2:1 wallet witness shifts framing;
both parent and risk still clip the product, so neither is a whole-product
catalogue example. The largest original target-score loss is1.3280887603759766;
this is a weak proxy difference, not an independent artistic verdict.
Head gating leaves every cave rectangle unchanged and restores the wallet's
baseline exactly. Actual pixels use owned resize/encoding and exact independent
PNG/source readback. No new labels are admitted from visual inspection.

Both applicability variants compare CPU reference, compact fused-normalization
and CUDA FP32 on the same cached feature inputs, through actual soft and hard
pair routing on both cohorts. Per variant:48 numerical rows/96 routing checks,
102432 endpoint comparisons, zero changed endpoints, zero offer/cut-flag changes
and zero three-grid edge violations. Maximum eligible logit difference is
3.814697265625e-6. Feature extraction, encoders and teacher inference are not
GPU/native-qualified by this head-level check. All five established quality
evaluator files remain byte-identical to9f126924.

## Resources and artifacts

Run-heavy i265 under4GiB/two threads; qualification CUDA allowance20 percent:

| Stage | Duration | Peak RSS | Minimum available RAM | Peak load |
|---|---:|---:|---:|---:|
|Relation invariants,2GiB|1s|0.60GiB|25079MiB|1.24|
|Twelve fits|22s|1.63GiB|23816MiB|1.91|
|Person-only routing/numerics|21s|1.78GiB|23385MiB|1.94|
|Actual-pixel witnesses|2s|1.35GiB|24030MiB|1.64|
|Head-gated routing/numerics|20s|1.78GiB|23536MiB|2.14|

These are research-run resources, not serving latency. The three relation
checks exercise boundary contact, fractional intersection, translation
invariance and explicit empty/duplicate inputs. Quality remains unsuitable,
so there is no runtime export or serving benchmark.

Local on i265: `/mnt/v/output/zensally/region-relationship-risk/2026-11-09/`.
NAS: `/mnt/tower/output/zensally/region-relationship-risk/2026-11-09/`.
The64MiB packet retains all396 epoch checkpoints,12 selected states/logits,
source proposals, donor/feature hashes, both complete84-point routing grids,
all numerical rows,10 PNG witnesses and full logs. No paid API spending or TEST.

Frozen489-member packet,44601786 bytes before manifest, complete NAS membership
and SHA256 readback verified. Manifest SHA256:
`0fbcd218ec4012a72113ab809fa8c489455a9182a65c1f54caf559c9f8968080`.
Advertised gallery HTML and all10 PNGs match local bytes over HTTP.
