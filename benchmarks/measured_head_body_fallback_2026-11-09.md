# Measured head locations with body fallback

Measured pose contributes useful head-protection information on common offers,
but the resulting policies remain unsafe. Across five frozen parents, adding
pose repairs15–20 common-offer cuts and introduces1–2 relative to face/body
fallback. The primary face+pose policies still cut26–31/127 offered accepted
DEV endpoints and withdraw12 of139. No crop model or safety policy is promoted.

## Registered source-only geometry

[Protocol](../docs/experiments/MEASURED_HEAD_BODY_FALLBACK.md) reuses seven
cached face detectors, pose-derived head estimates and Lite2 boxes.25-percent
padded measured heads stay protected. A detected body receives an upper45-percent
fallback only when no unpadded measured-head center lies inside its padded
full-body box. All measured heads remain, including unmatched ones. Deduplication
removes exact duplicates only. No reference nominations, chosen subjects or crop
windows supply proposals. This changes supplied regions, not the exact solver.

The five original parents replay7920 diagnostic endpoint rows exactly before
constraints. All7920 constrained pairs preserve maximum-area Minimal, strict
matching-aspect Maximal and the supplied-region interval condition whenever
both endpoints exist. Infeasible results stay explicit. No new inference,
fitting, paid calls, labels or TEST. This is a mechanism screen informed by
reused DEV pixels, not independent generalization evidence.

## Essential DEV outcomes

|Parent|Protected regions|Accepted offered|Accepted cuts|Matches|All head-known cuts|New cuts on common offers vs parent|
|---|---|---:|---:|---:|---:|---:|
|geometry|face-body|132/139|48|23|103/396|11|
|geometry|face-pose-body|127/139|30|20|70/396|5|
|person-all-pca8|face-body|132/139|48|23|100/396|14|
|person-all-pca8|face-pose-body|127/139|31|20|69/396|8|
|person-largest-pca8|face-body|132/139|42|23|83/396|13|
|person-largest-pca8|face-pose-body|127/139|27|23|53/396|7|
|person-all-raw|face-body|132/139|46|21|96/396|13|
|person-all-raw|face-pose-body|127/139|28|22|64/396|7|
|person-largest-raw|face-body|132/139|41|22|86/396|13|
|person-largest-raw|face-pose-body|127/139|26|22|55/396|7|

Face/body policies emit362/432 essential DEV endpoints and58/85 teacher-contain
requests. Face+pose/body emits350/432 and53/85. Source-only constraints do not
learn whether to offer a crop. The geometry parent's primary face+pose policy
repairs22 accepted cuts and introduces five on common offers relative to the
unconstrained parent; it loses one prior reference match. Its40 outer origins
are infeasible and two requests have no strict child among216 essential pairs.

### Pose information versus withdrawal

Both policies share127 offered accepted DEV endpoints for every parent.
Adding pose withdraws five additional accepted endpoints and offers no new
ones. The following cut changes therefore exclude those withdrawals:

|Parent|Face-only cuts on127 common offers|Face+pose cuts|Repairs|New cuts|
|---|---:|---:|---:|---:|
|geometry|48|30|20|2|
|person-all-pca8|47|31|18|2|
|person-largest-pca8|41|27|15|1|
|person-all-raw|44|28|18|2|
|person-largest-raw|40|26|15|1|

The pose signal reduces cuts on the same offered set. This does not establish
artistic quality or a safe learned student. Family splits and original quality
contracts stay fixed; original frozen-bank compatibility has not been rescored
by this integer geometry diagnostic. Detailed FIT/secondary outcomes remain
in the packet. Three accepted secondary DEV endpoints are insufficient to claim
extreme-ratio suitability.

## Actual pixels and preflight repair

All14 PNG panes were inspected with whole-source context. The gymnast at4:3
keeps the upside-down face rather than following the box-top heuristic; it
still has questionable framing of the hands and background person. The
baseball9:16 crop moves right and admits a partial standing official at the
edge. These visible failures remain despite zero supplied-region violations.
Incomplete or misassigned proposals limit the protection, and an artistic
subject/story decision remains necessary.

The first preparation assumes the pose gallery's sources.json describes all72
inference sources; it actually contains24 DEV render records. It fails before
writing preparation output. The repaired preflight verifies all72 identities
against the exact attention cache bound by pose protocol, plus all24 DEV gallery
identities separately. Original failed log is retained; no label, proposal,
threshold or solver rule changes. Source-only fallback counts are106 for faces
and98 for faces+pose. The remaining-witness caption was corrected to avoid
calling a previously reviewed photo fresh; original captions remain as bak,
with identical image bytes and statistical selection.

## Resources and artifact scope

Run-heavy i265, two threads unless noted:

| Stage | Duration | Peak RSS | Minimum available RAM | Peak load |
|---|---:|---:|---:|---:|
|Failed preparation,2GiB cap|1s|0.59GiB|25078MiB|1.55|
|Repaired preparation,2GiB cap|1s|0.61GiB|25077MiB|1.36|
|Ten controls,4GiB cap|3s|0.60GiB|24734MiB|1.36|
|Gallery,2GiB/one thread|1s|0.59GiB|25081MiB|1.38|
|Common-offer audit,2GiB/one thread|1s|0.59GiB|25078MiB|1.54|

These are research-run resources, not isolated runtime latency. No new native
runtime or full-network numerical comparison is claimed; parent head numerical
qualification remains separately frozen. All five quality evaluator files stay
unchanged. No new model export or promotion.

Local on i265: `/mnt/v/output/zensally/measured-head-body-fallback/2026-11-09/`.
NAS: `/mnt/tower/output/zensally/measured-head-body-fallback/2026-11-09/`.
Source/proposal hashes, all controls, statuses, common-offer rows, full logs and
14 PNG witnesses are retained under16MiB.

Frozen85-member packet,16420439 bytes before manifest, complete NAS membership
and hashes verified. Manifest SHA256:
`64ca9a0ac7813c5750831cfb2d4502ea85464601ec03a629d5fd13655c1cd088`.
Advertised gallery HTML and all14 PNGs match local bytes over HTTP.
