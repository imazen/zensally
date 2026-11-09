# Weak face-label coverage beyond accepted crop references

The earlier139-endpoint newer DEV score is conditional on accepted composition
references. Expanding the diagnostic to all396 essential requests on22 DEV
sources with nominated heads finds166 raw weak-head cuts, versus43 on the139
accepted-reference subset. The remaining two DEV sources have unknown head
coverage and remain outside these head diagnostics. No unknown composition is
admitted as an acceptable or unacceptable crop.

| Essential newer DEV cohort | Requests | Raw cuts | Joint cuts | Real pair137 cuts |
|---|---:|---:|---:|---:|
| Accepted reference |139|43|45|45|
| Explicit contain recommendation |85|54|52|51|
| Unknown composition |172|69|71|71|
| All nominated-head requests |396|166|168|167|

These are finite, unreviewed head-box intersection counts, not human face recall
or artistic-quality scores. Raw and nested rankers here always emit a candidate;
they do not apply a trained contain decision. The retained separate direct-crop
scout has an offer head, which previously failed suitability. Full72-source
cached teacher records contain525 accepted crop requests,471 explicit contain
recommendations and588 unknown compositions across all22 requests per source.

## Labels and geometric feasibility

The audit replays every candidate negative mask from existing nominated heads
and exact integer-native boxes, with zero disagreements and zero positive/cut
conflicts. All202752 candidate slots are checked.90 accepted-positive candidate
slots retain no nominated head: this is not automatically an incorrect label,
because selected non-head subjects and complete person exclusion can be valid.
It supplies no new safe-face or artistic certification.

Among198 essential DEV ratio pairs on22 head-annotated sources,174 have a
strict nested pair that avoids every nominated head intersection.161 also
retain at least one nominated head in the inner crop. Thus24 lack even a
weak-head-safe pair in this frozen bank, and37 lack a safe pair retaining a
head. This is a candidate-bank feasibility diagnostic; it does not prove no
other native crop could work. Requiring a retained head is a separate diagnostic,
not a universal composition requirement or changed runtime policy.

FIT has405 essential pairs on45 nominated-head sources,362 with a safe pair
and342 with a safe pair retaining a head. For secondary6:1/1:6 ratios, DEV has
44 pairs,40 safe and16 safe retaining a head; FIT has90,82 safe and43 retaining
a head. Extreme ratios require explicit availability/contain behavior.

The visible counter-group "repair" in the preceding pair study concerns one
endpoint; another endpoint remains face-cutting and is already flagged by the
weak geometry. It is not evidence of a new label disagreement. Labels are
incomplete, but these specific visible failures cannot be blamed on a measured
cache error.

## Reproduction and limits

`just composition-audit-face-label-coverage` binds the original source/role/
family identities, fixed common endpoint model, direct nominated-head cache
and frozen pair-correction packet. It writes scalar/per-query records, selected
head relation counts and source links; no metric maps or new image inference.
Two CPU threads on i265 under run-heavy3GiB:2s, peak RSS1.07GiB, minimum available
RAM22451MiB. This is an audit resource observation, not runtime latency.

No paid API calls, TEST access, altered quality gates, label admission, source
roles or model promotion. Original139-endpoint results reproduce unchanged.
The audit motivates a separate candidate face-cut classifier suitability test:
unknown candidates must remain unknown; avoiding rank regret does not establish
face safety, and abstention must report lost crop availability explicitly.

Frozen8-member packet,4041734 bytes before identity/manifest,16MiB cap.
Manifest SHA256 `160d693d91bae8dd858036c0bd198088af26be847122e50bdad8fa2ec336a34a`.
On i265: `/mnt/v/output/zensally/face-label-coverage/2026-11-09/`.
NAS: `/mnt/tower/output/zensally/face-label-coverage/2026-11-09/`.
Complete membership/hash readback verifies the mirror separately.
