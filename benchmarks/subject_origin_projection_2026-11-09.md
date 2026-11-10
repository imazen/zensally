# Exact per-person placement and automatic face inputs

Exact origin placement preserves supplied rectangles but cannot protect heads
that its inputs miss. The automatic versions still visibly truncate faces and
remain unsuitable as shipping guards. No new model or coverage guarantee is
promoted, no paid calls are made, and TEST remains closed.

The Python research solver holds crop extent fixed, contains at least one
supplied head and either wholly contains or wholly excludes every other head.
It therefore permits selecting a person or subgroup. It finds the nearest
integer origin by source-normalized squared displacement with deterministic
y/x ties. A sweep of vertical classification boundaries and interval subtraction
enumerates all legal origin classes.250 seeded small-grid cases agree with an
exhaustive integer-origin oracle. This is a geometric constraint, not artistic
composition or actual-face certification.

Three already-selected no-region image heads supply the proposals; no checkpoint
is reselected for these audits. Compare fixed predicted extent against a fallback
to the largest native Maximal extent only when the predicted extent is infeasible.
Minimal extent stays unchanged. Two identical results are deduplicated. Geometry
and learned offer/contain decisions are reported separately.

## Essential-ratio controls

These means balance22 reused DEV families,308 requested essential queries and
three seeds. They use largest-extent fallback and geometry-only offer decisions.
Cut percentages divide by all requests, not only offered crops. The reference
column matches either returned proposal against finite cached Sol windows.

| Regions / margin | Crop offered | Nominated-head cut | Reference match |
|---|---:|---:|---:|
| Supplied /0 |93.7229%|0 by construction|8.7662%|
| Supplied /10% |87.6623%|0 by construction|10.7143%|
| Supplied /25% |81.0606%|0 by construction|8.5498%|
| YuNet + pose /0 |69.8052%|36.5801%|8.1169%|
| YuNet + pose /10% |67.7489%|30.1948%|7.4675%|
| YuNet + pose /25% |65.5844%|22.6190%|7.5758%|

Positive margins expand each edge by ceil(region extent × percent/100) plus two
source pixels, clipped to the source. Zero adds nothing. Cut scoring still uses
the original, unexpanded Sol nominations. These are stricter geometric variants,
not a relaxed numerical gate. More padding reduces some cuts and feasible crop
coverage together; it does not solve missing heads. Neither margin is selected
as a production default from these reused DEV observations.

On17 DEV families/139 accepted essential queries, supplied-region10% fallback
offers94.0087% and matches either proposal23.8053%. This best2 figure must not
be compared as a single-choice ranker gain. Learned offer/contain over-abstains;
the complete artifacts retain both policies. Finite weak windows and unreviewed
regions cannot certify artistic quality or complete face safety.

All four audits compare CPU batch, CPU single-image and GPU batch geometry,
option counts and offer decisions. All pass the unchanged three-grid-pixel
edge gate; the three raw no-region heads have zero measured edge movement.
Padding10/25 audits each take3s with run-heavy peak RSS1.07/1.09GiB,4GiB/two-thread
caps and minimum available RAM21,429MiB. These are audit costs, not deployment
latencies. The complete solver has not been ported or benchmarked in Rust.

## Native detector comparison

The retained Rust probe evaluates UltraFace and YuNet on the same72 owned RGB
sources:144 new local calls, no new weights or runtime build. UltraFace returns
120 proposals on32 sources; YuNet237 on50. On22 structurally valid Sol DEV
sources with84 nominated heads, center-inside-box matching finds16 and24
respectively. This is annotation-scope overlap, not human recall, precision or
whole-head containment. Raw outputs, safe probe IDs and canonical source mapping
remain retained. The39,313,760-byte probe includes multiple models and is not the
binary cost of deploying these two models alone.

With the earlier fixed hybrid ranker and bank constraints, YuNet offers61.0390%
of all requested essential crops and cuts nominated heads on20.9957%; UltraFace
offers33.7662% and cuts on15.4762%. This is a separate bank experiment from
direct origin placement. Higher coverage alone does not establish better crops.
All144 calls take2s, run-heavy peak RSS0.13GiB; the bank audit takes2s,1.30GiB.
Initialization and IO are included; earlier per-image latency observations on
other cohorts are not added together or transferred here.

## Rendered review

Both viewers use eight independently selected DEV witnesses and fixed seed31.
Each passes704 browser crop/clip/aspect checks and retains ten bounded screenshots;
four screenshots per viewer were visually inspected:3:1 Maximal,1:3 Minimal,
square Maximal and1:6 Maximal. The10% viewer gives nominal heads more room but
still puts tiny waterfall subjects near the bottom; constraints alone do not
make good framing. The automatic3:1 singer still truncates the face, and crowd
coverage remains incomplete. Supplied-region infeasibility correctly uses contain.

Invalid annotations and unrequested ratios explicitly say no audit was performed;
their comparison panes show the full source. The first viewer omitted this status
distinction and is retained separately as `2026-11-09-initial-status`; its crop
coordinates did not change when the label was corrected. No screenshot ratings
or artistic gold labels are admitted.

Evidence roots under `/mnt/v/output/zensally/`:
`subject-native-faces/2026-11-09`, `subject-origin-projection/2026-11-09`,
`subject-origin-projection-native/2026-11-09`,
`subject-origin-projection-padded/2026-11-09/{p10,p25}`,
`subject-origin-review/2026-11-09` and
`subject-origin-review-padded/2026-11-09`.
