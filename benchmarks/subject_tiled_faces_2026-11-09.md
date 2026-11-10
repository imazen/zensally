# Four overlapping YuNet tiles

Tiling improves overlap with nominated head centers but still fails as a complete
crop guard. It adds288 local calls and no paid calls, training or TEST inference.
The [registered control](../docs/experiments/TILED_FACE_CONTROL.md) fixes the same
72 photos, native binary, four60%-width/height corner windows and both NMS policies.
Owned RGB rows are copied exactly, with129,288,372 bytes retained across288 tiles.

| Input | Sources with proposals /72 | Proposals | Matched Sol DEV centers /84 |
|---|---:|---:|---:|
| Cached single pass |50|237|24|
| Full source plus four tiles |67|556|33|
| Same, excluding internal-edge proposals |66|550|33|

Matching uses the same22 structurally valid DEV sources. The67 valid FIT/DEV
sources have340 nominated heads including backs and occlusions; FIT matches
increase from141/256 to156/256. These are incomplete unreviewed nominations,
not human face recall, precision or whole-head containment. Source families and
roles remain fixed; no reference windows are supplied to the detector.

With the same three fixed hybrid rankers and candidate bank, essential-ratio
metrics balance22 DEV families and308 requested queries. The two tiled policies
give identical DEV crop results. Cut percentages divide by all requests.

| Face input | Crop offered | Nominated-head cut | Reference match |
|---|---:|---:|---:|
| Single pass |61.0390%|20.9957%|7.1429%|
| Full source plus tiles |93.1818%|30.1948%|7.9004%|
| Supplied Sol heads |90.5844%|0 by construction|9.8485%|

More crops become feasible under the incomplete detector inputs; the total
fraction cutting nominated heads increases. The table alone does not isolate
per-offer risk, prove a statistically reliable quality gain or establish that
each newly detected region is a true face.

The separate exact-origin audit also passes CPU batch/single and GPU numerical
checks with25%+two-source-pixel margins and the unchanged three-grid-pixel gate.
With largest-Maximal fallback, tiled faces plus pose offer88.3117%, cut nominated
heads on33.0087% and match either proposal8.2251%. The untiled25% control offers
65.5844%, cuts22.6190% and matches7.5758%. It remains unsafe; no crop model,
detector/tiling default or complete-coverage flag is promoted.

Tile evaluation takes8s, run-heavy peak RSS0.16GiB under2GiB/two-thread limits.
Preparation and mapped-output audit each report0s and0.02GiB. The fixed-ranker
guard audit takes2s/1.30GiB; exact-origin audit3s/1.09GiB. These are batch audit
observations including initialization and IO, not paired per-image latency
measurements. No deployment runtime cost is inferred by division or summation.
The first guard command supplied a nonexistent cache path and failed before
inference; its log is retained. The corrected caller uses the unchanged pinned
cache hash and quality/geometry tests.

Evidence: `/mnt/v/output/zensally/subject-tiled-faces/2026-11-09/`.
All347 members/153,160,648 payload bytes have independent NAS membership, size
and hash readback at `/mnt/tower/output/zensally/subject-tiled-faces/2026-11-09/`.
The SHA256SUMS.json hash is
`c7872a992e607cb062089fdd678a0629a16d9f86841c831a4020cd9b8a579b29`.
Resolve that evidence again with `just composition-verify-remote-packet dev <NAS path>`.
