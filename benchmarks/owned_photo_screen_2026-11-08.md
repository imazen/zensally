# Additional owned-photo screen — 2026-11-08

No artistic model is promoted. The useful additions are 101 real photographic
inputs, a larger numerical gate, and concrete detector misses for refinement.
All new photos remain review-only: no crop grades, training admission, changed
family role or canonical holdout image access. No paid model API was called.
The [frozen evidence pointer](owned_photo_screen_2026-11-08.pointer.md) records
complete independent NAS verification of the output and original input caches.

## Population and pixels

The pinned current imazen-26 JXL-sRGB TRAIN export supplies 101 PD-own sources
in 98 families: 21 general, 11 interiors, 54 nature and 15 food. Selection
excludes every source/family and exact original, pixel or encoded hash in the
existing fit/development/evaluation/test selection. This subset adds no
people-category photos; incidental people do occur. It does not fill the
people/commerce coverage gap by itself.

All encoded hashes/dimensions and all canonical RGBA8 pixel hashes match.
Owned packed-RGB hashes differ from canonical RGBA hashes by representation;
the streamed oracle explicitly reconstructs opaque RGBA before comparison.
Independent PNG readback matches owned native/preview RGB for all 101 images.
The 256-square tensors use aspect-preserving letterboxes; 768-long-side
previews preserve aspect ratio. No stretching is introduced.

The 684,508,794 encoded bytes produce 6,238,314,864 bytes of frozen input-cache
payload. Its 1,012 members have independent NAS membership/size/hash readback;
checksum-file SHA256 is
`1d884d295ed9e958a74d18aaf105aedb896aa493b88eb3542dc60004ccc5e16f`.
The input copy is referenced by the output packet rather than duplicated.

## Geometry is incomplete

Existing sparse full-range face detection returns ten boxes on nine images;
Pose Landmarker Lite returns nine poses on eight. EfficientDet Lite0 proposes
19 person boxes on 12 images; SSD MobileNet v2 proposes 14 on nine. Their union
is 15 person-positive images. These are model counts, not precision/recall or
population counts.

Eight posthoc visual witnesses expose both false detections and misses. Boxes
land on crystals, clouds, flowers and artwork. In 11048, two foreground skiers
and distant people are visible but whole-frame face/pose detection returns one
each. In 11214, the sole face box lies in ceiling artwork while live visitors
occupy the lower edge; the pose runner returns none. EfficientDet proposes five
visitor boxes and SSD two. Printed/painted faces are recorded separately from
photographed people; no global protection policy for depictions is inferred.

An offline person-region audit tests 264 requests on the 12 EfficientDet-positive
photos. Of 258 offered crops, 167 remove some whole-person-box area; six requests
contain the full image. This is **not** a face-cut or artistic-failure count:
body cropping can be intentional, boxes are unreviewed, and nondetections are
not declared person-free. Candidate search protects only supplied head regions.

## Runtime arithmetic on real photos

The unchanged reference generates 2,222 requests over eleven ratios and both
crop modes: 2,214 crops and eight contain results. All remain undeployable
research proposals. Cached analysis/query execution matches the full graph
bit for bit on all 1,111 score arrays and 1,111 importance arrays, with no crop
or action changes. Importance planes are compared in memory, not saved.

Native ordered and algebraic query paths each pass the unchanged three-map-pixel
edge/action gate. Both change one selection: 11554, 2:1 Minimal, shifts from
`[1,1351,1561,2131]` to `[0,1351,1560,2131]`, a one-source-pixel horizontal move
or 0.08199871877001921 map pixels. No action or supplied-head violation changes.
Each mode compares 142,208 scores; maximum absolute errors are
0.000003814697265625 ordered and 0.000003337860107421875 algebraic. Scalar and
SIMD agree bit for bit in both modes. The native choice digest therefore differs
from ORT; this is allowed by the registered rectangle contract.

The numerical input fixture is 46,070,498 bytes, below its 64 MiB budget. Its
six image witnesses were metadata-selected before inspecting errors. The
101 feature arrays and six normalized images are required runtime inputs,
not metric diagnostic maps. No learned weight changes here.

## Crop preview correction

Both SVG research viewers previously used viewBox without clipping the image.
Letterbox gutters could expose source pixels outside the selected rectangle.
Explicit source-coordinate clipping now fixes both builders. Four independent
screenshot gutter probes fail on the retained old rendering and pass after
correction. Browser checks decode all 101 photos, validate 2,222 integer/head
requests, exercise 16 extreme-ratio/mode/overlay settings, and test pagination,
filters and ratings. Three older 56-image SVG viewers also pass their rebuilt
browser checks with identical stored selections.

Old SVG screenshots/ratings remain retained as unreliable visual evidence;
new storage namespaces prevent their automatic reuse. Actual rendered-PNG
fitting labels and the five-photo preference comparison gallery are unaffected.
An initial screenshot probe misaligned after scrollbar reflow; the retained
note explains the corrected fixed-viewport sampling. Thresholds are unchanged.
All four deployed HTML files return HTTP 200 and match their checked local bytes.

## Resource evidence and limits

All process measurements use run-heavy on i265 with explicit memory/thread
bounds. These screen durations are not isolated latency benchmarks.

| Completed stage | Wall seconds | Peak RSS GiB |
|---|---:|---:|
| Acquire/decode/owned previews, 101 sources | 301 | 0.23 |
| Independent native/preview PNG oracle | 28 | 0.41 |
| Canonical RGBA pixel identity oracle | 5 | 0.03 |
| Runtime input preparation | 1 | 0.71 |
| Face and pose coordinate extraction | 2 | 0.24 |
| EfficientDet screening | 3 | 0.23 |
| SSD screening | 3 | 0.22 |
| Candidate proposals | 5 | 0.68 |
| Full-versus-cached numerical gate | 31 | 0.71 |
| Native dispatched / scalar gates | 1 / 3 | 0.04 / 0.04 |
| Independent input-cache NAS hash readback | 65 | 0.02 |

The dispatched ordered target itself records 12,508 KiB maximum RSS by GNU time
for this 101-source fixture; it is separate from wrapper process-tree memory.
The [earlier runtime study](cached_crop_runtime_2026-11-08.md) contains paired
timings, allocation profiling and hardware counters on its frozen 56-source
fixture. Its timings are not extrapolated to these 101 images.

## Native person-region refinement

The [bounded protocol](../docs/experiments/PERSON_ROI_REFINEMENT.md) completes
18 views and 54 model calls on the two observed people witnesses. All 18 owned
crop PNGs match independent RGB readback. View payload is 9,022,288 bytes.
Preparation takes 1 s / 0.04 GiB peak RSS; reference inference 1 s / 0.22 GiB;
the independent PNG oracle takes 0 s / 0.03 GiB. Browser checks decode all views
and exercise ten source/overlay combinations, with no errors.

| Source | Sparse face, person / upper | Short face, person / upper | Pose, person / upper |
|---|---:|---:|---:|
| 11048 | 3 / 0 | 1 / 1 | 2 / 2 |
| 11214 | 5 / 2 | 3 / 3 | 2 / 2 |

Counts are raw, possibly duplicate detections across crops, **not recovered
people or recall**. In 11048-1-person the sparse detector's box
`[141,229,51,51]` lies on the upper torso, below the covered face, despite its
0.5796005725860596 confidence. Pose facial landmarks align with the covered
face in the inspected overlay. Short-range face detection misses this person.
In 11214-0-person, a face/pose corresponds to a background visitor; the visible
foreground person is not thereby protected. Restricting to upper-body views
also loses detections rather than consistently improving them.

This supports a broader measured pose/head-refinement study, while ruling out
automatic face-certification from positive ROI counts. It does not establish
complete face coverage or justify training admission. More representative
reviewed people/commerce labels are still needed before claiming artistic
composition quality.
