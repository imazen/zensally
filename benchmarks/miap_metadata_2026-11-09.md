# Broader people annotations: metadata suitability — 2026-11-09

The first metadata stage validates70,000 official TRAIN image IDs and373,386
person-box records, giving a broader candidate supervision source than the
current18 people fitting scenes. No images are acquired, source roles assigned
or new labels admitted yet. It supplies person annotations, not artistic crop
preferences or certified complete faces.

The [official MIAP release](https://storage.googleapis.com/openimages/web/extended.html)
describes human person boxes supplementing Open Images. Its
[data card](https://storage.googleapis.com/openimages/open_images_extended_miap/Open%20Images%20Extended%20-%20MIAP%20-%20Data%20Card.pdf)
documents annotation scope and CC BY4.0 annotation terms. The
[Open Images description](https://storage.googleapis.com/openimages/web/factsfigures_v7.html)
distinguishes annotation and image terms. The saved source records keep full
URLs, response headers, byte hashes, raw documentation and hyperlinks. Per-image
author/license/rotation metadata still needs joining before pixel acquisition.

Measured directly from the cached TRAIN file:

- 373,381 boxes pass strict finite, positive normalized geometry.
- Five boxes have zero width or height; their five images are quarantined.
  Original rows/coordinates remain retained; no bound is clamped or relaxed.
- 7,947 records represent groups; counts are box records, not individual people.
- 100,738 records are flagged occluded,239,859 truncated and20,814 depictions.
  These flags overlap and do not establish photographic eligibility by themselves.
- All70,000 IDs join the official TRAIN key list; no validation/TEST pixels open.
- The official boxable classes include separate Person and Human face labels.
  MIAP's person boxes must not be silently treated as face annotations.

The initial parser stops at degenerate geometry. Recovery retains all source
bytes, records invalid rows explicitly and quarantines their images while
completing a metadata inventory. Both failure and completed logs/source versions
are retained. Perceived age/gender fields are neither features nor targets and
are not used for selection. No face-coverage flag is enabled.

Source metadata and inventory occupy63,071,295 bytes before freezing, below the
128MiB allowance. run-heavy completed validation2s, peakRSS0.09GiB, minimum
available22,947MiB, peak-load6.34 under1GiB/one thread; GNU-time maxRSS91,444KiB.
The failed first attempt took3s, peakRSS0.05GiB. These are metadata resource
observations, not training or inference benchmarks.

The image-information CSV is only HEAD-probed in this stage: declared
638,407,721 bytes, GCS generation1526567626302755 and ETag
`a4ac0bcedb5c2df4d1b1230dc6b41b8f`. A subsequent bounded acquisition should pin
that identity, join attribution/rotation/URL records, reconcile author/source
families and existing holdouts, then freeze roles before selecting any pixels.
Any independent face-class annotations and artistic targets remain separate.
No API key, paid model call, image download, training, model promotion or metric
map is involved. See the [protocol](../docs/experiments/MIAP_METADATA.md) and
[artifact pointer](miap_metadata_2026-11-09.pointer.md).
