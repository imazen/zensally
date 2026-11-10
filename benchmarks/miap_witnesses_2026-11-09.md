# People-supervision pixel witnesses — 2026-11-09

Twelve fixed fitting-only sources, one per first-pilot cell, acquire successfully
from the official CVDF image channel. No development/TEST image is opened.
The source JPEGs total4,345,440 bytes and all have1024-pixel long edges and EXIF
orientation1. Every independent source-header/owned-preview dimension check
passes; all twelve owned PNGs have exact independent RGB sample readback.
The initial payload is27,201,754 bytes, below128MiB. No target is admitted yet.

Browser checks decode all12 previews and verify all46 projected person-box
rectangles against saved source annotations. One2,127,434-byte contact screenshot
fits the8MiB witness allowance and is visually inspected. The boxes broadly align
in the ordinary photographic witnesses, including a painter, distant people,
watercraft, event interactions and street pedestrians. This is a bounded assistant
observation, not a human annotation or exhaustive face/body certificate.

The review changes the next data step:

- `7df0a4df4df67865` is a magazine cover with printed portraits. Every returned
  depiction flag is0, so metadata filtering did not enforce the document exclusion.
- `46953247dc4e523a` has real people and a mounted statue. A person box labels the
  statue despite depiction flag0. Withhold its ambiguous targets pending per-box
  review; do not change the retained source annotations.
- `10cafdec9a42e709` is a crowd represented by one group box and one individual
  box. Two records must not be treated as two people. A group box can also dominate
  the maximum-area scale proxy despite its individual people being small.

Before larger acquisition/fitting, add group-aware sampling and visual content
screening. No split is rerolled and no old source role changes. Person annotations
remain auxiliary localization/relationship targets, separate from face protection
and artistic composition preferences.

Owned fingerprint comparison covers3,102 pairs:3,036 against253 stored prior
references and66 internal pairs. No distance is at or below the unchanged10-bit
flag threshold, and no available exact encoded/working-RGB match is found. This
does not prove complete duplicate recall, particularly for old sources without
fingerprints. Existing held-out source pixels remain closed.

run-heavy acquisition9s, peakRSS0.04GiB, minimum available22,377MiB, peak-load2.30
under2GiB/one thread. Browser1s, peakRSS0.21GiB, minimum22,082MiB, load2.48.
Fingerprint review1s, peakRSS0.02GiB, minimum23,251MiB, load2.52 under1GiB.
The earlier direct screenshot CLI produces no screenshot and is manually stopped
after63s; its retained log/rc0 is not browser-verification evidence. The existing
CDP harness completes the subsequent full check. These are preparation costs.

The [official download instructions](https://storage.googleapis.com/openimages/web/download_v7.html)
link the [source downloader](https://github.com/openimages/dataset/blob/master/downloader.py),
which identifies the public bucket/object naming; its bytes are cached as
provenance and never executed. CVDF derivatives have their own content hashes;
OriginalMD5 does not identify these recompressed downloads. Full source/author/
license links remain in per-source records and `links.jsonl`. No paid model API,
training, face qualification or artistic-quality claim.
