# Broader people supervision: metadata suitability first

The current photographic retention bank has18 people fitting scenes. Inspect
metadata for Open Images Extended MIAP before acquiring another pixel corpus.
The [official release](https://storage.googleapis.com/openimages/web/extended.html)
describes additional human person boxes; the
[data card](https://storage.googleapis.com/openimages/open_images_extended_miap/Open%20Images%20Extended%20-%20MIAP%20-%20Data%20Card.pdf)
documents CC BY4.0 annotation terms and group/occlusion/truncation flags.
These are person boxes, not face certificates or artistic crop preferences.
Images and annotations have separate provenance/terms; per-image author,
original URL, rotation and license metadata must be joined before acquisition.

Acquire only official TRAIN box metadata/key lists, class names and source
documentation. Cache complete bytes/headers/URLs/SHA256; preserve documentation
HTML and extracted hyperlinks. Probe the size of the larger image-information
CSV without downloading it in this stage. Budget128MiB for downloaded source
metadata and derived inventory,1GiB host/one thread; check disk headroom first.
No validation/TEST pixels, images, API keys or paid model calls.

Validate every TRAIN key/box ID, normalized finite bounds, class/confidence and
quality flags. Count box records, including group boxes; do not call record
counts individual-person counts. Summarize number/size/occlusion/truncation
variation per image to design later content coverage. Do not use perceived-age
or gender-presentation attributes as model inputs, selection features or targets.
Retain the original annotated file for provenance.

The initial full TRAIN scan finds five zero-area boxes. Keep strict geometry
bounds, retain each original row/line and quarantine every affected image;
do not clamp, repair or silently omit its annotation. Metadata inventory can
complete while such sources remain ineligible for any later fitting.

No new source role or training eligibility is assigned now. Before pixels or
fitting: join source/author/license/rotation records; reconcile original IDs,
provider URLs, author families and existing holdouts; freeze split/source roles;
choose a bounded content-balanced pilot; decode/resample only through owned
codecs/zenresize and independently verify outputs. Preserve existing corpora,
teacher caches, source roles and all face/numerical gates. Upstream pretraining
exposure remains a separate limitation. No complete-face, artistic-quality or
commercial-model qualification is inferred from metadata availability.
