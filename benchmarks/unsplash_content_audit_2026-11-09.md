# Photographic content coverage — 2026-11-09

The current development cohort does not cover the requested group/animal
scope. A visual source-preview review of all119 fit/development images finds
seven clearly visible group scenes and six primarily animal scenes, all in
fit. No frozen role, label, checkpoint or TEST source changes.

| Primary scene observation | Fit | Development | Total |
|---|---:|---:|---:|
| Product |38|12|50|
| People |18|2|20|
| Prepared food/drink |10|1|11|
| Animal |6|0|6|
| Plants/fungi |3|2|5|
| Scenery |8|1|9|
| Architecture/interior |7|5|12|
| Decorative still life |2|1|3|
| Apparent illustration/render |2|1|3|

These are broad Codex visual observations from320-long-edge source previews,
not verified semantic truth or population estimates. A different observation
field records people appearing in product/animal/craft scenes: fit has13 single-
person, three two-to-four-person, four larger-group and five partial-body views;
development has two single-person and two partial-body views. Three fit/three
development previews leave human presence unresolved. “None apparent” never
certifies absence of people or faces. Every face-certification flag stays false.

The cohort includes22 pale/white studio-style scenes,17 colored/dark studio
scenes and24 commercial/context scenes across its primary categories. The
three apparent renders and four rendering-uncertain sources stay flagged,
without retroactively rewriting the frozen photographic study. The source
with a moon/boat scene is one such apparent render. No document-focused image
is identified in this review; the painting/sketchbook image is a craft scene.

Native source aspect ratios range from0.5625 to1.777778; none reaches3:1 or1:3.
This does not remove existing extreme output-ratio tests, but it leaves genuinely
extreme photographic source framing unrepresented here. The synthetic sources
remain a separate source of wide/tall inputs and casual-family scenes.

Remote branch resolution on2026-11-09 confirms the same144-source
`unsplash-originals` commit`b8d7cd87562c271a7725bd446d656587a5886c01`.
The audit includes94 fit/81 families and25 development/21 families only; it
does not open25 TEST images. All119 preview PNGs independently read back to
the owned Lanczos output. Browser checks decode every preview, verify source
links/roles and capture12 fixed-order sheets without errors. Source URLs,
photographer metadata and all native/working/pixel identities remain attached.

Preview assets total12,026,040 bytes and12 screenshots total10,730,673 bytes,
below the registered64MiB/24MiB budgets. run-heavy on i265: build3s, peak RSS
0.02GiB, minimum available23421MiB, peak load5.03; browser3s/0.20GiB/23371MiB/
4.25. No metric diagnostic maps, paid API call, new image, training admission,
artistic grade or complete-face claim. See [protocol](../docs/experiments/UNSPLASH_CONTENT_AUDIT.md).

The useful next coverage check is the separate101-source screening-only
current-JXL cohort, with source/family and duplicate checks before reserving any
supplementary development scenes. Existing Unsplash fit/development/TEST roles
remain fixed. Further optimization of the tiny synthetic extent bank is not
supported by its [overfitting](florence_extent_finetune_2026-11-09.md) and
[augmentation controls](extent_augmentation_2026-11-09.md).
