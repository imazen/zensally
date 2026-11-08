# Acquired-photo relationship screen

Register before computing fingerprints or assigning new source roles. Use the
144 immutable owned strict-sRGB working RGB previews from the qualified Unsplash
packet. Canonical VALIDATE/TEST pixels remain closed. Original source IDs retain
case; all existing source/family roles remain unchanged.

Reuse zensimx's exposure-normalized linear-luma fingerprint and shared dHash bit
ordering from commit`ed956176e02e335d4a6efb3c31724e2aaf361d34`. Its existing
exposure/unrelated/nonfinite/black/geometry controls must pass. The frontend
reads declared sRGB RGB8 with width/rows/stride, converts rows with the owned
converter and keeps the same luma coefficients. Native source originals and
1280 working previews are distinct representations: these hashes form a new
era, never mixed with older native/encoded hash authorities.

Before the full screen, require identical hashes for the same source under
tight and padded-stride representations. Keep reduced float luma only as
bounded working memory; save scalar fingerprints/distances and identities,
not diagnostic images. Compare all10,296 unordered pairs and retain every
Hamming distance at most10 as a review flag, not a duplicate/no-duplicate label.
The threshold is a screening choice, not a measured recall claim. Flat/similar
compositions and cropped/edited relatives limit this whole-image fingerprint.

Start with124 conservative photographer groups. Merge graph components for
shared photographer, exact original hash and every flagged pair, retaining
the precise reason for each edge. A flagged edge means possible relationship,
not verified identical scene. This conservative barrier can merge unrelated
photos. Report component sizes and across-photographer flags explicitly.
The screen does not certify absence of all perceptual relatives or establish
the image quality/rights/label admission of a source.

Do not assign fit/dev/test roles in this fingerprint phase. A later registered
role freeze must preserve photographer/possible-lineage components, document
all prior exposure and source-ID derivation, and keep any canonical identity
conflict quarantined. Bound preparation to2GiB/two CPU threads; no model API,
quality-model promotion, training gradients or metric output maps.
