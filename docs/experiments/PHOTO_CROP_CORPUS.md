# Photographic crop corpus

Scope selected by the owner on 2026-11-08: people across subject counts,
distances and appearances, plus pets, other living subjects and scenery.
News/editorial, sales/lifestyle and ecommerce are the intended uses.
Documents, charts, screenshots, manuscript scans and artwork reproductions
are outside this successor corpus. The frozen all-content pilot is retained.

## Coverage and crop judgments

The sampling target is 70% people, 15% pets/other animals, 10% scenery/plants,
and 5% standalone product/food photography. These are design targets, not
achieved coverage or a reason to duplicate scarce source families. Product use,
workplaces, shops and lifestyle scenes can also occupy the people stratum.
Generated product renders remain separately labelled auxiliary data.

Cross the people stratum with these visible attributes:

- One person, two, groups of 3–5, and groups/crowds of 6 or more; record
  uncertain counts rather than pretending every distant person is resolved.
- Close portraits, medium/half-body, full-body/environmental, distant people,
  and mixed foreground/background scales. Record face size relative to the
  source and in the actual model input, rather than inferring distance in metres.
- Looking room, profile views, action, occlusion, edge faces and widely separated
  people; hands/products, gestures and relationships that explain the image.
- Varied visible ages, skin tones, body shapes, clothing and accessibility
  contexts, without guessing identity, ethnicity, religion or gender identity.
- Editorial events/interviews, everyday activity, retail/work, product use,
  outdoor/travel and indoor scenes, with varied lighting and source aspect.

Pet/wildlife groups, people with animals, vegetation and environmental scenery
need their own coverage reports. A filename/category/tag is only a screening
hint; visual review establishes which cells a photograph actually fills.

Preserve every visible face/head. Crops must also retain useful gestures,
subject/product relationships and editorial meaning. Animal faces and relevant
body/action context matter. Copy space is useful when appropriate, not a
mandatory empty side. Photographed signs or packaging may be important context
without turning the task into document cropping. Existing Minimal/Maximal
semantics and the essential 1:3–3:1 range remain unchanged; 1:6/6:1 are secondary.

Proposal and rendered-review tasks opt into `composition_profile` equal to
`people-commerce-v1`. Both requests use the photographic rubric. Historical
tasks without that field retain their original request bytes/cache identities;
new prompts cannot reuse old judgments as though they used this rubric.

## Source identities, admission and measurement

Use immutable provider photo IDs, original URLs, photographer attribution,
download hashes, licence/terms captures and owned decoded-pixel hashes.
Resizes and crops inherit the original source identity. Retain source-native
ID case. Join against all imazen-26 source aliases and past evaluation exposure
before assigning a new source family. Keep existing internal development,
canonical validation and TEST assignments fixed.

Group bursts/near-duplicates before freezing splits; photographer/shoot groups
are an additional conservative barrier. Conflicting previously fixed roles
require quarantine, not moving a held-out image into TRAIN. Exact hashes and
provider IDs do not establish a complete perceptual duplicate audit. New
source IDs and their family/role registry must be frozen before fitting.

Holdout role freezes do not imply that every legacy content category must
remain in the target-domain evaluation slice. Select and report the photographic
slice explicitly, retain the full historical results, and do not tune on the
previously inspected canonical evaluation images. Report quality and abstention
by ratio, mode, people count, subject scale and intended use, with source-level
sample sizes. An aggregate dominated by single close portraits is insufficient.

`just crop-photo-audit <frozen-corpus> <fresh-output>` inventories photographic
TRAIN screening candidates without opening any image. It excludes historical
holdout families and the already fixed internal-development families. Its
candidates are not training-admitted and still require visual and rights review.

## Unsplash acquisition routes

Checked on 2026-11-08: the ordinary
[API guideline](https://help.unsplash.com/en/articles/2511256-guideline-high-quality-authentic-experiences)
excludes AI training. The separate
[Lite dataset](https://unsplash.com/data) offers 25,000 photos and permits
internal business model training under its
[dataset terms](https://github.com/unsplash/datasets/blob/master/TERMS.md).
Those terms restrict disclosure and redistribution. Do not silently treat
Lite as permission for external Terra/Sol annotation, public dataset/gallery
distribution, or customer-facing weight distribution. Broader written rights
or clarification are required for those uses. The Full dataset's noncommercial
grant is not interchangeable with Lite.

Owned photographs and verified CC0/public-domain sources are the preferred
teacher-ready route when broader Unsplash rights are unavailable. Any Lite
download remains an internal corpus with separate admission flags. No new
Unsplash download or teacher disclosure is claimed by the scope/audit tools.
