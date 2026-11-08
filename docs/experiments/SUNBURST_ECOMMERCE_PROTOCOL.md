# Sunburst ecommerce image exploration

The owner requested GPT-Image-2.6 Sunburst at high quality, an exploration below
USD 1, cost-effective natural aspect ratios and detailed premium ecommerce
photographs with people/products. The owner will handle Unsplash and explicitly
approved a project API runner. This study changes no training inputs or models.

## Guidance and brief design

The [official prompting guide](https://developers.openai.com/api/docs/guides/image-prompting)
and [Images API guide](https://developers.openai.com/api/docs/guides/image-generation)
were fetched before prompt authoring. Briefs name intended use, physical scene,
subjects/actions, composition, light/materials and constraints. The imagery is
photorealistic, with ordinary skin/fabric detail and coherent product geometry.
People/product relationships, different subject scales, separated heads, useful
negative space and difficult narrow crops are explicit rather than every image
being an isolated centered product. Each call requests one coherent photograph.

`tools/crop-first/sunburst_briefs.py` owns 16 authored briefs: two square,
two 2:1, two 4:3, two 3:4, four 3:1 and four 1:3 images. Subjects include
outerwear, craftsmanship, skincare, adaptive travel, ceramics, furniture,
pet products, coffee equipment, lighting, cycling, jewelry, luggage, bags,
picnic products and hiking. People/animal counts in the plan are requested
content, not verified output annotations. Two ceramics images share exactly
the same prompt and scenario family for the square/wide pricing probe.

All outputs remain labelled synthetic and training-ineligible. Generated
people and settings are fictitious; these are commercial-photo fixtures,
not documentary evidence, human crop gold labels, or a replacement for real
photographic holdouts. Related scenario variants must share any future split.

## Calculator and monetary admission

Rates checked 2026-11-08 on the
[requested model page](https://developers.openai.com/api/docs/models/gpt-image-2.6-sunburst):
USD 30 per million output image tokens and USD 5 per million input text tokens.
This is text-only generation: no input-image or streaming-partial-image costs.
Use `quality=high`, `complexity=high`, `n=1`, opaque PNG output, explicit sizes,
and the exact requested model. Account model-list access is checked separately.

`sunburst_costs.py` reproduces the official calculator's high-quality formula
from its captured JavaScript: a long grid edge of 48, short grid edge rounded
ties-to-even from aspect ratio, and an area factor `(2,000,000 + pixels) /
4,000,000`, with the resulting tokens rounded up. This describes the estimator,
not an independently established model architecture. Its grid shrinks at wider
or taller ratios, so pixel count alone does not determine estimated cost.
Wide/tall estimates are symmetric.
The implementation preserves the JavaScript division order and binary64
rounding: exact rational arithmetic differs at four near-half-grid size pairs.
Dimensions satisfy the calculator's
multiple-of-16, pixel, edge and maximum-aspect constraints. The exact-ratio
minimum search also reports practical larger sizes.

The 16-image plan estimates USD 0.60876 for output alone. Actual native usage
must be checked before expanding the first probe. The project runner's local
admission ceiling is USD 0.95, with USD 0.25 reserved before each uncached POST.
The captured calculator's maximum high-quality output is 5,930 tokens (USD
0.17790); the reservation also covers the enforced 8,000-byte prompt bound
and margin. Reservations are replaced by native-usage cost after completion.
Missing usage, timeouts and unresolved attempts retain the reservation and
cannot be automatically posted again. This is conservative local admission,
not a provider-side billing cap or an invoice guarantee.

The cache hashes exact requests and preserves complete native responses,
usage and request IDs; API credentials are never recorded. Cache hits make
zero POSTs and verify stored response hashes. Every paid attempt is retained,
including errors. No retry, model substitution or quality reduction is implicit.
Native PNGs and complete evidence remain outside git with a tracked pointer.

## Reproduction

`just crop-sunburst-check` checks the calculator, cache tampering, retained
unknown-outcome reservations, budget refusal before network and native usage.
`crop-sunburst-costs` writes the estimate table; `crop-sunburst-generate` runs an
authored JSON plan against a persistent cache and study budget journal.
The configured dev API key is used only on dev. Inspect native dimensions,
photo quality and crop-relevant content before treating generated fixtures as
usable examples; generation itself does not validate crop quality.
