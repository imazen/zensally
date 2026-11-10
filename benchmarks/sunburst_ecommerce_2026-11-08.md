# Sunburst high-quality ecommerce exploration (2026-11-08)

Sixteen native PNG photographs were generated with `gpt-image-2.6-sunburst`,
`quality=high`, `complexity=high`, opaque backgrounds and no image inputs.
Native usage totals 20,292 output image tokens and 5,652 input text tokens:
**USD 0.637020** at published rates, below the owner's USD 1 exploration budget.
This is not invoice verification. No model/quality substitution, automatic
retry or separate text-model call was used. The owner is handling Unsplash.

The [official model page](https://developers.openai.com/api/docs/models/gpt-image-2.6-sunburst)
lists USD 30/million output image tokens and USD 5/million input text tokens.
The account model list exposed the requested ID; actual paid calls succeeded.
All 16 native output-token counts match the captured calculator exactly.

## Why the wider image can cost less

The [official calculator](https://developers.openai.com/api/docs/guides/image-generation#calculating-costs)
uses a high-quality long grid edge of 48. Its short grid edge decreases with
aspect ratio, followed by an area factor:

```text
short_grid = round_ties_to_even_binary64(48 / (long_edge / short_edge))
output_tokens = ceil(48 * short_grid * (2,000,000 + width * height) / 4,000,000)
```

This is the published estimator, not a claim about the model's architecture.
Increasing pixel area can be outweighed by the smaller grid. The wide/tall
calculation is symmetric.

The ceramics pair uses exactly the same authored prompt and the same 360 text
input tokens, with one generation at each size:

| Native output | Image tokens | Image output cost | Total including text |
|---|---:|---:|---:|
| 1024×1024 | 1,756 | $0.05268 | $0.05448 |
| 2048×1024 | 1,180 | $0.03540 | $0.03720 |

The cheaper wide estimate is supported by actual usage for this pair. The
scenes/background people differ: these are separate generations, not identical
scenes or resizes. No comparative artistic-quality claim follows from the price.

The Python estimator is checked against the captured original JavaScript for
**all 29,146 valid dimension pairs**: zero differences after preserving its
binary64 division order. An initial exact-rational version differed at four
near-half-grid sizes; that attempt is retained. No planned size or table entry
was affected. The largest calculator high-quality count is 5,930 output tokens.

## Cost-effective exact ratios

These are image-output-only calculator estimates. Text input is additional.
Every dimension is a multiple of 16. Minima apply to each exact ratio under
the captured pixel/edge limits; the smallest permitted image is not necessarily
the best source for small faces or product detail.

| Ratio | Lowest-cost exact size | Estimated output cost | Generated practical size | Native output cost |
|---|---|---:|---|---:|
| 1:1 | 816×816 | $0.04608 | 1024×1024 | $0.05268 |
| 4:3 | 960×720 | $0.03489 | 1536×1152 | $0.04887 |
| 3:4 | 720×960 | $0.03489 | 1152×1536 | $0.04887 |
| 2:1 | 1152×576 | $0.02304 | 2048×1024 | $0.03540 |
| 3:1 | 1440×480 | $0.01551 | 3072×1024 | $0.02964 |
| 1:3 | 480×1440 | $0.01551 | 1024×3072 | $0.02964 |

Among the requested ratios, 3:1 and 1:3 have the lowest estimated minimum cost.
Their generated practical sizes provide 3,145,728 native pixels each. Wider
ratios are not universally cheaper at every resolution: larger choices can cost
more than smaller images of another ratio. The complete estimate packet also
covers 3:2/2:3, 16:9/9:16 and 1:2.

## Photo briefs and crop lessons

The [official prompting guide](https://developers.openai.com/api/docs/guides/image-prompting)
was read before authoring. Briefs specify physical scenes, product materials,
people/actions, spatial relationships, lighting and source-frame constraints.
They request photorealistic texture and one coherent photograph, with no
overlays, invented logos or interface mockups. Final prompts are retained in
`plan.json`, the gallery, and `tools/crop-first/sunburst_briefs.py`.

The set contains two square, two 2:1, two 4:3, two 3:4, four 3:1 and four 1:3
images. Requested content spans solo people, pairs, four-person scenes and a
six-person group; three scenes include dogs, and one is product-only. These
are requested attributes, not certified annotations. The coordinator inspected
all 16 native sources and retained specific observations in `review.json`.

Useful cases for the crop generator:

- Outerwear, cycling, luggage and picnic scenes place faces at different
  positions/scales. A small background person remains meaningful.
- Bootmaker, jewelry and floor-lamp portraits separate faces, hands and products
  vertically. Face preservation alone does not establish a good crop.
- Skincare and ceramics make the face/product relationship central; adaptive
  travel adds product-use context that a tighter crop could remove.
- Furniture and pet-bed scenes make the product footprint important alongside
  people/animal faces. The dark tote tests a low-contrast main product against
  brighter environmental illumination.

Recommendation for later supervision: distinguish product-listing/thumbnail,
hero/lifestyle and deliberate product-detail intent. The same framing may be
useful for one and unsuitable for another. Preserve face constraints while
judging product identity, interaction, background subjects and composition.
This intent-conditioning proposal is not implemented or measured here.

Visible limits: some source frames clip hair or product edges despite the
prompt, and several scenes converge on similar models/hairstyles and warm
interior styling. The espresso machine reaches the right source boundary;
some foreground/background heads reach the top boundary. Distinguish generated
source truncation from later crop damage. No reference SKU was supplied, so
exact product identity, engineering and branding are unvalidated. Keep related
scenario variants grouped and audit cross-scene similarity before future use.
These synthetic fixtures remain training-ineligible and supply no human crop
gold labels, real-news evidence, complete face coverage or shipping quality claim.

## Verification and retained evidence

Native Images API responses/usage are content-addressed and immutable. The
monetary journal has 16 settled requests and zero unresolved reservations.
Credential-free replay returns all 16 from cache without new POSTs; the budget
journal is unchanged. Original worker source is pinned to `2c771ad1b505`;
subsequent calculator/gallery sources are separately identified.

Read-only independent Pillow decoding verifies every native PNG's exact size,
opacity and stored SHA256; no reader rewrites model-input pixels. Chromium
checks all 16 sources, all 112 centered-preview combinations and narrow-screen
layout with zero image/dimension/ratio failures or JavaScript exceptions.
Center previews are labelled baselines, never learned crops. Served HTML,
summary, review notes and calculator data match local SHA256s.

The API batch resource wrapper records peak RSS 0.09 GiB; the browser wrapper
records 0.19 GiB. These are resource checks, not controlled latency benchmarks.
Budget/cache/calculator controls pass locally and are in lightweight CI.

```text
API batch: run-heavy: done rc=0 528s | peak-RSS 0.09GiB | min-avail 41499MiB | peak-load 4.41
Browser: run-heavy: done rc=0 1s | peak-RSS 0.19GiB | min-avail 23978MiB | peak-load 13.70
```

Native images, complete responses, prompts, usage, calculator captures, source
snapshots, qualitative review, browser checks and hashes remain outside zensally
git. Verified lossless JXL derivatives are ordinary blobs in the separate private
dataset branch; see the [completed revision](sunburst_commerce_v2_2026-11-08.md).
See the [evidence pointer](sunburst_ecommerce_2026-11-08.pointer.md).
