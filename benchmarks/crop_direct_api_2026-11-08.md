# Direct ROI/crop-zone API pilot and response cache

Four original previews, four stateless **gpt-6.1-terra Responses API POSTs**,
no tools. Each response includes ROI boxes and all seven ratios × two modes
in one call. Coordinates are original-image percentages, not resized-pixel
coordinates. Calls use the configured API key on dev; credentials never enter
request artifacts, logs or this repository.

| Source | ROIs | Crop decisions | Contain decisions | Rendered alternatives |
|---|---:|---:|---:|---:|
| 1446, landscape | 4 | 14 | 0 | 20 |
| 1556, abstract ice | 3 | 14 | 0 | 14 |
| 1636, food | 7 | 6 | 8 | 6 |
| 2006, portrait | 4 | 4 | 10 | 4 |
| Total | 18 | 38 | 18 | 44 |

All four inherit TRAIN roles from the prior frozen corpus subset. No new
TRAIN/VALIDATE/TEST split is created. Existing labels remain unreviewed and
training-ineligible; this pilot is not human calibration or face certification.

Owned decode/zenresize/PNG rendering produces 44 crops and four source previews.
Both owned and independent decoders agree on all 48 RGB hashes. Chromium checks
four source sections, 56 decision panels, 44 crop images, all 48 unique PNGs and
source-bounded overlays without exceptions. Zero projected crops exclude a
**proposed** face ROI; incomplete/inaccurate VLM face boxes remain a limitation.

Raw percentage boxes and exact-aspect integer projections remain separate.
Minimal projection uses maximum integer area. Maximal projection preserves
the prior 0.5–1 scale range. These rendered pixels have not received a second
teacher judgment. Different answers from the earlier candidate-ranking pilot
are not measured artistic improvements.

The coordinator visually spot-checks the portrait's 9:16 composition, abstract
8:1 ice band and landscape 1:8 strip. They are inspectable proposals, not human
gold labels. Food crops deliberately truncate some surrounding plates; the
gallery exposes those choices rather than calling every proposal high quality.

## Local response cache

The cache identity hashes the complete canonical request, including image bytes,
model, reasoning/settings, prompt and strict schema. Immutable entries retain
the raw request/response, native usage, response/request IDs, source role,
annotation validity and content hashes. Cache hits validate retained bytes and
reuse the response with **zero network requests**. File locking prevents duplicate
concurrent requests for the same entry. Explicit revalidation does not repurchase
a completed response.

Four repeated requests succeed without an API key in the caller environment:
four cache hits, zero POSTs. Mocked tests cover image identity changes, tampering,
credential exclusion, incomplete responses, and completed but invalid labels.
Incomplete/failed attempts remain retained separately; no automatic retry occurs.

The first response described containment with a full-source option. The initial
parser rejected that representation. The correction requires any contain zone
to preserve 0–100% of both axes, and recovers the already-saved response without
another POST. Completed raw responses are cached independently of whether their
annotations pass downstream geometry validation.

## Native usage and token pricing

The four calls record **4,836 input tokens**, including **4,824 cache-write
tokens**, zero cached-read tokens, and **6,288 output tokens**. Reasoning tokens
are already included in output. First food call: 1,310 input / 1,752 output.
These are direct API counts; previous CLI-session comparisons have a different
scope/prompt and are not a controlled cost ratio.

At the verified standard [Terra rates](https://developers.openai.com/api/docs/models/gpt-6.1-terra),
the token-priced total is **$0.037482**. Per the [official caching guide](https://developers.openai.com/api/docs/guides/prompt-caching),
omitting both TTL settings preserves the 30-minute default: writes cost $1.25/M,
ordinary input $1/M, reads $0.05/M and output $5/M. Write tokens replace ordinary
input billing; they are not charged twice. The response's legacy `24h` maximum
retention field is independent of the newer minimum TTL. No long-context
multiplier applies. This is a calculation from native usage, not a verified
invoice or subscription balance. Reusing a local response makes no API request.

Reproduction: `just crop-direct-zones`, `crop-direct-zones-check`,
`crop-direct-render`, `crop-preview-oracle`, `crop-direct-browser`.
Owned rendering wrapper: 1 s, peak RSS **0.09 GiB** (4 GiB cap, one job).
Browser wrapper: 1 s, peak RSS **0.18 GiB** (4 GiB cap, two jobs).

Review:

http://dev.lan:3300/zensally/crop-direct-api/2026-11-08/index.html
