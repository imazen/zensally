# Source-aware local pair critic suitability

Register before inference. The unconditional PhotoFramer scalar protocol fails
its six-pair gate even in BF16. Keep that failure and its wider-panel restriction.
This separate diagnostic asks whether source context and direct comparison can
produce useful, position-consistent weak pair choices. It does not qualify a
new labeler or undo the previous screen.

Reuse the same six mixed-content historical calibration pairs, all still
training-ineligible. Each request contains whole-source context, candidate A
and candidate B, target aspect ratio and requested Minimal/Maximal mode. Whole
source previews use the owned resizer, at most336 pixels per axis, dimensions
rounded to multiples of28; crop inputs are the exact previous owned RGB bytes.
Record the slight source-preview aspect approximation. No third-party encoding.

Two deterministic orders per pair: accepted as A, then accepted as B. Neither
label nor historical decision is supplied in the prompt. Fixed prompt requests
one JSON choice A/B/tie and one short reason code, without reasoning text.
Seed31, deterministic generation,256-token cap, original pinned BF16 model,
10GiB GPU/10GiB CPU weight budgets,80-percent runtime GPU cap,16GiB host and
two threads. Preserve raw responses, inputs, placement, grids, token counts,
parse failures, truncations and cache protocol. Reference previous derived
processor files rather than duplicating tokenizer metadata. No downloads or
paid API calls, student gradients or TEST. First one-call probe, then explicit
12-call bound if resource-safe. Never restart a live process on observation timeout.

Predeclare success only if all12 parse without truncation, all six pair
choices agree with the weak preference in both orders, and all six are swap
consistent. Ties fail the preference screen. Even success supplies only reused
weak-calibration evidence; no score, excellent endpoint or interval is admitted.
Report both orders and all failures, not just aggregate accuracy. Total new
artifact budget4MiB. No output images or diagnostic maps needed.
