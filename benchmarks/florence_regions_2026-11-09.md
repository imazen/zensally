# Local phrase-grounded regions — 2026-11-09

Florence-2 is useful for querying named products in this small scout. It does
not reliably isolate requested product details or certify complete subjects.
All inference is local; no paid API calls or source-role changes occur.

The pinned native Transformers conversion is
`florence-community/Florence-2-base@00921df66db728a9ceb750f5eca43e5c203a2051`:
231,443,968 parameters, 463,178,864 weight bytes, FP16/eager. All12 catalogued
asset files total468,546,278 bytes, including the original Microsoft MIT license.
Native Transformers classes load with zero missing/unexpected/mismatched keys,
local files only and remote code disabled. The acquisition's initial license
whitespace assertion failure is retained; resume verifies the exact license
hash without redownloading weights. Owned Lanczos stretch768 supplies geometry;
processor spatial resize/crop is disabled. This differs from canonical PIL
bicubic preprocessing and is not claimed pixel-equivalent.

Six previously reviewed Unsplash development sources each receive detailed
caption, dense-region and explicit phrase-grounding tasks. Six additional
feature queries follow. All24 unique generations finish with EOS, no truncation
or parser failure. All58 returned region boxes have valid native coordinates.
These checks establish output validity, not semantic accuracy or recall.

On three approximate, unreviewed catalogue product marks, explicit noun queries
have mean agreement IoU0.852348, compared with0.171284 for the first matching
dense-caption label. Oracle-best matching dense labels reach0.439552; this is
an upper-bound diagnostic, not a usable selector. The wallet noun box agrees
closely with its rough mark, whereas the dense caption associates a partial
wallet with the hand. Phrase grounding finds a distant person in the restaurant
passage that dense captions omit, but also returns a nearly full-frame passage
box. Unioning context indiscriminately would prevent useful cropping.

The embossed-wallet-brand query returns a separate plausible brand region
and a parent wallet region. The circular-home-button query instead returns the
whole phone while labelling it as the button. The branch query is broad. Caption
prose also contains visible contradictions. Neither labels nor fluent prose are
automatically admitted as factual gold; fine-feature and whole-object targets
must remain separate. No face-completeness flag is set.

run-heavy evidence on i265: bounded probe5s/2.06GiB peak RSS; complete object
scout7s/2.05GiB/min-available22879MiB/peak-load12.42; feature stage5s/2.02GiB/
min-available22767MiB/peak-load1.44. Scout uses three cached calls and generates
15 new ones. These are whole evaluation stages including model loading, not
per-image deployment latency. Scout CUDA peak allocated648,516,608 bytes and
reserved713,031,680 bytes are allocator counters, separate from host RSS.

See the [registered scout](../docs/experiments/FLORENCE_REGION_SUITABILITY.md)
and [subsequent controlled extent fits](florence_extents_2026-11-09.md).
The [native model documentation](https://huggingface.co/docs/transformers/main/model_doc/florence2),
[conversion](https://huggingface.co/florence-community/Florence-2-base) and
[Microsoft model card](https://huggingface.co/microsoft/Florence-2-base) identify
the task formats, model and license. Exact assets, failures, inputs, cached
tokens/raw text/entities and producer source remain protected outside Git.
