# People-pilot content screening

The twelve pixel witnesses expose a magazine cover and a statue target despite
depiction flags0. Use local captions to prioritize review, never as automatic
photographic admission, human annotation, face completeness or crop-quality gold.

Bind the existing native Florence-2-base assets at revision
`00921df66db728a9ceb750f5eca43e5c203a2051`. Verify every local asset; no download,
remote code or paid API. Keep FP16 eager inference, TF32 off, seed31, two host
threads,4GiB host cap and35% CUDA allocator allowance. Generate one deterministic
`<MORE_DETAILED_CAPTION>` per source, one beam, at most256 new tokens. Cache complete
tokens, raw text, parsing, normalization error, source role/pixel identity and
generation/model/library identities. A capped/invalid output remains retained
and unsuitable, not silently retried or admitted.

First run all twelve fitting-only witnesses. Fixed document/depiction phrases
in `caption_miap_content.py` produce review hints. Compare those with the already
recorded magazine/statue and ordinary-photo observations; report missed hints
and false flags. Do not tune a claim of general recall on these twelve images.
Then use the unchanged contract on all288 fit/96 development pilot images.
Source roles are fixed before either stage; TEST pixels remain closed.

Use the pinned owned resizer to prepare768x768 inputs from the owned working RGB.
This matches the scout's earlier stretch convention and is not a crop-runtime
preprocessing recommendation. Disable processor resize/crop and check pointwise
normalization against the independent formula at the unchanged1e-6 limit.
Retain inputs/responses within1GiB; identical source/input/task/model contracts
reuse complete responses, including the eight selected earlier witnesses.

The next visual review still checks photographic scope, group semantics and
ambiguous person/depiction targets. It must not infer individual counts from
group boxes or turn absence of a caption hint into acceptance. Human person
boxes support auxiliary relationships/localization; artistic preferences and
independent face protection remain separate requirements for crop fitting.
