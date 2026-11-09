# Phrase-grounded primary-extent fitting follow-up

Register before full synthetic inference or gradients. The six-image Florence
scout returns valid geometry for all 50 object/dense-caption boxes. Explicit
product queries agree more closely with three approximate catalogue marks than
first matching dense-caption labels (mean IoU 0.852348 versus 0.171284); even
oracle-best matching dense labels reach only 0.439552. These are three unreviewed
marks, not independent quality truth. Feature grounding can instead return a
parent object: the phone-button query returns the phone. Caption prose is not
admitted as factual supervision.

Reuse the existing 56 synthetic sources, their exact owned native RGBs and the
already frozen primary-object queries from the Grounding-DINO extent study.
Keep all 48-fit/27-family and eight-development/eight-family assignments.
Fictional-brand and scenario groups never cross roles. Join the legacy auxiliary
admission and runtime manifest by source ID and RGB SHA256 before new inference;
do not process the runtime manifest's other 20 sources. No canonical holdout
or new Unsplash TEST inference occurs.

Run only Florence's caption-to-phrase grounding, with the existing pinned native
model/assets, owned Lanczos stretch768, FP16/eager, TF32 disabled, deterministic
three-beam decoding and 512-new-token limit. Cache complete generation contracts,
tokens, raw text, parsed entities, EOS/truncation and source-role identities.
All 56 are local calls, not paid API calls. Inputs/responses remain protected.

First report geometry, labels and missing/invalid sources. No detection, parser
failure, token-cap result or invalid coordinate means unknown, not a negative.
The weak target is the union of returned valid entity boxes, matching the old
primary-query intent. Compound queries/false positives can inflate it; preserve
all entities and inspect disagreement witnesses. No source is face-certified.

This follow-up separately admits these new primary-extent soft targets for
fitting-only sources after identity/geometry checks. Development targets supply
selection diagnostics, never gradients. It does not admit old crop judgments,
gaze, face labels or photographic quality grades, and does not rewrite the older
Grounding-DINO cache/results.

Reuse the old frozen 512-visual/four-geometry descriptor cache from the epoch11
small encoder, preserving every descriptor/source/encoder identity. Replace only
targets/presence and their provenance in a new cache namespace. Reuse the owned
516→64→6 ExtentHead and existing fit-mean, aspect-only, image and role/aspect
permutation controls: seeds31/137/911, 200 epochs, FP32/TF32-off, batch16,
AdamW0.001/weight decay0.01, gradient clip5, edge MSE and epoch-zero selection.
This first comparison isolates the changed target source. Existing source-based
weighting/permutation remain unchanged and their family imbalance is reported.
Do not relabel a reused development comparison as independent confirmation.

Retain all nine fits, checkpoints, histories, predictions and the mean control.
Use 6GiB host/two threads and at most35% GPU memory for teacher inference;
4GiB host/two threads for fitting, with the same GPU bound. Keep state artifacts
below4GiB. No new public API, model promotion, paid inference or diagnostic maps.
Any later end-to-end training or training-path optimization needs a separate
registered comparison after this suitability result.
