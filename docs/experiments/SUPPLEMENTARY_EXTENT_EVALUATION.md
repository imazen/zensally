# Frozen extent models on supplementary development

Register model identities before reference generation. Use the fifteen-source
development registry SHA256
`a52144833a5824386577f63626e599f8dc602f8b77ee9d4d7f0c183574cebf70`.
Every source remains excluded from gradients; original canonical TRAIN and all
earlier fit/development/TEST assignments remain unchanged.

Freeze all 48 selected checkpoints from the completed Florence-target studies:
small/FocalNet/photographic frozen-descriptor MSE heads, nine each; small frozen
L1/GIoU heads, nine; end-to-end and augmented end-to-end heads, six each. Record
checkpoint SHA256, original selection epoch/arm/seed and source packet identities.
This is evaluation of fixed selections, not choosing epochs on this new cohort.

Use the existing pinned native Florence assets, owned Lanczos stretch768,
FP16/eager/TF32-off, deterministic three-beam decoding and512-new-token bound.
Run only the registry's frozen People/Monkeys phrase queries. Cache complete
input/task/version/source-role contracts, raw tokens/text, parsed boxes and
EOS/truncation status. Invalid/empty references stay unknown, never negative.
Valid entity unions are weak evaluation references, not human artistic labels
or complete face bounds. Preserve all entities and uncertainty.

After references are cached, compare the fixed models with original fit-only
normalization, source geometry, old mean-box/geometry and shuffled controls.
Qualify new descriptor extraction against prior cached inputs before interpreting
scores. Summarize source-family overlap and uncertainty; no final-TEST or
human-quality claim. No model or arithmetic gate is promoted or relaxed.
Teacher host cap6GiB/two threads, CUDA35%; evaluation4GiB/two threads.
No new paid API calls, training, source-role changes or diagnostic maps.
