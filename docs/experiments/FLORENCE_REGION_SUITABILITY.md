# Local region-label suitability scout

Use a local model to test product/feature region discovery without paid API
inference. Native Transformers documentation supports Florence-2 object
detection, dense-region captions and phrase grounding:
[Florence-2 documentation](https://huggingface.co/docs/transformers/main/model_doc/florence2).
This is a region-label scout, not an aesthetic scorer or a face certificate.

On 2026-11-09 the public, ungated
[`florence-community/Florence-2-base`](https://huggingface.co/florence-community/Florence-2-base)
resolves to `00921df66db728a9ceb750f5eca43e5c203a2051`, advertised MIT.
Its safetensors file is 463,178,864 bytes, SHA256
`62f3e696da74f8869a68ddb529a9b3e14eb25b21c592cb3dea6179bf944df6a0`.
The original [Microsoft model](https://huggingface.co/microsoft/Florence-2-base)
resolves to `5ca5edf5bd017b9919c05d08aebef5e4c7ac3bac`; preserve its license
and the converted model card/provenance. These are dated observations; resolve
`HfApi(token=False).model_info(repo, files_metadata=True)` before acquisition.

Use installed native `Florence2ForConditionalGeneration`/`Florence2Processor`,
with remote model code disabled. Pin all asset files to the resolved revision,
check sizes/LFS SHA256s and record URLs. Bound model assets to 600MiB and preserve
failed downloads/outputs. Do not change the existing teacher/student dependency
environment merely to make this scout load.

Start with the wallet source from the fixed six-image development review.
Only after a valid bounded call succeeds, evaluate all six development sources.
Keep source IDs/roles and all historical labels unchanged. Record complete raw
token sequences, generated text, parsed boxes, EOS/truncation status, generation
parameters and full source/model/owned-input identities. Cache by the complete
inference contract; repeated inspection must not re-run generation.

Use owned working RGB and the pinned owned resizer for the model's documented
input geometry, then disable processor resizing. Do not introduce third-party
image resampling into labels. Record the processor's actual normalization and
source-coordinate restoration. No rendered diagnostic maps or masks.

Scout tasks are more-detailed caption, dense-region caption, and phrase grounding
for a nominated source object. Evaluate agreement with the existing approximate
regions, preserve unexpected labels/boxes and inspect actual output overlays.
Those unreviewed boxes do not establish recall, complete product bounds or
acceptable crops. No targets are admitted to fitting by this experiment.

Run batch1 FP16 (the model card's trained precision), deterministic decoding,
maximum512 new tokens and three beams. Use a 6GiB host cap/two CPU threads and
35% of device memory, checking current work first. Retain CUDA allocated/reserved
bytes and real target RSS. No inference-latency claim from a profiler/setup run;
profile before optimizing only if the scout proves suitable.
