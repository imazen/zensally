# Preview token probe and Astra consultation, 2026-11-07

Three fresh GPT-6.1 Terra CLI requests attach the same source at different
zenresize resolutions. Each uses the same prompt, low reasoning, no tools,
one image and an OK-only reply. Non-image message content matches after
normalizing the attachment path; image filenames differ. Input below is the
whole CLI request, including instructions, not an image-only API usage meter.

| Preview | Total input tokens | Cached input | Output |
|---|---:|---:|---:|
| 1024×768 | 31,991 | 12,032 | 5 |
| 768×576 | 31,588 | 12,032 | 5 |
| 512×384 | 31,300 | 12,032 | 5 |

The matched request reduction is 403 tokens at 768×576 and 691 at 512×384.
These observations cover one source, not every corpus aspect or a quality
study. Cropping accuracy/preference and face coverage at the reduced sizes
are not established. The expanded pilot uses 768-pixel source previews and
keeps its judgments unreviewed and excluded from training.

The [official vision guide](https://developers.openai.com/api/docs/guides/images-vision)
was checked 2026-11-07. Its sizing table does not explicitly list Terra 6.1;
we do not silently assign another model's rule to it. No API-key-backed
image-only meter or actual Codex credit charge was available.

Reproduce the audit with `just crop-token-compare CASES OUTPUT`; it verifies
one tool-free turn and matching normalized non-image messages. Retained
metadata includes native usage, normalized-prompt SHA256 and image SHA256.

The user-requested [Astra design consultation](../docs/experiments/CROP_NETWORK_ASTRA_2026-11-07.md)
recommends a shared convolutional encoder, projected stride-8/16 features,
fractional area pooling for retained/discarded/boundary regions, and separate
ranking/acceptability heads. Deterministic face-constrained search stays outside
learned scores. Fixed aspect-preserving input shapes and optional local views
are proposed ablations. No new network was trained or profiled.

Astra inspected source f2465355 read-only. The full recommendation and native
usage are preserved; it is model advice, not a benchmark or an independent
human review. Current crop-ranker architecture, weights and runtime contracts
remain unchanged.
