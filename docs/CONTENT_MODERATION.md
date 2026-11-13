# Local image safety classifier assessment

Research checked 2026-11-12. Scope: evaluate local inference candidates for
automated image safety and policy classification, focusing on the accuracy/speed
tradeoff on CPU. This is a technical candidate assessment, not a compliance
determination.

The [2026-11-13 local evaluation](../benchmarks/local_moderation_2026-11-13.md)
provides measured CPU latency and runtime parity for all four candidates across
ONNX Runtime 1.31.1, `tract 0.23.10`, and the custom zero-allocation NHWC AVX2+FMA
`zensally-micro` SwiftFormer executor (**9.144 ms p50 / 9.419 ms p95** for SwiftFormer XS
and **14.729 ms p50 / 15.183 ms p95** for SwiftFormer S on one core of `i265`, beating both
`tract` and ONNX Runtime while preserving `<= 0.0001` probability parity).
It does not establish classifier accuracy or category recall on held-out labeled data.
Statements below about unperformed inference describe the initial research phase on
2026-11-12.

## Recommendation and coverage gap

Start an open-weight comparison with **OwenElliott SwiftFormer XS/S** and the
**Marqo ViT-Tiny-384** checkpoint. Use the **Freepik EVA-02-Large-448** checkpoint as
a larger-model reference. Published model cards report aggregate validation metrics on
their respective training distributions, which do not establish recall across all
downstream policy categories or edge cases.

Proprietary self-hosted classifiers exist for specialized safety taxonomies, but
public documentation does not establish fully offline operation, redistribution
rights, a pure-Rust embedding path, or comparable latency and recall at a stated
operating point. Services that transmit perceptual hashes to an external API do not
meet a strict no-outbound-request requirement.

## Open-weight shortlist

| Model key | Architecture / Input | Output classes | Why compare it |
|---|---|---|---|
| `swift-xs` ([OwenElliott/image-safety-classifier-xs](https://huggingface.co/OwenElliott/image-safety-classifier-xs)) | SwiftFormer-XS, 224 x 224 | 3 classes | Smallest hybrid CNN-ViT candidate; publisher supplies ONNX |
| `swift-s` ([OwenElliott/image-safety-classifier-s](https://huggingface.co/OwenElliott/image-safety-classifier-s)) | SwiftFormer-S, 224 x 224 | 3 classes | Same 3-class taxonomy with a larger SwiftFormer backbone |
| `marqo` (`vit_tiny_patch16_384`) | ViT-Tiny, 384 x 384 | 2 classes | Small ViT alternative trained on photos, illustrations, memes and synthetic imagery |
| `freepik` (`eva02_large_patch14_448`) | EVA-02-Large, 448 x 448 | 4 severity classes | Larger EVA-02 model with multi-level severity labels |

The SwiftFormer cards list MIT; the ViT-Tiny-384 checkpoint lists Apache-2.0; the
EVA-02-Large-448 model card lists MIT (while its Python wrapper repository separately
lists Apache-2.0: do not substitute the wrapper's license for the model's stated
license). Final model packaging still needs provenance and license review.
[SwiftFormer card](https://huggingface.co/OwenElliott/image-safety-classifier-s).

### Published quality evidence and limits

SwiftFormer reports 97.76% (XS) and 97.99% (S) accuracy on its internal validation
set. Because training data curation overlapped with the ViT-Tiny-384 workflow, an
ensemble should not assume independent errors. Secondary classes are underrepresented
in the training split, and held-out category recall across specialized policy domains
is unreported.
[Publisher's evaluation](https://huggingface.co/OwenElliott/image-safety-classifier-s).

The ViT-Tiny-384 checkpoint reports 98.56% accuracy on its own balanced 20,000-image
test set. This is not directly comparable to the SwiftFormer figures, and its binary
labels provide no fine-grained subcategory breakdown.

The EVA-02-Large-448 checkpoint reports internal category results and a separate
small manually labeled comparison, including synthetic images. Its four severity
labels and evaluation rules differ from the 2-class and 3-class classifiers. It is a
candidate for richer severity thresholds, not an established accuracy winner on the
target task.

### Speed evidence

Parameter counts and input resolution alone do not establish CPU latency rankings.

The original SwiftFormer authors report 0.7 ms for XS and 0.8 ms for S on iPhone
14 using Xcode's benchmark tooling. These are ImageNet backbone results on mobile
NPU/GPU hardware, not measurements of these classification checkpoints, Rust
inference, image decoding, or server CPU latency.
[Original architecture and measurement conditions](https://github.com/Amshaker/SwiftFormer).

The EVA-02-Large-448 card reports batch-one inference of 10 ms with tensor input and
28 ms with PIL input, using BF16 on an RTX 3090 GPU. These vendor GPU measurements
cannot be compared directly with single-core CPU inference.

## Pinned artifact identities

The following revisions, tensor counts and ONNX file identities were read from
the Hugging Face API. Tensor counts are Hub metadata, not independently counted
from downloaded weights.

| Model key | Revision | Hub safetensors parameter count |
|---|---|---:|
| `swift-xs` (`OwenElliott/image-safety-classifier-xs`) | `54f4560bd9c5ee92d45dc30418a8f8680e80de6d` | 3,040,646 FP32 |
| `swift-s` (`OwenElliott/image-safety-classifier-s`) | `015042b0eab17f1b17f2986527386346fb0d94be` | 5,652,374 FP32 |
| `marqo` (`vit_tiny_patch16_384`) | `0c26ec22111b83f106d72a55f611ec35962bcb65` | 5,597,762 FP32 |
| `freepik` (`eva02_large_patch14_448`) | `15b85477e4fd2000db76ae9aae0f89a72f95e2e3` | 86,351,620 BF16 |

Resolve metadata with `https://huggingface.co/api/models/<repository>`; resolve
configurations with `https://huggingface.co/<repository>/raw/<revision>/config.json`.
The pinned SwiftFormer tensors differ in count from the original backbone sizes
quoted in the cards; retain the distinction.

Publisher ONNX artifacts, from
`https://huggingface.co/api/models/<repository>/tree/<revision>/onnx`:

| Model | File | Bytes | LFS SHA-256 |
|---|---|---:|---|
| XS | `onnx/image-safety-classifier-xs.onnx` | 13,137,569 | `8c28c49d9075f3ad15ebdc2961f02d5b3f99be944815b848b49c9f0e6f3fb689` |
| S | `onnx/image-safety-classifier-s.onnx` | 23,701,765 | `fef443ed68ae25ed693b6fef9e456071692ed3963cff4168acb39c3de6f017e7` |

The cards' example download paths omit the `onnx/` directory. They also advertise
FP16 variants, but the inspected repository file lists contain only the ONNX
files above. Do not assume an unlisted FP16 artifact exists.

### Preprocessing must be verified

SwiftFormer native configurations specify bicubic interpolation, center cropping
with `crop_pct = 0.95`, and ImageNet channel normalization. The documented ONNX
example instead stretches RGB to 224 x 224 with bilinear interpolation and passes
0-255 float values, saying normalization and softmax are inside the graph.
Establish output parity and the intended input contract before benchmarking either
path; double normalization or a crop/resize change would invalidate the comparison.
[ONNX instructions](https://huggingface.co/OwenElliott/image-safety-classifier-s),
[native configuration](https://huggingface.co/OwenElliott/image-safety-classifier-s/blob/015042b0eab17f1b17f2986527386346fb0d94be/config.json).

The ViT-Tiny-384 configuration uses bicubic interpolation and center cropping, with
channel mean/std 0.5. The EVA-02-Large-448 configuration specifies bicubic squash
resizing and its own channel normalization. Whole-frame coverage matters for
classification; changing the default crop to padding, stretching or tiles is a new
quality experiment, not a neutral optimization.

## Local accuracy/speed selection

The evaluation protocol is finite: the four checkpoints above, one fixed labeled
evaluation manifest, a frozen set of thresholds selected on a separate calibration
split, and the target deployment hardware. Retain checkpoint, graph, runtime,
preprocessing, image and split identities with scalar results.

1. Establish graph loading and reference-output parity. Local inference does not
   imply pure Rust support: verify `tract` and custom executor compatibility against
   native FP32 outputs.
2. Measure positive-class recall at matched false-positive rates, with uncertainty
   intervals and separate results for photos, illustrations, synthetic images,
   and visually ambiguous negative examples (such as medical or fine-art imagery).
   Include small subjects, image edges, extreme aspect ratios and compression
   artifacts.
3. Record decoding/preprocessing, warm inference and total latency separately;
   report p50/p95, batch size, thread count, cold startup, and measured peak RSS.
   Compare FP32/BF16/INT8 variants separately if available. Quantization needs
   quality revalidation, especially for decisions near thresholds.
4. Prefer the smallest model that meets the selected recall and false-positive
   requirements. A larger model on ambiguous cases can reduce average work, but
   it cannot recover first-stage misses that were never escalated. Measure the
   complete cascade, its escalation rate, and its low-score misses.

## Perceptual hash matching is separate

For strictly local known-item matching, evaluate PDQ with an authorized compatible
hash feed and a local index. A hashing algorithm alone supplies no reference index,
and perceptual matching does not classify novel imagery. Meta publishes a PDQ
reference implementation and links a Rust implementation; compatibility,
preprocessing, feed permissions and matching thresholds require validation.
[Meta's PDQ documentation](https://github.com/meta/ThreatExchange/tree/main/pdq).

Keep perceptual hash matching independent of the neural classifier: a low classifier
score must not bypass required hash matching or override a match. A classifier score,
a known-hash match, and a downstream policy action should remain distinct records.

## Stopping point

This assessment establishes a local-model shortlist and reproducible artifact
identities. See the [2026-11-13 local benchmark report](../benchmarks/local_moderation_2026-11-13.md)
for measured CPU latencies and numerical parity across ONNX Runtime, `tract`, and
`zensally-micro`.
