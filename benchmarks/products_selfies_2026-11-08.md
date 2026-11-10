# Additional products/selfies and ratio-training plan (2026-11-08)

The owner requested more products and selfies, while accepting small AI-detail
artifacts. Eight additional high-quality Sunburst sources completed: four products
(speakers, tea packaging, watch-in-context and a dog harness) and four selfies
(solo bus, two people at night, family hike and mirror outfit). Three products use
white backgrounds and one uses context. Six sources use 960x720 / 720x960 and two
products use 1440x480 / 480x1440. These are smaller native generations, not resized
copies of previous images; all remain synthetic and supply no new crop labels.

Native published-rate usage is $0.249285, including prompt tokens. Total image
usage for all 56 sources is $2.344965 under the approved $2.50 cap, not invoice
billing. No retry occurred. All eight complete responses are cached, and a
credential-free replay makes eight hits, zero POSTs and no budget-journal change.

An opt-in size-specific reservation uses the published requested-size output
count and a conservative prompt UTF-8 byte bound, retaining the default global
envelope's $0.0321 headroom. The default $0.25 reservation is unchanged. Preflight
prior spend + all eight output/byte bounds + one pending margin is $2.415565,
below the runner's $2.45 ceiling. Actual native usage is independently reconciled
with the carried previous journal; no pending reservation remains.

All eight native PNGs agree with the owned PNG reader. Every lossless JXL recovers
exact source RGB through both owned and independent libjxl decoders. Payloads:
9,235,064 native PNG bytes, 5,225,479 lossless JXL bytes, 1,567,776 d0.5 JXL bytes.
The d0.5 variants remain separate comparisons with crop equivalence unmeasured.
Native PNGs retain embedded generation provenance in the archive.

The private dataset [ai-originals branch](https://github.com/imazen/zensally-datasets/tree/ai-originals)
now has 56 ordinary JXL blobs, with 101,887,093 total image bytes. Commit
`a3606f90964885d641c3a79f5098e7170609667a` is verified by complete local payload
hashes and remote GitHub branch/tree objects. Git attributes disable LFS filtering.
Per-source metadata keeps exact prompts, hashes, synthetic IDs/scenario families,
requested/observed primary-person counts and coordinator inspection. No family
split, face annotation or fitted crop-student result is invented by acquisition.

Eight budget/cache/calculator/scope/gallery tests pass. Browser checks cover all
eight sources and 56 centered baseline previews, with zero errors, overflow or
source distortion. Network gallery/inspection/usage/sample-image SHA256 readback
matches the frozen source manifest. The API wrapper reports:

```text
run-heavy: done rc=0 264s | peak-RSS 0.04GiB | min-avail 44552MiB | peak-load 3.56
```

The JXL encode/decode/oracle comparison wrapper reports:

```text
run-heavy: done rc=0 14s | peak-RSS 0.08GiB | min-avail 22245MiB | peak-load 15.34
```

These are bounded resource observations, not latency benchmarks.

The [next ratio-training design](../docs/experiments/CROP_FIRST_PROTOCOL.md#photographic-ratio-training-plan-2026-11-08)
uses the existing aspect-conditioned candidate ranker, source/task-balanced
boundary sampling, actual rendered teacher comparisons and a separate crop-
acceptability objective. It compares aspect-preserving rectangular whole-source
inputs and bounded local refinement while keeping face vetoes and contain
available. Existing fitting/holdout roles remain frozen. This design has not yet
been fitted or exported; the earlier 3:1/1:3 weaknesses remain the measured result.

Complete source/JXL evidence and independent NAS hashes:
[artifact pointer](products_selfies_2026-11-08.pointer.md).
