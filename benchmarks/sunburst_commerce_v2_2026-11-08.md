# Branded commerce and smartphone sources (2026-11-08)

All 32 requested sources completed with `gpt-image-2.6-sunburst`, high quality /
high complexity, native opaque PNG output and text-only input. No cosmetic retry
or extra text-model call was made. The owner accepts small generated anatomy,
utensil, lettering and other detail quirks for the target network.

The set contains 20 ecommerce photographs: 16 white seamless backgrounds and
four full-product-in-context scenes. Ten use 4:3/3:4 (five each); the other ten
span 3:1, 1:3, 2:1 and 1:1. Twelve amateur smartphone sources cover six birthdays
and six vacations, eight at 3:4 and four at 4:3. The coordinator inspected all
32 sources. Faces, people counts and source-edge margins are qualitative notes,
not exhaustive face labels. Some product-only prompts yielded extra people.

Four fictional brand wordmarks appear on products/packaging: Orvemelle,
Neralvique, Tavrovelle and Cevranel. Exact-name web searches found no results;
direct Google queries encountered a robot challenge, retained in the evidence.
Names remain provisional. Lettering errors and other small quirks are retained.

## Native usage and caching

| Study | Images | Published-rate USD, including text |
|---|---:|---:|
| Initial unbranded pilot | 16 | 0.637020 |
| Branded commerce and smartphone revision | 32 | 1.458660 |
| Total | 48 | 2.095680 |

This remains below the approved $2.50 total. Native usage is priced at the
[verified published rates](https://developers.openai.com/api/docs/models/gpt-image-2.6-sunburst),
not invoice billing. Each new native PNG/usage record matches its complete cached
response and the calculator output-token count. The predecessor journal is bound
by SHA256 and carried exactly once. No pending reservations remain. A replay with
credentials absent produced 32 cache hits, zero POSTs and an unchanged journal.

## JXL storage

All images retain native dimensions. Owned zenpng decoding agrees with an
independent PNG reader; owned and independent libjxl JXL decoding recover exactly
the original RGB pixels for every lossless file.

| Sources | PNG bytes | Lossless JXL bytes | Distance-0.5 JXL bytes |
|---|---:|---:|---:|
| Initial 16 | 67,332,833 | 39,696,986 | 11,824,443 |
| New 32 | 100,617,954 | 56,964,628 | 17,200,242 |
| All 48 | 167,950,787 | 96,661,614 | 29,024,685 |

Lossless reduces total image payload by 42.45%; d0.5 by 82.72%. Choose lossless
for original dataset blobs, with d0.5 kept separately for comparison. The new
32 d0.5 readbacks have RGB PSNR 39.5650–49.0734 dB. Displayed face/lettering spot
checks looked close; no blind rating or crop-coordinate equivalence test was run.
Native PNGs and embedded generation provenance remain in the archive. Signed PNG
metadata is not transplanted to a changed file format.

The private dataset's [ai-originals branch](https://github.com/imazen/zensally-datasets/tree/ai-originals)
contains all 48 lossless files as ordinary Git blobs, explicitly disabling LFS
filters. Commit `32fd7f59166a8667cc9680dda1ed76ea2cbde3b0` was verified locally
against every source SHA256 and through GitHub's remote branch/tree APIs: 48 blob
objects, 96,661,614 image bytes. Metadata carries prompts, native usage, synthetic
IDs, scenario families, original/pixel hashes and inspection notes. Crop labels,
trusted face boxes and split admission are not supplied by image generation.
No new crop-student fit or export is implied by this source acquisition.

## Verification and artifacts

Seven budget/cache/calculator/scope/gallery tests pass; both standalone Rust
storage tools pass clippy with warnings denied. Browser checks cover all 32
sources and 224 centered baseline previews, with no JavaScript exceptions,
mobile overflow or source distortion. Centered previews are explicitly labelled
as baselines, not crop-generator recommendations. Native network gallery HTML,
review, verification report and a sample image match local SHA256 readback.

```text
API batch: run-heavy: done rc=0 1216s | peak-RSS 0.08GiB | min-avail 42367MiB | peak-load 5.96
Native verification: run-heavy: done rc=0 2s | peak-RSS 0.08GiB | min-avail 24093MiB | peak-load 13.98
Browser: run-heavy: done rc=0 1s | peak-RSS 0.19GiB | min-avail 24053MiB | peak-load 14.41
```

Incremental JXL encoding/readback logs report at most 0.30 GiB peak RSS per
bounded wrapper (5 GiB cap, two jobs). These are resource observations, not
controlled encoder-speed benchmarks. Source/worker identities, every intermediate
readback, all native responses, captured searches, usage and full logs are in the
[artifact packet](sunburst_commerce_v2_2026-11-08.pointer.md).
