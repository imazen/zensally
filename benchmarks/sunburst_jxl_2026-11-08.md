# AI-source JXL storage experiment (2026-11-08)

All 16 pilot images were encoded at native dimensions using the owned encoder,
effort 7. This measures storage and decoded samples, not runtime crop equivalence.

| Format | Total bytes | Savings versus native PNG |
|---|---:|---:|
| Native PNG | 67,332,833 | — |
| JXL lossless | 39,696,986 | 41.04% |
| JXL distance 0.5 | 11,824,443 | 82.44% |

Lossless: 16/16 exactly recover native RGB8 pixels through both zenjxl-decoder
and independent libjxl. The owned PNG reader also matched an independent PNG
reader on every source. Distance 0.5 changes samples; per-image RGB PSNR spans
39.1545–45.1395 dB, and the two JXL decoders differed by at most one sample value.
The source/d0.5 jewelry portrait looked similar at displayed resolution; no blind
ratings, complete logo-detail review or crop-coordinate stability test was run.

Choose lossless JXL for dataset originals. Keep d0.5 variants as comparison data.
Native PNGs retain embedded generation provenance in the archived API packet;
the JXL derivatives do not transplant signed PNG metadata.

The 16 lossless files are regular Git blobs on
[ai-originals](https://github.com/imazen/zensally-datasets/tree/ai-originals/ai-originals/sunburst-pilot-v1),
commit `f58d42423fcc72536216611d21664a3a5b712677` (39,696,986 image bytes).
Each committed blob's length and SHA256 was checked against the compression
manifest. Git attributes explicitly disable LFS filtering. The remote branch
commit was independently checked through GitHub's API.

The 89-second wrapper reported `peak-RSS 0.32GiB`, 5 GiB cap, two jobs on i265.
Encoder build: 21 seconds, 0.81 GiB peak; PNG reader build: 11 seconds, 0.43 GiB.
Both standalone tools passed clippy with warnings denied. Source revision:
`3fb3ddf14b78513f90f97778ac83a55f839f49c4`; pinned codec revisions, features,
binary hashes, raw readbacks, independent oracle images and complete logs are in
`_MANIFEST.json` / the [artifact packet](sunburst_jxl_2026-11-08.pointer.md).
