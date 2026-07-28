# preprocess_nchw cost — 2026-07-28

Platform: Apple Silicon (aarch64, NEON), darwin 25.5.0
Bench: `crates/zensally/benches/preprocess.rs` (zenbench)

`preprocess_nchw` runs once per image before ONNX inference: bilinear resample + normalize +
NCHW planar scatter, fully scalar. This crate had no benchmarks, so its cost was unknown.

| source → target | time |
|---|---|
| 4032×3024 → 320×240 | 519 µs |
| 4032×3024 → 320×320 | 529 µs |
| 1920×1080 → 320×240 | 401 µs |
| 1920×1080 → 320×320 | 397 µs |

## The shape of the cost — and why NEON does not help

Cost tracks the **source** size, not the output size. 320×240 and 320×320 outputs cost the
same (519 vs 529 µs) despite a 33% difference in output pixels, while halving the source
linear dimension drops it ~23%. So the loop is not bound by the arithmetic per output pixel.

It is bound by the source access pattern. Downscaling 4032→320 is a 12.6× reduction, so
consecutive output pixels sample source pixels ~12.6 px (50 bytes) apart. A 64-byte cache line
holds 16 RGBA pixels, so nearly every sample lands in a different line and ~60 of every 64
fetched bytes are discarded. Effective throughput is ~15 GB/s against this host's ~118 GB/s
ceiling, which is the tell.

**That is a gather, and AArch64 has no gather instruction.** Vectorizing the arithmetic would
not move a loop that is waiting on scattered loads. There is no NEON win here; this is
recorded so the next session does not spend the time re-deriving it.

## The real issue is quality, and it belongs to zenresize

A 12.6× downscale sampled with a 2×2 bilinear kernel reads 4 of the ~160 source pixels that
map to each output pixel. That is severe aliasing — the model sees a badly undersampled image.

`zenresize` is the workspace owner for resampling (31 filter kernels, SIMD for u8/i16/f32),
and the one-owner rule says this should call it rather than hand-roll bilinear.

A two-stage approach would be both faster and better: a box reduction to within 2× of target
(fully sequential, reads each source byte once, SIMD-friendly, ~118 GB/s) followed by bilinear
from the small cached intermediate. That fixes the cache-line waste AND the aliasing.

**Not done here, deliberately.** Any of these changes the pixels fed to the detector, which
changes detection and saliency output. That needs validation against the models
(UltraFace RFB-320, MicroSalNet, U2NetP) with a face/saliency accuracy check — it is a quality
change requiring evidence, not a drop-in optimization, and it is not something to slip into a
NEON sweep.
