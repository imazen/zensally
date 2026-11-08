# Owned corpus preview adapter

Private research executable around pinned zenpipe `ImageJob`. It decodes JPEG,
converts to sRGB with strict ICC handling, preserves aspect ratio and writes a
new lossless PNG of at most the requested long edge. Lanczos3 downsampling,
automatic orientation, no upscaling. All144 current Unsplash source orientations
were independently observed as1; record native and working dimensions separately.

Per-job limits permit the screened160MP image (200MP /32768 dimensions /4GiB
single-allocation policy /one frame /120s). Library defaults are unchanged.
Always invoke under run-heavy with an explicit process cap. Encoded input is
fallibly bounded to64MiB; output to16MiB and dimensions128–2048. Existing
output files are refused. This does not preserve archival descriptive metadata;
source originals and provenance remain immutable outside the preview cache.

Reuse source: imazen/zenpipe at `0b3274f8ce661333af5bbdb3c0d5e5a2f00fa1f7`.
That revision propagates caller-selected JPEG header budgets; the earlier
120MP probe refusal is retained as a failed attempt in the study logs.
The root repeats that source graph's required patches because dependency patch
tables do not propagate, pinning floating entries to its existing lockfile.
The application contains no codec or resizing kernel implementation. Verify
every output independently before admitting it to a study.
The shared codec output budget permits800MB of decoded pixel bytes; the final
encoded preview is separately bounded to16MiB before writing.
