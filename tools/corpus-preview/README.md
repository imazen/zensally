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

Reuse source: imazen/zenpipe at `6ed32dea4df1d086b5674dbbc54e53aada14c3f5`.
The root repeats that source graph's required patches because dependency patch
tables do not propagate, pinning floating entries to its existing lockfile.
The application contains no codec or resizing kernel implementation. Verify
every output independently before admitting it to a study.
