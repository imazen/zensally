# AI lossless JXL branch verification — 2026-11-08

All56 lossless JXL images are ordinary Git blobs on
[ai-originals](https://github.com/imazen/zensally-datasets/tree/ai-originals),
commit`a3606f90964885d641c3a79f5098e7170609667a`.
Their image payload totals101,887,093 bytes. Every local owned JXL matches the
remote Git blob SHA1 and manifest SHA256/size, with a valid JXL signature and
zero LFS pointers. Branch attributes explicitly disable JXL/PNG LFS filters.
The existing dirty dataset checkout was inspected read-only and not modified.

The seven-member proof packet has complete independent NAS readback:

- i265: `/mnt/v/output/zensally/ai-remote-verification/2026-11-08/`
- NAS: `/mnt/tower/output/zensally/ai-remote-verification/2026-11-08/`
- SHA256SUMS.json SHA256:
  `78f1dd6ad80c50a44db77227a6bd74fe53c8afc4cbc5000c41b26dbf28d11893`.

The proof contains captured remote tree/manifests, per-file checks and exact
verification source; source image bytes remain in their existing dataset and
earlier immutable image-generation packets. No new image/API spending.
