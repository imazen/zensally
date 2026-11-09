# Generalization and source-coverage evidence — 2026-11-09

These immutable packets have complete independent NAS membership, size and
SHA256 readback. Local roots were checked on i265; NAS roots through dev.
Each local root is `/mnt/v/output/zensally/<name>/2026-11-09/`; its NAS mirror
is `/mnt/tower/output/zensally/<name>/2026-11-09/`.

| Packet name | Members | Payload bytes | SHA256SUMS.json SHA256 |
|---|---:|---:|---|
| `florence-finetune` |1,185|488,016,594|`c0d0c178120812341488d7ac7e2bf57b3768c91c6197d86e3a8472f2120f609f`|
| `florence-extent-augmentation` |326|327,817,885|`c0bad7374e08722c2ed308f2e412d7ea7f78614ea54eff3ed86afb42c11ddd19`|
| `unsplash-content-audit` |151|23,545,880|`2440fb879fe7bd8df6dc0e6fb740459627bf5c221962a8fc2e07132f0e02ac6a`|
| `owned-content-audit` |134|26,092,407|`d11ec3aa607f2ab92250465115d46a919142864c2dc2a86d062f6a364ed52d71`|

Checksum files are additional to payload sizes. The first two packets contain
all21 fits, selected/final checkpoint audits, optimizer/RNG states, targets,
source tensors, exact producer source, unit checks and full command logs.
They freeze from19e761a3 and8ce5f603 respectively. Earlier source/teacher
identities remain in the [phrase-target packet](florence_extents_2026-11-09.pointer.md).

The two content packets freeze from13b64c65 and21f4bf7e, retaining all220
source previews,23 bounded sheets, source hyperlinks, broad unreviewed visual
observations and exact producers. No content tag is a training label or face
certificate. The original source images remain in their existing frozen input
packets; no holdout pixels are opened for these new content reviews.

Readable findings: [encoder overfitting](florence_extent_finetune_2026-11-09.md),
[augmentation controls](extent_augmentation_2026-11-09.md),
[Unsplash coverage](unsplash_content_audit_2026-11-09.md),
[supplementary screening coverage](owned_content_audit_2026-11-09.md).
