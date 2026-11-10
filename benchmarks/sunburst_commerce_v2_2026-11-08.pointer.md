# Branded commerce and candid artifacts

Native sources, full API cache snapshot, usage, exact prompts, inspection,
credential-free replay and gallery:

- Primary on i265: `/mnt/v/output/zensally/sunburst-commerce-v2/2026-11-08/`.
- Served copy on dev: the corresponding `/mnt/v/output/zensally/` path.
- Independent NAS: `/mnt/tower/output/zensally/sunburst-commerce-v2/2026-11-08/` via dev.
- All 190 files passed full NAS readback. SHA256SUMS.json digest:
  `2fc0c62b89dc28f78bd49d6b42956cdcf37e7f895c65ecfc7f784262910e0ee9`.

JXL variants, source/decoded RGB, independent oracle readbacks and codec logs:

- Primary on i265: `/mnt/v/output/zensally/sunburst-jxl/2026-11-08-commerce-v2/`.
- Independent NAS: `/mnt/tower/output/zensally/sunburst-jxl/2026-11-08-commerce-v2/` via dev.
- All 244 files passed full NAS readback. SHA256SUMS.json digest:
  `7f0c2453bacf7b188ab2a571df1799067ce9d5a9e12730e3599c1d4637d1629f`.

Generation worker: `3fb3ddf14b78513f90f97778ac83a55f839f49c4`.
Verification/publication tooling: `33c4176725cdb5a87c43050b4501c5f0b78461f1`.
Actual worker and codec binary hashes are retained, not inferred from names.
Dataset branch commit: `32fd7f59166a8667cc9680dda1ed76ea2cbde3b0`.

http://dev.lan:3300/zensally/sunburst-commerce-v2/2026-11-08/index.html
