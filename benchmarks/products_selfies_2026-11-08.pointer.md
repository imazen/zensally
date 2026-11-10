# Additional product/selfie evidence

Native source packet and cached responses:

- Primary on i265: `/mnt/v/output/zensally/sunburst-products-selfies-v3/2026-11-08/`.
- Served on dev: corresponding `/mnt/v/output/zensally/` path.
- Independent NAS: `/mnt/tower/output/zensally/sunburst-products-selfies-v3/2026-11-08/` via dev.
- All 58 files passed full NAS readback. SHA256SUMS.json digest:
  `5c3530bc7b32eaa8c8a7bbe80196d6bcb52532fbd7eacb9b9237ba957ac046e7`.

JXL variants, source/decoded RGB and independent oracle readbacks:

- Primary on i265: `/mnt/v/output/zensally/sunburst-jxl/2026-11-08-products-selfies-v3/`.
- Independent NAS: `/mnt/tower/output/zensally/sunburst-jxl/2026-11-08-products-selfies-v3/` via dev.
- All 62 files passed full NAS readback. SHA256SUMS.json digest:
  `274b60ec480275b8fbade83927f92d54c9ac257c8faaff306cc0c4ed0ccedcbf`.

Generation source: `1375e40ca5fd1c440ae438be2f6dce68ad6b86bf`.
Ratio-training design: `cc22ec229c86ea6034e8c03ce382eadd78873e39`.
Dataset branch: `a3606f90964885d641c3a79f5098e7170609667a`.
Worker source and actual codec binary hashes are retained in each packet.

http://dev.lan:3300/zensally/sunburst-products-selfies-v3/2026-11-08/index.html
