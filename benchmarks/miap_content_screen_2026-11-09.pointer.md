# Full people-screen evidence — frozen 2026-11-09

All five packets have complete independent NAS membership, size and SHA256 readback.
Locations were resolved on i265 and through dev for NAS. Prior packets remain immutable.

## Acquisition

- Local: `/mnt/v/input/zensally/miap-fit-development-gray/2026-11-09/`.
- NAS: `/mnt/tower/input/zensally/miap-fit-development-gray/2026-11-09/`.
- Members: 3072; payload 943570531 bytes.
- SHA256SUMS.json SHA256: `93f63d69180d82f197e34a733005118de260eda728e18998ced3194959581bb1`.

## Captions

- Local: `/mnt/v/output/zensally/miap-content-captions/2026-11-09/`.
- NAS: `/mnt/tower/output/zensally/miap-content-captions/2026-11-09/`.
- Members: 1181; payload 689895475 bytes.
- SHA256SUMS.json SHA256: `136b516fc75aa7364fe49119644dc325eff2ebc5b800c48171dcd6ce622918e8`.

## Relationships

- Local: `/mnt/v/output/zensally/miap-pilot-relationships/2026-11-09/`.
- NAS: `/mnt/tower/output/zensally/miap-pilot-relationships/2026-11-09/`.
- Members: 11; payload 27470278 bytes.
- SHA256SUMS.json SHA256: `a735c55ce6ce117da9149fb6cdc748f5917a4e84efb08eaeb17b5af75c1d76b5`.

## Review

- Local: `/mnt/v/output/zensally/miap-content-review/2026-11-09/`.
- NAS: `/mnt/tower/output/zensally/miap-content-review/2026-11-09/`.
- Members: 555; payload 300155951 bytes.
- SHA256SUMS.json SHA256: `75a75ee2df2c85ecac6b18cf6c875d3bfb0c6d625bf26542efc57fcbb4d2bca9`.

## Detectors

- Local: `/mnt/v/output/zensally/miap-detector-screen/2026-11-09/`.
- NAS: `/mnt/tower/output/zensally/miap-detector-screen/2026-11-09/`.
- Members: 30; payload 8057510 bytes.
- SHA256SUMS.json SHA256: `e4cfeccba09896fd0f6c6e0f6a084966fb535d4e3da5ca753f10ffd71ff34947`.

Frozen from828e9784; exact stage producer commits and source copies are retained.
Acquisition includes the stopped RGB-only attempt, helper parity and grayscale probe.
Captions include388 unique full responses and the capped invalid response.
Review includes384 observations, explicit corrections, all16 full sheets, frozen masks
and the separate119-source legacy gallery replay. Detectors include all1,920 outputs,
scoring protocol, per-source matches and paired uncertainty. Model weights remain in
their earlier immutable parent packets. No labels or paid API calls are added.

See [source findings](miap_content_screen_2026-11-09.md) and
[detector findings](miap_detectors_2026-11-09.md).
