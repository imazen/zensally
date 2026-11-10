# Photographic feature transfer evidence — frozen 2026-11-09

All 891 members have independent NAS membership, size and SHA256 readback.
Locations checked on i265 and through dev for the NAS:

- Local: `/mnt/v/output/zensally/florence-photo-transfer/2026-11-09/`.
- NAS: `/mnt/tower/output/zensally/florence-photo-transfer/2026-11-09/`.
- Payload: 170,506,316 bytes; checksum file additional.
- SHA256SUMS.json SHA256:
  `dd979a789ff17c6d791f1164f8d8731b9bc07accf2ff469bb612c1093728a300`.
- Frozen from `3cc8c80f`, retaining the preselected photographic checkpoint,
  exact producer source, bound descriptors/targets, all nine fits, histories,
  checkpoints, predictions, paired analysis and the original guard-failure log.

The [original target packet](florence_extents_2026-11-09.pointer.md) supplies
cached source/query/teacher identities and its complete local-region model
dependency. No teacher is rerun for this comparison. Source tensors remain in
the original auxiliary cache with its pinned hash; fitting uses the descriptor
bytes copied into this packet. See [negative findings](photo_student_extent_transfer_2026-11-09.md).
