# Phrase-target extent fits — frozen 2026-11-09

All 1,910 members have independent NAS size and SHA256 readback. Locations checked
on i265 and through dev for the NAS:

- Local: `/mnt/v/output/zensally/florence-primary56/2026-11-09/`.
- NAS: `/mnt/tower/output/zensally/florence-primary56/2026-11-09/`.
- Payload 606,242,329 bytes; checksum file additional.
- SHA256SUMS.json SHA256:
  `684d74cf1070af1378dc49b45bc14104c1e7be93356161b22dfbb0548ada3b3c`.
- Frozen from 1a57a65e, retaining 56 source/query bindings, full cached local
  generations, owned inputs, original/rebound descriptor caches, all 18 fits,
  checkpoints/histories/predictions, family analysis and native-source viewer.

Model/assets and resize binary are dependencies in the independently frozen
[region packet](florence_regions_2026-11-09.pointer.md). Frozen source snapshots
overlay the complete research source in the earlier
[rank packet](unsplash_rank_distillation_2026-11-08.pointer.md); unchanged
`training/crop_fit.py` and `training/composition_head.py` match that packet's
SHA256s exactly. The original descriptor bytes are copied into this fit packet,
so fitting does not require re-extracting either encoder.

Only 48 fit sources supply new noisy primary-extent gradients; eight development
families supply selection targets. No human artistic labels, complete-face
certification, new TEST inference, paid API calls or promoted model. See
[measured findings and limits](florence_extents_2026-11-09.md).
