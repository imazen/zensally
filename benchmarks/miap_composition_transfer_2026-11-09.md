# Person-relationship transfer controls — 2026-11-09

All twelve registered fits finish, but broader person-statistics supervision
does not improve the existing crop ranker. Every arm/seed selects epoch zero.
The unchanged starting model remains the result; no crop model is promoted.
See [registration](../docs/experiments/MIAP_COMPOSITION_TRANSFER.md) and
[evidence pointer](miap_composition_transfer_2026-11-09.pointer.md).

The new cache contains366 owned256-square letterboxed uint8 inputs and human
person-record statistics:275 fit/91 development, one image per frozen component.
Its SHA256 is`b986f2740585449a390177e199fda71eefd1bb4034c49fce50e4d667fbdb4bc4`.
The earlier94-fit/25-development crop-ranking cache and C002 initializer are
pinned. All five original model/evaluation source files match their frozen
selection commit byte for byte. No TEST pixels, detector outputs, caption labels,
human artistic labels or face certificates enter training.

Four arms × three seeds use44 epochs,16-source batches, FP32/TF32 off, AdamW,
encoder lr0.0001, head lr0.001, decay0.01 and clipping5. BatchNorm running
statistics are fixed. The rank-only arm receives no auxiliary updates; the
joint arms use true images, aspect-bin-permuted images or geometry only. Every
crop source/query is visited twice. One fitting auxiliary source remains unmoved
in its permutation singleton; all91 development donors move. Task-isolation
checks preserve inactive warmed AdamW state in both directions.

| Arm | Selected epochs | Selected essential regret | Final auxiliary union IoU |
|---|---|---:|---:|
| Rank only | 0,0,0 | 0.247001 | 0.079402 |
| Joint image | 0,0,0 | 0.247001 | 0.462483 |
| Joint permuted | 0,0,0 | 0.247001 | 0.431306 |
| Joint geometry | 0,0,0 | 0.247001 | 0.472314 |

IoUs are three-seed means on91 auxiliary development photos. Rank-only's
auxiliary head is untrained. The image arm learns the new targets, but does not
beat the geometry control's union overlap. More importantly, all selected crop
metrics remain identical:86.5480% strong-pair agreement and40.7641% choices
within3 normalized128-grid pixels of the weak teacher on essential ratios.
Paired21-family crop-regret differences and their bootstrap intervals are zero.
The registered positive-transfer gate fails. These are teacher-agreement results,
not artistic quality judgments. Extra body supervision is insufficient here.

Every selected native checkpoint replay preserves all550 development source/
query decisions and metrics exactly,12/12. Final optimizer/RNG/visit states,
all selected intermediate states, per-epoch metrics and predictions are retained.
No numerical or face expectation is relaxed.

Producer6a171d73 runs `just composition-fit-miap-transfer`; analyzer3560d881
runs `just composition-analyze-miap-transfer`. Exact commands, source hashes
and GNU time output remain in the packet. The full12-fit run takes98 seconds
under a4 GiB host cap and two threads. Run-heavy reports peak RSS **2.17 GiB**,
minimum available RAM22,618 MiB and peak load2.42. CUDA peak allocated/reserved
bytes are655,719,936 /864,026,624; allocator counters exclude some driver memory.
This is a bounded fit, with no GPU-saturation claim. Cache construction takes
2 seconds with peak RSS0.63 GiB; analysis takes1 second with peak RSS0.03 GiB.

The next composition experiment needs ranking supervision on the broader
photographic sources, with real-image versus geometry/permuted controls and the
same independent role barriers. Better auxiliary fit alone cannot qualify it.
