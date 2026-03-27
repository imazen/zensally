# MicroSalNet v3 — Plan for improved saliency model

## What we know

### Benchmark results (DUTS-TE, 5019 images)

| Model | Params | MAE ↓ | F_β ↑ | Precision | Recall | tract ms |
|-------|--------|-------|-------|-----------|--------|----------|
| U2-Netp | 1.13M | 0.054 | 0.744 | 0.751 | 0.869 | 1828 |
| MicroSalNet v2 (w24) | 524K | 0.094 | 0.619 | 0.605 | 0.815 | 12.3 |
| MicroSalNet v1 (w16) | 237K | 0.120 | 0.550 | 0.537 | 0.780 | 7.5 |
| CSNet (patched) | 211K | ~0.056* | ~0.77* | — | — | 201 |

*published numbers, not benchmarked in our eval harness

### Why CSNet is accurate but slow in tract

CSNet processes at **full resolution (224×224) for 8 blocks** in stages 0-1,
keeping 1M+ elements per feature map. Its octave convolutions also maintain
parallel feature maps at 2-3 resolutions, connected by Resize ops.

Total compute: 397 ONNX ops, 66 depthwise convs at large spatial sizes.

### Why MicroSalNet is fast but less accurate

MicroSalNet aggressively downsamples: 256→128→64→32→16→8 through the encoder,
then decodes back to 128. The bottleneck at 8×8 (12K elements) is 80× smaller
than CSNet's full-res processing. This means:

- **Fast**: small feature maps = fast convolutions
- **Inaccurate**: 8×8 bottleneck loses spatial detail. The decoder reconstructs
  from very low-res features — fine structure (edges, small objects) is blurred.

### What makes CSNet accurate (the ideas, not the implementation)

1. **Multi-scale dilated convolutions (MSBlock)**: parallel Conv2d at
   dilations 1,2,4,8,16, concatenated. Large receptive field without
   downsampling — the network "sees" context at many scales while
   maintaining spatial resolution.

2. **No ImageNet backbone needed**: 100K params trained from scratch on
   DUTS-TR achieves near-SOTA. The task doesn't need deep features from
   classification pretraining.

3. **Cross-stage fusion**: features from stages 2,3,4 are fused at a
   shared resolution, giving the output head multi-scale context.

### Tract performance profile (what's fast/slow)

| Op type | CSNet ms | MSN v2 ms | Notes |
|---------|----------|-----------|-------|
| DeconvSum (ConvTranspose) | 83 (35%) | 4.5 (33%) | scales with spatial size × channels |
| DepthWiseConv | 77 (32%) | 3.3 (24%) | scales with spatial size × channels |
| OptMatMul (pointwise Conv) | 44 (18%) | 3.3 (24%) | scales with channels² × spatial |
| PReLU (> + Iff + MulScalar) | 24 (10%) | — | CSNet uses PReLU (3 ops); MSN uses ReLU (1 op) |
| Other | 12 (5%) | 2.5 (18%) | — |

**Key constraint**: depthwise conv at 224×224 with 20 channels costs ~4ms per block.
The same op at 64×64 with 24 channels costs ~0.4ms. Spatial size dominates.

### Feature map size comparison

MicroSalNet keeps things small:
```
stem:  [24, 128, 128]  →  393K elements  (largest)
enc1:  [24,  64,  64]  →   98K
enc2:  [48,  32,  32]  →   49K
enc3:  [96,  16,  16]  →   25K
enc4:  [192,  8,   8]  →   12K  (bottleneck)
```

CSNet stays large:
```
stg0:  [20, 224, 224]  → 1,003K elements  (2.5× larger than MSN's largest)
stg1:  [20, 224, 224]  → 1,003K  ← 3 blocks at this size!
stg2:  [40, 112, 112]  →   502K
stg3:  [80,  56,  56]  →   251K
stg4:  [80,  28,  28]  →    63K
```

## Design: MicroSalNet v3

### Core insight

MicroSalNet's bottleneck is too aggressive (8×8 = 64 pixels to represent
the whole scene). CSNet's full-res processing is too expensive. The sweet
spot: **keep spatial resolution higher in the bottleneck, and add dilated
convolutions for receptive field instead of depth**.

### Architecture

```
Input [3, 256, 256]
  ↓ Conv2d stride=2
Stem [C, 128, 128]          ← same as v2
  ↓ InvertedResidual stride=2
Enc1 [C, 64, 64]            ← same as v2
  ↓ InvertedResidual stride=2
Enc2 [2C, 32, 32]           ← same as v2
  ↓ DilatedMSBlock (no downsample!)
Enc3 [2C, 32, 32]  ★NEW     ← stay at 32×32 instead of going to 16×16
  ↓ DilatedMSBlock (no downsample!)
Enc4 [2C, 32, 32]  ★NEW     ← dilated convs give receptive field of 16→8
  ↓
Dec3 = Enc4 + skip(Enc2)     ← concat at 32×32, ConvTranspose2d to 64×64
  ↓ ConvTranspose2d stride=2
Dec2 [C, 64, 64] + skip(Enc1)
  ↓ ConvTranspose2d stride=2
Dec1 [C, 128, 128] + skip(Stem)
  ↓ Conv2d 1×1
Output [1, 128, 128]
```

### DilatedMSBlock (from CSNet, tract-compatible)

```python
class DilatedMSBlock(nn.Module):
    """Multi-scale dilated convolutions — CSNet's key ingredient."""
    def __init__(self, channels, dilations=[1, 2, 4, 8]):
        # 4 parallel depthwise Conv2d at different dilation rates
        # Concat → 1×1 pointwise → BN → ReLU
        # Receptive field at dilation=8 on 32×32: covers 17×17 = half the map
```

All standard ops: Conv2d (depthwise + pointwise), Concat, BN, ReLU.
No Resize. No PReLU (use ReLU — saves 2 ops per activation).

### Why this should work

1. **Bottleneck at 32×32 (1024 pixels)** instead of 8×8 (64 pixels) —
   16× more spatial information preserved through the bottleneck.
   Still 10× fewer pixels than CSNet's full-res 224×224.

2. **Dilated convolutions at 32×32** give equivalent receptive field to
   processing at 8×8 and 4×4. Dilation=8 on a 32×32 map covers 17×17 —
   more than half the feature map in one conv.

3. **Fewer decoder stages**: only 2 upsamples (32→64→128) instead of
   4 (8→16→32→64→128). Fewer ConvTranspose2d = less DeconvSum time.

4. **Same encoder depth as v2** for the early stages (proven fast),
   swapping the deep narrow bottleneck for a wide dilated one.

### Expected performance

Spatial sizes → compute profile:
- Encoder: 128→64→32 (same as v2 through enc2, ~5ms)
- Dilated blocks at 32×32 with 2C channels: depthwise conv at 32×32 costs
  ~0.1ms each; with 4 dilations × 2 blocks = ~1ms total
- Decoder: 32→64→128 = 2 ConvTranspose2d: ~2ms
- Total: ~8-10ms (similar to v2 w16, faster than v2 w24)

Quality:
- The 32×32 bottleneck with dilated convolutions should capture both
  local detail and global context
- Target: MAE < 0.08, F_β > 0.65 (between v2 and U2-Netp)

### Param budget

- Encoder through enc2: ~180K (same as v2 w24 through enc2)
- 2× DilatedMSBlock at 2C=48: ~50K (4 dilated depthwise + pointwise each)
- Decoder (2 stages): ~50K
- **Total: ~280K params, ~1.1MB ONNX**

### Training plan

- Dataset: DUTS-TR (10553 images + masks), teacher from U2-Netp
- Loss: structure_loss (weighted BCE + weighted IoU) — proven in v2
- Teacher: alpha=0.7 GT + 0.3 teacher (same as v2)
- Epochs: 120, batch=32, AdamW with cosine LR
- Eval: DUTS-TE MAE + F_β, tract latency benchmark

### Success criteria

1. MAE < 0.085 on DUTS-TE (better than v2's 0.094)
2. F_β > 0.65 on DUTS-TE (better than v2's 0.619)
3. tract latency < 15ms (not worse than v2 w24's 12.3ms)
4. All standard ONNX ops, no Resize

### Fallback

If dilated convolutions don't help enough, try:
- Bottleneck at 16×16 instead of 32×32 (compromise)
- Wider channels (3C instead of 2C in bottleneck)
- ASPP-style parallel dilations (1,6,12,18) instead of (1,2,4,8)
