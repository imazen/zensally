# Saliency Model Exploration — March 2026

## Goal

Improve saliency detection quality for smart image cropping in zenpipe,
targeting two latency budgets: 15ms (real-time crop) and 50ms (batch/preview).

## Baseline Models (DUTS-TE, 5019 images)

| Model | Params | ONNX | MAE ↓ | F_β ↑ | Precision | Recall | tract ms |
|-------|--------|------|-------|-------|-----------|--------|----------|
| U2-Netp | 1.13M | 4.1MB | 0.054 | 0.744 | 0.751 | 0.869 | 1828 |
| MicroSalNet v1 (w16) | 237K | 858KB | 0.120 | 0.550 | 0.537 | 0.780 | 7.5 |
| SelfieSeg | 106K | 259KB | 0.107 | 0.564 | 0.610 | 0.654 | 31 |

v1 had two problems: (1) low precision (0.537) — marks too much as salient,
(2) high MAE (0.120) — poor boundary accuracy. Both matter for crop quality.

## Exploration 1: External models

### CSNet (SOD100K, ECCV 2020)

**Hypothesis**: CSNet achieves 0.056 MAE with only 100K params. If we can
use it, it would match U2-Netp quality at tiny model size.

**Result**: Dead end for tract. Architecture uses Octave Convolutions that
process features at multiple resolutions connected by bilinear Resize ops.

- ONNX export succeeded (211K params, 662KB)
- tract latency: **360ms** (21 Resize ops = 200ms, 56% of runtime)
- Replaced Resize with ConvTranspose2d (bilinear weights): **201ms** — 44% faster but still 16× too slow
- The remaining cost is 66 depthwise convolutions at 224×224 (full resolution through 8 blocks)

**Lesson**: tract's Resize op is slow, but the real problem is CSNet's
architecture processes at full resolution for too many layers. The
Resize→ConvTranspose2d trick is useful but can't fix an architecture
that does 1M-element feature maps through 8 sequential blocks.

### TurboQuant (Google, March 2026)

Investigated for potential model compression. TurboQuant compresses
LLM KV caches using polar coordinate quantization — not applicable to
small CNN saliency models. Wrong problem domain.

## Exploration 2: tract performance profiling

Profiled all models to understand where time is spent:

### Op-level breakdown

| Op type | CSNet % | MicroSalNet v2 % | Notes |
|---------|---------|-------------------|-------|
| DeconvSum (ConvTranspose) | 35% | 33% | Scales with spatial size × channels |
| DepthWiseConv | 32% | 24% | Scales with spatial size × channels |
| OptMatMul (pointwise Conv) | 19% | 24% | Scales with channels² × spatial |
| Activations (PReLU/ReLU) | 10% | 8% | PReLU costs 3 ops; ReLU costs 1 |

**Key insight**: spatial size dominates. A depthwise conv at 224×224 costs
~4ms; at 64×64 it costs ~0.4ms; at 32×32 it costs ~0.1ms. This means
the bottleneck resolution choice is the critical architecture decision.

### Feature map size analysis

MicroSalNet v2 (fast, inaccurate):
```
stem:  [24, 128, 128]  →  393K elements
enc1:  [24,  64,  64]  →   98K
enc2:  [48,  32,  32]  →   49K
enc3:  [96,  16,  16]  →   25K
enc4:  [192,  8,   8]  →   12K  ← bottleneck, 64 pixels total
```

CSNet (accurate, slow):
```
stg0:  [20, 224, 224]  → 1,003K elements
stg1:  [20, 224, 224]  → 1,003K  ← 3 blocks at full resolution
stg2:  [40, 112, 112]  →   502K
stg3:  [80,  56,  56]  →   251K
stg4:  [80,  28,  28]  →    63K
```

**The accuracy gap is spatial**: v2's 8×8 bottleneck (64 pixels) can't
represent where objects are. CSNet keeps full resolution but at enormous
compute cost. The sweet spot is between them.

## Exploration 3: Architecture iterations

### v2 — Better training, same architecture (w24, 524K params)

Changed training only:
- Structure loss (weighted BCE + weighted IoU) instead of plain BCE
- Higher GT weight (alpha=0.7 vs 0.5)
- More augmentation (random scale, color jitter)
- Wider model (w24 vs w16)

Result: MAE 0.094 (from 0.120), F_β 0.619 (from 0.550).
Precision improved from 0.537→0.605 (structure loss worked).
But still limited by 8×8 bottleneck.

### v3 — 32×32 bottleneck with dilated blocks (w24, 115K params)

First attempt at CSNet-inspired architecture:
- Encoder stops at 32×32 (3 downsample stages)
- 8 DilatedBlock at 32×32 with dilations (1,2,4,8,1,2,4,8)
- 2 decoder stages (32→64→128)

Result: **20.7ms in tract** — too slow for 15ms budget.
Depthwise convolutions at 32×32 cost ~1.1ms each × 8 blocks = ~9ms.

### v3d — 16×16 bottleneck with dilated blocks (w24, 191K params)

Compromise:
- Encoder goes to 16×16 (4 downsample stages)
- 4 DilatedBlock at 16×16 with dilations (1,2,4,8)
- 3 decoder stages (16→32→64→128)
- SE attention in enc3

Result: **15.7ms in tract**, MAE 0.086, F_β 0.640.
4× more spatial info than v2 (16×16 vs 8×8), dilated convs for receptive
field. First model to beat v2 on quality AND be smaller (191K vs 524K).

### v3ds — v3d + deep supervision (w24, 191K params)

Same architecture, better training:
- Auxiliary saliency heads at 32×32, 64×64, 128×128 decoder stages
- Each supervised with structure_loss against downscaled GT
- Aux weight ramps 0→0.4 over first 20 epochs
- Stronger augmentation: rotation ±10°, Gaussian blur
- 150 epochs (up from 120)
- Aux heads removed for ONNX export — zero latency impact

Result: MAE **0.080** (from 0.086), F_β **0.652** (from 0.640).
Deep supervision didn't change val MAE but improved test generalization
(gap: 0.023→0.017). Same ONNX, same speed, strictly better.

### v3ds w28 — wider v3ds for 50ms budget (257K params)

Width 28 instead of 24, otherwise identical to v3ds.

Result: MAE **0.078**, F_β **0.654**, **21ms in tract**.
Modest improvement from extra capacity. Best quality-per-ms model.

### v3ds32 w40 — 32×32 bottleneck for 50ms budget (208K params)

Tested whether 32×32 bottleneck with wider channels beats 16×16 with
more depth, given the same latency budget.

- 32×32 bottleneck, 4 dilated blocks, width 40
- 48ms in tract

Result: MAE **0.078**, F_β **0.657**.
**Tie with v3ds w28** (0.078 MAE) at 2.3× the compute.

**Conclusion**: 16×16 with dilated convolutions is the right bottleneck
resolution. The skip connections recover spatial detail sufficiently.
Going to 32×32 wastes compute without improving quality.

## Final Results

### 15ms budget (real-time smart crop)

**Recommended: MicroSalNet v3ds w24** — 191K params, 751KB ONNX (690KB gz)

| Metric | v1 (original) | v3ds w24 | Improvement |
|--------|---------------|----------|-------------|
| MAE | 0.120 | 0.080 | **33% better** |
| F_β | 0.550 | 0.652 | **19% better** |
| Precision | 0.537 | 0.640 | **19% better** |
| Recall | 0.780 | 0.830 | **6% better** |
| Params | 237K | 191K | **19% smaller** |
| ONNX | 858KB | 751KB | **12% smaller** |
| tract | 7.5ms | 15.7ms | 2× slower (acceptable) |

### 50ms budget (batch/preview)

**Recommended: MicroSalNet v3ds w28** — 257K params, 1.0MB ONNX (928KB gz)

| Metric | v1 (original) | v3ds w28 | Improvement |
|--------|---------------|----------|-------------|
| MAE | 0.120 | 0.078 | **35% better** |
| F_β | 0.550 | 0.654 | **19% better** |
| Precision | 0.537 | 0.641 | **19% better** |
| Recall | 0.780 | 0.833 | **7% better** |
| tract | 7.5ms | 21ms | within budget |

### Gap to U2-Netp closed

| Metric | v1 | v3ds w24 | v3ds w28 | U2-Netp | Gap closed |
|--------|-----|----------|----------|---------|------------|
| MAE | 0.120 | 0.080 | 0.078 | 0.054 | 64% |
| F_β | 0.550 | 0.652 | 0.654 | 0.744 | 54% |

## Key Learnings

1. **Bottleneck resolution matters more than parameter count.** v3ds w24
   (191K params, 16×16 bottleneck) beats v2 w24 (524K params, 8×8 bottleneck)
   on every quality metric despite having 2.7× fewer parameters.

2. **Dilated convolutions are the right tool for receptive field in tract.**
   They give CSNet-like context awareness without Resize ops or full-res
   processing. All-standard-ops ONNX that tract handles efficiently.

3. **Deep supervision helps generalization, not val MAE.** The auxiliary
   losses force intermediate features to be saliency-aware, which matters
   more on unseen test data than on the training distribution.

4. **32×32 vs 16×16 bottleneck: 16×16 wins on efficiency.** At equal
   compute budget, 16×16 with wider channels matches 32×32 quality.
   Skip connections from the encoder recover spatial detail.

5. **Structure loss (wBCE + wIoU) is essential.** Boundary-aware weighting
   fixed the precision problem that plagued v1. Every SOD paper uses it
   for good reason.

6. **Resize→ConvTranspose2d is a valid ONNX graph rewrite** for models
   with bilinear upsample ops. Saved 44% on CSNet (360→201ms). Useful
   toolkit for importing external models into tract.

## Files

### Embedded models (production)
- `crates/zensally-tract/models/microsalnet_v3ds_w24.onnx.gz` — 15ms budget
- `crates/zensally-tract/models/microsalnet_v3ds_w28.onnx.gz` — 50ms budget
- `crates/zensally-tract/models/microsalnet.onnx.gz` — original v1 (kept for comparison)

### Training code
- `training/model.py` — v1/v2 MicroSalNet (MobileNetV3 encoder-decoder)
- `training/model_v3.py` — v3/v3d architecture (dilated bottleneck, no deep supervision)
- `training/model_v3ds.py` — v3ds architecture (+ deep supervision, production)
- `training/train.py` — v1 training (BCE + teacher distillation)
- `training/train_v2.py` — v2 training (structure loss + augmentation)
- `training/train_v3.py` — v3d training
- `training/train_v3ds.py` — v3ds training (deep supervision)
- `training/train_v3ds32.py` — v3ds32 training (32×32 bottleneck experiment)
- `training/export_onnx.py` — ONNX export + tract patching + gzip

### Checkpoints (backed up to /mnt/v/output/zensally/training-v3/)
- `microsalnet_v3ds_w24_s256_best.pth` — production 15ms model
- `microsalnet_v3ds_w28_s256_best.pth` — production 50ms model
- `microsalnet_v3d_w24_s256_best.pth` — v3d without deep supervision
- `microsalnet_w24_s256_v2_best.pth` — v2 baseline
- `microsalnet_v3ds32_w40_d4_s256_best.pth` — 32×32 bottleneck experiment

### Benchmark results (backed up to /mnt/v/output/zensally/training-v3/)
- `duts_te_full.txt` — U2-Netp baseline
- `duts_te_microsalnet.txt` — v1 w16 baseline
- `duts_te_microsalnet_v2_w24.txt` — v2 results
- `duts_te_microsalnet_v3d_w24.txt` — v3d results
- `duts_te_v3ds_w24.txt` — v3ds w24 results
- `duts_te_selfie_seg.txt` — SelfieSeg baseline
