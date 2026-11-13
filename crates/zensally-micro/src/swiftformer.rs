//! Zero-allocation NHWC SIMD executor for SwiftFormer-XS and SwiftFormer-S.
//!
//! Designed from profiler evidence on `moderation-probe-release` (`tract 0.23.10`)
//! and ONNX Runtime `1.31.1`:
//! - **85–89% of MACs** are `1x1` pointwise linear projections (`pwconv1`,
//!   `pwconv2`, `fc1`, `fc2`, and `EfficientAdditiveAttention` projections).
//! - **10–13.5% of MACs** are `3x3` stride-2 dense convolutions (`stem.0`,
//!   `stem.3`, and `stages.{1,2,3}.downsample`).
//! - **~1% of MACs** are `3x3` stride-1 depthwise convolutions (`dwconv`).
//! - Both `tract` and `ORT` pay heavy penalties from materializing the full
//!   `4x`-expanded hidden activation tensor (`[56x56x192] = 2.4 MiB` in Stage 0)
//!   before running a separate `Erf`/`Gelu` pass, plus `Nchwc <-> NCHW <-> NLC`
//!   layout transposes around every stage's `EfficientAdditiveAttention` block
//!   and unfused `BatchNorm2d` before `Block.linear.fc1`.
//!
//! This module implements a dedicated FP32 NHWC executor over `archmage` +
//! `magetypes` (`f32x8` across AVX-512, AVX2+FMA, NEON, WASM128, and scalar):
//! 1. **Load-time FP64 algebraic weight folding**:
//!    - All `LayerScale2d` (`gamma`) vectors fold into the preceding `pwconv2`,
//!      `attn.final`, and `linear.fc2` weights and biases.
//!    - `Block.linear.norm1` (`BatchNorm2d`) folds into the following `fc1`
//!      weights and biases (`W' = scale[:, None] * W`, `b' = offset @ W + b`).
//!    - `attn.w_g` absorbs `scale_factor = dim^-0.5`.
//!    - Final `norm` (`BatchNorm2d`), spatial mean (`1/49`), and dual classification
//!      heads `0.5 * (head + head_dist)` fold into a single `dim -> 3` linear layer.
//! 2. **Padded multiple-of-16/8 NHWC channel storage**:
//!    - SwiftFormer-XS Stage 3 (`dim = 220`) is zero-padded to `dim_pad = 224`
//!      (`14 x 16 = 28 x 8`) by zeroing output weights and biases for channels
//!      `220..224`. Padded channels stay identically `0.0` through `dwconv`,
//!      `GELU(0) = 0`, `ReduceL2` (`0^2 = 0`), and `Div` (`0 / norm = 0`), while
//!      `scale_factor` preserves the exact `dim = 220` normalization.
//! 3. **6x16 / 4x16 register-blocked FMA GEMM microkernel & L1d token tiling**:
//!    - Weights are pre-packed into `16`-channel panels (`[nv16][in_c][16]`) plus
//!      an optional `8`-channel panel (`[in_c][8]`).
//!    - `pwconv1 + GeluExact -> pwconv2 + ResidualAdd` executes in tiles of
//!      `M_TILE = 24` tokens (`4 x 6`), keeping the `4x`-expanded intermediate
//!      tensor inside L1d/L1.5 cache (`18–86 KiB`) instead of writing 2.4 MiB to L2/L3.
//! 4. **Zero-transpose `EfficientAdditiveAttention`**:
//!    - In NHWC, `[1, H, W, C]` and `[1, H*W, C]` share the exact same memory
//!      layout, eliminating all `Transpose`/`ReorderInput`/`ReorderOutput` ops.

use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;
use archmage::prelude::*;

/// Tile size (in tokens/pixels) for cache-resident MLP and `3x3` stride-2 GEMM.
/// Must be a multiple of 6 (`6x16` microkernel row block).
const M_TILE: usize = 24;

/// Supported SwiftFormer architecture variants.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SwiftFormerKind {
    /// `SwiftFormer-XS`: dims `[48, 56, 112, 220]`, layers `[3, 3, 6, 4]`.
    Xs,
    /// `SwiftFormer-S`: dims `[48, 64, 168, 224]`, layers `[3, 3, 9, 6]`.
    S,
}

impl SwiftFormerKind {
    /// True unpadded embedding dimensions per stage.
    #[must_use]
    pub const fn dims(self) -> [usize; 4] {
        match self {
            Self::Xs => [48, 56, 112, 220],
            Self::S => [48, 64, 168, 224],
        }
    }

    /// Padded embedding dimensions (multiples of 8; XS stage 3 `220 -> 224`).
    #[must_use]
    pub const fn dims_pad(self) -> [usize; 4] {
        match self {
            Self::Xs => [48, 56, 112, 224],
            Self::S => [48, 64, 168, 224],
        }
    }

    /// Number of blocks per stage (`layers[s] - 1` `ConvEncoder`s + `1` `Block`).
    #[must_use]
    pub const fn layers(self) -> [usize; 4] {
        match self {
            Self::Xs => [3, 3, 6, 4],
            Self::S => [3, 3, 9, 6],
        }
    }
}

/// Error returned when loading or validating a SwiftFormer ONNX model.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SwiftFormerError(pub String);

impl core::fmt::Display for SwiftFormerError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(&self.0)
    }
}

#[cfg(feature = "std")]
impl std::error::Error for SwiftFormerError {}

/// Pre-packed weight matrix for the `6x16` / `4x16` / `1x16` (+ `rem8`) GEMM microkernel.
#[derive(Clone, Debug)]
struct PackedGemm {
    /// Active input channels accumulated in the inner loop.
    in_c: usize,
    /// Input row stride in `f32` elements (`>= in_c`; `224` when `in_c == 220`).
    in_stride: usize,
    /// Padded output channel count (multiple of 8).
    out_c_pad: usize,
    /// Number of 16-channel panels (`out_c_pad / 16`).
    nv16: usize,
    /// Whether a trailing 8-channel panel follows the 16-channel panels.
    has_rem8: bool,
    /// Packed weights: `[nv16][in_c][16]` followed by optional `[in_c][8]`.
    packed_w: Vec<f32>,
    /// Bias vector of length `out_c_pad` (padded channels set to `0.0`).
    bias: Vec<f32>,
}

impl PackedGemm {
    /// Pack a row-major `[in_c, out_c]` matrix and `[out_c]` bias into 16-wide
    /// and 8-wide SIMD panels, zero-padding output channels `out_c..out_c_pad`.
    fn pack(w_io: &[f32], bias: &[f32], in_c: usize, in_stride: usize, out_c: usize) -> Self {
        assert_eq!(w_io.len(), in_c * out_c);
        assert_eq!(bias.len(), out_c);
        assert!(in_stride >= in_c);
        let out_c_pad = out_c.next_multiple_of(8);
        let nv16 = out_c_pad / 16;
        let has_rem8 = (out_c_pad % 16) == 8;
        let mut packed_w = vec![0.0f32; in_c * out_c_pad];

        for p16 in 0..nv16 {
            let co = p16 * 16;
            let panel_base = p16 * in_c * 16;
            for ic in 0..in_c {
                let row = ic * out_c;
                let dst = panel_base + ic * 16;
                for l in 0..16 {
                    let oc = co + l;
                    if oc < out_c {
                        packed_w[dst + l] = w_io[row + oc];
                    }
                }
            }
        }
        if has_rem8 {
            let co = nv16 * 16;
            let panel_base = nv16 * in_c * 16;
            for ic in 0..in_c {
                let row = ic * out_c;
                let dst = panel_base + ic * 8;
                for l in 0..8 {
                    let oc = co + l;
                    if oc < out_c {
                        packed_w[dst + l] = w_io[row + oc];
                    }
                }
            }
        }

        let mut bias_pad = vec![0.0f32; out_c_pad];
        bias_pad[..out_c].copy_from_slice(bias);

        Self {
            in_c,
            in_stride,
            out_c_pad,
            nv16,
            has_rem8,
            packed_w,
            bias: bias_pad,
        }
    }
}

/// Packed weights for a `3x3` stride-1 pad-1 depthwise convolution.
/// Layout is `[nv][9][8]` (`nv = c_pad / 8`), so all 9 spatial taps for one
/// 8-channel vector sit in 288 contiguous bytes and stay in registers across a row.
#[derive(Clone, Debug)]
struct PackedDepthwise {
    c_pad: usize,
    nv: usize,
    /// `[nv][9][8]` packed weights.
    packed_w: Vec<f32>,
    /// `[c_pad]` bias (zero for padded channels).
    bias: Vec<f32>,
}

impl PackedDepthwise {
    fn pack(w_c9: &[f32], bias: &[f32], c: usize) -> Self {
        assert_eq!(w_c9.len(), c * 9);
        assert_eq!(bias.len(), c);
        let c_pad = c.next_multiple_of(8);
        let nv = c_pad / 8;
        let mut packed_w = vec![0.0f32; nv * 9 * 8];
        for cv in 0..nv {
            for k in 0..9 {
                for l in 0..8 {
                    let ch = cv * 8 + l;
                    if ch < c {
                        packed_w[(cv * 9 + k) * 8 + l] = w_c9[ch * 9 + k];
                    }
                }
            }
        }
        let mut bias_pad = vec![0.0f32; c_pad];
        bias_pad[..c].copy_from_slice(bias);
        Self {
            c_pad,
            nv,
            packed_w,
            bias: bias_pad,
        }
    }
}

/// One `ConvEncoder` (or `LocalRepresentation`) block:
/// `x + layer_scale * pwconv2(gelu(pwconv1(dwconv(x))))` with `BatchNorm` and
/// `layer_scale` folded into `dwconv` and `pwconv2` at load time.
#[derive(Clone, Debug)]
struct ConvEncoderWeights {
    dw: PackedDepthwise,
    pw1: PackedGemm,
    pw2: PackedGemm,
}

/// `EfficientAdditiveAttention` + `Mlp` block at the end of each stage.
#[derive(Clone, Debug)]
struct StageAttentionBlock {
    local_rep: ConvEncoderWeights,
    to_query: PackedGemm,
    to_key: PackedGemm,
    /// `w_g * (dim^-0.5)` padded to `dim_pad` with zeros.
    w_g_scaled: Vec<f32>,
    proj: PackedGemm,
    /// `attn.final` with `layer_scale_1` folded into weights and bias.
    final_proj: PackedGemm,
    /// `linear.fc1` with preceding `linear.norm1` (`BatchNorm2d`) folded in.
    mlp_fc1: PackedGemm,
    /// `linear.fc2` with `layer_scale_2` folded in.
    mlp_fc2: PackedGemm,
}

#[derive(Clone, Debug)]
struct StageWeights {
    /// Present for stages `1..4` (`3x3` stride-2 pad-1 downsample conv).
    downsample: Option<PackedGemm>,
    in_h: usize,
    in_w: usize,
    dim_pad: usize,
    conv_blocks: Vec<ConvEncoderWeights>,
    attn_block: StageAttentionBlock,
}

/// Pre-allocated scratch workspace for zero-allocation warm SwiftFormer inference.
pub struct SwiftFormerWorkspace {
    /// Primary activation carrier (`max = 112 * 112 * 24 = 301,056` floats).
    buf_a: Vec<f32>,
    /// Secondary buffer (`dwconv` output, downsample target, `attn_q`; `max = 150,528` floats).
    buf_b: Vec<f32>,
    /// Attention `K` buffer (`max = 56 * 56 * 48 = 150,528` floats).
    buf_c: Vec<f32>,
    /// L1d/L1.5 tile for `pwconv1`/`fc1` output (`M_TILE * 896 = 21,504` floats).
    hidden_tile: Vec<f32>,
    /// L1d/L1.5 im2col patch tile for `3x3` stride-2 convs (`M_TILE * 9 * 168 = 36,288` floats).
    patch_tile: Vec<f32>,
    /// Attention token weights `a` (`max = 56 * 56 = 3,136` floats).
    attn_a: Vec<f32>,
    /// Global attention context vector `g` (`max = 224` floats).
    attn_g: Vec<f32>,
}

impl SwiftFormerWorkspace {
    /// Allocate a reusable workspace sized for 224x224 SwiftFormer-XS and SwiftFormer-S.
    #[must_use]
    pub fn new() -> Self {
        Self {
            buf_a: vec![0.0f32; 112 * 112 * 24],
            buf_b: vec![0.0f32; 56 * 56 * 48],
            buf_c: vec![0.0f32; 56 * 56 * 48],
            hidden_tile: vec![0.0f32; M_TILE * 896],
            patch_tile: vec![0.0f32; M_TILE * 9 * 168],
            attn_a: vec![0.0f32; 56 * 56],
            attn_g: vec![0.0f32; 224],
        }
    }

    /// Total scratch memory in bytes (`~2.65 MiB`).
    #[must_use]
    pub fn byte_footprint(&self) -> usize {
        (self.buf_a.len()
            + self.buf_b.len()
            + self.buf_c.len()
            + self.hidden_tile.len()
            + self.patch_tile.len()
            + self.attn_a.len()
            + self.attn_g.len())
            * 4
    }
}

impl Default for SwiftFormerWorkspace {
    fn default() -> Self {
        Self::new()
    }
}

/// Folded, pre-packed NHWC SIMD executor for SwiftFormer-XS and SwiftFormer-S.
#[derive(Clone, Debug)]
pub struct SwiftFormerModel {
    kind: SwiftFormerKind,
    stem0: PackedGemm,
    stem3: PackedGemm,
    stages: [StageWeights; 4],
    /// Folded final `BatchNorm2d` + spatial mean (`1/49`) + `0.5 * (head + head_dist)`
    /// weights `[3][dim_pad]` and biases `[3]`.
    head_w: [[f32; 224]; 3],
    head_b: [f32; 3],
}

impl SwiftFormerModel {
    /// Architecture variant (`Xs` or `S`).
    #[must_use]
    pub const fn kind(&self) -> SwiftFormerKind {
        self.kind
    }

    /// Parse an ONNX model byte slice (`native-fp32.onnx`), detect whether it is
    /// `SwiftFormer-XS` or `SwiftFormer-S`, fold all affine norms/layer-scales in
    /// FP64, and pack all weights into 16-wide AVX2/SIMD panels.
    pub fn from_onnx_bytes(onnx_bytes: &[u8]) -> Result<Self, SwiftFormerError> {
        let graph = OnnxGraph::parse(onnx_bytes)?;
        let kind = graph.detect_kind()?;
        let dims = kind.dims();
        let dims_pad = kind.dims_pad();
        let layers = kind.layers();

        // Stem 0: Conv2d(3 -> 24, k=3, s=2, p=1) + ReLU
        let stem0 = graph.pack_conv3x3_s2("/stem/stem.0/Conv", 3, 3, 24)?;
        // Stem 3: Conv2d(24 -> 48, k=3, s=2, p=1) + ReLU
        let stem3 = graph.pack_conv3x3_s2("/stem/stem.3/Conv", 24, 24, dims[0])?;

        let spatial = [(56usize, 56usize), (28, 28), (14, 14), (7, 7)];
        let mut stages_vec = Vec::with_capacity(4);

        for s in 0..4 {
            let dim = dims[s];
            let dim_pad = dims_pad[s];
            let hidden = dim * 4;
            let nblocks = layers[s];
            let (in_h, in_w) = spatial[s];

            let downsample = if s > 0 {
                let prev_dim = dims[s - 1];
                let prev_pad = dims_pad[s - 1];
                Some(graph.pack_conv3x3_s2(
                    &alloc::format!("/stages/stages.{s}/downsample/proj/Conv"),
                    prev_dim,
                    prev_pad,
                    dim,
                )?)
            } else {
                None
            };

            let mut conv_blocks = Vec::with_capacity(nblocks - 1);
            for b in 0..(nblocks - 1) {
                let pfx_node = alloc::format!("/stages/stages.{s}/blocks/blocks.{b}");
                let pfx_init = alloc::format!("stages.{s}.blocks.{b}");
                conv_blocks
                    .push(graph.pack_conv_encoder(&pfx_node, &pfx_init, dim, dim_pad, hidden)?);
            }

            let b = nblocks - 1;
            let pfx_node = alloc::format!("/stages/stages.{s}/blocks/blocks.{b}");
            let pfx_init = alloc::format!("stages.{s}.blocks.{b}");

            let local_rep = graph.pack_conv_encoder(
                &alloc::format!("{pfx_node}/local_representation"),
                &alloc::format!("{pfx_init}.local_representation"),
                dim,
                dim_pad,
                dim,
            )?;

            let wq = graph.node_init(&alloc::format!("{pfx_node}/attn/to_query/MatMul"), 1)?;
            let bq = graph.init(&alloc::format!("{pfx_init}.attn.to_query.bias"))?;
            let to_query = PackedGemm::pack(wq, bq, dim, dim_pad, dim);

            let wk = graph.node_init(&alloc::format!("{pfx_node}/attn/to_key/MatMul"), 1)?;
            let bk = graph.init(&alloc::format!("{pfx_init}.attn.to_key.bias"))?;
            let to_key = PackedGemm::pack(wk, bk, dim, dim_pad, dim);

            let wg_raw = graph.init(&alloc::format!("{pfx_init}.attn.w_g"))?;
            if wg_raw.len() != dim {
                return Err(SwiftFormerError(alloc::format!(
                    "unexpected w_g len {} != {dim}",
                    wg_raw.len()
                )));
            }
            let scale_factor = (dim as f32).powf(-0.5);
            let mut w_g_scaled = vec![0.0f32; dim_pad];
            for c in 0..dim {
                w_g_scaled[c] = wg_raw[c] * scale_factor;
            }

            let wproj = graph.node_init(&alloc::format!("{pfx_node}/attn/proj/MatMul"), 1)?;
            let bproj = graph.init(&alloc::format!("{pfx_init}.attn.proj.bias"))?;
            let proj = PackedGemm::pack(wproj, bproj, dim, dim_pad, dim);

            let wfin_raw = graph.node_init(&alloc::format!("{pfx_node}/attn/final/MatMul"), 1)?;
            let bfin_raw = graph.init(&alloc::format!("{pfx_init}.attn.final.bias"))?;
            let ls1 = graph.init(&alloc::format!("{pfx_init}.layer_scale_1.gamma"))?;
            let (wfin_folded, bfin_folded) = fold_output_scale(wfin_raw, bfin_raw, ls1, dim, dim)?;
            let final_proj = PackedGemm::pack(&wfin_folded, &bfin_folded, dim, dim_pad, dim);

            // Fold linear.norm1 (BatchNorm2d) into linear.fc1 (1x1 Conv):
            let bn_w = graph.init(&alloc::format!("{pfx_init}.linear.norm1.weight"))?;
            let bn_b = graph.init(&alloc::format!("{pfx_init}.linear.norm1.bias"))?;
            let bn_m = graph.init(&alloc::format!("{pfx_init}.linear.norm1.running_mean"))?;
            let bn_v = graph.init(&alloc::format!("{pfx_init}.linear.norm1.running_var"))?;
            let fc1_w_oi = graph.init(&alloc::format!("{pfx_init}.linear.fc1.weight"))?;
            let fc1_b_raw = graph.init(&alloc::format!("{pfx_init}.linear.fc1.bias"))?;
            let fc1_w_io = oi_to_io(fc1_w_oi, dim, hidden)?;
            let (fc1_w_folded, fc1_b_folded) =
                fold_input_batchnorm(&fc1_w_io, fc1_b_raw, bn_w, bn_b, bn_m, bn_v, dim, hidden)?;
            let mlp_fc1 = PackedGemm::pack(&fc1_w_folded, &fc1_b_folded, dim, dim_pad, hidden);

            // Fold layer_scale_2 into linear.fc2:
            let fc2_w_oi = graph.init(&alloc::format!("{pfx_init}.linear.fc2.weight"))?;
            let fc2_b_raw = graph.init(&alloc::format!("{pfx_init}.linear.fc2.bias"))?;
            let ls2 = graph.init(&alloc::format!("{pfx_init}.layer_scale_2.gamma"))?;
            let fc2_w_io = oi_to_io(fc2_w_oi, hidden, dim)?;
            let (fc2_w_folded, fc2_b_folded) =
                fold_output_scale(&fc2_w_io, fc2_b_raw, ls2, hidden, dim)?;
            let mlp_fc2 = PackedGemm::pack(&fc2_w_folded, &fc2_b_folded, hidden, hidden, dim);

            stages_vec.push(StageWeights {
                downsample,
                in_h,
                in_w,
                dim_pad,
                conv_blocks,
                attn_block: StageAttentionBlock {
                    local_rep,
                    to_query,
                    to_key,
                    w_g_scaled,
                    proj,
                    final_proj,
                    mlp_fc1,
                    mlp_fc2,
                },
            });
        }

        let stages: [StageWeights; 4] = stages_vec
            .try_into()
            .map_err(|_| SwiftFormerError(String::from("expected 4 stages")))?;

        // Classifier head: fold norm (BatchNorm2d) + 0.5 * (head + head_dist)
        let last_dim = dims[3];
        let norm_w = graph.init("norm.weight")?;
        let norm_b = graph.init("norm.bias")?;
        let norm_m = graph.init("norm.running_mean")?;
        let norm_v = graph.init("norm.running_var")?;
        let head_w_raw = graph.init("head.weight")?;
        let head_dist_w_raw = graph.init("head_dist.weight")?;
        let head_b_raw = graph.init("head.bias")?;
        let head_dist_b_raw = graph.init("head_dist.bias").unwrap_or(head_b_raw);

        if head_w_raw.len() != 3 * last_dim || head_dist_w_raw.len() != 3 * last_dim {
            return Err(SwiftFormerError(String::from(
                "unexpected head weight shape",
            )));
        }

        let mut head_w = [[0.0f32; 224]; 3];
        let mut head_b = [0.0f32; 3];
        for cls in 0..3 {
            let mut b_acc = 0.5 * (f64::from(head_b_raw[cls]) + f64::from(head_dist_b_raw[cls]));
            for c in 0..last_dim {
                let scale = f64::from(norm_w[c]) / (f64::from(norm_v[c]) + 1e-5).sqrt();
                let offset = f64::from(norm_b[c]) - f64::from(norm_m[c]) * scale;
                let w_avg = 0.5
                    * (f64::from(head_w_raw[cls * last_dim + c])
                        + f64::from(head_dist_w_raw[cls * last_dim + c]));
                b_acc += w_avg * offset;
                // Fold the 1/49 spatial average directly into head_w so the
                // runtime head only needs a plain sum over the 49 tokens.
                head_w[cls][c] = ((w_avg * scale) / 49.0) as f32;
            }
            head_b[cls] = b_acc as f32;
        }

        Ok(Self {
            kind,
            stem0,
            stem3,
            stages,
            head_w,
            head_b,
        })
    }

    /// Run a single warm forward pass on an NCHW `[1, 3, 224, 224]` input slice,
    /// returning the 3 class logits (`[class_0, class_1, class_2]`).
    /// Zero heap allocations occur during this call.
    pub fn forward_nchw(&self, input_nchw: &[f32], ws: &mut SwiftFormerWorkspace) -> [f32; 3] {
        assert_eq!(input_nchw.len(), 3 * 224 * 224);
        let mut logits = [0.0f32; 3];
        incant!(
            forward_impl(self, input_nchw, ws, &mut logits),
            [v4x(cfg(avx512)), v4(cfg(avx512)), v3, neon, wasm128, scalar]
        );
        logits
    }
}

// ---------------------------------------------------------------------------
// Load-time FP64 algebraic folding helpers
// ---------------------------------------------------------------------------

fn oi_to_io(w_oi: &[f32], in_c: usize, out_c: usize) -> Result<Vec<f32>, SwiftFormerError> {
    if w_oi.len() != in_c * out_c {
        return Err(SwiftFormerError(alloc::format!(
            "weight len {} != in_c({in_c}) * out_c({out_c})",
            w_oi.len()
        )));
    }
    let mut out = vec![0.0f32; in_c * out_c];
    for oc in 0..out_c {
        for ic in 0..in_c {
            out[ic * out_c + oc] = w_oi[oc * in_c + ic];
        }
    }
    Ok(out)
}

fn fold_output_scale(
    w_io: &[f32],
    bias: &[f32],
    gamma: &[f32],
    in_c: usize,
    out_c: usize,
) -> Result<(Vec<f32>, Vec<f32>), SwiftFormerError> {
    if w_io.len() != in_c * out_c || bias.len() != out_c || gamma.len() != out_c {
        return Err(SwiftFormerError(String::from(
            "fold_output_scale dimension mismatch",
        )));
    }
    let mut w_out = vec![0.0f32; in_c * out_c];
    let mut b_out = vec![0.0f32; out_c];
    for oc in 0..out_c {
        let g = f64::from(gamma[oc]);
        b_out[oc] = (f64::from(bias[oc]) * g) as f32;
        for ic in 0..in_c {
            w_out[ic * out_c + oc] = (f64::from(w_io[ic * out_c + oc]) * g) as f32;
        }
    }
    Ok((w_out, b_out))
}

#[allow(clippy::too_many_arguments)]
fn fold_input_batchnorm(
    w_io: &[f32],
    bias: &[f32],
    bn_w: &[f32],
    bn_b: &[f32],
    bn_m: &[f32],
    bn_v: &[f32],
    in_c: usize,
    out_c: usize,
) -> Result<(Vec<f32>, Vec<f32>), SwiftFormerError> {
    if w_io.len() != in_c * out_c
        || bias.len() != out_c
        || bn_w.len() != in_c
        || bn_b.len() != in_c
        || bn_m.len() != in_c
        || bn_v.len() != in_c
    {
        return Err(SwiftFormerError(String::from(
            "fold_input_batchnorm dimension mismatch",
        )));
    }
    let mut scale = vec![0.0f64; in_c];
    let mut offset = vec![0.0f64; in_c];
    for ic in 0..in_c {
        let s = f64::from(bn_w[ic]) / (f64::from(bn_v[ic]) + 1e-5).sqrt();
        scale[ic] = s;
        offset[ic] = f64::from(bn_b[ic]) - f64::from(bn_m[ic]) * s;
    }
    let mut w_out = vec![0.0f32; in_c * out_c];
    let mut b_out = vec![0.0f32; out_c];
    for oc in 0..out_c {
        let mut b_acc = f64::from(bias[oc]);
        for ic in 0..in_c {
            let w_val = f64::from(w_io[ic * out_c + oc]);
            w_out[ic * out_c + oc] = (w_val * scale[ic]) as f32;
            b_acc += w_val * offset[ic];
        }
        b_out[oc] = b_acc as f32;
    }
    Ok((w_out, b_out))
}

// ---------------------------------------------------------------------------
// Multi-tier SIMD dispatch entry points (single boundary per forward pass)
// ---------------------------------------------------------------------------

#[magetypes(v3, neon, wasm128, scalar)]
fn forward_impl(
    token: Token,
    model: &SwiftFormerModel,
    input_nchw: &[f32],
    ws: &mut SwiftFormerWorkspace,
    out_logits: &mut [f32; 3],
) {
    k8::run_forward(token, model, input_nchw, ws, out_logits);
}

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
#[arcane]
fn forward_impl_v4x(
    token: X64V4xToken,
    model: &SwiftFormerModel,
    input_nchw: &[f32],
    ws: &mut SwiftFormerWorkspace,
    out_logits: &mut [f32; 3],
) {
    k8::run_forward(token, model, input_nchw, ws, out_logits);
}

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
#[arcane]
fn forward_impl_v4(
    token: X64V4Token,
    model: &SwiftFormerModel,
    input_nchw: &[f32],
    ws: &mut SwiftFormerWorkspace,
    out_logits: &mut [f32; 3],
) {
    k8::run_forward(token, model, input_nchw, ws, out_logits);
}

// ---------------------------------------------------------------------------
// SIMD kernel body (shared across all tiers via F32x8Convert)
// ---------------------------------------------------------------------------

mod k8 {
    use super::{
        ConvEncoderWeights, M_TILE, PackedDepthwise, PackedGemm, StageAttentionBlock,
        SwiftFormerModel, SwiftFormerWorkspace,
    };
    use magetypes::simd::backends::F32x8Convert as Backend;
    use magetypes::simd::generic::f32x8 as V;

    #[derive(Clone, Copy, PartialEq, Eq)]
    pub(crate) enum GemmOutMode {
        Write,
        ReLU,
        AddInPlace,
    }

    #[inline(always)]
    fn max_x86<T: Backend>(a: V<T>, b: V<T>) -> V<T> {
        V::<T>::blend(a.simd_gt(b), a, b)
    }

    #[inline(always)]
    fn min_x86<T: Backend>(a: V<T>, b: V<T>) -> V<T> {
        V::<T>::blend(a.simd_lt(b), a, b)
    }

    /// Vectorized in-place exact-ERF GELU:
    /// `0.5 * x * (1 + erf(x / sqrt(2)))` using Abramowitz & Stegun 7.1.26
    /// (matching `tract-linalg`'s `x86_64_fma_erf_f32` polynomial).
    /// Unrolled 2x (`16` floats/iter) so all 9 constants + 6 working registers
    /// fit in 15 `ymm` registers without stack spills.
    #[inline(always)]
    pub(crate) fn gelu_exact_in_place<T: Backend>(token: T, buf: &mut [f32]) {
        let sp = |c: f32| V::<T>::splat_t(token, c);
        let zero = V::<T>::zero_t(token);
        let one = sp(1.0);
        let half = sp(0.5);
        let inv_sqrt2 = sp(core::f32::consts::FRAC_1_SQRT_2);
        let max_a = sp(5.0);
        let a1 = sp(0.070_523_08);
        let a2 = sp(0.042_282_01);
        let a3 = sp(0.009_270_527);
        let a4 = sp(0.000_152_014_3);
        let a5 = sp(0.000_276_567_2);
        let a6 = sp(0.000_043_063_8);

        let n = buf.len();
        let mut i = 0usize;
        while i + 16 <= n {
            let x0 = V::<T>::from_slice_t(token, &buf[i..]);
            let x1 = V::<T>::from_slice_t(token, &buf[i + 8..]);

            let z0 = x0 * inv_sqrt2;
            let z1 = x1 * inv_sqrt2;
            let abs0 = min_x86(z0.abs(), max_a);
            let abs1 = min_x86(z1.abs(), max_a);

            let mut y0 = abs0.mul_add_relaxed(a6, a5);
            let mut y1 = abs1.mul_add_relaxed(a6, a5);
            y0 = y0.mul_add_relaxed(abs0, a4);
            y1 = y1.mul_add_relaxed(abs1, a4);
            y0 = y0.mul_add_relaxed(abs0, a3);
            y1 = y1.mul_add_relaxed(abs1, a3);
            y0 = y0.mul_add_relaxed(abs0, a2);
            y1 = y1.mul_add_relaxed(abs1, a2);
            y0 = y0.mul_add_relaxed(abs0, a1);
            y1 = y1.mul_add_relaxed(abs1, a1);
            y0 = y0.mul_add_relaxed(abs0, one);
            y1 = y1.mul_add_relaxed(abs1, one);

            y0 = y0 * y0;
            y1 = y1 * y1;
            y0 = y0 * y0;
            y1 = y1 * y1;
            y0 = y0 * y0;
            y1 = y1 * y1;
            y0 = y0 * y0;
            y1 = y1 * y1;

            let e0_abs = one - (one / y0);
            let e1_abs = one - (one / y1);
            let e0 = V::<T>::blend(z0.simd_lt(zero), zero - e0_abs, e0_abs);
            let e1 = V::<T>::blend(z1.simd_lt(zero), zero - e1_abs, e1_abs);

            let r0 = (x0 * (e0 + one)) * half;
            let r1 = (x1 * (e1 + one)) * half;

            let d0: &mut [f32; 8] = (&mut buf[i..i + 8]).try_into().unwrap();
            r0.store(d0);
            let d1: &mut [f32; 8] = (&mut buf[i + 8..i + 16]).try_into().unwrap();
            r1.store(d1);

            i += 16;
        }
        while i + 8 <= n {
            let x0 = V::<T>::from_slice_t(token, &buf[i..]);
            let z0 = x0 * inv_sqrt2;
            let abs0 = min_x86(z0.abs(), max_a);
            let mut y0 = abs0.mul_add_relaxed(a6, a5);
            y0 = y0.mul_add_relaxed(abs0, a4);
            y0 = y0.mul_add_relaxed(abs0, a3);
            y0 = y0.mul_add_relaxed(abs0, a2);
            y0 = y0.mul_add_relaxed(abs0, a1);
            y0 = y0.mul_add_relaxed(abs0, one);
            y0 = y0 * y0;
            y0 = y0 * y0;
            y0 = y0 * y0;
            y0 = y0 * y0;
            let e0_abs = one - (one / y0);
            let e0 = V::<T>::blend(z0.simd_lt(zero), zero - e0_abs, e0_abs);
            let r0 = (x0 * (e0 + one)) * half;
            let d0: &mut [f32; 8] = (&mut buf[i..i + 8]).try_into().unwrap();
            r0.store(d0);
            i += 8;
        }
    }

    #[inline(always)]
    fn store_vec8<T: Backend>(token: T, acc: V<T>, mode: GemmOutMode, dst_slice: &mut [f32]) {
        let zero = V::<T>::zero_t(token);
        let v = match mode {
            GemmOutMode::Write => acc,
            GemmOutMode::ReLU => max_x86(acc, zero),
            GemmOutMode::AddInPlace => V::<T>::from_slice_t(token, dst_slice) + acc,
        };
        let dst: &mut [f32; 8] = (&mut dst_slice[..8]).try_into().unwrap();
        v.store(dst);
    }

    /// Execute `inp[0..m_count, 0..in_c] @ W + bias` into `out[0..m_count, 0..out_c_pad]`
    /// using `6x16`, `4x16`, and `1x16` register-blocked FMA microkernels (plus `6x8`/`4x8`/`1x8`
    /// when `out_c_pad % 16 == 8`).
    ///
    /// Outer loop is over 16-channel panels (`p16 in 0..nv16`) and inner loop is over
    /// the `m_count` rows in the L1d tile so each weight panel is loaded once per tile.
    #[inline(always)]
    pub(crate) fn gemm_tile<T: Backend>(
        token: T,
        inp: &[f32],
        m_count: usize,
        gemm: &PackedGemm,
        mode: GemmOutMode,
        out: &mut [f32],
    ) {
        let in_c = gemm.in_c;
        let in_stride = gemm.in_stride;
        let out_c = gemm.out_c_pad;

        for p16 in 0..gemm.nv16 {
            let co = p16 * 16;
            let b0 = V::<T>::from_slice_t(token, &gemm.bias[co..]);
            let b1 = V::<T>::from_slice_t(token, &gemm.bias[co + 8..]);
            let w_panel = &gemm.packed_w[p16 * in_c * 16..(p16 + 1) * in_c * 16];
            let w_chunks = &w_panel.as_chunks::<16>().0[..in_c];

            let mut m = 0usize;
            while m + 6 <= m_count {
                let p0 = &inp[m * in_stride..m * in_stride + in_c];
                let p1 = &inp[(m + 1) * in_stride..(m + 1) * in_stride + in_c];
                let p2 = &inp[(m + 2) * in_stride..(m + 2) * in_stride + in_c];
                let p3 = &inp[(m + 3) * in_stride..(m + 3) * in_stride + in_c];
                let p4 = &inp[(m + 4) * in_stride..(m + 4) * in_stride + in_c];
                let p5 = &inp[(m + 5) * in_stride..(m + 5) * in_stride + in_c];

                let mut a0_0 = b0;
                let mut a0_1 = b1;
                let mut a1_0 = b0;
                let mut a1_1 = b1;
                let mut a2_0 = b0;
                let mut a2_1 = b1;
                let mut a3_0 = b0;
                let mut a3_1 = b1;
                let mut a4_0 = b0;
                let mut a4_1 = b1;
                let mut a5_0 = b0;
                let mut a5_1 = b1;

                for (ic, wf) in w_chunks.iter().enumerate() {
                    let w0 = V::<T>::from_slice_t(token, &wf[0..]);
                    let w1 = V::<T>::from_slice_t(token, &wf[8..]);

                    let s0 = V::<T>::splat_t(token, p0[ic]);
                    a0_0 = s0.mul_add_relaxed(w0, a0_0);
                    a0_1 = s0.mul_add_relaxed(w1, a0_1);

                    let s1 = V::<T>::splat_t(token, p1[ic]);
                    a1_0 = s1.mul_add_relaxed(w0, a1_0);
                    a1_1 = s1.mul_add_relaxed(w1, a1_1);

                    let s2 = V::<T>::splat_t(token, p2[ic]);
                    a2_0 = s2.mul_add_relaxed(w0, a2_0);
                    a2_1 = s2.mul_add_relaxed(w1, a2_1);

                    let s3 = V::<T>::splat_t(token, p3[ic]);
                    a3_0 = s3.mul_add_relaxed(w0, a3_0);
                    a3_1 = s3.mul_add_relaxed(w1, a3_1);

                    let s4 = V::<T>::splat_t(token, p4[ic]);
                    a4_0 = s4.mul_add_relaxed(w0, a4_0);
                    a4_1 = s4.mul_add_relaxed(w1, a4_1);

                    let s5 = V::<T>::splat_t(token, p5[ic]);
                    a5_0 = s5.mul_add_relaxed(w0, a5_0);
                    a5_1 = s5.mul_add_relaxed(w1, a5_1);
                }

                store_vec8(token, a0_0, mode, &mut out[m * out_c + co..]);
                store_vec8(token, a0_1, mode, &mut out[m * out_c + co + 8..]);
                store_vec8(token, a1_0, mode, &mut out[(m + 1) * out_c + co..]);
                store_vec8(token, a1_1, mode, &mut out[(m + 1) * out_c + co + 8..]);
                store_vec8(token, a2_0, mode, &mut out[(m + 2) * out_c + co..]);
                store_vec8(token, a2_1, mode, &mut out[(m + 2) * out_c + co + 8..]);
                store_vec8(token, a3_0, mode, &mut out[(m + 3) * out_c + co..]);
                store_vec8(token, a3_1, mode, &mut out[(m + 3) * out_c + co + 8..]);
                store_vec8(token, a4_0, mode, &mut out[(m + 4) * out_c + co..]);
                store_vec8(token, a4_1, mode, &mut out[(m + 4) * out_c + co + 8..]);
                store_vec8(token, a5_0, mode, &mut out[(m + 5) * out_c + co..]);
                store_vec8(token, a5_1, mode, &mut out[(m + 5) * out_c + co + 8..]);

                m += 6;
            }

            if m + 4 <= m_count {
                let p0 = &inp[m * in_stride..m * in_stride + in_c];
                let p1 = &inp[(m + 1) * in_stride..(m + 1) * in_stride + in_c];
                let p2 = &inp[(m + 2) * in_stride..(m + 2) * in_stride + in_c];
                let p3 = &inp[(m + 3) * in_stride..(m + 3) * in_stride + in_c];

                let mut a0_0 = b0;
                let mut a0_1 = b1;
                let mut a1_0 = b0;
                let mut a1_1 = b1;
                let mut a2_0 = b0;
                let mut a2_1 = b1;
                let mut a3_0 = b0;
                let mut a3_1 = b1;

                for (ic, wf) in w_chunks.iter().enumerate() {
                    let w0 = V::<T>::from_slice_t(token, &wf[0..]);
                    let w1 = V::<T>::from_slice_t(token, &wf[8..]);

                    let s0 = V::<T>::splat_t(token, p0[ic]);
                    a0_0 = s0.mul_add_relaxed(w0, a0_0);
                    a0_1 = s0.mul_add_relaxed(w1, a0_1);

                    let s1 = V::<T>::splat_t(token, p1[ic]);
                    a1_0 = s1.mul_add_relaxed(w0, a1_0);
                    a1_1 = s1.mul_add_relaxed(w1, a1_1);

                    let s2 = V::<T>::splat_t(token, p2[ic]);
                    a2_0 = s2.mul_add_relaxed(w0, a2_0);
                    a2_1 = s2.mul_add_relaxed(w1, a2_1);

                    let s3 = V::<T>::splat_t(token, p3[ic]);
                    a3_0 = s3.mul_add_relaxed(w0, a3_0);
                    a3_1 = s3.mul_add_relaxed(w1, a3_1);
                }

                store_vec8(token, a0_0, mode, &mut out[m * out_c + co..]);
                store_vec8(token, a0_1, mode, &mut out[m * out_c + co + 8..]);
                store_vec8(token, a1_0, mode, &mut out[(m + 1) * out_c + co..]);
                store_vec8(token, a1_1, mode, &mut out[(m + 1) * out_c + co + 8..]);
                store_vec8(token, a2_0, mode, &mut out[(m + 2) * out_c + co..]);
                store_vec8(token, a2_1, mode, &mut out[(m + 2) * out_c + co + 8..]);
                store_vec8(token, a3_0, mode, &mut out[(m + 3) * out_c + co..]);
                store_vec8(token, a3_1, mode, &mut out[(m + 3) * out_c + co + 8..]);

                m += 4;
            }

            while m < m_count {
                let p0 = &inp[m * in_stride..m * in_stride + in_c];
                let mut a0_0 = b0;
                let mut a0_1 = b1;
                for (ic, wf) in w_chunks.iter().enumerate() {
                    let w0 = V::<T>::from_slice_t(token, &wf[0..]);
                    let w1 = V::<T>::from_slice_t(token, &wf[8..]);
                    let s0 = V::<T>::splat_t(token, p0[ic]);
                    a0_0 = s0.mul_add_relaxed(w0, a0_0);
                    a0_1 = s0.mul_add_relaxed(w1, a0_1);
                }
                store_vec8(token, a0_0, mode, &mut out[m * out_c + co..]);
                store_vec8(token, a0_1, mode, &mut out[m * out_c + co + 8..]);
                m += 1;
            }
        }

        if gemm.has_rem8 {
            let co = gemm.nv16 * 16;
            let b0 = V::<T>::from_slice_t(token, &gemm.bias[co..]);
            let w_panel = &gemm.packed_w[gemm.nv16 * in_c * 16..];
            let w_chunks = &w_panel.as_chunks::<8>().0[..in_c];

            let mut m = 0usize;
            while m + 6 <= m_count {
                let p0 = &inp[m * in_stride..m * in_stride + in_c];
                let p1 = &inp[(m + 1) * in_stride..(m + 1) * in_stride + in_c];
                let p2 = &inp[(m + 2) * in_stride..(m + 2) * in_stride + in_c];
                let p3 = &inp[(m + 3) * in_stride..(m + 3) * in_stride + in_c];
                let p4 = &inp[(m + 4) * in_stride..(m + 4) * in_stride + in_c];
                let p5 = &inp[(m + 5) * in_stride..(m + 5) * in_stride + in_c];

                let mut a0 = b0;
                let mut a1 = b0;
                let mut a2 = b0;
                let mut a3 = b0;
                let mut a4 = b0;
                let mut a5 = b0;

                for (ic, wf) in w_chunks.iter().enumerate() {
                    let w0 = V::<T>::from_slice_t(token, wf);
                    a0 = V::<T>::splat_t(token, p0[ic]).mul_add_relaxed(w0, a0);
                    a1 = V::<T>::splat_t(token, p1[ic]).mul_add_relaxed(w0, a1);
                    a2 = V::<T>::splat_t(token, p2[ic]).mul_add_relaxed(w0, a2);
                    a3 = V::<T>::splat_t(token, p3[ic]).mul_add_relaxed(w0, a3);
                    a4 = V::<T>::splat_t(token, p4[ic]).mul_add_relaxed(w0, a4);
                    a5 = V::<T>::splat_t(token, p5[ic]).mul_add_relaxed(w0, a5);
                }

                store_vec8(token, a0, mode, &mut out[m * out_c + co..]);
                store_vec8(token, a1, mode, &mut out[(m + 1) * out_c + co..]);
                store_vec8(token, a2, mode, &mut out[(m + 2) * out_c + co..]);
                store_vec8(token, a3, mode, &mut out[(m + 3) * out_c + co..]);
                store_vec8(token, a4, mode, &mut out[(m + 4) * out_c + co..]);
                store_vec8(token, a5, mode, &mut out[(m + 5) * out_c + co..]);
                m += 6;
            }
            while m < m_count {
                let p0 = &inp[m * in_stride..m * in_stride + in_c];
                let mut a0 = b0;
                for (ic, wf) in w_chunks.iter().enumerate() {
                    let w0 = V::<T>::from_slice_t(token, wf);
                    a0 = V::<T>::splat_t(token, p0[ic]).mul_add_relaxed(w0, a0);
                }
                store_vec8(token, a0, mode, &mut out[m * out_c + co..]);
                m += 1;
            }
        }
    }

    /// Run a full-tensor GEMM by tiling over `M_TILE` rows at a time.
    #[inline(always)]
    fn gemm_all<T: Backend>(
        token: T,
        inp: &[f32],
        npix: usize,
        gemm: &PackedGemm,
        mode: GemmOutMode,
        out: &mut [f32],
    ) {
        let in_stride = gemm.in_stride;
        let out_c = gemm.out_c_pad;
        let mut m0 = 0usize;
        while m0 < npix {
            let mc = (npix - m0).min(M_TILE);
            gemm_tile(
                token,
                &inp[m0 * in_stride..(m0 + mc) * in_stride],
                mc,
                gemm,
                mode,
                &mut out[m0 * out_c..(m0 + mc) * out_c],
            );
            m0 += mc;
        }
    }

    /// Run fused `pwconv1 (GeluExact) -> pwconv2 (AddInPlace)` in L1d-resident tiles
    /// of `M_TILE = 24` tokens so the `4x` expanded tensor never leaves L1d/L1.5 cache.
    #[inline(always)]
    pub(crate) fn mlp_tiled_residual<T: Backend>(
        token: T,
        inp: &[f32],
        npix: usize,
        pw1: &PackedGemm,
        pw2: &PackedGemm,
        hidden_tile: &mut [f32],
        res_inout: &mut [f32],
    ) {
        let in_stride = pw1.in_stride;
        let hidden = pw1.out_c_pad;
        let out_stride = pw2.out_c_pad;
        let mut m0 = 0usize;
        while m0 < npix {
            let mc = (npix - m0).min(M_TILE);
            let h_slice = &mut hidden_tile[..mc * hidden];
            gemm_tile(
                token,
                &inp[m0 * in_stride..(m0 + mc) * in_stride],
                mc,
                pw1,
                GemmOutMode::Write,
                h_slice,
            );
            gelu_exact_in_place(token, h_slice);
            gemm_tile(
                token,
                h_slice,
                mc,
                pw2,
                GemmOutMode::AddInPlace,
                &mut res_inout[m0 * out_stride..(m0 + mc) * out_stride],
            );
            m0 += mc;
        }
    }

    /// Depthwise `3x3` stride-1 pad-1 on NHWC `[h, w, c_pad]`.
    /// For interior rows (`1..h-1`) and interior columns (`1..w-1`), keeps all
    /// 9 tap weight vectors `w00..w22` + `bias` in 10 `ymm` registers across the
    /// row and unrolls 2 horizontal pixels (`ox, ox+1`) at a time (12 loads for 18 FMAs).
    #[inline(always)]
    pub(crate) fn depthwise_3x3_s1<T: Backend>(
        token: T,
        inp: &[f32],
        h: usize,
        w: usize,
        dw: &PackedDepthwise,
        out: &mut [f32],
    ) {
        let c = dw.c_pad;
        let nv = dw.nv;
        let row_len = w * c;

        for oy in 0..h {
            let out_row = &mut out[oy * row_len..(oy + 1) * row_len];
            if oy > 0 && oy + 1 < h {
                let r0 = &inp[(oy - 1) * row_len..oy * row_len];
                let r1 = &inp[oy * row_len..(oy + 1) * row_len];
                let r2 = &inp[(oy + 1) * row_len..(oy + 2) * row_len];

                for cv in 0..nv {
                    let co = cv * 8;
                    let b_v = V::<T>::from_slice_t(token, &dw.bias[co..]);
                    let wb = &dw.packed_w[cv * 9 * 8..(cv + 1) * 9 * 8];
                    let w00 = V::<T>::from_slice_t(token, &wb[0..]);
                    let w01 = V::<T>::from_slice_t(token, &wb[8..]);
                    let w02 = V::<T>::from_slice_t(token, &wb[16..]);
                    let w10 = V::<T>::from_slice_t(token, &wb[24..]);
                    let w11 = V::<T>::from_slice_t(token, &wb[32..]);
                    let w12 = V::<T>::from_slice_t(token, &wb[40..]);
                    let w20 = V::<T>::from_slice_t(token, &wb[48..]);
                    let w21 = V::<T>::from_slice_t(token, &wb[56..]);
                    let w22 = V::<T>::from_slice_t(token, &wb[64..]);

                    // Left border ox = 0 (kx = 1, 2 only)
                    {
                        let mut acc = b_v;
                        acc = V::<T>::from_slice_t(token, &r0[co..]).mul_add_relaxed(w01, acc);
                        acc = V::<T>::from_slice_t(token, &r0[c + co..]).mul_add_relaxed(w02, acc);
                        acc = V::<T>::from_slice_t(token, &r1[co..]).mul_add_relaxed(w11, acc);
                        acc = V::<T>::from_slice_t(token, &r1[c + co..]).mul_add_relaxed(w12, acc);
                        acc = V::<T>::from_slice_t(token, &r2[co..]).mul_add_relaxed(w21, acc);
                        acc = V::<T>::from_slice_t(token, &r2[c + co..]).mul_add_relaxed(w22, acc);
                        let dst: &mut [f32; 8] = (&mut out_row[co..co + 8]).try_into().unwrap();
                        acc.store(dst);
                    }

                    // Interior columns ox in 1..w-1, unrolled by 2
                    let mut ox = 1usize;
                    while ox + 2 < w {
                        let base = (ox - 1) * c + co;
                        let mut acc0 = b_v;
                        let mut acc1 = b_v;

                        let v00 = V::<T>::from_slice_t(token, &r0[base..]);
                        let v01 = V::<T>::from_slice_t(token, &r0[base + c..]);
                        let v02 = V::<T>::from_slice_t(token, &r0[base + 2 * c..]);
                        let v03 = V::<T>::from_slice_t(token, &r0[base + 3 * c..]);
                        acc0 = v00.mul_add_relaxed(w00, acc0);
                        acc1 = v01.mul_add_relaxed(w00, acc1);
                        acc0 = v01.mul_add_relaxed(w01, acc0);
                        acc1 = v02.mul_add_relaxed(w01, acc1);
                        acc0 = v02.mul_add_relaxed(w02, acc0);
                        acc1 = v03.mul_add_relaxed(w02, acc1);

                        let v10 = V::<T>::from_slice_t(token, &r1[base..]);
                        let v11 = V::<T>::from_slice_t(token, &r1[base + c..]);
                        let v12 = V::<T>::from_slice_t(token, &r1[base + 2 * c..]);
                        let v13 = V::<T>::from_slice_t(token, &r1[base + 3 * c..]);
                        acc0 = v10.mul_add_relaxed(w10, acc0);
                        acc1 = v11.mul_add_relaxed(w10, acc1);
                        acc0 = v11.mul_add_relaxed(w11, acc0);
                        acc1 = v12.mul_add_relaxed(w11, acc1);
                        acc0 = v12.mul_add_relaxed(w12, acc0);
                        acc1 = v13.mul_add_relaxed(w12, acc1);

                        let v20 = V::<T>::from_slice_t(token, &r2[base..]);
                        let v21 = V::<T>::from_slice_t(token, &r2[base + c..]);
                        let v22 = V::<T>::from_slice_t(token, &r2[base + 2 * c..]);
                        let v23 = V::<T>::from_slice_t(token, &r2[base + 3 * c..]);
                        acc0 = v20.mul_add_relaxed(w20, acc0);
                        acc1 = v21.mul_add_relaxed(w20, acc1);
                        acc0 = v21.mul_add_relaxed(w21, acc0);
                        acc1 = v22.mul_add_relaxed(w21, acc1);
                        acc0 = v22.mul_add_relaxed(w22, acc0);
                        acc1 = v23.mul_add_relaxed(w22, acc1);

                        let d0: &mut [f32; 8] = (&mut out_row[ox * c + co..ox * c + co + 8])
                            .try_into()
                            .unwrap();
                        acc0.store(d0);
                        let d1: &mut [f32; 8] = (&mut out_row
                            [(ox + 1) * c + co..(ox + 1) * c + co + 8])
                            .try_into()
                            .unwrap();
                        acc1.store(d1);

                        ox += 2;
                    }
                    while ox < w - 1 {
                        let base = (ox - 1) * c + co;
                        let mut acc = b_v;
                        acc = V::<T>::from_slice_t(token, &r0[base..]).mul_add_relaxed(w00, acc);
                        acc =
                            V::<T>::from_slice_t(token, &r0[base + c..]).mul_add_relaxed(w01, acc);
                        acc = V::<T>::from_slice_t(token, &r0[base + 2 * c..])
                            .mul_add_relaxed(w02, acc);
                        acc = V::<T>::from_slice_t(token, &r1[base..]).mul_add_relaxed(w10, acc);
                        acc =
                            V::<T>::from_slice_t(token, &r1[base + c..]).mul_add_relaxed(w11, acc);
                        acc = V::<T>::from_slice_t(token, &r1[base + 2 * c..])
                            .mul_add_relaxed(w12, acc);
                        acc = V::<T>::from_slice_t(token, &r2[base..]).mul_add_relaxed(w20, acc);
                        acc =
                            V::<T>::from_slice_t(token, &r2[base + c..]).mul_add_relaxed(w21, acc);
                        acc = V::<T>::from_slice_t(token, &r2[base + 2 * c..])
                            .mul_add_relaxed(w22, acc);
                        let d0: &mut [f32; 8] = (&mut out_row[ox * c + co..ox * c + co + 8])
                            .try_into()
                            .unwrap();
                        acc.store(d0);
                        ox += 1;
                    }

                    // Right border ox = w - 1 (kx = 0, 1 only)
                    {
                        let base = (w - 2) * c + co;
                        let mut acc = b_v;
                        acc = V::<T>::from_slice_t(token, &r0[base..]).mul_add_relaxed(w00, acc);
                        acc =
                            V::<T>::from_slice_t(token, &r0[base + c..]).mul_add_relaxed(w01, acc);
                        acc = V::<T>::from_slice_t(token, &r1[base..]).mul_add_relaxed(w10, acc);
                        acc =
                            V::<T>::from_slice_t(token, &r1[base + c..]).mul_add_relaxed(w11, acc);
                        acc = V::<T>::from_slice_t(token, &r2[base..]).mul_add_relaxed(w20, acc);
                        acc =
                            V::<T>::from_slice_t(token, &r2[base + c..]).mul_add_relaxed(w21, acc);
                        let dst: &mut [f32; 8] = (&mut out_row
                            [(w - 1) * c + co..(w - 1) * c + co + 8])
                            .try_into()
                            .unwrap();
                        acc.store(dst);
                    }
                }
            } else {
                // Top (oy == 0) or bottom (oy == h - 1) border row
                for ox in 0..w {
                    for cv in 0..nv {
                        let co = cv * 8;
                        let mut acc = V::<T>::from_slice_t(token, &dw.bias[co..]);
                        let wb = &dw.packed_w[cv * 9 * 8..(cv + 1) * 9 * 8];
                        for ky in 0..3usize {
                            let iy = oy + ky;
                            if iy == 0 || iy > h {
                                continue;
                            }
                            let in_row = &inp[(iy - 1) * row_len..iy * row_len];
                            for kx in 0..3usize {
                                let ix = ox + kx;
                                if ix == 0 || ix > w {
                                    continue;
                                }
                                let v = V::<T>::from_slice_t(token, &in_row[(ix - 1) * c + co..]);
                                let wv = V::<T>::from_slice_t(token, &wb[(ky * 3 + kx) * 8..]);
                                acc = v.mul_add_relaxed(wv, acc);
                            }
                        }
                        let dst: &mut [f32; 8] = (&mut out_row[ox * c + co..ox * c + co + 8])
                            .try_into()
                            .unwrap();
                        acc.store(dst);
                    }
                }
            }
        }
    }

    /// Stem 0 (`3x3` stride-2 pad-1, NCHW `[3, 224, 224]` -> NHWC `[112, 112, 24]`, ReLU)
    /// via L1d patch gathering + `6x16` / `6x8` GEMM microkernel.
    #[inline(always)]
    fn stem0_nchw_to_nhwc<T: Backend>(
        token: T,
        input_nchw: &[f32],
        stem0: &PackedGemm,
        patch_tile: &mut [f32],
        out: &mut [f32],
    ) {
        const H: usize = 224;
        const W: usize = 224;
        const OH: usize = 112;
        const OW: usize = 112;
        const K: usize = 27;
        let out_c = stem0.out_c_pad;
        let r_plane = &input_nchw[0..H * W];
        let g_plane = &input_nchw[H * W..2 * H * W];
        let b_plane = &input_nchw[2 * H * W..3 * H * W];

        let total_out = OH * OW;
        let mut m0 = 0usize;
        while m0 < total_out {
            let mc = (total_out - m0).min(M_TILE);
            for m in 0..mc {
                let idx = m0 + m;
                let oy = idx / OW;
                let ox = idx % OW;
                let iy0 = oy * 2;
                let ix0 = ox * 2;
                let patch = &mut patch_tile[m * K..(m + 1) * K];
                for ky in 0..3usize {
                    let iy = iy0 + ky;
                    for kx in 0..3usize {
                        let ix = ix0 + kx;
                        let dst_off = (ky * 3 + kx) * 3;
                        if iy == 0 || iy > H || ix == 0 || ix > W {
                            patch[dst_off] = 0.0;
                            patch[dst_off + 1] = 0.0;
                            patch[dst_off + 2] = 0.0;
                        } else {
                            let p = (iy - 1) * W + (ix - 1);
                            patch[dst_off] = r_plane[p];
                            patch[dst_off + 1] = g_plane[p];
                            patch[dst_off + 2] = b_plane[p];
                        }
                    }
                }
            }
            gemm_tile(
                token,
                &patch_tile[..mc * K],
                mc,
                stem0,
                GemmOutMode::ReLU,
                &mut out[m0 * out_c..(m0 + mc) * out_c],
            );
            m0 += mc;
        }
    }

    /// Dense `3x3` stride-2 pad-1 convolution on NHWC `[h, w, c_in]` -> `[h/2, w/2, out_c_pad]`
    /// via L1d row-slice gathering + `6x16` / `4x16` GEMM microkernel.
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    fn conv3x3_s2_nhwc_gemm<T: Backend>(
        token: T,
        inp: &[f32],
        h: usize,
        w: usize,
        c_in: usize,
        gemm: &PackedGemm,
        mode: GemmOutMode,
        patch_tile: &mut [f32],
        out: &mut [f32],
    ) {
        let oh = h / 2;
        let ow = w / 2;
        let k_len = 9 * c_in;
        let row_stride = w * c_in;
        let out_c = gemm.out_c_pad;
        let total_out = oh * ow;

        let mut m0 = 0usize;
        while m0 < total_out {
            let mc = (total_out - m0).min(M_TILE);
            for m in 0..mc {
                let idx = m0 + m;
                let oy = idx / ow;
                let ox = idx % ow;
                let iy0 = oy * 2;
                let ix0 = ox * 2;
                let patch = &mut patch_tile[m * k_len..(m + 1) * k_len];

                for ky in 0..3usize {
                    let iy = iy0 + ky;
                    let row_dst = &mut patch[ky * 3 * c_in..(ky + 1) * 3 * c_in];
                    if iy == 0 || iy > h {
                        row_dst.fill(0.0);
                    } else {
                        let in_row = &inp[(iy - 1) * row_stride..iy * row_stride];
                        if ix0 >= 1 && ix0 + 1 < w {
                            // All 3 horizontal pixels (ix0-1, ix0, ix0+1) are contiguous in NHWC!
                            let src_off = (ix0 - 1) * c_in;
                            row_dst.copy_from_slice(&in_row[src_off..src_off + 3 * c_in]);
                        } else {
                            for kx in 0..3usize {
                                let ix = ix0 + kx;
                                let tap_dst = &mut row_dst[kx * c_in..(kx + 1) * c_in];
                                if ix == 0 || ix > w {
                                    tap_dst.fill(0.0);
                                } else {
                                    let src_off = (ix - 1) * c_in;
                                    tap_dst.copy_from_slice(&in_row[src_off..src_off + c_in]);
                                }
                            }
                        }
                    }
                }
            }
            gemm_tile(
                token,
                &patch_tile[..mc * k_len],
                mc,
                gemm,
                mode,
                &mut out[m0 * out_c..(m0 + mc) * out_c],
            );
            m0 += mc;
        }
    }

    #[inline(always)]
    fn run_conv_encoder<T: Backend>(
        token: T,
        x: &mut [f32],
        h: usize,
        w: usize,
        enc: &ConvEncoderWeights,
        dw_buf: &mut [f32],
        hidden_tile: &mut [f32],
    ) {
        let npix = h * w;
        let dim_pad = enc.dw.c_pad;
        let dw_slice = &mut dw_buf[..npix * dim_pad];
        depthwise_3x3_s1(token, x, h, w, &enc.dw, dw_slice);
        mlp_tiled_residual(token, dw_slice, npix, &enc.pw1, &enc.pw2, hidden_tile, x);
    }

    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    fn run_stage_attention_block<T: Backend>(
        token: T,
        x: &mut [f32],
        h: usize,
        w: usize,
        blk: &StageAttentionBlock,
        buf_b: &mut [f32],
        buf_c: &mut [f32],
        hidden_tile: &mut [f32],
        attn_a: &mut [f32],
        attn_g: &mut [f32],
    ) {
        let npix = h * w;
        let dim_pad = blk.local_rep.dw.c_pad;
        let nv = dim_pad / 8;

        // 1. LocalRepresentation (1x expansion ConvEncoder)
        run_conv_encoder(token, x, h, w, &blk.local_rep, buf_b, hidden_tile);

        // 2. EfficientAdditiveAttention (zero layout transposes!)
        let q_buf = &mut buf_b[..npix * dim_pad];
        let k_buf = &mut buf_c[..npix * dim_pad];
        let a_buf = &mut attn_a[..npix];
        let g_buf = &mut attn_g[..dim_pad];

        // 2a. Q = L2Normalize(x @ W_q + b_q), and a_raw[n] = Q[n] . (w_g * scale_factor)
        gemm_all(token, x, npix, &blk.to_query, GemmOutMode::Write, q_buf);
        let mut a_norm_sq = 0.0f32;
        for n in 0..npix {
            let q_row = &mut q_buf[n * dim_pad..(n + 1) * dim_pad];
            let mut sum_sq = V::<T>::zero_t(token);
            for cv in 0..nv {
                let v = V::<T>::from_slice_t(token, &q_row[cv * 8..]);
                sum_sq = v.mul_add_relaxed(v, sum_sq);
            }
            let norm = sum_sq.reduce_add().sqrt().max(1e-12);
            let inv_norm = V::<T>::splat_t(token, 1.0 / norm);
            let mut dot_wg = V::<T>::zero_t(token);
            for cv in 0..nv {
                let co = cv * 8;
                let qn = V::<T>::from_slice_t(token, &q_row[co..]) * inv_norm;
                let wg = V::<T>::from_slice_t(token, &blk.w_g_scaled[co..]);
                dot_wg = qn.mul_add_relaxed(wg, dot_wg);
                let dst: &mut [f32; 8] = (&mut q_row[co..co + 8]).try_into().unwrap();
                qn.store(dst);
            }
            let an = dot_wg.reduce_add();
            a_buf[n] = an;
            a_norm_sq += an * an;
        }

        // 2b. a = a_raw / max(||a_raw||_2, 1e-12), g = sum_n (a[n] * Q[n])
        let a_inv_norm = 1.0 / a_norm_sq.sqrt().max(1e-12);
        g_buf.fill(0.0);
        for n in 0..npix {
            let an = V::<T>::splat_t(token, a_buf[n] * a_inv_norm);
            let q_row = &q_buf[n * dim_pad..(n + 1) * dim_pad];
            for cv in 0..nv {
                let co = cv * 8;
                let qv = V::<T>::from_slice_t(token, &q_row[co..]);
                let gv = V::<T>::from_slice_t(token, &g_buf[co..]);
                let res = an.mul_add_relaxed(qv, gv);
                let dst: &mut [f32; 8] = (&mut g_buf[co..co + 8]).try_into().unwrap();
                res.store(dst);
            }
        }

        // 2c. K = L2Normalize(x @ W_k + b_k) * g (in-place fused normalization + context gating)
        gemm_all(token, x, npix, &blk.to_key, GemmOutMode::Write, k_buf);
        for n in 0..npix {
            let k_row = &mut k_buf[n * dim_pad..(n + 1) * dim_pad];
            let mut sum_sq = V::<T>::zero_t(token);
            for cv in 0..nv {
                let v = V::<T>::from_slice_t(token, &k_row[cv * 8..]);
                sum_sq = v.mul_add_relaxed(v, sum_sq);
            }
            let norm = sum_sq.reduce_add().sqrt().max(1e-12);
            let inv_norm = V::<T>::splat_t(token, 1.0 / norm);
            for cv in 0..nv {
                let co = cv * 8;
                let kn = V::<T>::from_slice_t(token, &k_row[co..]) * inv_norm;
                let gv = V::<T>::from_slice_t(token, &g_buf[co..]);
                let kg = kn * gv;
                let dst: &mut [f32; 8] = (&mut k_row[co..co + 8]).try_into().unwrap();
                kg.store(dst);
            }
        }

        // 2d. q_buf += proj(k_buf)  (fused + Q residual in epilogue!)
        gemm_all(
            token,
            k_buf,
            npix,
            &blk.proj,
            GemmOutMode::AddInPlace,
            q_buf,
        );

        // 2e. x += final_proj(q_buf) (with layer_scale_1 folded into final_proj!)
        gemm_all(
            token,
            q_buf,
            npix,
            &blk.final_proj,
            GemmOutMode::AddInPlace,
            x,
        );

        // 3. Mlp (norm1 folded into mlp_fc1, layer_scale_2 folded into mlp_fc2)
        // Copy x into q_buf as MLP input so mlp_tiled_residual can accumulate into x in-place.
        q_buf.copy_from_slice(x);
        mlp_tiled_residual(
            token,
            q_buf,
            npix,
            &blk.mlp_fc1,
            &blk.mlp_fc2,
            hidden_tile,
            x,
        );
    }

    #[inline(always)]
    pub(crate) fn run_forward<T: Backend>(
        token: T,
        model: &SwiftFormerModel,
        input_nchw: &[f32],
        ws: &mut SwiftFormerWorkspace,
        out_logits: &mut [f32; 3],
    ) {
        // Stem 0: [3, 224, 224] -> buf_a [112, 112, 24]
        stem0_nchw_to_nhwc(
            token,
            input_nchw,
            &model.stem0,
            &mut ws.patch_tile,
            &mut ws.buf_a[..112 * 112 * 24],
        );

        // Stem 3: buf_a [112, 112, 24] -> buf_b [56, 56, 48], then copy to buf_a
        let s0_len = 56 * 56 * 48;
        conv3x3_s2_nhwc_gemm(
            token,
            &ws.buf_a[..112 * 112 * 24],
            112,
            112,
            24,
            &model.stem3,
            GemmOutMode::ReLU,
            &mut ws.patch_tile,
            &mut ws.buf_b[..s0_len],
        );
        ws.buf_a[..s0_len].copy_from_slice(&ws.buf_b[..s0_len]);

        // Stages 0..4
        for s in 0..4 {
            let stg = &model.stages[s];
            let npix = stg.in_h * stg.in_w;
            let cur_len = npix * stg.dim_pad;

            if let Some(ref ds) = stg.downsample {
                let prev = &model.stages[s - 1];
                let prev_len = prev.in_h * prev.in_w * prev.dim_pad;
                conv3x3_s2_nhwc_gemm(
                    token,
                    &ws.buf_a[..prev_len],
                    prev.in_h,
                    prev.in_w,
                    prev.dim_pad,
                    ds,
                    GemmOutMode::Write,
                    &mut ws.patch_tile,
                    &mut ws.buf_b[..cur_len],
                );
                ws.buf_a[..cur_len].copy_from_slice(&ws.buf_b[..cur_len]);
            }

            let x = &mut ws.buf_a[..cur_len];
            for enc in &stg.conv_blocks {
                run_conv_encoder(
                    token,
                    x,
                    stg.in_h,
                    stg.in_w,
                    enc,
                    &mut ws.buf_b,
                    &mut ws.hidden_tile,
                );
            }
            run_stage_attention_block(
                token,
                x,
                stg.in_h,
                stg.in_w,
                &stg.attn_block,
                &mut ws.buf_b,
                &mut ws.buf_c,
                &mut ws.hidden_tile,
                &mut ws.attn_a,
                &mut ws.attn_g,
            );
        }

        // Classifier head: spatial sum over 7x7 = 49 tokens, then dot with folded head_w + head_b
        let last_pad = model.stages[3].dim_pad; // 224 for both XS and S
        let nv = last_pad / 8; // 28
        let x_final = &ws.buf_a[..49 * last_pad];
        let sum_buf = &mut ws.attn_g[..last_pad];
        sum_buf.fill(0.0);
        for n in 0..49 {
            let row = &x_final[n * last_pad..(n + 1) * last_pad];
            for cv in 0..nv {
                let co = cv * 8;
                let s = V::<T>::from_slice_t(token, &sum_buf[co..])
                    + V::<T>::from_slice_t(token, &row[co..]);
                let dst: &mut [f32; 8] = (&mut sum_buf[co..co + 8]).try_into().unwrap();
                s.store(dst);
            }
        }

        for cls in 0..3 {
            let mut acc = V::<T>::zero_t(token);
            let hw = &model.head_w[cls];
            for cv in 0..nv {
                let co = cv * 8;
                let sv = V::<T>::from_slice_t(token, &sum_buf[co..]);
                let wv = V::<T>::from_slice_t(token, &hw[co..]);
                acc = sv.mul_add_relaxed(wv, acc);
            }
            out_logits[cls] = model.head_b[cls] + acc.reduce_add();
        }
    }
}

// ---------------------------------------------------------------------------
// Minimal zero-dependency ONNX Protobuf reader
// ---------------------------------------------------------------------------

struct OnnxNode<'a> {
    name: &'a str,
    op_type: &'a str,
    inputs: Vec<&'a str>,
    outputs: Vec<&'a str>,
}

struct OnnxGraph<'a> {
    nodes: Vec<OnnxNode<'a>>,
    initializers: Vec<(&'a str, Vec<f32>)>,
}

impl<'a> OnnxGraph<'a> {
    fn parse(bytes: &'a [u8]) -> Result<Self, SwiftFormerError> {
        let mut pos = 0usize;
        let mut graph_bytes: Option<&'a [u8]> = None;
        while pos < bytes.len() {
            let tag = read_varint(bytes, &mut pos)
                .ok_or_else(|| SwiftFormerError(String::from("truncated ModelProto tag")))?;
            let field_num = tag >> 3;
            let wire_type = tag & 7;
            if field_num == 7 && wire_type == 2 {
                let len = read_varint(bytes, &mut pos)
                    .ok_or_else(|| SwiftFormerError(String::from("truncated GraphProto len")))?
                    as usize;
                let end = pos
                    .checked_add(len)
                    .filter(|&e| e <= bytes.len())
                    .ok_or_else(|| SwiftFormerError(String::from("GraphProto out of bounds")))?;
                graph_bytes = Some(&bytes[pos..end]);
                pos = end;
            } else {
                skip_field(bytes, &mut pos, wire_type)
                    .ok_or_else(|| SwiftFormerError(String::from("invalid ModelProto field")))?;
            }
        }
        let gbytes =
            graph_bytes.ok_or_else(|| SwiftFormerError(String::from("no GraphProto found")))?;

        let mut gpos = 0usize;
        let mut nodes = Vec::new();
        let mut initializers = Vec::new();

        while gpos < gbytes.len() {
            let tag = read_varint(gbytes, &mut gpos)
                .ok_or_else(|| SwiftFormerError(String::from("truncated GraphProto tag")))?;
            let field_num = tag >> 3;
            let wire_type = tag & 7;
            if field_num == 1 && wire_type == 2 {
                let slice = read_len_delimited(gbytes, &mut gpos)?;
                nodes.push(parse_node_proto(slice)?);
            } else if field_num == 5 && wire_type == 2 {
                let slice = read_len_delimited(gbytes, &mut gpos)?;
                initializers.push(parse_tensor_proto(slice)?);
            } else {
                skip_field(gbytes, &mut gpos, wire_type)
                    .ok_or_else(|| SwiftFormerError(String::from("invalid GraphProto field")))?;
            }
        }

        // Resolve Identity aliases (e.g. head.bias -> head_dist.bias)
        for node in &nodes {
            if node.op_type == "Identity"
                && let (Some(&inp), Some(&out)) = (node.inputs.first(), node.outputs.first())
                && initializers.iter().all(|(k, _)| *k != out)
                && let Some((_, vals)) = initializers.iter().find(|(k, _)| *k == inp)
            {
                let cloned = vals.clone();
                initializers.push((out, cloned));
            }
        }

        Ok(Self {
            nodes,
            initializers,
        })
    }

    fn init(&self, name: &str) -> Result<&[f32], SwiftFormerError> {
        self.initializers
            .iter()
            .find(|(k, _)| *k == name)
            .map(|(_, v)| v.as_slice())
            .ok_or_else(|| SwiftFormerError(alloc::format!("missing initializer `{name}`")))
    }

    fn node_init(&self, node_name: &str, input_idx: usize) -> Result<&[f32], SwiftFormerError> {
        let node = self
            .nodes
            .iter()
            .find(|n| n.name == node_name)
            .ok_or_else(|| SwiftFormerError(alloc::format!("missing node `{node_name}`")))?;
        let init_name = node.inputs.get(input_idx).copied().ok_or_else(|| {
            SwiftFormerError(alloc::format!(
                "node `{node_name}` missing input {input_idx}"
            ))
        })?;
        self.init(init_name)
    }

    fn detect_kind(&self) -> Result<SwiftFormerKind, SwiftFormerError> {
        let b1 = self.init("stages.1.blocks.0.pwconv1.bias")?;
        let b3 = self.init("stages.3.blocks.0.pwconv1.bias")?;
        match (b1.len(), b3.len()) {
            (224, 880) => Ok(SwiftFormerKind::Xs),
            (256, 896) => Ok(SwiftFormerKind::S),
            (a, b) => Err(SwiftFormerError(alloc::format!(
                "unsupported SwiftFormer stage hidden widths ({a}, {b})"
            ))),
        }
    }

    fn pack_conv3x3_s2(
        &self,
        node_name: &str,
        in_c: usize,
        in_stride: usize,
        out_c: usize,
    ) -> Result<PackedGemm, SwiftFormerError> {
        let w_oihw = self.node_init(node_name, 1)?;
        let bias = self.node_init(node_name, 2)?;
        if w_oihw.len() != out_c * in_c * 9 || bias.len() != out_c {
            return Err(SwiftFormerError(alloc::format!(
                "conv3x3 `{node_name}` shape mismatch"
            )));
        }
        // Transpose [out_c, in_c, 3, 3] -> [(ky * 3 + kx) * in_stride + ic, out_c]
        let k_len = 9 * in_stride;
        let mut w_ko = vec![0.0f32; k_len * out_c];
        for oc in 0..out_c {
            for ic in 0..in_c {
                for k in 0..9 {
                    let k_idx = k * in_stride + ic;
                    w_ko[k_idx * out_c + oc] = w_oihw[(oc * in_c + ic) * 9 + k];
                }
            }
        }
        Ok(PackedGemm::pack(&w_ko, bias, k_len, k_len, out_c))
    }

    fn pack_conv_encoder(
        &self,
        pfx_node: &str,
        pfx_init: &str,
        dim: usize,
        dim_pad: usize,
        hidden: usize,
    ) -> Result<ConvEncoderWeights, SwiftFormerError> {
        let dw_node = alloc::format!("{pfx_node}/dwconv/Conv");
        let dw_w = self.node_init(&dw_node, 1)?;
        let dw_b = self.node_init(&dw_node, 2)?;
        let dw = PackedDepthwise::pack(dw_w, dw_b, dim);

        let pw1_w_oi = self.init(&alloc::format!("{pfx_init}.pwconv1.weight"))?;
        let pw1_b = self.init(&alloc::format!("{pfx_init}.pwconv1.bias"))?;
        let pw1_w_io = oi_to_io(pw1_w_oi, dim, hidden)?;
        let pw1 = PackedGemm::pack(&pw1_w_io, pw1_b, dim, dim_pad, hidden);

        let pw2_w_oi = self.init(&alloc::format!("{pfx_init}.pwconv2.weight"))?;
        let pw2_b_raw = self.init(&alloc::format!("{pfx_init}.pwconv2.bias"))?;
        let gamma = self.init(&alloc::format!("{pfx_init}.layer_scale.gamma"))?;
        let pw2_w_io = oi_to_io(pw2_w_oi, hidden, dim)?;
        let (pw2_w_folded, pw2_b_folded) =
            fold_output_scale(&pw2_w_io, pw2_b_raw, gamma, hidden, dim)?;
        let hidden_pad = hidden.next_multiple_of(8);
        let pw2 = PackedGemm::pack(&pw2_w_folded, &pw2_b_folded, hidden, hidden_pad, dim);

        Ok(ConvEncoderWeights { dw, pw1, pw2 })
    }
}

fn read_varint(bytes: &[u8], pos: &mut usize) -> Option<u64> {
    let mut result = 0u64;
    let mut shift = 0u32;
    loop {
        if *pos >= bytes.len() || shift >= 64 {
            return None;
        }
        let b = bytes[*pos];
        *pos += 1;
        result |= u64::from(b & 0x7f) << shift;
        if (b & 0x80) == 0 {
            return Some(result);
        }
        shift += 7;
    }
}

fn read_len_delimited<'a>(bytes: &'a [u8], pos: &mut usize) -> Result<&'a [u8], SwiftFormerError> {
    let len = read_varint(bytes, pos)
        .ok_or_else(|| SwiftFormerError(String::from("truncated length varint")))?
        as usize;
    let end = pos
        .checked_add(len)
        .filter(|&e| e <= bytes.len())
        .ok_or_else(|| SwiftFormerError(String::from("length-delimited slice out of bounds")))?;
    let slice = &bytes[*pos..end];
    *pos = end;
    Ok(slice)
}

fn skip_field(bytes: &[u8], pos: &mut usize, wire_type: u64) -> Option<()> {
    match wire_type {
        0 => {
            read_varint(bytes, pos)?;
            Some(())
        }
        1 => {
            *pos = pos.checked_add(8)?;
            (*pos <= bytes.len()).then_some(())
        }
        2 => {
            let len = read_varint(bytes, pos)? as usize;
            *pos = pos.checked_add(len)?;
            (*pos <= bytes.len()).then_some(())
        }
        5 => {
            *pos = pos.checked_add(4)?;
            (*pos <= bytes.len()).then_some(())
        }
        _ => None,
    }
}

fn parse_node_proto<'a>(bytes: &'a [u8]) -> Result<OnnxNode<'a>, SwiftFormerError> {
    let mut pos = 0usize;
    let mut name = "";
    let mut op_type = "";
    let mut inputs = Vec::new();
    let mut outputs = Vec::new();

    while pos < bytes.len() {
        let tag = read_varint(bytes, &mut pos)
            .ok_or_else(|| SwiftFormerError(String::from("truncated NodeProto tag")))?;
        let field_num = tag >> 3;
        let wire_type = tag & 7;
        match (field_num, wire_type) {
            (1, 2) => {
                let s = core::str::from_utf8(read_len_delimited(bytes, &mut pos)?)
                    .map_err(|_| SwiftFormerError(String::from("invalid utf8 in node input")))?;
                inputs.push(s);
            }
            (2, 2) => {
                let s = core::str::from_utf8(read_len_delimited(bytes, &mut pos)?)
                    .map_err(|_| SwiftFormerError(String::from("invalid utf8 in node output")))?;
                outputs.push(s);
            }
            (3, 2) => {
                name = core::str::from_utf8(read_len_delimited(bytes, &mut pos)?)
                    .map_err(|_| SwiftFormerError(String::from("invalid utf8 in node name")))?;
            }
            (4, 2) => {
                op_type = core::str::from_utf8(read_len_delimited(bytes, &mut pos)?)
                    .map_err(|_| SwiftFormerError(String::from("invalid utf8 in node op_type")))?;
            }
            _ => {
                skip_field(bytes, &mut pos, wire_type)
                    .ok_or_else(|| SwiftFormerError(String::from("invalid NodeProto field")))?;
            }
        }
    }

    Ok(OnnxNode {
        name,
        op_type,
        inputs,
        outputs,
    })
}

fn parse_tensor_proto<'a>(bytes: &'a [u8]) -> Result<(&'a str, Vec<f32>), SwiftFormerError> {
    let mut pos = 0usize;
    let mut name = "";
    let mut data_type = 0i32;
    let mut raw_data: Option<&'a [u8]> = None;
    let mut float_data: Vec<f32> = Vec::new();

    while pos < bytes.len() {
        let tag = read_varint(bytes, &mut pos)
            .ok_or_else(|| SwiftFormerError(String::from("truncated TensorProto tag")))?;
        let field_num = tag >> 3;
        let wire_type = tag & 7;
        match (field_num, wire_type) {
            (2, 0) => {
                data_type = read_varint(bytes, &mut pos)
                    .ok_or_else(|| SwiftFormerError(String::from("truncated data_type")))?
                    as i32;
            }
            (4, 5) => {
                if pos + 4 > bytes.len() {
                    return Err(SwiftFormerError(String::from("truncated float_data")));
                }
                let b: [u8; 4] = bytes[pos..pos + 4].try_into().unwrap();
                float_data.push(f32::from_le_bytes(b));
                pos += 4;
            }
            (4, 2) => {
                let slice = read_len_delimited(bytes, &mut pos)?;
                for chunk in slice.as_chunks::<4>().0 {
                    float_data.push(f32::from_le_bytes(*chunk));
                }
            }
            (8, 2) => {
                name = core::str::from_utf8(read_len_delimited(bytes, &mut pos)?)
                    .map_err(|_| SwiftFormerError(String::from("invalid utf8 in tensor name")))?;
            }
            (9, 2) => {
                raw_data = Some(read_len_delimited(bytes, &mut pos)?);
            }
            _ => {
                skip_field(bytes, &mut pos, wire_type)
                    .ok_or_else(|| SwiftFormerError(String::from("invalid TensorProto field")))?;
            }
        }
    }

    let values = if data_type == 1 {
        if let Some(raw) = raw_data {
            raw.as_chunks::<4>()
                .0
                .iter()
                .map(|b| f32::from_le_bytes(*b))
                .collect()
        } else {
            float_data
        }
    } else {
        Vec::new()
    };

    Ok((name, values))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gelu_exact_matches_reference_formula() {
        let mut vals: Vec<f32> = (-60..=60).map(|i| i as f32 * 0.1).collect();
        while !vals.len().is_multiple_of(8) {
            vals.push(0.0);
        }
        let orig = vals.clone();
        k8::gelu_exact_in_place(archmage::ScalarToken, &mut vals);

        for (&x, &got) in orig.iter().zip(vals.iter()) {
            // Reference A&S 7.1.26 exact-ERF GELU
            let z = x * core::f32::consts::FRAC_1_SQRT_2;
            let a = z.abs().min(5.0);
            let mut y = 0.000_043_063_8f32 * a + 0.000_276_567_2;
            y = y * a + 0.000_152_014_3;
            y = y * a + 0.009_270_527;
            y = y * a + 0.042_282_01;
            y = y * a + 0.070_523_08;
            y = y * a + 1.0;
            let y16 = (y * y).powi(8);
            let erf = (1.0 - 1.0 / y16).copysign(z);
            let expected = 0.5 * x * (1.0 + erf);
            let diff = (got - expected).abs();
            assert!(
                diff <= 1e-6,
                "x={x}: got={got}, expected={expected}, diff={diff}"
            );
        }
        // GELU(0.0) must be identically 0.0 so padded channels never leak.
        let mut zero_vec = [0.0f32; 8];
        k8::gelu_exact_in_place(archmage::ScalarToken, &mut zero_vec);
        assert_eq!(zero_vec, [0.0f32; 8]);
    }

    #[test]
    fn gemm_tile_handles_padded_220_to_224_and_tail_rows() {
        // Test m_count = 11 (6 + 4 + 1 rows) with in_c = 20, in_stride = 24, out_c = 20 -> out_c_pad = 24 (16 + 8)
        let in_c = 20usize;
        let in_stride = 24usize;
        let out_c = 20usize;
        let out_c_pad = 24usize;
        let m_count = 11usize;

        let mut w_io = vec![0.0f32; in_c * out_c];
        for ic in 0..in_c {
            for oc in 0..out_c {
                w_io[ic * out_c + oc] = ((ic * 7 + oc * 3) as f32) * 0.01 - 0.5;
            }
        }
        let bias: Vec<f32> = (0..out_c).map(|oc| oc as f32 * 0.05 - 0.2).collect();
        let gemm = PackedGemm::pack(&w_io, &bias, in_c, in_stride, out_c);

        let mut inp = vec![999.0f32; m_count * in_stride]; // non-zero in padded in_c..in_stride slots!
        for m in 0..m_count {
            for ic in 0..in_c {
                inp[m * in_stride + ic] = ((m * 5 + ic) as f32) * 0.02 - 0.3;
            }
        }

        let mut out = vec![-1.0f32; m_count * out_c_pad];
        k8::gemm_tile(
            archmage::ScalarToken,
            &inp,
            m_count,
            &gemm,
            k8::GemmOutMode::Write,
            &mut out,
        );

        for m in 0..m_count {
            for oc in 0..out_c {
                let mut expected = bias[oc];
                for ic in 0..in_c {
                    expected += inp[m * in_stride + ic] * w_io[ic * out_c + oc];
                }
                let got = out[m * out_c_pad + oc];
                assert!(
                    (got - expected).abs() < 1e-5,
                    "m={m}, oc={oc}: got={got}, expected={expected}"
                );
            }
            // Padded output channels 20..24 must be identically 0.0!
            for oc in out_c..out_c_pad {
                assert_eq!(out[m * out_c_pad + oc], 0.0, "padded channel {oc} non-zero");
            }
        }
    }

    #[test]
    fn mlp_tiled_residual_matches_reference_for_padded_hidden_220_to_224() {
        let dim = 220usize;
        let dim_pad = 224usize;
        let hidden = 220usize;
        let hidden_pad = 224usize;
        let npix = 49usize; // 24 + 24 + 1 rows

        let w1: Vec<f32> = (0..dim * hidden)
            .map(|i| ((i % 97) as f32 - 48.0) * 0.005)
            .collect();
        let b1: Vec<f32> = (0..hidden).map(|i| (i as f32 - 110.0) * 0.002).collect();
        let w2: Vec<f32> = (0..hidden * dim)
            .map(|i| ((i % 89) as f32 - 44.0) * 0.004)
            .collect();
        let b2: Vec<f32> = (0..dim).map(|i| (i as f32 - 100.0) * 0.001).collect();

        let pw1 = PackedGemm::pack(&w1, &b1, dim, dim_pad, hidden);
        let pw2 = PackedGemm::pack(&w2, &b2, hidden, hidden_pad, dim);

        let mut inp = vec![0.0f32; npix * dim_pad];
        let mut res = vec![0.0f32; npix * dim_pad];
        for p in 0..npix {
            for c in 0..dim {
                inp[p * dim_pad + c] = ((p * 13 + c * 7) % 101) as f32 * 0.02 - 1.0;
                res[p * dim_pad + c] = ((p * 5 + c * 3) % 73) as f32 * 0.01 - 0.35;
            }
        }
        let orig_res = res.clone();
        let mut hidden_tile = vec![0.0f32; M_TILE * hidden_pad];

        k8::mlp_tiled_residual(
            archmage::ScalarToken,
            &inp,
            npix,
            &pw1,
            &pw2,
            &mut hidden_tile,
            &mut res,
        );

        for p in 0..npix {
            let mut h_ref = vec![0.0f32; hidden_pad];
            for hc in 0..hidden {
                let mut acc = b1[hc];
                for ic in 0..dim {
                    acc += inp[p * dim_pad + ic] * w1[ic * hidden + hc];
                }
                h_ref[hc] = acc;
            }
            k8::gelu_exact_in_place(archmage::ScalarToken, &mut h_ref);
            for oc in 0..dim {
                let mut acc = orig_res[p * dim_pad + oc] + b2[oc];
                for hc in 0..hidden {
                    acc += h_ref[hc] * w2[hc * dim + oc];
                }
                let got = res[p * dim_pad + oc];
                assert!(
                    (got - acc).abs() < 1e-4,
                    "p={p}, oc={oc}: got={got}, expected={acc}"
                );
            }
            for oc in dim..dim_pad {
                assert_eq!(res[p * dim_pad + oc], 0.0);
            }
        }
    }

    #[test]
    fn depthwise_3x3_s1_matches_reference_for_padded_dim_220_to_224() {
        let h = 7usize;
        let w = 7usize;
        let dim = 220usize;
        let dim_pad = 224usize;

        let weights: Vec<f32> = (0..dim * 9)
            .map(|i| ((i % 71) as f32 - 35.0) * 0.01)
            .collect();
        let bias: Vec<f32> = (0..dim).map(|i| (i as f32 - 110.0) * 0.005).collect();
        let dw = PackedDepthwise::pack(&weights, &bias, dim);
        assert_eq!(dw.c_pad, dim_pad);

        let mut inp = vec![0.0f32; h * w * dim_pad];
        for y in 0..h {
            for x in 0..w {
                for c in 0..dim {
                    inp[(y * w + x) * dim_pad + c] =
                        ((y * 17 + x * 11 + c * 3) % 83) as f32 * 0.02 - 0.8;
                }
            }
        }
        let mut out = vec![-1.0f32; h * w * dim_pad];
        k8::depthwise_3x3_s1(archmage::ScalarToken, &inp, h, w, &dw, &mut out);

        for oy in 0..h {
            for ox in 0..w {
                for c in 0..dim {
                    let mut acc = bias[c];
                    for ky in 0..3usize {
                        let iy = oy + ky;
                        if iy == 0 || iy > h {
                            continue;
                        }
                        for kx in 0..3usize {
                            let ix = ox + kx;
                            if ix == 0 || ix > w {
                                continue;
                            }
                            let v = inp[((iy - 1) * w + (ix - 1)) * dim_pad + c];
                            let wv = weights[c * 9 + ky * 3 + kx];
                            acc += v * wv;
                        }
                    }
                    let got = out[(oy * w + ox) * dim_pad + c];
                    assert!(
                        (got - acc).abs() < 1e-5,
                        "oy={oy}, ox={ox}, c={c}: got={got}, expected={acc}"
                    );
                }
                for c in dim..dim_pad {
                    assert_eq!(out[(oy * w + ox) * dim_pad + c], 0.0);
                }
            }
        }
    }
}
