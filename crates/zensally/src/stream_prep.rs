//! Fused streaming downsample + classical structural energy extraction.
//!
//! Single-pass pipeline that
//!
//! 1. resamples a strided source image into a normalized NHWC `f32` tensor
//!    (`target_w * target_h * 3`, e.g. 512×384×3), and
//! 2. simultaneously accumulates the stride-8 classical features consumed by
//!    [`crate::sat_crop`]:
//!
//! * `classical_energy` (`grid_w * grid_h` cells, `grid_w = target_w / 8`):
//!   per-cell mean luminance gradient magnitude (3×3 Sobel on interior
//!   pixels, one-sided 2×2 differences on the border) plus OKLab chroma
//!   dispersion — the RMS distance of each pixel's `a`/`b` chroma from the
//!   cell mean (`E[x²] − E[x]²` accumulated in the same pass).
//! * `horizon_profile` (`grid_h` entries): per-row horizontal edge
//!   continuity `Σx |∂y L| − 0.5·|∂x L|` clamped to ≥ 0 per row, so
//!   landscape crops can align horizons on rule-of-thirds lines.
//!
//! Everything here is `no_std + alloc` compatible: the row kernels use only
//! `core` math (the `*32` helpers below), perform zero heap allocations
//! inside the pass, and operate entirely on caller-owned buffers. The energy
//! grid covers the whole target including letterbox padding (pad pixels are
//! black: L = a = b = 0); remainder pixels past `grid_*8` fold into the last
//! cell of each axis.
//!
//! [`StreamingPrepSink`] adds a row-at-a-time source interface for streaming
//! decoders (zenjpeg strip output): push source rows in order, call
//! [`StreamingPrepSink::finish`], get the same outputs as the one-shot
//! [`preprocess_nhwc_strided`].
//!
//! Callers holding a zenjpeg decode can additionally fold the codec's
//! per-8×8-block DCT AC energy into `classical_energy` via
//! [`fold_jpeg_dct_ac_energy`] — a free high-frequency focus map.

extern crate alloc;

use alloc::vec::Vec;

use crate::PixelFormat;
use crate::preprocess::{LetterboxInfo, Normalization, PreprocessConfig, ResizeMode};

// ---------------------------------------------------------------------------
// no_std-compatible scalar math (core only; no libm dependency)
// ---------------------------------------------------------------------------

/// `f32::sqrt` for `no_std`: bit-hack seed + 3 Newton iterations.
/// Relative error ~1e-7 on positive inputs; `sqrt32(x<=0) == 0`.
#[inline]
pub(crate) fn sqrt32(x: f32) -> f32 {
    if x <= 0.0 {
        return 0.0;
    }
    let seed = f32::from_bits((x.to_bits().wrapping_add(0x3f80_0000)) >> 1);
    let y = 0.5 * (seed + x / seed);
    let y = 0.5 * (y + x / y);
    0.5 * (y + x / y)
}

/// `f32::cbrt` for `no_std`: bit-hack seed + 3 Newton iterations.
/// Signed; relative error ~1e-6 away from zero.
#[inline]
pub(crate) fn cbrt32(x: f32) -> f32 {
    if x == 0.0 {
        return 0.0;
    }
    let ax = x.abs();
    // ix/3 + B seed — the classic cube-root trick (B = 709958130).
    let seed = f32::from_bits(ax.to_bits() / 3 + 709_958_130);
    let y = (2.0 * seed + ax / (seed * seed)) / 3.0;
    let y = (2.0 * y + ax / (y * y)) / 3.0;
    let y = (2.0 * y + ax / (y * y)) / 3.0;
    y.copysign(x)
}

/// `f32::floor` for `no_std` (truncation + fixup). Exact for |x| < 2^62.
#[inline]
pub(crate) fn floor32(x: f32) -> f32 {
    let t = x as i64 as f32;
    if t > x { t - 1.0 } else { t }
}

/// `f32::round` for `no_std` (half away from zero), matching std `round`.
#[inline]
pub(crate) fn round32(x: f32) -> f32 {
    floor32(x.abs() + 0.5).copysign(x)
}

/// Scalar `base.powf(exp)` for `no_std` via `exp2(exp * log2(base))`.
///
/// Ported from `zenfilters::fast_math::fast_powf` (same workspace;
/// `zenpipe/zenfilters/src/fast_math.rs`, ~1% max relative error).
/// `base <= 0` returns 0.0. Used once per prep call to build the sRGB→linear
/// LUT — never on the per-pixel path.
#[inline]
pub(crate) fn powf32(base: f32, exp: f32) -> f32 {
    if base <= 0.0 {
        return 0.0;
    }
    let log2_base = {
        const P0: f32 = -1.850_383_3e-6;
        const P1: f32 = 1.428_716_1;
        const P2: f32 = 0.742_458_7;
        const Q0: f32 = 0.990_328_14;
        const Q1: f32 = 1.009_671_8;
        const Q2: f32 = 0.174_093_43;

        let x_bits = base.to_bits() as i32;
        let offset = 0x3f2a_aaab_u32 as i32;
        let exp_shifted = x_bits.wrapping_sub(offset) >> 23;
        let mantissa = f32::from_bits(x_bits.wrapping_sub(exp_shifted << 23) as u32);
        let m = mantissa - 1.0;
        (P2 * m + P1).mul_add(m, P0) / (Q2 * m + Q1).mul_add(m, Q0) + exp_shifted as f32
    };

    let product = (exp * log2_base).clamp(-126.0, 126.0);
    let xi = floor32(product);
    let xf = product - xi;
    let poly = (0.055_504_11 * xf + 0.240_226_5)
        .mul_add(xf, core::f32::consts::LN_2)
        .mul_add(xf, 1.0);
    poly * f32::from_bits(((xi as i32 + 127) << 23) as u32)
}

// ---------------------------------------------------------------------------
// Scratch layout
// ---------------------------------------------------------------------------

const LUT_OFF: usize = 0;
const LUT_LEN: usize = 256;

/// `f32` scratch elements [`preprocess_nhwc_strided`] and
/// [`StreamingPrepSink`] require: a 256-entry sRGB→linear LUT,
/// `3 * target_w` rolling luminance rows, and `4 * grid_w * grid_h` per-cell
/// chroma accumulators (`Σa`, `Σb`, `Σa²`, `Σb²`).
///
/// 512×384 target (64×48 grid): 256 + 1536 + 12288 ≈ 14k `f32`s ≈ 55 KiB of
/// reusable scratch.
pub fn prep_scratch_len(target_w: usize, grid_w: usize, grid_h: usize) -> usize {
    LUT_LEN + 3 * target_w + 4 * grid_w * grid_h
}

/// Caller-owned output + scratch buffers for the fused pass.
///
/// `classical_energy` and `horizon_profile` are *overwritten* (zeroed then
/// filled); fold additional sources in afterwards with
/// [`fold_jpeg_dct_ac_energy`].
pub struct PrepOutputs<'a> {
    /// Normalized interleaved RGB, `>= target_w * target_h * 3`.
    pub output_nhwc: &'a mut [f32],
    /// Stride-8 classical energy, `>= grid_w * grid_h`.
    pub classical_energy: &'a mut [f32],
    /// Per-grid-row horizontal edge continuity, `>= grid_h`.
    pub horizon_profile: &'a mut [f32],
    /// Work buffer of at least [`prep_scratch_len`] elements.
    pub scratch: &'a mut [f32],
}

// ---------------------------------------------------------------------------
// OKLab (Ottosson 2020) from linear sRGB
// ---------------------------------------------------------------------------

#[inline]
fn lut_linear(lut: &[f32], v255: f32) -> f32 {
    let i = (v255 + 0.5).clamp(0.0, 255.0) as usize;
    lut[i]
}

/// Linear sRGB triplet → OKLab `(L, a, b)`.
#[inline]
fn linear_rgb_to_oklab(lr: f32, lg: f32, lb: f32) -> (f32, f32, f32) {
    let l = 0.412_221_46 * lr + 0.536_332_55 * lg + 0.051_445_995 * lb;
    let m = 0.211_903_5 * lr + 0.680_699_5 * lg + 0.107_396_96 * lb;
    let s = 0.088_302_46 * lr + 0.281_718_85 * lg + 0.629_978_7 * lb;
    let l_ = cbrt32(l);
    let m_ = cbrt32(m);
    let s_ = cbrt32(s);
    (
        0.210_454_26 * l_ + 0.793_617_8 * m_ - 0.004_072_047 * s_,
        1.977_998_5 * l_ - 2.428_592_2 * m_ + 0.450_593_7 * s_,
        0.025_904_037 * l_ + 0.782_771_77 * m_ - 0.808_675_77 * s_,
    )
}

// ---------------------------------------------------------------------------
// Shared single-pass engine (zero heap allocation)
// ---------------------------------------------------------------------------

struct PrepCore<'a> {
    // source geometry
    src_w: usize,
    src_h: usize,
    bpp: usize,
    ri: usize,
    gi: usize,
    bi: usize,
    // target geometry
    tw: usize,
    th: usize,
    rw: usize,
    rh: usize,
    pad_left: usize,
    pad_top: usize,
    ratio: f32,
    norm: Normalization,
    x_ratio: f32,
    y_ratio: f32,
    // stride-8 feature grid
    gw: usize,
    gh: usize,
    ring_off: usize,
    acc_off: usize,
    // caller state
    scratch: &'a mut [f32],
    out: &'a mut [f32],
    energy: &'a mut [f32],
    horizon: &'a mut [f32],
    // progress
    emitted: usize,
    pending: usize, // next luma row awaiting its gradient pass
}

impl<'a> PrepCore<'a> {
    fn new(
        src_w: u32,
        src_h: u32,
        format: PixelFormat,
        config: &PreprocessConfig,
        outputs: PrepOutputs<'a>,
    ) -> Self {
        let PrepOutputs {
            output_nhwc,
            classical_energy,
            horizon_profile,
            scratch,
        } = outputs;

        let tw = config.target_w;
        let th = config.target_h;
        assert!(tw >= 1 && th >= 1, "target dims must be >= 1");
        let gw = tw / 8;
        let gh = th / 8;
        let need = prep_scratch_len(tw, gw, gh);
        assert!(
            output_nhwc.len() >= 3 * tw * th,
            "output_nhwc too small: need {} got {}",
            3 * tw * th,
            output_nhwc.len()
        );
        assert!(
            classical_energy.len() >= gw * gh,
            "classical_energy too small: need {} got {}",
            gw * gh,
            classical_energy.len()
        );
        assert!(
            horizon_profile.len() >= gh,
            "horizon_profile too small: need {gh} got {}",
            horizon_profile.len()
        );
        assert!(
            scratch.len() >= need,
            "scratch too small: need {need} got {}",
            scratch.len()
        );

        let bpp = format.bytes_per_pixel();
        let (ri, gi, bi) = match format {
            PixelFormat::Bgra => (2, 1, 0),
            PixelFormat::Rgba | PixelFormat::Rgb => (0, 1, 2),
        };

        let (rw, rh, pad_left, pad_top, ratio) = match config.mode {
            ResizeMode::Letterbox => {
                let ratio = (tw as f32 / src_w as f32).min(th as f32 / src_h as f32);
                let rw = round32(src_w as f32 * ratio) as usize;
                let rh = round32(src_h as f32 * ratio) as usize;
                (rw, rh, (tw - rw) / 2, (th - rh) / 2, ratio)
            }
            ResizeMode::Stretch => (tw, th, 0, 0, 1.0),
        };
        let x_ratio = if rw > 1 {
            (src_w as f32 - 1.0) / (rw as f32 - 1.0)
        } else {
            0.0
        };
        let y_ratio = if rh > 1 {
            (src_h as f32 - 1.0) / (rh as f32 - 1.0)
        } else {
            0.0
        };

        // sRGB EOTF LUT at u8 resolution (deterministic, built once per call).
        for (i, e) in scratch[LUT_OFF..LUT_OFF + LUT_LEN].iter_mut().enumerate() {
            let u = i as f32 / 255.0;
            *e = if u <= 0.040_45 {
                u / 12.92
            } else {
                powf32((u + 0.055) / 1.055, 2.4)
            };
        }
        // Chroma accumulators start at zero; the luma ring is fully rewritten.
        let acc_off = LUT_OFF + LUT_LEN + 3 * tw;
        for e in &mut scratch[acc_off..acc_off + 4 * gw * gh] {
            *e = 0.0;
        }
        for e in classical_energy.iter_mut() {
            *e = 0.0;
        }
        for e in horizon_profile.iter_mut() {
            *e = 0.0;
        }

        Self {
            src_w: src_w as usize,
            src_h: src_h as usize,
            bpp,
            ri,
            gi,
            bi,
            tw,
            th,
            rw,
            rh,
            pad_left,
            pad_top,
            ratio,
            norm: config.norm,
            x_ratio,
            y_ratio,
            gw,
            gh,
            ring_off: LUT_OFF + LUT_LEN,
            acc_off,
            scratch,
            out: output_nhwc,
            energy: classical_energy,
            horizon: horizon_profile,
            emitted: 0,
            pending: 0,
        }
    }

    fn info(&self) -> LetterboxInfo {
        LetterboxInfo {
            ratio: self.ratio,
            pad_left: self.pad_left as f32,
            pad_top: self.pad_top as f32,
        }
    }

    /// Source row index target row `ty` needs (and whether it needs one at
    /// all — letterbox pad rows never do).
    fn needed_src_row(&self, ty: usize) -> Option<usize> {
        if ty < self.pad_top || ty >= self.pad_top + self.rh || self.rh == 0 {
            return None;
        }
        let dst_y = ty - self.pad_top;
        let y0 = (dst_y as f32 * self.y_ratio) as usize;
        Some((y0 + 1).min(self.src_h.saturating_sub(1)))
    }

    /// Emit target row `ty`. `rows` holds the two source rows the bilinear
    /// kernel interpolates between (each `>= src_w*bpp` bytes); `None` marks
    /// a letterbox pad row (or a missing source row in the streaming path).
    fn emit(&mut self, ty: usize, rows: Option<(&[u8], &[u8])>) {
        let cy = (ty / 8).min(self.gh.saturating_sub(1));
        let acc_n = self.gw * self.gh;

        // Split the scratch into the disjoint regions this row touches.
        let (front, acc) = self.scratch.split_at_mut(self.acc_off);
        let (lut, ring_area) = front.split_at_mut(self.ring_off);
        let lut = &lut[..LUT_LEN];
        let ring_row = &mut ring_area[(ty % 3) * self.tw..(ty % 3 + 1) * self.tw];
        let out_row = &mut self.out[ty * self.tw * 3..(ty + 1) * self.tw * 3];

        let Some((row0, row1)) = rows else {
            // Pad / missing row: normalized zero + black luma. Neutral chroma
            // (a = b = 0) accumulates nothing; cell sample counts live in
            // `finalize`'s fixed cw*ch coverage.
            for px in out_row.iter_mut() {
                *px = 0.0;
            }
            for l in ring_row.iter_mut() {
                *l = 0.0;
            }
            self.after_emit(ty);
            return;
        };

        let dst_y = ty - self.pad_top;
        let yf = dst_y as f32 * self.y_ratio;
        let fy = yf - (yf as usize) as f32;
        let fy_inv = 1.0 - fy;
        let pad_val = 0.0f32; // all current Normalization variants pad at 0

        for (tx, luma) in ring_row.iter_mut().enumerate() {
            let px3 = tx * 3;
            if tx < self.pad_left || tx >= self.pad_left + self.rw {
                out_row[px3] = pad_val;
                out_row[px3 + 1] = pad_val;
                out_row[px3 + 2] = pad_val;
                *luma = 0.0;
                continue;
            }

            let dst_x = tx - self.pad_left;
            let xf = dst_x as f32 * self.x_ratio;
            let x0 = xf as usize;
            let x1 = (x0 + 1).min(self.src_w - 1);
            let fx = xf - x0 as f32;
            let fx_inv = 1.0 - fx;
            let o0 = x0 * self.bpp;
            let o1 = x1 * self.bpp;

            let w00 = fx_inv * fy_inv;
            let w10 = fx * fy_inv;
            let w01 = fx_inv * fy;
            let w11 = fx * fy;

            let r = row0[o0 + self.ri] as f32 * w00
                + row0[o1 + self.ri] as f32 * w10
                + row1[o0 + self.ri] as f32 * w01
                + row1[o1 + self.ri] as f32 * w11;
            let g = row0[o0 + self.gi] as f32 * w00
                + row0[o1 + self.gi] as f32 * w10
                + row1[o0 + self.gi] as f32 * w01
                + row1[o1 + self.gi] as f32 * w11;
            let b = row0[o0 + self.bi] as f32 * w00
                + row0[o1 + self.bi] as f32 * w10
                + row1[o0 + self.bi] as f32 * w01
                + row1[o1 + self.bi] as f32 * w11;

            let (nr, ng, nb) = match self.norm {
                Normalization::CenterScale => (
                    (r - 127.0) / 128.0,
                    (g - 127.0) / 128.0,
                    (b - 127.0) / 128.0,
                ),
                Normalization::UnitScale => (r / 255.0, g / 255.0, b / 255.0),
                Normalization::MeanSubtract {
                    r: mr,
                    g: mg,
                    b: mb,
                } => (r - mr, g - mg, b - mb),
            };
            out_row[px3] = nr;
            out_row[px3 + 1] = ng;
            out_row[px3 + 2] = nb;

            // Un-normalized sRGB → OKLab for luma + chroma features.
            let (l, a, bb) =
                linear_rgb_to_oklab(lut_linear(lut, r), lut_linear(lut, g), lut_linear(lut, b));
            *luma = l;
            if acc_n > 0 {
                let cx = (tx / 8).min(self.gw - 1);
                let c = cy * self.gw + cx;
                acc[c] += a;
                acc[acc_n + c] += b;
                acc[2 * acc_n + c] += a * a;
                acc[3 * acc_n + c] += bb * bb;
            }
        }
        self.after_emit(ty);
    }

    /// Post-emit bookkeeping: every luma row below `ty` now has the
    /// neighbors its gradient pass needs.
    fn after_emit(&mut self, ty: usize) {
        self.emitted = ty + 1;
        while self.pending < ty && self.pending < self.th {
            self.gradient_row(self.pending);
            self.pending += 1;
        }
    }

    /// Gradient magnitude for luma row `r`: 3×3 Sobel on interior rows with
    /// edge-column replication, one-sided differences on the first/last row.
    /// Accumulates `classical_energy` (raw sum; `finalize` normalizes) and
    /// `horizon_profile`.
    fn gradient_row(&mut self, r: usize) {
        if self.gw == 0 || self.gh == 0 {
            return;
        }
        let ring = self.ring_off;
        let row_m =
            &self.scratch[ring + ((r + 2) % 3) * self.tw..ring + ((r + 2) % 3 + 1) * self.tw];
        let row_c = &self.scratch[ring + (r % 3) * self.tw..ring + (r % 3 + 1) * self.tw];
        let row_p =
            &self.scratch[ring + ((r + 1) % 3) * self.tw..ring + ((r + 1) % 3 + 1) * self.tw];
        let has_prev = r > 0;
        let has_next = r + 1 < self.th;
        let cy = (r / 8).min(self.gh - 1);
        let mut row_term = 0.0f32;

        for x in 0..self.tw {
            let xl = x.saturating_sub(1);
            let xr = (x + 1).min(self.tw - 1);
            let (gx, gy) = if has_prev && has_next {
                // 3×3 Sobel, kernel weight sum 8.
                let gx = (row_m[xr] + 2.0 * row_c[xr] + row_p[xr])
                    - (row_m[xl] + 2.0 * row_c[xl] + row_p[xl]);
                let gy = (row_p[xl] + 2.0 * row_p[x] + row_p[xr])
                    - (row_m[xl] + 2.0 * row_m[x] + row_m[xr]);
                (gx / 8.0, gy / 8.0)
            } else {
                // Border row: 2×2 differences. gx is the centered one-sided
                // pair where possible; gy is the available side only.
                let gx = if self.tw < 2 {
                    0.0
                } else if x == 0 {
                    2.0 * (row_c[1] - row_c[0])
                } else if x + 1 == self.tw {
                    2.0 * (row_c[self.tw - 1] - row_c[self.tw - 2])
                } else {
                    row_c[x + 1] - row_c[x - 1]
                };
                let gy = if has_next {
                    row_p[x] - row_c[x]
                } else if has_prev {
                    row_c[x] - row_m[x]
                } else {
                    0.0
                };
                (gx, gy)
            };
            let cx = (x / 8).min(self.gw - 1);
            self.energy[cy * self.gw + cx] += sqrt32(gx * gx + gy * gy);
            row_term += gy.abs() - 0.5 * gx.abs();
        }
        if row_term > 0.0 {
            self.horizon[cy] += row_term;
        }
    }

    /// Final per-cell transform: mean Sobel magnitude + OKLab chroma
    /// dispersion `sqrt(E[a²]−E[a]² + E[b²]−E[b]²)`.
    fn finalize(&mut self) {
        if self.gw == 0 || self.gh == 0 {
            return;
        }
        let n = self.gw * self.gh;
        let base = self.acc_off;
        for cy in 0..self.gh {
            let ch = if cy + 1 == self.gh {
                self.th - 8 * (self.gh - 1)
            } else {
                8
            };
            for cx in 0..self.gw {
                let cw = if cx + 1 == self.gw {
                    self.tw - 8 * (self.gw - 1)
                } else {
                    8
                };
                let cnt = (cw * ch) as f32;
                let c = cy * self.gw + cx;
                let sa = self.scratch[base + c] / cnt;
                let sb = self.scratch[base + n + c] / cnt;
                let sa2 = self.scratch[base + 2 * n + c] / cnt;
                let sb2 = self.scratch[base + 3 * n + c] / cnt;
                let var = (sa2 - sa * sa) + (sb2 - sb * sb);
                self.energy[c] = self.energy[c] / cnt + sqrt32(var.max(0.0));
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Public entry points
// ---------------------------------------------------------------------------

/// Resample `pixels` (`src_w × src_h`, explicit `src_stride` bytes ≥
/// `src_w * bpp`) into a normalized NHWC `f32` tensor while accumulating the
/// stride-8 `classical_energy` and `horizon_profile` maps — one pass, zero
/// heap allocations.
///
/// Returns the [`LetterboxInfo`] needed to map tensor coordinates back to
/// source pixels (trivial for [`ResizeMode::Stretch`]).
pub fn preprocess_nhwc_strided(
    pixels: &[u8],
    src_w: u32,
    src_h: u32,
    src_stride: usize,
    format: PixelFormat,
    config: &PreprocessConfig,
    outputs: &mut PrepOutputs<'_>,
) -> LetterboxInfo {
    let bpp = format.bytes_per_pixel();
    let packed = src_w as usize * bpp;
    assert!(src_w >= 1 && src_h >= 1, "source dims must be >= 1");
    assert!(src_stride >= packed, "src_stride {src_stride} < {packed}");
    assert!(
        pixels.len() >= (src_h as usize - 1) * src_stride + packed,
        "pixel buffer too small for {src_w}x{src_h} @ stride {src_stride}"
    );

    let outputs = PrepOutputs {
        output_nhwc: &mut outputs.output_nhwc[..],
        classical_energy: &mut outputs.classical_energy[..],
        horizon_profile: &mut outputs.horizon_profile[..],
        scratch: &mut outputs.scratch[..],
    };
    let mut core = PrepCore::new(src_w, src_h, format, config, outputs);
    let src_h = src_h as usize;

    for ty in 0..core.th {
        if core.needed_src_row(ty).is_some() {
            let dst_y = ty - core.pad_top;
            let yf = dst_y as f32 * core.y_ratio;
            let y0 = (yf as usize).min(src_h - 1);
            let y1 = (y0 + 1).min(src_h - 1);
            core.emit(
                ty,
                Some((&pixels[y0 * src_stride..], &pixels[y1 * src_stride..])),
            );
        } else {
            core.emit(ty, None);
        }
    }
    core.gradient_row(core.th - 1);
    core.finalize();
    core.info()
}

/// Fold a zenjpeg 8×8-block DCT AC energy grid into `classical_energy`.
///
/// `dct_ac_grid` must hold `grid_w * grid_h` values in the same stride-8
/// cell order as the prep output. Negative entries clamp to zero; the fold
/// is a plain elementwise add — the caller decides units/normalization.
pub fn fold_jpeg_dct_ac_energy(
    dct_ac_grid: &[f32],
    grid_w: usize,
    grid_h: usize,
    classical_energy: &mut [f32],
) {
    let n = grid_w * grid_h;
    assert_eq!(
        dct_ac_grid.len(),
        n,
        "dct_ac_grid len {} != grid {grid_w}x{grid_h}",
        dct_ac_grid.len()
    );
    assert!(
        classical_energy.len() >= n,
        "classical_energy too small: need {n} got {}",
        classical_energy.len()
    );
    for (e, &d) in classical_energy.iter_mut().zip(dct_ac_grid.iter()) {
        *e += d.max(0.0);
    }
}

// ---------------------------------------------------------------------------
// StreamingPrepSink — push source rows, get identical output
// ---------------------------------------------------------------------------

/// Streaming row sink for [`preprocess_nhwc_strided`].
///
/// Push source rows in top-down order via [`push_row`](Self::push_row) (each
/// `>= src_w * bpp` bytes — the sink caches the only two rows a bilinear
/// pass can ever need) and call [`finish`](Self::finish) after the last one.
/// Produces bit-identical output to the one-shot function.
pub struct StreamingPrepSink<'a> {
    core: PrepCore<'a>,
    packed_row: usize,
    cache: Vec<u8>,
    /// Source row index stored in cache slot `i & 1`, or `u32::MAX`.
    tags: [u32; 2],
    pushed: u32,
}

impl<'a> StreamingPrepSink<'a> {
    /// Create a sink for a `src_w × src_h` source delivered row-by-row.
    ///
    /// Allocates the 2-row source cache (`2 * src_w * bpp` bytes) once; all
    /// feature state lives in the caller-owned `outputs`.
    pub fn new(
        src_w: u32,
        src_h: u32,
        src_stride: usize,
        format: PixelFormat,
        config: &PreprocessConfig,
        outputs: PrepOutputs<'a>,
    ) -> Self {
        let packed_row = src_w as usize * format.bytes_per_pixel();
        assert!(src_stride >= packed_row, "src_stride < src_w*bpp");
        Self {
            core: PrepCore::new(src_w, src_h, format, config, outputs),
            packed_row,
            cache: alloc::vec![0u8; 2 * packed_row],
            tags: [u32::MAX; 2],
            pushed: 0,
        }
    }

    /// Push the next source row (leading `src_w * bpp` bytes used).
    pub fn push_row(&mut self, row: &[u8]) {
        assert!(
            row.len() >= self.packed_row,
            "row too short: need {} got {}",
            self.packed_row,
            row.len()
        );
        let r = self.pushed;
        let slot = (r & 1) as usize;
        self.cache[slot * self.packed_row..(slot + 1) * self.packed_row]
            .copy_from_slice(&row[..self.packed_row]);
        self.tags[slot] = r;
        self.pushed += 1;

        while self.core.emitted < self.core.th {
            let ty = self.core.emitted;
            match self.core.needed_src_row(ty) {
                None => self.core.emit(ty, None),
                Some(need) => {
                    if need as u32 > r {
                        break;
                    }
                    self.emit_from_cache(ty);
                }
            }
        }
    }

    fn emit_from_cache(&mut self, ty: usize) {
        let dst_y = ty - self.core.pad_top;
        let yf = dst_y as f32 * self.core.y_ratio;
        let y0 = (yf as usize).min(self.core.src_h - 1);
        let y1 = (y0 + 1).min(self.core.src_h - 1);
        // Disjoint field borrows: rows read `self.cache`, emit mutates `self.core`.
        let (pr, cache, tags) = (self.packed_row, &self.cache, &self.tags);
        let get = |row: usize| -> Option<&[u8]> {
            let slot = row & 1;
            (tags[slot] == row as u32).then(|| &cache[slot * pr..(slot + 1) * pr])
        };
        let rows = get(y0).zip(get(y1));
        self.core.emit(ty, rows);
    }

    /// Emit any rows still waiting on source data (unpushed rows emit as
    /// pad rows), flush the final gradient row, and finalize the grid.
    ///
    /// Call after pushing all `src_h` rows; shorter streams produce pad rows
    /// for the missing tail (documented behavior, not an error).
    pub fn finish(&mut self) -> LetterboxInfo {
        while self.core.emitted < self.core.th {
            let ty = self.core.emitted;
            if self.core.needed_src_row(ty).is_some() {
                self.emit_from_cache(ty);
            } else {
                self.core.emit(ty, None);
            }
        }
        self.core.gradient_row(self.core.th - 1);
        self.core.finalize();
        self.core.info()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg_pixels(w: usize, h: usize, stride: usize, bpp: usize) -> Vec<u8> {
        let mut s = 0x9e37_79b9u32;
        let mut v = alloc::vec![0u8; (h - 1) * stride + w * bpp];
        for i in 0..h {
            for j in 0..w * bpp {
                s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                v[i * stride + j] = (s >> 24) as u8;
            }
        }
        v
    }

    struct Buffers {
        out: Vec<f32>,
        energy: Vec<f32>,
        horizon: Vec<f32>,
        scratch: Vec<f32>,
    }

    fn buffers(tw: usize, th: usize) -> Buffers {
        let (gw, gh) = (tw / 8, th / 8);
        Buffers {
            out: alloc::vec![0.0; 3 * tw * th],
            energy: alloc::vec![f32::NAN; gw * gh],
            horizon: alloc::vec![f32::NAN; gh],
            scratch: alloc::vec![0.0; prep_scratch_len(tw, gw, gh)],
        }
    }

    fn run(
        px: &[u8],
        sw: u32,
        sh: u32,
        stride: usize,
        tw: usize,
        th: usize,
        mode: ResizeMode,
    ) -> (LetterboxInfo, Buffers) {
        let mut b = buffers(tw, th);
        let mut outs = PrepOutputs {
            output_nhwc: &mut b.out,
            classical_energy: &mut b.energy,
            horizon_profile: &mut b.horizon,
            scratch: &mut b.scratch,
        };
        let info = preprocess_nhwc_strided(
            px,
            sw,
            sh,
            stride,
            PixelFormat::Rgb,
            &PreprocessConfig {
                target_w: tw,
                target_h: th,
                mode,
                norm: Normalization::UnitScale,
            },
            &mut outs,
        );
        (info, b)
    }

    #[test]
    fn strided_parity_bit_identical() {
        let (sw, sh, tw, th) = (200usize, 120usize, 64usize, 48usize);
        let packed = lcg_pixels(sw, sh, sw * 3, 3);
        let strided = lcg_pixels(sw, sh, sw * 3 + 37, 3);
        let (_ia, a) = run(
            &packed,
            sw as u32,
            sh as u32,
            sw * 3,
            tw,
            th,
            ResizeMode::Stretch,
        );
        let (_ib, b) = run(
            &strided,
            sw as u32,
            sh as u32,
            sw * 3 + 37,
            tw,
            th,
            ResizeMode::Stretch,
        );
        assert_eq!(a.out, b.out, "NHWC output differs under padding");
        assert_eq!(a.energy, b.energy, "energy differs under padding");
        assert_eq!(a.horizon, b.horizon, "horizon differs under padding");
    }

    #[test]
    fn sink_matches_one_shot_bit_identical() {
        let (sw, sh, tw, th) = (160usize, 100usize, 64usize, 48usize);
        for mode in [ResizeMode::Stretch, ResizeMode::Letterbox] {
            let px = lcg_pixels(sw, sh, sw * 3, 3);
            let cfg = PreprocessConfig {
                target_w: tw,
                target_h: th,
                mode,
                norm: Normalization::UnitScale,
            };
            let (_i1, one) = run(&px, sw as u32, sh as u32, sw * 3, tw, th, mode);

            let mut b = buffers(tw, th);
            let mut sink = StreamingPrepSink::new(
                sw as u32,
                sh as u32,
                sw * 3,
                PixelFormat::Rgb,
                &cfg,
                PrepOutputs {
                    output_nhwc: &mut b.out,
                    classical_energy: &mut b.energy,
                    horizon_profile: &mut b.horizon,
                    scratch: &mut b.scratch,
                },
            );
            for i in 0..sh {
                sink.push_row(&px[i * sw * 3..(i + 1) * sw * 3]);
            }
            sink.finish();
            assert_eq!(one.out, b.out, "NHWC differs ({mode:?})");
            assert_eq!(one.energy, b.energy, "energy differs ({mode:?})");
            assert_eq!(one.horizon, b.horizon, "horizon differs ({mode:?})");
        }
    }

    #[test]
    fn uniform_image_has_zero_energy() {
        // Constant color: no gradients, no chroma dispersion.
        let (sw, sh, tw, th) = (32usize, 24usize, 32usize, 24usize);
        let px = alloc::vec![200u8; sw * sh * 3];
        let (_i, b) = run(
            &px,
            sw as u32,
            sh as u32,
            sw * 3,
            tw,
            th,
            ResizeMode::Stretch,
        );
        for (i, &e) in b.energy.iter().enumerate() {
            assert!(e.abs() < 1e-3, "energy[{i}] = {e} on flat image");
        }
        for (i, &h) in b.horizon.iter().enumerate() {
            assert!(h.abs() < 1e-3, "horizon[{i}] = {h} on flat image");
        }
    }

    #[test]
    fn horizontal_edge_lights_horizon_profile() {
        // Bright sky over dark ground: one strong horizontal luminance step.
        let (sw, sh, tw, th) = (64usize, 48usize, 64usize, 48usize);
        let mut px = alloc::vec![0u8; sw * sh * 3];
        for y in 0..sh {
            for x in 0..sw {
                let v = if y < 24 { 230u8 } else { 20u8 };
                px[(y * sw + x) * 3] = v;
                px[(y * sw + x) * 3 + 1] = v;
                px[(y * sw + x) * 3 + 2] = v;
            }
        }
        let (_i, b) = run(
            &px,
            sw as u32,
            sh as u32,
            sw * 3,
            tw,
            th,
            ResizeMode::Stretch,
        );
        let best = b
            .horizon
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap();
        // The step sits at target row 24 → grid row 3.
        assert_eq!(best, 3, "horizon peak should sit on the step");
        assert!(b.horizon[3] > 4.0 * b.horizon[0].max(1e-3));
    }

    #[test]
    fn fold_dct_energy_adds() {
        let mut e = alloc::vec![1.0f32; 6];
        fold_jpeg_dct_ac_energy(&[0.5, -1.0, 2.0, 0.0, 3.0, 1.0], 3, 2, &mut e);
        assert_eq!(e, alloc::vec![1.5, 1.0, 3.0, 1.0, 4.0, 2.0]);
    }

    #[test]
    fn letterbox_pads_fill_output() {
        // 4:1 source into a square target: content letterboxed, pads zeroed.
        let (sw, sh, tw, th) = (256usize, 64usize, 64usize, 64usize);
        let px = alloc::vec![128u8; sw * sh * 3];
        let (info, b) = run(
            &px,
            sw as u32,
            sh as u32,
            sw * 3,
            tw,
            th,
            ResizeMode::Letterbox,
        );
        assert!(info.pad_top > 0.0);
        for v in &b.out[..tw * 3] {
            assert_eq!(*v, 0.0, "pad row must be zero");
        }
    }

    #[test]
    fn math_helpers_match_std() {
        for i in 0..512u32 {
            let x = (i as f32 + 0.5) * 0.37;
            assert!(
                (sqrt32(x) - x.sqrt()).abs() < 1e-5 * x.sqrt().max(1.0),
                "sqrt {x}"
            );
            let cx = if i % 2 == 0 { x } else { -x };
            assert!(
                (cbrt32(cx) - cx.cbrt()).abs() < 1e-4 * cx.cbrt().abs().max(1.0),
                "cbrt {cx}"
            );
            assert_eq!(floor32(cx), cx.floor(), "floor {cx}");
            assert_eq!(round32(cx), cx.round(), "round {cx}");
        }
        for &base in &[0.01f32, 0.5, 1.0, 2.0, 7.7] {
            for &exp in &[0.5f32, 1.0, 2.4, 3.0] {
                let got = powf32(base, exp);
                let want = base.powf(exp);
                assert!(
                    ((got - want) / want).abs() < 0.02,
                    "powf {base}^{exp}: {got} vs {want}"
                );
            }
        }
    }
}
