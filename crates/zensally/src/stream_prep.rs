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
//! borrowed 8×8 zigzag DCT blocks (`(coeffs: &[i16], blocks_wide, blocks_high)`)
//! into `classical_energy` via [`JpegDctBlocks`] and [`fold_jpeg_dct_blocks`]
//! (or a pre-reduced `f32` grid via [`fold_jpeg_dct_ac_energy`]) — a free
//! high-frequency focus map.

extern crate alloc;

use alloc::vec::Vec;
use archmage::prelude::*;

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
#[cfg(test)]
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
// OKLab (Ottosson 2020) from linear sRGB — SIMD row kernels (archmage + magetypes)
// ---------------------------------------------------------------------------

#[allow(clippy::needless_range_loop)]
mod k8 {
    use crate::PixelFormat;
    use crate::preprocess::Normalization;
    use magetypes::simd::backends::F32x8Convert as Backend;
    use magetypes::simd::generic::{f32x8 as V, i32x8 as I};

    #[inline(always)]
    fn max_x86<T: Backend>(a: V<T>, b: V<T>) -> V<T> {
        V::<T>::blend(a.simd_gt(b), a, b)
    }

    #[inline(always)]
    fn min_x86<T: Backend>(a: V<T>, b: V<T>) -> V<T> {
        V::<T>::blend(a.simd_lt(b), a, b)
    }

    /// Vectorized cube root for non-negative linear-sRGB LMS mixtures (`x >= 0`).
    ///
    /// Computes Kahan's `x.to_bits() / 3 + MAGIC` seed entirely in `i32x8`
    /// registers via 4 shift-add steps (`1/3 = (1/4 + 1/16)(1 + 2^-4)(1 + 2^-8)(1 + 2^-16)`),
    /// followed by 2 Halley iterations using a 1-Newton-step `rcp_approx`
    /// instead of hardware division (`vdivps`). Zero `to_array()` roundtrips,
    /// zero scalar divisions, and zero vector divisions.
    #[inline(always)]
    fn cbrt_pos<T: Backend>(token: T, x: V<T>) -> V<T> {
        const MAGIC: i32 = 0x2a50_8c2d_u32 as i32;
        let zero = V::<T>::zero_t(token);
        let two = V::<T>::splat_t(token, 2.0);
        let x_pos = max_x86(x, V::<T>::splat_t(token, 1.0e-20));

        let u = x_pos.bitcast_to_i32();
        let mut d = u.shr_arithmetic_const::<2>() + u.shr_arithmetic_const::<4>();
        d = d + d.shr_arithmetic_const::<4>();
        d = d + d.shr_arithmetic_const::<8>();
        d = d + d.shr_arithmetic_const::<16>();
        let mut y = V::<T>::from_i32_bitcast_t(token, d + I::<T>::splat_t(token, MAGIC));

        let y3 = y * y * y;
        let num = x_pos.mul_add_relaxed(two, y3);
        let den = y3.mul_add_relaxed(two, x_pos);
        let r0 = den.rcp_approx();
        let r = r0 * (-den).mul_add_relaxed(r0, two);
        y = y * num * r;
        V::<T>::blend(x.simd_le(zero), zero, y)
    }

    #[inline(always)]
    fn linear_rgb_to_oklab_v<T: Backend>(
        token: T,
        lr: V<T>,
        lg: V<T>,
        lb: V<T>,
    ) -> (V<T>, V<T>, V<T>) {
        let sp = |c: f32| V::<T>::splat_t(token, c);
        let l = lr.mul_add_relaxed(
            sp(0.412_221_46),
            lg.mul_add_relaxed(sp(0.536_332_55), lb * sp(0.051_445_995)),
        );
        let m = lr.mul_add_relaxed(
            sp(0.211_903_5),
            lg.mul_add_relaxed(sp(0.680_699_5), lb * sp(0.107_396_96)),
        );
        let s = lr.mul_add_relaxed(
            sp(0.088_302_46),
            lg.mul_add_relaxed(sp(0.281_718_85), lb * sp(0.629_978_7)),
        );
        let l_ = cbrt_pos(token, l);
        let m_ = cbrt_pos(token, m);
        let s_ = cbrt_pos(token, s);
        let l_out = l_.mul_add_relaxed(
            sp(0.210_454_26),
            m_.mul_add_relaxed(sp(0.793_617_8), s_ * sp(-0.004_072_047)),
        );
        let a = l_.mul_add_relaxed(
            sp(1.977_998_5),
            m_.mul_add_relaxed(sp(-2.428_592_2), s_ * sp(0.450_593_7)),
        );
        let bb = l_.mul_add_relaxed(
            sp(0.025_904_037),
            m_.mul_add_relaxed(sp(0.782_771_77), s_ * sp(-0.808_675_77)),
        );
        (l_out, a, bb)
    }

    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn emit_row<T: Backend>(
        token: T,
        row0: &[u8],
        row1: &[u8],
        src_w: usize,
        format: PixelFormat,
        tw: usize,
        rw: usize,
        pad_left: usize,
        x_ratio: f32,
        fy: f32,
        norm: Normalization,
        gw: usize,
        cy: usize,
        acc_n: usize,
        lut: &[f32],
        out_row: &mut [f32],
        ring_row: &mut [f32],
        acc: &mut [f32],
    ) {
        let x_start = pad_left.min(tw);
        let x_end = (pad_left + rw).min(tw);
        out_row[..x_start * 3].fill(0.0);
        ring_row[..x_start].fill(0.0);
        out_row[x_end * 3..tw * 3].fill(0.0);
        ring_row[x_end..tw].fill(0.0);
        if x_start >= x_end {
            return;
        }

        let bpp = format.bytes_per_pixel();
        let (ri, gi, bi) = match format {
            PixelFormat::Bgra => (2usize, 1usize, 0usize),
            PixelFormat::Rgba | PixelFormat::Rgb => (0, 1, 2),
        };
        let row0 = &row0[..src_w * bpp];
        let row1 = &row1[..src_w * bpp];
        let lut256: &[f32; 256] = lut[..256].try_into().unwrap();

        let sp = |c: f32| V::<T>::splat_t(token, c);
        let zero = V::<T>::zero_t(token);
        let one = sp(1.0);
        let half = sp(0.5);
        let max_255 = sp(255.0);
        let inv_255 = sp(1.0 / 255.0);
        let c_127 = sp(127.0);
        let inv_128 = sp(1.0 / 128.0);
        let x_ratio_v = sp(x_ratio);
        let fy_v = sp(fy);
        let fy_inv_v = sp(1.0 - fy);
        let lanes = V::<T>::from_array_t(token, [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
        let src_w_m1 = src_w - 1;

        // Step over 8-aligned cell blocks so all 8 lanes in a block map to the
        // same grid cell `cx = (b8 / 8).min(gw - 1)`.
        let mut b8 = x_start & !7;
        while b8 < x_end {
            let lo = b8.max(x_start);
            let hi = (b8 + 8).min(x_end);
            let full_block = lo == b8 && hi == b8 + 8;

            let dst_xv = if full_block {
                sp((b8 - pad_left) as f32) + lanes
            } else {
                let arr: [f32; 8] =
                    core::array::from_fn(|k| ((b8 + k).clamp(lo, hi - 1) - pad_left) as f32);
                V::<T>::from_array_t(token, arr)
            };
            let xfv = dst_xv * x_ratio_v;
            let x0_i32 = xfv.to_i32();
            let fx = xfv - x0_i32.to_f32();
            let fx_inv = one - fx;
            let w00 = fx_inv * fy_inv_v;
            let w10 = fx * fy_inv_v;
            let w01 = fx_inv * fy_v;
            let w11 = fx * fy_v;
            let x0_arr = x0_i32.to_array();

            let mut r00_a = [0.0f32; 8];
            let mut g00_a = [0.0f32; 8];
            let mut b00_a = [0.0f32; 8];
            let mut r10_a = [0.0f32; 8];
            let mut g10_a = [0.0f32; 8];
            let mut b10_a = [0.0f32; 8];
            let mut r01_a = [0.0f32; 8];
            let mut g01_a = [0.0f32; 8];
            let mut b01_a = [0.0f32; 8];
            let mut r11_a = [0.0f32; 8];
            let mut g11_a = [0.0f32; 8];
            let mut b11_a = [0.0f32; 8];

            if (x0_arr[7] as usize) + 1 < src_w && x0_arr[0] >= 0 {
                if bpp == 3 {
                    for k in 0..8 {
                        let o0 = (x0_arr[k] as usize) * 3;
                        let p0 = &row0[o0..o0 + 6];
                        let p1 = &row1[o0..o0 + 6];
                        r00_a[k] = p0[0] as f32;
                        g00_a[k] = p0[1] as f32;
                        b00_a[k] = p0[2] as f32;
                        r10_a[k] = p0[3] as f32;
                        g10_a[k] = p0[4] as f32;
                        b10_a[k] = p0[5] as f32;
                        r01_a[k] = p1[0] as f32;
                        g01_a[k] = p1[1] as f32;
                        b01_a[k] = p1[2] as f32;
                        r11_a[k] = p1[3] as f32;
                        g11_a[k] = p1[4] as f32;
                        b11_a[k] = p1[5] as f32;
                    }
                } else {
                    for k in 0..8 {
                        let o0 = (x0_arr[k] as usize) * 4;
                        let p0 = &row0[o0..o0 + 8];
                        let p1 = &row1[o0..o0 + 8];
                        r00_a[k] = p0[ri] as f32;
                        g00_a[k] = p0[gi] as f32;
                        b00_a[k] = p0[bi] as f32;
                        r10_a[k] = p0[4 + ri] as f32;
                        g10_a[k] = p0[4 + gi] as f32;
                        b10_a[k] = p0[4 + bi] as f32;
                        r01_a[k] = p1[ri] as f32;
                        g01_a[k] = p1[gi] as f32;
                        b01_a[k] = p1[bi] as f32;
                        r11_a[k] = p1[4 + ri] as f32;
                        g11_a[k] = p1[4 + gi] as f32;
                        b11_a[k] = p1[4 + bi] as f32;
                    }
                }
            } else {
                for k in 0..8 {
                    let x0 = (x0_arr[k].max(0) as usize).min(src_w_m1);
                    let x1 = (x0 + 1).min(src_w_m1);
                    let o0 = x0 * bpp;
                    let o1 = x1 * bpp;
                    let p00 = &row0[o0..o0 + bpp];
                    let p10 = &row0[o1..o1 + bpp];
                    let p01 = &row1[o0..o0 + bpp];
                    let p11 = &row1[o1..o1 + bpp];
                    r00_a[k] = p00[ri] as f32;
                    g00_a[k] = p00[gi] as f32;
                    b00_a[k] = p00[bi] as f32;
                    r10_a[k] = p10[ri] as f32;
                    g10_a[k] = p10[gi] as f32;
                    b10_a[k] = p10[bi] as f32;
                    r01_a[k] = p01[ri] as f32;
                    g01_a[k] = p01[gi] as f32;
                    b01_a[k] = p01[bi] as f32;
                    r11_a[k] = p11[ri] as f32;
                    g11_a[k] = p11[gi] as f32;
                    b11_a[k] = p11[bi] as f32;
                }
            }

            let r = V::<T>::from_array_t(token, r00_a).mul_add_relaxed(
                w00,
                V::<T>::from_array_t(token, r10_a).mul_add_relaxed(
                    w10,
                    V::<T>::from_array_t(token, r01_a)
                        .mul_add_relaxed(w01, V::<T>::from_array_t(token, r11_a) * w11),
                ),
            );
            let g = V::<T>::from_array_t(token, g00_a).mul_add_relaxed(
                w00,
                V::<T>::from_array_t(token, g10_a).mul_add_relaxed(
                    w10,
                    V::<T>::from_array_t(token, g01_a)
                        .mul_add_relaxed(w01, V::<T>::from_array_t(token, g11_a) * w11),
                ),
            );
            let b = V::<T>::from_array_t(token, b00_a).mul_add_relaxed(
                w00,
                V::<T>::from_array_t(token, b10_a).mul_add_relaxed(
                    w10,
                    V::<T>::from_array_t(token, b01_a)
                        .mul_add_relaxed(w01, V::<T>::from_array_t(token, b11_a) * w11),
                ),
            );

            let (nr, ng, nb) = match norm {
                Normalization::CenterScale => (
                    (r - c_127) * inv_128,
                    (g - c_127) * inv_128,
                    (b - c_127) * inv_128,
                ),
                Normalization::UnitScale => (r * inv_255, g * inv_255, b * inv_255),
                Normalization::MeanSubtract {
                    r: mr,
                    g: mg,
                    b: mb,
                } => (r - sp(mr), g - sp(mg), b - sp(mb)),
            };

            // LUT lookup via scalar array indexing (faster than microcoded vgatherdps).
            let ri_idx = min_x86(max_x86(r + half, zero), max_255)
                .to_i32()
                .to_array();
            let gi_idx = min_x86(max_x86(g + half, zero), max_255)
                .to_i32()
                .to_array();
            let bi_idx = min_x86(max_x86(b + half, zero), max_255)
                .to_i32()
                .to_array();
            let lr = V::<T>::from_array_t(
                token,
                core::array::from_fn(|k| lut256[(ri_idx[k] as usize) & 255]),
            );
            let lg = V::<T>::from_array_t(
                token,
                core::array::from_fn(|k| lut256[(gi_idx[k] as usize) & 255]),
            );
            let lb = V::<T>::from_array_t(
                token,
                core::array::from_fn(|k| lut256[(bi_idx[k] as usize) & 255]),
            );

            let (l_out, a, bb) = linear_rgb_to_oklab_v(token, lr, lg, lb);

            if full_block {
                let nr_a = nr.to_array();
                let ng_a = ng.to_array();
                let nb_a = nb.to_array();
                let dst = &mut out_row[b8 * 3..b8 * 3 + 24];
                for k in 0..8 {
                    dst[k * 3] = nr_a[k];
                    dst[k * 3 + 1] = ng_a[k];
                    dst[k * 3 + 2] = nb_a[k];
                }
                ring_row[b8..b8 + 8].copy_from_slice(&l_out.to_array());
                if acc_n > 0 {
                    let cx = (b8 / 8).min(gw - 1);
                    let c = cy * gw + cx;
                    let a2 = a * a;
                    let b2 = bb * bb;
                    acc[c] += a.reduce_add();
                    acc[acc_n + c] += bb.reduce_add();
                    acc[2 * acc_n + c] += a2.reduce_add();
                    acc[3 * acc_n + c] += b2.reduce_add();
                }
            } else {
                let nr_a = nr.to_array();
                let ng_a = ng.to_array();
                let nb_a = nb.to_array();
                let l_a = l_out.to_array();
                let a_a = a.to_array();
                let b_a = bb.to_array();
                for tx in lo..hi {
                    let k = tx - b8;
                    let px3 = tx * 3;
                    out_row[px3] = nr_a[k];
                    out_row[px3 + 1] = ng_a[k];
                    out_row[px3 + 2] = nb_a[k];
                    ring_row[tx] = l_a[k];
                    if acc_n > 0 {
                        let cx = (tx / 8).min(gw - 1);
                        let c = cy * gw + cx;
                        let av = a_a[k];
                        let bv = b_a[k];
                        acc[c] += av;
                        acc[acc_n + c] += bv;
                        acc[2 * acc_n + c] += av * av;
                        acc[3 * acc_n + c] += bv * bv;
                    }
                }
            }

            b8 += 8;
        }
    }

    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn gradient_row<T: Backend>(
        token: T,
        row_m: &[f32],
        row_c: &[f32],
        row_p: &[f32],
        tw: usize,
        gw: usize,
        has_prev: bool,
        has_next: bool,
        energy_row: &mut [f32],
    ) -> f32 {
        let sp = |c: f32| V::<T>::splat_t(token, c);
        let two = sp(2.0);
        let eighth = sp(0.125);
        let half = sp(0.5);
        let mut row_term = 0.0f32;

        if has_prev && has_next && tw >= 16 {
            let mut row_term_vec = V::<T>::zero_t(token);
            // Interior 8-pixel chunks where [x - 1 .. x + 9] lies inside [0 .. tw].
            let full_cells = (tw - 1) / 8;
            for cx in 1..full_cells {
                let x = cx * 8;
                let rm_l = V::<T>::from_slice_t(token, &row_m[x - 1..]);
                let rm_c = V::<T>::from_slice_t(token, &row_m[x..]);
                let rm_r = V::<T>::from_slice_t(token, &row_m[x + 1..]);

                let rc_l = V::<T>::from_slice_t(token, &row_c[x - 1..]);
                let rc_r = V::<T>::from_slice_t(token, &row_c[x + 1..]);

                let rp_l = V::<T>::from_slice_t(token, &row_p[x - 1..]);
                let rp_c = V::<T>::from_slice_t(token, &row_p[x..]);
                let rp_r = V::<T>::from_slice_t(token, &row_p[x + 1..]);

                let col_r = rc_r.mul_add_relaxed(two, rm_r + rp_r);
                let col_l = rc_l.mul_add_relaxed(two, rm_l + rp_l);
                let gx = (col_r - col_l) * eighth;

                let row_bot = rp_c.mul_add_relaxed(two, rp_l + rp_r);
                let row_top = rm_c.mul_add_relaxed(two, rm_l + rm_r);
                let gy = (row_bot - row_top) * eighth;

                let mag = gx.mul_add_relaxed(gx, gy * gy).sqrt();
                energy_row[cx.min(gw - 1)] += mag.reduce_add();
                row_term_vec = (gy.abs() - gx.abs() * half) + row_term_vec;
            }
            row_term += row_term_vec.reduce_add();

            // Left boundary cell (x in 0..8) and right tail (x in full_cells * 8 .. tw).
            for range in [0..8, (full_cells * 8)..tw] {
                for x in range {
                    let xl = x.saturating_sub(1);
                    let xr = (x + 1).min(tw - 1);
                    let gx = ((row_m[xr] + 2.0 * row_c[xr] + row_p[xr])
                        - (row_m[xl] + 2.0 * row_c[xl] + row_p[xl]))
                        * 0.125;
                    let gy = ((row_p[xl] + 2.0 * row_p[x] + row_p[xr])
                        - (row_m[xl] + 2.0 * row_m[x] + row_m[xr]))
                        * 0.125;
                    let cx = (x / 8).min(gw - 1);
                    energy_row[cx] += super::sqrt32(gx * gx + gy * gy);
                    row_term += gy.abs() - 0.5 * gx.abs();
                }
            }
        } else {
            for x in 0..tw {
                let xl = x.saturating_sub(1);
                let xr = (x + 1).min(tw - 1);
                let (gx, gy) = if has_prev && has_next {
                    let gx = (row_m[xr] + 2.0 * row_c[xr] + row_p[xr])
                        - (row_m[xl] + 2.0 * row_c[xl] + row_p[xl]);
                    let gy = (row_p[xl] + 2.0 * row_p[x] + row_p[xr])
                        - (row_m[xl] + 2.0 * row_m[x] + row_m[xr]);
                    (gx * 0.125, gy * 0.125)
                } else {
                    let gx = if tw < 2 {
                        0.0
                    } else if x == 0 {
                        2.0 * (row_c[1] - row_c[0])
                    } else if x + 1 == tw {
                        2.0 * (row_c[tw - 1] - row_c[tw - 2])
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
                let cx = (x / 8).min(gw - 1);
                energy_row[cx] += super::sqrt32(gx * gx + gy * gy);
                row_term += gy.abs() - 0.5 * gx.abs();
            }
        }
        row_term
    }
}

#[magetypes(v3, neon, wasm128, scalar)]
#[allow(clippy::too_many_arguments)]
fn emit_row_impl(
    token: Token,
    row0: &[u8],
    row1: &[u8],
    src_w: usize,
    format: PixelFormat,
    tw: usize,
    rw: usize,
    pad_left: usize,
    x_ratio: f32,
    fy: f32,
    norm: Normalization,
    gw: usize,
    cy: usize,
    acc_n: usize,
    lut: &[f32],
    out_row: &mut [f32],
    ring_row: &mut [f32],
    acc: &mut [f32],
) {
    k8::emit_row(
        token, row0, row1, src_w, format, tw, rw, pad_left, x_ratio, fy, norm, gw, cy, acc_n, lut,
        out_row, ring_row, acc,
    );
}

#[magetypes(v3, neon, wasm128, scalar)]
#[allow(clippy::too_many_arguments)]
fn gradient_row_impl(
    token: Token,
    row_m: &[f32],
    row_c: &[f32],
    row_p: &[f32],
    tw: usize,
    gw: usize,
    has_prev: bool,
    has_next: bool,
    energy_row: &mut [f32],
) -> f32 {
    k8::gradient_row(
        token, row_m, row_c, row_p, tw, gw, has_prev, has_next, energy_row,
    )
}

// ---------------------------------------------------------------------------
// Shared single-pass engine (zero heap allocation)
// ---------------------------------------------------------------------------

struct PrepCore<'a> {
    // source geometry
    src_w: usize,
    src_h: usize,
    format: PixelFormat,
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
            format,
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
            out_row.fill(0.0);
            ring_row.fill(0.0);
            self.after_emit(ty);
            return;
        };

        let dst_y = ty - self.pad_top;
        let yf = dst_y as f32 * self.y_ratio;
        let fy = yf - (yf as usize) as f32;

        incant!(
            emit_row_impl(
                row0,
                row1,
                self.src_w,
                self.format,
                self.tw,
                self.rw,
                self.pad_left,
                self.x_ratio,
                fy,
                self.norm,
                self.gw,
                cy,
                acc_n,
                lut,
                out_row,
                ring_row,
                acc
            ),
            [v3, neon, wasm128, scalar]
        );
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
        let energy_row = &mut self.energy[cy * self.gw..(cy + 1) * self.gw];

        let row_term = incant!(
            gradient_row_impl(
                row_m, row_c, row_p, self.tw, self.gw, has_prev, has_next, energy_row
            ),
            [v3, neon, wasm128, scalar]
        );
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

/// Borrowed 8×8 JPEG DCT coefficient grid in standard 64-element zigzag order.
///
/// Matches the zero-copy `(coeffs: &[i16], blocks_wide, blocks_high)` layout
/// exposed by `zenjpeg` (`ComponentCoefficients`) and consumed by `zensr`
/// (`zensr::source::Component`), so a single decoded coefficient plane can
/// feed both restoration and smart-crop AC focus folding without copying or
/// reordering coefficients.
#[derive(Clone, Copy, Debug)]
pub struct JpegDctBlocks<'a> {
    /// Quantized (or dequantized) 8×8 block coefficients in JPEG zigzag order:
    /// `blocks_wide * blocks_high * 64` `i16` elements, row-major by block.
    /// Within each 64-element block slice `b`, `b[0]` is DC and `b[1..64]` are
    /// the 63 AC coefficients in zigzag order.
    pub coeffs: &'a [i16],
    /// Number of 8×8 blocks horizontally.
    pub blocks_wide: usize,
    /// Number of 8×8 blocks vertically.
    pub blocks_high: usize,
    /// Optional 64-element quantization table (in the same zigzag order as
    /// `coeffs`) used to weight each AC coefficient by its quantizer step `Q[k]`
    /// before summing energy. When `None`, raw quantized `|c[k]|` values are used.
    pub quant_table: Option<&'a [u16; 64]>,
}

impl<'a> JpegDctBlocks<'a> {
    /// Construct a borrowed JPEG 8×8 zigzag DCT block grid, validating that
    /// `coeffs.len() >= blocks_wide * blocks_high * 64`.
    pub fn new(
        coeffs: &'a [i16],
        blocks_wide: usize,
        blocks_high: usize,
    ) -> Result<Self, &'static str> {
        let needed = blocks_wide
            .checked_mul(blocks_high)
            .and_then(|n| n.checked_mul(64))
            .ok_or("JPEG block grid dimensions overflow")?;
        if coeffs.len() < needed {
            return Err("JPEG DCT coefficient slice too small for blocks_wide * blocks_high * 64");
        }
        Ok(Self {
            coeffs,
            blocks_wide,
            blocks_high,
            quant_table: None,
        })
    }

    /// Attach a 64-element zigzag quantization table so AC energy is computed
    /// in dequantized coefficient units (`|c[k]| * q[k]`).
    #[must_use]
    pub fn with_quant_table(mut self, quant_table: &'a [u16; 64]) -> Self {
        self.quant_table = Some(quant_table);
        self
    }
}

/// Fold a borrowed `zenjpeg` 8×8 zigzag DCT coefficient grid ([`JpegDctBlocks`])
/// into `classical_energy` (`grid_w × grid_h` cells) scaled by `weight`.
///
/// For each destination cell `(gx, gy)` in `0..grid_w × 0..grid_h`, averages
/// the per-block L1 AC energy over the corresponding source block window
/// `[bx0..bx1) × [by0..by1)` in `0..blocks_wide × 0..blocks_high` with zero
/// heap allocations.
///
/// Within each 64-element zigzag block `b`:
/// - When `quant_table` is `Some(q)`, per-block AC energy is
///   `(1 / (63 * 255)) * Σ_{k=1..63} |b[k]| * q[k]`.
/// - When `quant_table` is `None`, per-block AC energy is
///   `(1 / (63 * 32)) * Σ_{k=1..63} |b[k]|`.
pub fn fold_jpeg_dct_blocks(
    blocks: &JpegDctBlocks<'_>,
    grid_w: usize,
    grid_h: usize,
    weight: f32,
    classical_energy: &mut [f32],
) {
    let n = grid_w * grid_h;
    assert!(
        classical_energy.len() >= n,
        "classical_energy too small: need {n} got {}",
        classical_energy.len()
    );
    let bw = blocks.blocks_wide;
    let bh = blocks.blocks_high;
    if bw == 0 || bh == 0 || grid_w == 0 || grid_h == 0 || weight <= 0.0 {
        return;
    }
    assert!(
        blocks.coeffs.len() >= bw * bh * 64,
        "JpegDctBlocks coeffs len {} < {} * {} * 64",
        blocks.coeffs.len(),
        bw,
        bh
    );

    let norm_scale = if blocks.quant_table.is_some() {
        weight * (1.0 / (63.0 * 255.0))
    } else {
        weight * (1.0 / (63.0 * 32.0))
    };

    for gy in 0..grid_h {
        let by0 = (gy * bh) / grid_h;
        let by1 = (((gy + 1) * bh).div_ceil(grid_h)).max(by0 + 1).min(bh);
        for gx in 0..grid_w {
            let bx0 = (gx * bw) / grid_w;
            let bx1 = (((gx + 1) * bw).div_ceil(grid_w)).max(bx0 + 1).min(bw);
            let mut sum = 0.0f32;
            let mut count = 0u32;
            for by in by0..by1 {
                for bx in bx0..bx1 {
                    let b_off = (by * bw + bx) * 64;
                    let blk = &blocks.coeffs[b_off..b_off + 64];
                    let blk_ac: u32 = if let Some(qt) = blocks.quant_table {
                        let mut s = 0u32;
                        for (&c, &q) in blk[1..64].iter().zip(qt[1..64].iter()) {
                            s += (c.unsigned_abs() as u32) * (q as u32);
                        }
                        s
                    } else {
                        let mut s = 0u32;
                        for &c in &blk[1..64] {
                            s += c.unsigned_abs() as u32;
                        }
                        s
                    };
                    sum += blk_ac as f32;
                    count += 1;
                }
            }
            if count > 0 {
                classical_energy[gy * grid_w + gx] += (sum / count as f32) * norm_scale;
            }
        }
    }
}

/// Fold a pre-reduced `grid_w × grid_h` `f32` DCT AC energy grid into
/// `classical_energy`.
///
/// `dct_ac_grid` must hold `grid_w * grid_h` values in the same stride-8
/// cell order as the prep output. Negative entries clamp to zero; the fold
/// is a plain elementwise add — the caller decides units/normalization.
/// For zero-copy borrowed `zenjpeg` `(coeffs: &[i16], blocks_wide, blocks_high)`
/// zigzag block grids at arbitrary source dimensions, use [`fold_jpeg_dct_blocks`].
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

    #[test]
    fn simd_matches_scalar_within_tolerance() {
        let (sw, sh, tw, th) = (173usize, 113usize, 64usize, 48usize);
        for format in [PixelFormat::Rgb, PixelFormat::Rgba, PixelFormat::Bgra] {
            let bpp = format.bytes_per_pixel();
            let px = lcg_pixels(sw, sh, sw * bpp, bpp);
            for mode in [ResizeMode::Stretch, ResizeMode::Letterbox] {
                let cfg = PreprocessConfig {
                    target_w: tw,
                    target_h: th,
                    mode,
                    norm: Normalization::UnitScale,
                };
                let mut b_simd = buffers(tw, th);
                preprocess_nhwc_strided(
                    &px,
                    sw as u32,
                    sh as u32,
                    sw * bpp,
                    format,
                    &cfg,
                    &mut PrepOutputs {
                        output_nhwc: &mut b_simd.out,
                        classical_energy: &mut b_simd.energy,
                        horizon_profile: &mut b_simd.horizon,
                        scratch: &mut b_simd.scratch,
                    },
                );

                // Run the ScalarToken path explicitly row by row.
                let mut b_scalar = buffers(tw, th);
                let mut core = PrepCore::new(
                    sw as u32,
                    sh as u32,
                    format,
                    &cfg,
                    PrepOutputs {
                        output_nhwc: &mut b_scalar.out,
                        classical_energy: &mut b_scalar.energy,
                        horizon_profile: &mut b_scalar.horizon,
                        scratch: &mut b_scalar.scratch,
                    },
                );
                let tok = archmage::ScalarToken;
                for ty in 0..core.th {
                    if core.needed_src_row(ty).is_some() {
                        let dst_y = ty - core.pad_top;
                        let yf = dst_y as f32 * core.y_ratio;
                        let y0 = (yf as usize).min(sh - 1);
                        let y1 = (y0 + 1).min(sh - 1);
                        let fy = yf - (yf as usize) as f32;
                        let cy = (ty / 8).min(core.gh.saturating_sub(1));
                        let acc_n = core.gw * core.gh;
                        let (front, acc) = core.scratch.split_at_mut(core.acc_off);
                        let (lut, ring_area) = front.split_at_mut(core.ring_off);
                        let lut = &lut[..LUT_LEN];
                        let ring_row = &mut ring_area[(ty % 3) * core.tw..(ty % 3 + 1) * core.tw];
                        let out_row = &mut core.out[ty * core.tw * 3..(ty + 1) * core.tw * 3];
                        emit_row_impl_scalar(
                            tok,
                            &px[y0 * sw * bpp..],
                            &px[y1 * sw * bpp..],
                            core.src_w,
                            core.format,
                            core.tw,
                            core.rw,
                            core.pad_left,
                            core.x_ratio,
                            fy,
                            core.norm,
                            core.gw,
                            cy,
                            acc_n,
                            lut,
                            out_row,
                            ring_row,
                            acc,
                        );
                    } else {
                        let ring_row = &mut core.scratch[core.ring_off + (ty % 3) * core.tw
                            ..core.ring_off + (ty % 3 + 1) * core.tw];
                        let out_row = &mut core.out[ty * core.tw * 3..(ty + 1) * core.tw * 3];
                        out_row.fill(0.0);
                        ring_row.fill(0.0);
                    }
                    core.emitted = ty + 1;
                    while core.pending < ty && core.pending < core.th {
                        let r = core.pending;
                        let ring = core.ring_off;
                        let row_m = &core.scratch
                            [ring + ((r + 2) % 3) * core.tw..ring + ((r + 2) % 3 + 1) * core.tw];
                        let row_c =
                            &core.scratch[ring + (r % 3) * core.tw..ring + (r % 3 + 1) * core.tw];
                        let row_p = &core.scratch
                            [ring + ((r + 1) % 3) * core.tw..ring + ((r + 1) % 3 + 1) * core.tw];
                        let cy = (r / 8).min(core.gh - 1);
                        let energy_row = &mut core.energy[cy * core.gw..(cy + 1) * core.gw];
                        let rt = gradient_row_impl_scalar(
                            tok,
                            row_m,
                            row_c,
                            row_p,
                            core.tw,
                            core.gw,
                            r > 0,
                            r + 1 < core.th,
                            energy_row,
                        );
                        if rt > 0.0 {
                            core.horizon[cy] += rt;
                        }
                        core.pending += 1;
                    }
                }
                {
                    let r = core.th - 1;
                    let ring = core.ring_off;
                    let row_m = &core.scratch
                        [ring + ((r + 2) % 3) * core.tw..ring + ((r + 2) % 3 + 1) * core.tw];
                    let row_c =
                        &core.scratch[ring + (r % 3) * core.tw..ring + (r % 3 + 1) * core.tw];
                    let row_p = &core.scratch
                        [ring + ((r + 1) % 3) * core.tw..ring + ((r + 1) % 3 + 1) * core.tw];
                    let cy = (r / 8).min(core.gh - 1);
                    let energy_row = &mut core.energy[cy * core.gw..(cy + 1) * core.gw];
                    let rt = gradient_row_impl_scalar(
                        tok,
                        row_m,
                        row_c,
                        row_p,
                        core.tw,
                        core.gw,
                        r > 0,
                        false,
                        energy_row,
                    );
                    if rt > 0.0 {
                        core.horizon[cy] += rt;
                    }
                }
                core.finalize();

                for (a, b) in b_simd.out.iter().zip(b_scalar.out.iter()) {
                    assert!((a - b).abs() < 1e-5, "out diff: {a} vs {b}");
                }
                for (a, b) in b_simd.energy.iter().zip(b_scalar.energy.iter()) {
                    assert!((a - b).abs() < 1e-4, "energy diff: {a} vs {b}");
                }
                for (a, b) in b_simd.horizon.iter().zip(b_scalar.horizon.iter()) {
                    assert!((a - b).abs() < 5e-3, "horizon diff: {a} vs {b}");
                }
            }
        }
    }

    #[test]
    fn fold_jpeg_dct_blocks_zigzag_with_and_without_qtable() {
        // 4x4 block grid folded into a 2x2 classical_energy grid (each cell averages 2x2 blocks).
        let bw = 4usize;
        let bh = 4usize;
        let mut coeffs = vec![0i16; bw * bh * 64];
        // Put DC = 500 everywhere (must be ignored by AC fold!) and high AC in top-left 2x2 blocks.
        for by in 0..bh {
            for bx in 0..bw {
                let off = (by * bw + bx) * 64;
                coeffs[off] = 500; // DC coefficient (index 0)
                if bx < 2 && by < 2 {
                    for k in 1..64 {
                        coeffs[off + k] = if k % 2 == 0 { 16 } else { -16 };
                    }
                }
            }
        }
        let blocks = JpegDctBlocks::new(&coeffs, bw, bh).unwrap();
        let mut energy = vec![0.10f32; 4];
        fold_jpeg_dct_blocks(&blocks, 2, 2, 1.0, &mut energy);

        // Top-left cell (0,0) has |c[k]| = 16 for all 63 AC coeffs => 16/32 = 0.5 added to 0.10.
        assert!((energy[0] - 0.60).abs() < 1e-5, "energy[0] = {}", energy[0]);
        // Other 3 cells have 0 AC energy (DC=500 ignored) => stay at 0.10.
        assert!((energy[1] - 0.10).abs() < 1e-6);
        assert!((energy[2] - 0.10).abs() < 1e-6);
        assert!((energy[3] - 0.10).abs() < 1e-6);

        // Now test with a quant table of Q[k] = 10 -> 16 * 10 / 255 = 160 / 255.
        let qtable = [10u16; 64];
        let blocks_q = blocks.with_quant_table(&qtable);
        let mut energy_q = vec![0.0f32; 4];
        fold_jpeg_dct_blocks(&blocks_q, 2, 2, 0.5, &mut energy_q);
        let expected = 0.5 * (160.0 / 255.0);
        assert!(
            (energy_q[0] - expected).abs() < 1e-5,
            "energy_q[0] = {} vs {expected}",
            energy_q[0]
        );
        assert_eq!(energy_q[3], 0.0);
    }
}
