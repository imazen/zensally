//! NHWC operator building blocks for the static micro-graph, with
//! `archmage` + `magetypes` multi-tier dispatch.
//!
//! Layout is NHWC everywhere: `buf[(y * w + x) * C + c]`. Internal channel
//! counts are multiples of 8, so one `f32x8` is one channel-vector and
//! depthwise taps are 9 contiguous FMAs with no shuffles or gathers.
//!
//! All tiers run the SAME `f32x8` kernel body (k8): on AVX2 it is one ymm,
//! on AVX-512 halves of a zmm, on NEON/WASM128 pairs of 128-bit lanes, and
//! lane-wise on scalar. Using `mul_add_relaxed` emits single-instruction
//! hardware FMA on x86 v3/v4, ARM64 NEON, and WASM `relaxed-simd` (and fast
//! `mul + add` on baseline WASM/scalar without software FMA emulation),
//! while keeping identical sequential `ic` accumulation order across pixels.

use crate::arena::{BufId, ScratchArena, StorageMode};
use alloc::vec::Vec;
use archmage::prelude::*;

/// Pointwise activation fused into conv outputs.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Activation {
    Identity,
    ReLU,
    ReLU6,
    /// `x * relu6(x + 3) / 6` (MobileNetV3).
    HardSwish,
    /// `1 / (1 + exp(-x))`, evaluated with the portable `exp` approximation.
    Sigmoid,
    /// `clamp(x / 6 + 0.5, 0, 1)` (MobileNetV3 hard approximation).
    HardSigmoid,
}

impl Activation {
    /// Scalar evaluation (test references, edge paths).
    pub fn scalar(self, x: f32) -> f32 {
        match self {
            Activation::Identity => x,
            Activation::ReLU => x.max(0.0),
            Activation::ReLU6 => x.clamp(0.0, 6.0),
            Activation::HardSwish => x * (x + 3.0).clamp(0.0, 6.0) / 6.0,
            Activation::Sigmoid => {
                #[cfg(feature = "std")]
                {
                    1.0 / (1.0 + (-x).exp())
                }
                #[cfg(not(feature = "std"))]
                {
                    1.0 / (1.0 + k8::exp_hp_scalar(-x))
                }
            }
            Activation::HardSigmoid => (x / 6.0 + 0.5).clamp(0.0, 1.0),
        }
    }
}

/// One decoded CenterNet-style peak on the stride-8 head.
#[derive(Clone, Debug, PartialEq)]
pub struct Peak {
    /// Cell column on the stride-8 grid.
    pub x: usize,
    /// Cell row on the stride-8 grid.
    pub y: usize,
    /// Heat score at the peak cell.
    pub score: f32,
    /// `dl, dt, dr, db` box-edge regressions (channels 5..9).
    pub bbox: [f32; 4],
    /// Five keypoint offset regressions (channels 9..14).
    pub kps: [f32; 5],
}

/// Output side for a 3x3 conv: `n` for stride 1, `ceil(n/2)` for stride 2.
fn out_dim(n: usize, stride: usize) -> usize {
    if stride == 2 { n.div_ceil(2) } else { n }
}

// ---------------------------------------------------------------------------
// Weight packing (offline-style helpers; call once per model load)
// ---------------------------------------------------------------------------

/// Pack raw `[oc][ic=3][ky][kx]` stem weights into `[(ic,k)][cv][8]` blocks
/// for contiguous vector loads. `OUT_C` must be a multiple of 8.
pub fn pack_stem_w(w: &[f32], out_c: usize) -> Vec<f32> {
    assert!(out_c.is_multiple_of(8) && w.len() == out_c * 27);
    let mut out = alloc::vec![0.0f32; out_c * 27];
    let nv = out_c / 8;
    for ic in 0..3 {
        for k in 0..9 {
            for cv in 0..nv {
                for l in 0..8 {
                    out[((ic * 9 + k) * nv + cv) * 8 + l] = w[(cv * 8 + l) * 27 + ic * 9 + k];
                }
            }
        }
    }
    out
}

/// Pack raw `[c][ky][kx]` depthwise weights into `[k][cv][8]` blocks.
pub fn pack_depthwise_w(w: &[f32], c: usize) -> Vec<f32> {
    assert!(c.is_multiple_of(8) && w.len() == c * 9);
    let mut out = alloc::vec![0.0f32; c * 9];
    let nv = c / 8;
    for k in 0..9 {
        for cv in 0..nv {
            for l in 0..8 {
                out[(k * nv + cv) * 8 + l] = w[(cv * 8 + l) * 9 + k];
            }
        }
    }
    out
}

/// Pack raw `[ic][oc]` 1x1 weights into `[cv][ic][8]` blocks, followed by the
/// raw `[ic][tail]` rows for the non-8 `out_c` tail (the 14-channel head).
/// Total floats: `(out_c/8) * in_c * 8 + in_c * (out_c % 8)`.
pub fn pack_pointwise_w(w: &[f32], in_c: usize, out_c: usize) -> Vec<f32> {
    assert!(w.len() == in_c * out_c);
    let nv = out_c / 8;
    let tail = out_c % 8;
    let mut out = alloc::vec![0.0f32; nv * in_c * 8 + in_c * tail];
    for cv in 0..nv {
        for ic in 0..in_c {
            for l in 0..8 {
                out[(cv * in_c + ic) * 8 + l] = w[ic * out_c + cv * 8 + l];
            }
        }
    }
    let base = nv * in_c * 8;
    for ic in 0..in_c {
        for t in 0..tail {
            out[base + ic * tail + t] = w[ic * out_c + nv * 8 + t];
        }
    }
    out
}

/// Pack IOHW `[in_c][out_c][2][2]` ConvTranspose weights into tap-major
/// pointwise blocks: `out[k * per_tap + ..]` where `k = ky * 2 + kx` selects
/// the `[in_c][out_c]` sub-matrix `w[i][o][ky][kx]` and `per_tap` is
/// `pack_pointwise_w`'s block size for that matrix.
pub fn pack_tconv_w(w: &[f32], in_c: usize, out_c: usize) -> Vec<f32> {
    assert!(w.len() == in_c * out_c * 4);
    let nv = out_c / 8;
    let per_tap = nv * in_c * 8 + in_c * (out_c % 8);
    let mut out = alloc::vec![0.0f32; 4 * per_tap];
    for k in 0..4usize {
        let (ky, kx) = (k / 2, k % 2);
        for cv in 0..nv {
            for ic in 0..in_c {
                for l in 0..8 {
                    out[k * per_tap + (cv * in_c + ic) * 8 + l] =
                        w[((ic * out_c) + cv * 8 + l) * 4 + ky * 2 + kx];
                }
            }
        }
        let base = nv * in_c * 8;
        for ic in 0..in_c {
            for t in 0..out_c % 8 {
                out[k * per_tap + base + ic * (out_c % 8) + t] =
                    w[((ic * out_c) + nv * 8 + t) * 4 + ky * 2 + kx];
            }
        }
    }
    out
}

// ---------------------------------------------------------------------------
// f32x8 kernel family — one body for every dispatch tier
// ---------------------------------------------------------------------------

mod k8 {
    use super::Activation;
    use magetypes::simd::backends::F32x8Convert as Backend;
    use magetypes::simd::generic::f32x8 as V;

    /// `max`/`min` with x86 `maxps` semantics on every backend (a NaN `a`
    /// yields `b`); keeps clamp edges identical across tiers.
    #[inline(always)]
    fn max_x86<T: Backend>(a: V<T>, b: V<T>) -> V<T> {
        V::<T>::blend(a.simd_gt(b), a, b)
    }
    #[inline(always)]
    fn min_x86<T: Backend>(a: V<T>, b: V<T>) -> V<T> {
        V::<T>::blend(a.simd_lt(b), a, b)
    }

    /// `exp(x)` — Cody-Waite range reduction `x = k*ln2 + r` with a degree-7
    /// Taylor polynomial on `r ∈ [-ln2/2, ln2/2]`; every step is one-rounding
    /// `mul_add_portable`, so all backends produce identical bits. Measured
    /// 1.25e-7 worst relative error on `x ∈ [-24, 24]` through the sigmoid
    /// (see `tests/parity.rs::sigmoid_exp_precision`). Inputs are clamped to
    /// `[-87.3, 88.5]`, which also guarantees the `(k+127)<<23` scale trick
    /// stays in normal floats.
    #[inline(always)]
    pub(crate) fn exp_hp_portable<T: Backend>(token: T, v: V<T>) -> V<T> {
        // ln2 split into a big exactly-representable part (0x3F317000) and a
        // remainder, so k*LN2_HI is exact and the |r| residual error ~1e-9.
        #[allow(clippy::excessive_precision)]
        const LN2_HI: f32 = 0.693_359_375;
        const LN2_LO: f32 = -2.121_944_4e-4;
        // 1/7! .. 1/1! — degree-7 Taylor for exp(r).
        const C7: f32 = 1.0 / 5040.0;
        const C6: f32 = 1.0 / 720.0;
        const C5: f32 = 1.0 / 120.0;
        const C4: f32 = 1.0 / 24.0;
        const C3: f32 = 1.0 / 6.0;
        const C2: f32 = 0.5;
        let sp = |c: f32| V::<T>::splat_t(token, c);
        let x = min_x86(max_x86(v, sp(-87.3)), sp(88.5));
        let k = (x * sp(core::f32::consts::LOG2_E)).round();
        let r = k.mul_add_portable(sp(-LN2_HI), x);
        let r = k.mul_add_portable(sp(-LN2_LO), r);
        let poly = r.mul_add_portable(sp(C7), sp(C6));
        let poly = poly.mul_add_portable(r, sp(C5));
        let poly = poly.mul_add_portable(r, sp(C4));
        let poly = poly.mul_add_portable(r, sp(C3));
        let poly = poly.mul_add_portable(r, sp(C2));
        let poly = poly.mul_add_portable(r, sp(1.0));
        let poly = poly.mul_add_portable(r, sp(1.0));
        let scale = (k + sp(127.0))
            .to_i32_round()
            .shl_const::<23>()
            .bitcast_to_f32();
        poly * scale
    }

    /// Scalar twin of [`exp_hp_portable`] (same algorithm, `no_std`-compatible).
    /// `Activation::scalar` uses `f32::exp` (correctly rounded) when `std` is
    /// enabled; this helper is used by `conv1x1_cat_head` and `no_std` builds.
    #[inline(always)]
    pub(crate) fn exp_hp_scalar(x: f32) -> f32 {
        #[allow(clippy::excessive_precision)]
        const LN2_HI: f32 = 0.693_359_375;
        const LN2_LO: f32 = -2.121_944_4e-4;
        let x = x.clamp(-87.3, 88.5);
        let y = x * core::f32::consts::LOG2_E;
        let ki = if y >= 0.0 {
            (y + 0.5) as i32
        } else {
            (y - 0.5) as i32
        };
        let k = ki as f32;
        let r = x - k * LN2_HI;
        let r = r - k * LN2_LO;
        let poly = r * (1.0 / 5040.0) + (1.0 / 720.0);
        let poly = poly * r + (1.0 / 120.0);
        let poly = poly * r + (1.0 / 24.0);
        let poly = poly * r + (1.0 / 6.0);
        let poly = poly * r + 0.5;
        let poly = poly * r + 1.0;
        let poly = poly * r + 1.0;
        poly * f32::from_bits(((ki + 127) as u32) << 23)
    }

    /// Activation on one channel-vector. Every branch is a fixed sequence of
    /// single-rounding ops, identical across tiers.
    #[inline(always)]
    fn act<T: Backend>(token: T, v: V<T>, a: Activation) -> V<T> {
        let zero = V::<T>::zero_t(token);
        match a {
            Activation::Identity => v,
            Activation::ReLU => max_x86(v, zero),
            Activation::ReLU6 => min_x86(max_x86(v, zero), V::<T>::splat_t(token, 6.0)),
            Activation::HardSwish => {
                let r6 = min_x86(
                    max_x86(v + V::<T>::splat_t(token, 3.0), zero),
                    V::<T>::splat_t(token, 6.0),
                );
                v * r6 / V::<T>::splat_t(token, 6.0)
            }
            Activation::Sigmoid => {
                let one = V::<T>::splat_t(token, 1.0);
                one / (one + exp_hp_portable(token, zero - v))
            }
            Activation::HardSigmoid => {
                let lo = V::<T>::splat_t(token, 0.5);
                min_x86(max_x86(v / V::<T>::splat_t(token, 6.0) + lo, zero), lo + lo)
            }
        }
    }

    /// Stem 3x3 stride-2 pad-1, RGB(3ch) -> OUT_C, fused bias + activation.
    /// `wts` is [`super::pack_stem_w`] layout. Input pixels are 3 floats, so
    /// the FMA is `splat(in[ic]) * wvec` per (ic, tap, channel-block).
    #[inline(always)]
    pub(crate) fn conv3x3_s2<T: Backend>(
        token: T,
        inp: &[f32],
        w: usize,
        h: usize,
        wts: &[f32],
        bias: &[f32],
        act_a: Activation,
        out: &mut [f32],
        out_c: usize,
    ) {
        let nv = out_c / 8;
        let ow = super::out_dim(w, 2);
        let oh = super::out_dim(h, 2);
        for oy in 0..oh {
            for ox in 0..ow {
                let mut acc = [V::<T>::zero_t(token); 9];
                for cv in 0..nv {
                    acc[cv] = V::<T>::from_slice_t(token, &bias[cv * 8..]);
                }
                let iy0 = oy * 2;
                let ix0 = ox * 2;
                for ic in 0..3 {
                    for ky in 0..3usize {
                        let iy = iy0 + ky;
                        if iy == 0 || iy > h {
                            continue;
                        }
                        for kx in 0..3usize {
                            let ix = ix0 + kx;
                            if ix == 0 || ix > w {
                                continue;
                            }
                            let xin = V::<T>::splat_t(token, inp[((iy - 1) * w + ix - 1) * 3 + ic]);
                            let wb = &wts[((ic * 9 + ky * 3 + kx) * nv) * 8..];
                            for cv in 0..nv {
                                let wv = V::<T>::from_slice_t(token, &wb[cv * 8..]);
                                acc[cv] = xin.mul_add_relaxed(wv, acc[cv]);
                            }
                        }
                    }
                }
                let ob = (oy * ow + ox) * out_c;
                for cv in 0..nv {
                    let dst: &mut [f32; 8] =
                        (&mut out[ob + cv * 8..ob + cv * 8 + 8]).try_into().unwrap();
                    act(token, acc[cv], act_a).store(dst);
                }
            }
        }
    }

    /// Depthwise 3x3, pad = `dil`, stride 1|2, dilation 1|2|4|8..., fused
    /// bias + activation. `wts` is [`super::pack_depthwise_w`] layout.
    /// Interior channel-vectors load as contiguous `f32x8`s — 9 FMAs per
    /// output pixel, no shuffles.
    #[inline(always)]
    pub(crate) fn depthwise<T: Backend>(
        token: T,
        inp: &[f32],
        w: usize,
        h: usize,
        stride: usize,
        dil: usize,
        c: usize,
        wts: &[f32],
        bias: &[f32],
        act_a: Activation,
        out: &mut [f32],
    ) {
        let nv = c / 8;
        let ow = super::out_dim(w, stride);
        let oh = super::out_dim(h, stride);
        for oy in 0..oh {
            for ox in 0..ow {
                let mut acc = [V::<T>::zero_t(token); 20];
                for cv in 0..nv {
                    acc[cv] = V::<T>::from_slice_t(token, &bias[cv * 8..]);
                }
                let cy = oy * stride;
                let cx = ox * stride;
                for ky in 0..3usize {
                    let iy = cy + ky * dil;
                    if iy < dil || iy >= dil + h {
                        continue;
                    }
                    for kx in 0..3usize {
                        let ix = cx + kx * dil;
                        if ix < dil || ix >= dil + w {
                            continue;
                        }
                        let px = ((iy - dil) * w + ix - dil) * c;
                        let wb = &wts[((ky * 3 + kx) * nv) * 8..];
                        for cv in 0..nv {
                            let v = V::<T>::from_slice_t(token, &inp[px + cv * 8..]);
                            let wv = V::<T>::from_slice_t(token, &wb[cv * 8..]);
                            acc[cv] = v.mul_add_relaxed(wv, acc[cv]);
                        }
                    }
                }
                let ob = (oy * ow + ox) * c;
                for cv in 0..nv {
                    let dst: &mut [f32; 8] =
                        (&mut out[ob + cv * 8..ob + cv * 8 + 8]).try_into().unwrap();
                    act(token, acc[cv], act_a).store(dst);
                }
            }
        }
    }

    /// Pointwise 1x1, fused bias + activation, optional residual add.
    /// `wts` is [`super::pack_pointwise_w`] layout. `res` empty = no add.
    /// An `out_c % 8` tail (the 14-channel head) runs the identical ic-loop
    /// in scalar so the same accumulation order applies everywhere.
    ///
    /// Tiled across 4 pixels (`a0..a3` in registers) with `cv` outer and `ic`
    /// inner so weight reads (`wts[(cv * in_c + ic) * 8..]`) are unit-stride
    /// sequential, reused 4x across pixels, and 4 independent accumulator
    /// chains hide the 4-cycle FMA pipeline latency.
    #[inline(always)]
    pub(crate) fn pointwise<T: Backend>(
        token: T,
        inp: &[f32],
        npix: usize,
        in_c: usize,
        out_c: usize,
        wts: &[f32],
        bias: &[f32],
        res: &[f32],
        act_a: Activation,
        out: &mut [f32],
    ) {
        let nv = out_c / 8;
        let tail = out_c % 8;
        let has_res = !res.is_empty();
        let mut px = 0usize;
        while px + 4 <= npix {
            let p0 = &inp[px * in_c..px * in_c + in_c];
            let p1 = &inp[(px + 1) * in_c..(px + 1) * in_c + in_c];
            let p2 = &inp[(px + 2) * in_c..(px + 2) * in_c + in_c];
            let p3 = &inp[(px + 3) * in_c..(px + 3) * in_c + in_c];
            let ob0 = px * out_c;
            let ob1 = (px + 1) * out_c;
            let ob2 = (px + 2) * out_c;
            let ob3 = (px + 3) * out_c;
            for cv in 0..nv {
                let co = cv * 8;
                let b_cv = V::<T>::from_slice_t(token, &bias[co..]);
                let (mut a0, mut a1, mut a2, mut a3) = if has_res {
                    (
                        b_cv + V::<T>::from_slice_t(token, &res[ob0 + co..]),
                        b_cv + V::<T>::from_slice_t(token, &res[ob1 + co..]),
                        b_cv + V::<T>::from_slice_t(token, &res[ob2 + co..]),
                        b_cv + V::<T>::from_slice_t(token, &res[ob3 + co..]),
                    )
                } else {
                    (b_cv, b_cv, b_cv, b_cv)
                };
                let w_cv = &wts[cv * in_c * 8..(cv + 1) * in_c * 8];
                for ic in 0..in_c {
                    let wv = V::<T>::from_slice_t(token, &w_cv[ic * 8..]);
                    a0 = V::<T>::splat_t(token, p0[ic]).mul_add_relaxed(wv, a0);
                    a1 = V::<T>::splat_t(token, p1[ic]).mul_add_relaxed(wv, a1);
                    a2 = V::<T>::splat_t(token, p2[ic]).mul_add_relaxed(wv, a2);
                    a3 = V::<T>::splat_t(token, p3[ic]).mul_add_relaxed(wv, a3);
                }
                let d0: &mut [f32; 8] = (&mut out[ob0 + co..ob0 + co + 8]).try_into().unwrap();
                act(token, a0, act_a).store(d0);
                let d1: &mut [f32; 8] = (&mut out[ob1 + co..ob1 + co + 8]).try_into().unwrap();
                act(token, a1, act_a).store(d1);
                let d2: &mut [f32; 8] = (&mut out[ob2 + co..ob2 + co + 8]).try_into().unwrap();
                act(token, a2, act_a).store(d2);
                let d3: &mut [f32; 8] = (&mut out[ob3 + co..ob3 + co + 8]).try_into().unwrap();
                act(token, a3, act_a).store(d3);
            }
            if tail > 0 {
                for sub in 0..4usize {
                    let p_sub = px + sub;
                    let ob = p_sub * out_c;
                    for t in 0..tail {
                        let oc = nv * 8 + t;
                        let mut s = bias[oc] + if has_res { res[ob + oc] } else { 0.0 };
                        for ic in 0..in_c {
                            s += inp[p_sub * in_c + ic] * wts_tail(wts, ic, oc, in_c, out_c, nv);
                        }
                        out[ob + oc] = act_a.scalar(s);
                    }
                }
            }
            px += 4;
        }
        while px < npix {
            let p0 = &inp[px * in_c..px * in_c + in_c];
            let ob = px * out_c;
            for cv in 0..nv {
                let co = cv * 8;
                let mut a0 = V::<T>::from_slice_t(token, &bias[co..]);
                if has_res {
                    a0 += V::<T>::from_slice_t(token, &res[ob + co..]);
                }
                let w_cv = &wts[cv * in_c * 8..(cv + 1) * in_c * 8];
                for ic in 0..in_c {
                    let wv = V::<T>::from_slice_t(token, &w_cv[ic * 8..]);
                    a0 = V::<T>::splat_t(token, p0[ic]).mul_add_relaxed(wv, a0);
                }
                let dst: &mut [f32; 8] = (&mut out[ob + co..ob + co + 8]).try_into().unwrap();
                act(token, a0, act_a).store(dst);
            }
            for t in 0..tail {
                let oc = nv * 8 + t;
                let mut s = bias[oc] + if has_res { res[ob + oc] } else { 0.0 };
                for ic in 0..in_c {
                    s += p0[ic] * wts_tail(wts, ic, oc, in_c, out_c, nv);
                }
                out[ob + oc] = act_a.scalar(s);
            }
            px += 1;
        }
    }

    /// Raw-layout weight read for the scalar tail (`pack_pointwise_w` only
    /// covers the multiple-of-8 blocks).
    #[inline(always)]
    fn wts_tail(
        wts_raw: &[f32],
        ic: usize,
        oc: usize,
        in_c: usize,
        out_c: usize,
        nv: usize,
    ) -> f32 {
        // The packed region holds nv*8 channels; tail weights live in the
        // raw layout appended after it by the caller. Layout: packed blocks
        // first, then raw [ic][oc-tail] rows.
        let packed = nv * in_c * 8;
        wts_raw[packed + ic * (out_c - nv * 8) + (oc - nv * 8)]
    }

    /// 2x bilinear upsample + elementwise skip add, NHWC.
    /// Corner convention: `out[2y][2x] = p(y,x)`; half pixels average the
    /// two adjacent taps; edges clamp (replicate the border pixel).
    #[inline(always)]
    pub(crate) fn upsample_add<T: Backend>(
        token: T,
        low: &[f32],
        w: usize,
        h: usize,
        c: usize,
        skip: &[f32],
        out: &mut [f32],
    ) {
        let nv = c / 8;
        let (ow, oh) = (w * 2, h * 2);
        let half = V::<T>::splat_t(token, 0.5);
        for oy2 in 0..oh {
            let iy = oy2 / 2;
            let iy1 = (iy + 1).min(h - 1);
            let fy = oy2 & 1;
            for ox2 in 0..ow {
                let ix = ox2 / 2;
                let ix1 = (ix + 1).min(w - 1);
                let fx = ox2 & 1;
                let p00 = (iy * w + ix) * c;
                let p01 = (iy * w + ix1) * c;
                let p10 = (iy1 * w + ix) * c;
                let p11 = (iy1 * w + ix1) * c;
                let ob = (oy2 * ow + ox2) * c;
                for cv in 0..nv {
                    let o = cv * 8;
                    let a = V::<T>::from_slice_t(token, &low[p00 + o..]);
                    let v = match (fy, fx) {
                        (0, 0) => a,
                        (0, 1) => (a + V::<T>::from_slice_t(token, &low[p01 + o..])) * half,
                        (1, 0) => (a + V::<T>::from_slice_t(token, &low[p10 + o..])) * half,
                        _ => {
                            let b = V::<T>::from_slice_t(token, &low[p01 + o..]);
                            let c10 = V::<T>::from_slice_t(token, &low[p10 + o..]);
                            let d = V::<T>::from_slice_t(token, &low[p11 + o..]);
                            (a + b + c10 + d) * V::<T>::splat_t(token, 0.25)
                        }
                    };
                    let sk = V::<T>::from_slice_t(token, &skip[ob + o..]);
                    let dst: &mut [f32; 8] = (&mut out[ob + o..ob + o + 8]).try_into().unwrap();
                    (v + sk).store(dst);
                }
            }
        }
    }

    /// Squeeze-excitation gate: per-channel spatial mean -> fc1 (ReLU) ->
    /// fc2 (Sigmoid) -> broadcast channel multiply. `fc1_w` is `[m][c]` raw,
    /// `fc2_w` is `[c][m]` raw. The tiny FC sums run scalar (deterministic);
    /// pooling and the final multiply are vector.
    #[inline(always)]
    pub(crate) fn se_gate<T: Backend>(
        token: T,
        inp: &[f32],
        w: usize,
        h: usize,
        c: usize,
        mid_c: usize,
        fc1_w: &[f32],
        fc1_b: &[f32],
        fc2_w: &[f32],
        fc2_b: &[f32],
        out: &mut [f32],
    ) {
        let nv = c / 8;
        let npix = w * h;
        // Global average pooling per channel-vector chunk.
        let mut sums = [V::<T>::zero_t(token); 20];
        for px in 0..npix {
            for cv in 0..nv {
                sums[cv] += V::<T>::from_slice_t(token, &inp[px * c + cv * 8..]);
            }
        }
        let mut mean = [0.0f32; 160];
        for cv in 0..nv {
            let arr = sums[cv].to_array();
            for l in 0..8 {
                mean[cv * 8 + l] = arr[l] / npix as f32;
            }
        }
        // fc1 -> ReLU, fc2 -> Sigmoid on channel-vector chunks.
        let mut gate = [0.0f32; 160];
        for m in 0..mid_c {
            let mut s = fc1_b[m];
            for ch in 0..c {
                s += mean[ch] * fc1_w[m * c + ch];
            }
            gate[m] = s.max(0.0);
        }
        let mut scale = [0.0f32; 160];
        for ch in 0..c {
            let mut s = fc2_b[ch];
            for m in 0..mid_c {
                s += gate[m] * fc2_w[ch * mid_c + m];
            }
            scale[ch] = s;
        }
        // Vector sigmoid over the scale buffer (C % 8 == 0 chunks).
        for cv in 0..nv {
            let v = V::<T>::from_slice_t(token, &scale[cv * 8..]);
            let dst: &mut [f32; 8] = (&mut scale[cv * 8..cv * 8 + 8]).try_into().unwrap();
            act(token, v, Activation::Sigmoid).store(dst);
        }
        for px in 0..npix {
            for cv in 0..nv {
                let v = V::<T>::from_slice_t(token, &inp[px * c + cv * 8..]);
                let g = V::<T>::from_slice_t(token, &scale[cv * 8..]);
                let dst: &mut [f32; 8] = (&mut out[px * c + cv * 8..px * c + cv * 8 + 8])
                    .try_into()
                    .unwrap();
                (v * g).store(dst);
            }
        }
    }

    /// ConvTranspose 2x2 stride-2 pad-0: every input pixel projects through
    /// four `in_c x out_c` sub-matrices into a non-overlapping 2x2 output
    /// block. Two input slices provide channel-concat operands — `in_b`
    /// empty means a single input. `wts` is [`super::pack_tconv_w`] layout
    /// (tap-major: 4 x [`super::pack_pointwise_w`] block).
    ///
    /// Tiled across 4 input columns (`a0..a3` in registers) with `cv` outer
    /// and `ic` split into `0..ca` and `0..cb` so weight reads are sequential,
    /// branch-free, reused 4x, and hide FMA pipeline latency.
    #[inline(always)]
    pub(crate) fn conv_t2x2<T: Backend>(
        token: T,
        in_a: &[f32],
        in_b: &[f32],
        w: usize,
        h: usize,
        ca: usize,
        cb: usize,
        out_c: usize,
        wts: &[f32],
        bias: &[f32],
        act_a: Activation,
        out: &mut [f32],
    ) {
        let in_c = ca + cb;
        let nv = out_c / 8;
        let tail = out_c % 8;
        let per_tap = nv * in_c * 8 + in_c * tail;
        for iy in 0..h {
            let mut ix = 0usize;
            while ix + 4 <= w {
                let pa0 = (iy * w + ix) * ca;
                let pa1 = (iy * w + ix + 1) * ca;
                let pa2 = (iy * w + ix + 2) * ca;
                let pa3 = (iy * w + ix + 3) * ca;
                let pb0 = (iy * w + ix) * cb;
                let pb1 = (iy * w + ix + 1) * cb;
                let pb2 = (iy * w + ix + 2) * cb;
                let pb3 = (iy * w + ix + 3) * cb;
                let sa0 = &in_a[pa0..pa0 + ca];
                let sa1 = &in_a[pa1..pa1 + ca];
                let sa2 = &in_a[pa2..pa2 + ca];
                let sa3 = &in_a[pa3..pa3 + ca];
                for k in 0..4usize {
                    let oy = 2 * iy + k / 2;
                    let kx = k % 2;
                    let ob0 = (oy * 2 * w + 2 * ix + kx) * out_c;
                    let ob1 = (oy * 2 * w + 2 * (ix + 1) + kx) * out_c;
                    let ob2 = (oy * 2 * w + 2 * (ix + 2) + kx) * out_c;
                    let ob3 = (oy * 2 * w + 2 * (ix + 3) + kx) * out_c;
                    let wk = &wts[k * per_tap..];
                    for cv in 0..nv {
                        let co = cv * 8;
                        let b_cv = V::<T>::from_slice_t(token, &bias[co..]);
                        let (mut a0, mut a1, mut a2, mut a3) = (b_cv, b_cv, b_cv, b_cv);
                        let w_cv = &wk[cv * in_c * 8..(cv + 1) * in_c * 8];
                        for ic in 0..ca {
                            let wv = V::<T>::from_slice_t(token, &w_cv[ic * 8..]);
                            a0 = V::<T>::splat_t(token, sa0[ic]).mul_add_relaxed(wv, a0);
                            a1 = V::<T>::splat_t(token, sa1[ic]).mul_add_relaxed(wv, a1);
                            a2 = V::<T>::splat_t(token, sa2[ic]).mul_add_relaxed(wv, a2);
                            a3 = V::<T>::splat_t(token, sa3[ic]).mul_add_relaxed(wv, a3);
                        }
                        if cb > 0 {
                            let sb0 = &in_b[pb0..pb0 + cb];
                            let sb1 = &in_b[pb1..pb1 + cb];
                            let sb2 = &in_b[pb2..pb2 + cb];
                            let sb3 = &in_b[pb3..pb3 + cb];
                            let w_cb = &w_cv[ca * 8..];
                            for ic in 0..cb {
                                let wv = V::<T>::from_slice_t(token, &w_cb[ic * 8..]);
                                a0 = V::<T>::splat_t(token, sb0[ic]).mul_add_relaxed(wv, a0);
                                a1 = V::<T>::splat_t(token, sb1[ic]).mul_add_relaxed(wv, a1);
                                a2 = V::<T>::splat_t(token, sb2[ic]).mul_add_relaxed(wv, a2);
                                a3 = V::<T>::splat_t(token, sb3[ic]).mul_add_relaxed(wv, a3);
                            }
                        }
                        let d0: &mut [f32; 8] =
                            (&mut out[ob0 + co..ob0 + co + 8]).try_into().unwrap();
                        act(token, a0, act_a).store(d0);
                        let d1: &mut [f32; 8] =
                            (&mut out[ob1 + co..ob1 + co + 8]).try_into().unwrap();
                        act(token, a1, act_a).store(d1);
                        let d2: &mut [f32; 8] =
                            (&mut out[ob2 + co..ob2 + co + 8]).try_into().unwrap();
                        act(token, a2, act_a).store(d2);
                        let d3: &mut [f32; 8] =
                            (&mut out[ob3 + co..ob3 + co + 8]).try_into().unwrap();
                        act(token, a3, act_a).store(d3);
                    }
                    if tail > 0 {
                        for sub in 0..4usize {
                            let ix_s = ix + sub;
                            let pa = (iy * w + ix_s) * ca;
                            let pb = (iy * w + ix_s) * cb;
                            let ob = (oy * 2 * w + 2 * ix_s + kx) * out_c;
                            for t in 0..tail {
                                let oc = nv * 8 + t;
                                let mut s = bias[oc];
                                for ic in 0..in_c {
                                    let x = if ic < ca {
                                        in_a[pa + ic]
                                    } else {
                                        in_b[pb + ic - ca]
                                    };
                                    s += x * wk[nv * in_c * 8 + ic * tail + t];
                                }
                                out[ob + oc] = act_a.scalar(s);
                            }
                        }
                    }
                }
                ix += 4;
            }
            while ix < w {
                let pa = (iy * w + ix) * ca;
                let pb = (iy * w + ix) * cb;
                let sa0 = &in_a[pa..pa + ca];
                for k in 0..4usize {
                    let oy = 2 * iy + k / 2;
                    let ox = 2 * ix + k % 2;
                    let ob = (oy * 2 * w + ox) * out_c;
                    let wk = &wts[k * per_tap..];
                    for cv in 0..nv {
                        let co = cv * 8;
                        let mut a0 = V::<T>::from_slice_t(token, &bias[co..]);
                        let w_cv = &wk[cv * in_c * 8..(cv + 1) * in_c * 8];
                        for ic in 0..ca {
                            let wv = V::<T>::from_slice_t(token, &w_cv[ic * 8..]);
                            a0 = V::<T>::splat_t(token, sa0[ic]).mul_add_relaxed(wv, a0);
                        }
                        if cb > 0 {
                            let sb0 = &in_b[pb..pb + cb];
                            let w_cb = &w_cv[ca * 8..];
                            for ic in 0..cb {
                                let wv = V::<T>::from_slice_t(token, &w_cb[ic * 8..]);
                                a0 = V::<T>::splat_t(token, sb0[ic]).mul_add_relaxed(wv, a0);
                            }
                        }
                        let dst: &mut [f32; 8] =
                            (&mut out[ob + co..ob + co + 8]).try_into().unwrap();
                        act(token, a0, act_a).store(dst);
                    }
                    for t in 0..tail {
                        let oc = nv * 8 + t;
                        let mut s = bias[oc];
                        for ic in 0..in_c {
                            let x = if ic < ca {
                                in_a[pa + ic]
                            } else {
                                in_b[pb + ic - ca]
                            };
                            s += x * wk[nv * in_c * 8 + ic * tail + t];
                        }
                        out[ob + oc] = act_a.scalar(s);
                    }
                }
                ix += 1;
            }
        }
    }

    /// Final saliency head: `sigmoid(bias + Wa·a + Wb·b)` per pixel on two
    /// channel slices (the `cat([d1, s0])` input). `w` is `[ca + cb]` raw
    /// weights — first `ca` apply to `in_a`, last `cb` to `in_b`. The dot
    /// products accumulate over channel-vector lanes; the sigmoid is the
    /// portable high-precision exp path (same code every tier).
    #[inline(always)]
    pub(crate) fn conv1x1_cat_head<T: Backend>(
        token: T,
        in_a: &[f32],
        in_b: &[f32],
        npix: usize,
        ca: usize,
        cb: usize,
        w: &[f32],
        bias0: f32,
        out: &mut [f32],
    ) {
        let va = ca / 8;
        let vb = cb / 8;
        let mut px = 0usize;
        while px + 8 <= npix {
            let mut logits = [0.0f32; 8];
            for sub in 0..8usize {
                let p = px + sub;
                let mut acc = V::<T>::zero_t(token);
                let mut tail = 0.0f32;
                for cv in 0..va {
                    let x = V::<T>::from_slice_t(token, &in_a[p * ca + cv * 8..]);
                    let wv = V::<T>::from_slice_t(token, &w[cv * 8..]);
                    acc = x.mul_add_relaxed(wv, acc);
                }
                for ic in va * 8..ca {
                    tail += in_a[p * ca + ic] * w[ic];
                }
                for cv in 0..vb {
                    let x = V::<T>::from_slice_t(token, &in_b[p * cb + cv * 8..]);
                    let wv = V::<T>::from_slice_t(token, &w[ca + cv * 8..]);
                    acc = x.mul_add_relaxed(wv, acc);
                }
                for ic in vb * 8..cb {
                    tail += in_b[p * cb + ic] * w[ca + ic];
                }
                logits[sub] = bias0 + acc.reduce_add() + tail;
            }
            let v = V::<T>::from_slice_t(token, &logits);
            let dst: &mut [f32; 8] = (&mut out[px..px + 8]).try_into().unwrap();
            act(token, v, Activation::Sigmoid).store(dst);
            px += 8;
        }
        while px < npix {
            let mut acc = V::<T>::zero_t(token);
            let mut tail = 0.0f32;
            for cv in 0..va {
                let x = V::<T>::from_slice_t(token, &in_a[px * ca + cv * 8..]);
                let wv = V::<T>::from_slice_t(token, &w[cv * 8..]);
                acc = x.mul_add_relaxed(wv, acc);
            }
            for ic in va * 8..ca {
                tail += in_a[px * ca + ic] * w[ic];
            }
            for cv in 0..vb {
                let x = V::<T>::from_slice_t(token, &in_b[px * cb + cv * 8..]);
                let wv = V::<T>::from_slice_t(token, &w[ca + cv * 8..]);
                acc = x.mul_add_relaxed(wv, acc);
            }
            for ic in vb * 8..cb {
                tail += in_b[px * cb + ic] * w[ca + ic];
            }
            let s = bias0 + acc.reduce_add() + tail;
            out[px] = 1.0 / (1.0 + exp_hp_scalar(-s));
            px += 1;
        }
    }
}

// ---------------------------------------------------------------------------
// Per-operator multi-tier dispatch — `#[magetypes]` variants for
// v3/neon/wasm128/scalar plus `#[arcane]` AVX-512 entries (v4x/v4), all
// calling the same k8 kernel body.
// ---------------------------------------------------------------------------

#[magetypes(v3, neon, wasm128, scalar)]
#[allow(clippy::too_many_arguments)]
fn stem_impl(
    token: Token,
    inp: &[f32],
    w: usize,
    h: usize,
    wts: &[f32],
    bias: &[f32],
    act: Activation,
    out: &mut [f32],
    out_c: usize,
) {
    k8::conv3x3_s2(token, inp, w, h, wts, bias, act, out, out_c);
}
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
#[arcane]
#[allow(clippy::too_many_arguments)]
fn stem_impl_v4x(
    token: X64V4xToken,
    inp: &[f32],
    w: usize,
    h: usize,
    wts: &[f32],
    bias: &[f32],
    act: Activation,
    out: &mut [f32],
    out_c: usize,
) {
    k8::conv3x3_s2(token, inp, w, h, wts, bias, act, out, out_c);
}
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
#[arcane]
#[allow(clippy::too_many_arguments)]
fn stem_impl_v4(
    token: X64V4Token,
    inp: &[f32],
    w: usize,
    h: usize,
    wts: &[f32],
    bias: &[f32],
    act: Activation,
    out: &mut [f32],
    out_c: usize,
) {
    k8::conv3x3_s2(token, inp, w, h, wts, bias, act, out, out_c);
}

#[magetypes(v3, neon, wasm128, scalar)]
#[allow(clippy::too_many_arguments)]
fn depthwise_impl(
    token: Token,
    inp: &[f32],
    w: usize,
    h: usize,
    stride: usize,
    dil: usize,
    c: usize,
    wts: &[f32],
    bias: &[f32],
    act: Activation,
    out: &mut [f32],
) {
    k8::depthwise(token, inp, w, h, stride, dil, c, wts, bias, act, out);
}
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
#[arcane]
#[allow(clippy::too_many_arguments)]
fn depthwise_impl_v4x(
    token: X64V4xToken,
    inp: &[f32],
    w: usize,
    h: usize,
    stride: usize,
    dil: usize,
    c: usize,
    wts: &[f32],
    bias: &[f32],
    act: Activation,
    out: &mut [f32],
) {
    k8::depthwise(token, inp, w, h, stride, dil, c, wts, bias, act, out);
}
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
#[arcane]
#[allow(clippy::too_many_arguments)]
fn depthwise_impl_v4(
    token: X64V4Token,
    inp: &[f32],
    w: usize,
    h: usize,
    stride: usize,
    dil: usize,
    c: usize,
    wts: &[f32],
    bias: &[f32],
    act: Activation,
    out: &mut [f32],
) {
    k8::depthwise(token, inp, w, h, stride, dil, c, wts, bias, act, out);
}

#[magetypes(v3, neon, wasm128, scalar)]
#[allow(clippy::too_many_arguments)]
fn pointwise_impl(
    token: Token,
    inp: &[f32],
    npix: usize,
    in_c: usize,
    out_c: usize,
    wts: &[f32],
    bias: &[f32],
    res: &[f32],
    act: Activation,
    out: &mut [f32],
) {
    k8::pointwise(token, inp, npix, in_c, out_c, wts, bias, res, act, out);
}
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
#[arcane]
#[allow(clippy::too_many_arguments)]
fn pointwise_impl_v4x(
    token: X64V4xToken,
    inp: &[f32],
    npix: usize,
    in_c: usize,
    out_c: usize,
    wts: &[f32],
    bias: &[f32],
    res: &[f32],
    act: Activation,
    out: &mut [f32],
) {
    k8::pointwise(token, inp, npix, in_c, out_c, wts, bias, res, act, out);
}
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
#[arcane]
#[allow(clippy::too_many_arguments)]
fn pointwise_impl_v4(
    token: X64V4Token,
    inp: &[f32],
    npix: usize,
    in_c: usize,
    out_c: usize,
    wts: &[f32],
    bias: &[f32],
    res: &[f32],
    act: Activation,
    out: &mut [f32],
) {
    k8::pointwise(token, inp, npix, in_c, out_c, wts, bias, res, act, out);
}

#[magetypes(v3, neon, wasm128, scalar)]
fn upsample_impl(
    token: Token,
    low: &[f32],
    w: usize,
    h: usize,
    c: usize,
    skip: &[f32],
    out: &mut [f32],
) {
    k8::upsample_add(token, low, w, h, c, skip, out);
}
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
#[arcane]
fn upsample_impl_v4x(
    token: X64V4xToken,
    low: &[f32],
    w: usize,
    h: usize,
    c: usize,
    skip: &[f32],
    out: &mut [f32],
) {
    k8::upsample_add(token, low, w, h, c, skip, out);
}
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
#[arcane]
fn upsample_impl_v4(
    token: X64V4Token,
    low: &[f32],
    w: usize,
    h: usize,
    c: usize,
    skip: &[f32],
    out: &mut [f32],
) {
    k8::upsample_add(token, low, w, h, c, skip, out);
}

#[magetypes(v3, neon, wasm128, scalar)]
#[allow(clippy::too_many_arguments)]
fn se_gate_impl(
    token: Token,
    inp: &[f32],
    w: usize,
    h: usize,
    c: usize,
    mid_c: usize,
    fc1_w: &[f32],
    fc1_b: &[f32],
    fc2_w: &[f32],
    fc2_b: &[f32],
    out: &mut [f32],
) {
    k8::se_gate(token, inp, w, h, c, mid_c, fc1_w, fc1_b, fc2_w, fc2_b, out);
}
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
#[arcane]
#[allow(clippy::too_many_arguments)]
fn se_gate_impl_v4x(
    token: X64V4xToken,
    inp: &[f32],
    w: usize,
    h: usize,
    c: usize,
    mid_c: usize,
    fc1_w: &[f32],
    fc1_b: &[f32],
    fc2_w: &[f32],
    fc2_b: &[f32],
    out: &mut [f32],
) {
    k8::se_gate(token, inp, w, h, c, mid_c, fc1_w, fc1_b, fc2_w, fc2_b, out);
}
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
#[arcane]
#[allow(clippy::too_many_arguments)]
fn se_gate_impl_v4(
    token: X64V4Token,
    inp: &[f32],
    w: usize,
    h: usize,
    c: usize,
    mid_c: usize,
    fc1_w: &[f32],
    fc1_b: &[f32],
    fc2_w: &[f32],
    fc2_b: &[f32],
    out: &mut [f32],
) {
    k8::se_gate(token, inp, w, h, c, mid_c, fc1_w, fc1_b, fc2_w, fc2_b, out);
}

// ---------------------------------------------------------------------------
// Public API — runtime dispatch to the best tier via `incant!`
// ---------------------------------------------------------------------------

/// Stem 3x3 stride-2 pad-1 conv: RGB `w*h*3` NHWC input -> `(w/2)*(h/2)*OUT_C`
/// output, fused bias + `Activation`. `OUT_C` must be a multiple of 8.
/// `packed_w` is [`pack_stem_w`] output (`OUT_C*27` floats).
pub fn conv3x3_s2_nhwc<const OUT_C: usize>(
    inp: &[f32],
    w: usize,
    h: usize,
    packed_w: &[f32],
    bias: &[f32],
    act: Activation,
    out: &mut [f32],
) {
    assert!(OUT_C.is_multiple_of(8) && OUT_C <= 72);
    assert!(inp.len() >= w * h * 3 && bias.len() >= OUT_C && packed_w.len() >= OUT_C * 27);
    incant!(
        stem_impl(inp, w, h, packed_w, bias, act, out, OUT_C),
        [v4x(cfg(avx512)), v4(cfg(avx512)), v3, neon, wasm128, scalar]
    );
}

/// Depthwise 3x3, pad = `DILATION`, `STRIDE` in {1, 2}, `DILATION` covering
/// {1, 2, 4, 8} (standard FPN stages and MicroSalNet v3ds dilated
/// bottlenecks). `C` must be a multiple of 8. `packed_w` is
/// [`pack_depthwise_w`] output (`C*9` floats); output is
/// `out_dim(w,STRIDE) * out_dim(h,STRIDE) * C`.
pub fn depthwise_conv3x3_nhwc<const C: usize, const STRIDE: usize, const DILATION: usize>(
    inp: &[f32],
    w: usize,
    h: usize,
    packed_w: &[f32],
    bias: &[f32],
    act: Activation,
    out: &mut [f32],
) {
    assert!(C.is_multiple_of(8) && C <= 160);
    assert!(STRIDE == 1 || STRIDE == 2);
    assert!(DILATION >= 1);
    assert!(inp.len() >= w * h * C && bias.len() >= C && packed_w.len() >= C * 9);
    incant!(
        depthwise_impl(inp, w, h, STRIDE, DILATION, C, packed_w, bias, act, out),
        [v4x(cfg(avx512)), v4(cfg(avx512)), v3, neon, wasm128, scalar]
    );
}

/// Pointwise 1x1 conv over `npix` NHWC pixels, fused bias + `Activation`.
/// `packed_w` is [`pack_pointwise_w`] output. `OUT_C` is a multiple of 8 for
/// hidden layers; the 14-channel head tail runs the scalar path internally.
pub fn pointwise_conv1x1_nhwc<const IN_C: usize, const OUT_C: usize>(
    inp: &[f32],
    npix: usize,
    packed_w: &[f32],
    bias: &[f32],
    act: Activation,
    out: &mut [f32],
) {
    pointwise_run::<IN_C, OUT_C>(inp, npix, packed_w, bias, &[], act, out);
}

/// Pointwise 1x1 with a residual add fused before the activation —
/// inverted-residual projection stages need no extra pass. `res` is the same
/// shape as the output (`npix * OUT_C`).
pub fn pointwise_conv1x1_residual_nhwc<const IN_C: usize, const OUT_C: usize>(
    inp: &[f32],
    npix: usize,
    packed_w: &[f32],
    bias: &[f32],
    res: &[f32],
    act: Activation,
    out: &mut [f32],
) {
    assert!(res.len() >= npix * OUT_C);
    pointwise_run::<IN_C, OUT_C>(inp, npix, packed_w, bias, res, act, out);
}

#[allow(clippy::too_many_arguments)]
fn pointwise_run<const IN_C: usize, const OUT_C: usize>(
    inp: &[f32],
    npix: usize,
    packed_w: &[f32],
    bias: &[f32],
    res: &[f32],
    act: Activation,
    out: &mut [f32],
) {
    assert!(IN_C >= 1 && OUT_C >= 1 && OUT_C <= 160);
    assert!(inp.len() >= npix * IN_C && bias.len() >= OUT_C && out.len() >= npix * OUT_C);
    incant!(
        pointwise_impl(inp, npix, IN_C, OUT_C, packed_w, bias, res, act, out),
        [v4x(cfg(avx512)), v4(cfg(avx512)), v3, neon, wasm128, scalar]
    );
}

/// 2x bilinear upsample of a `w*h*C` map plus elementwise skip addition —
/// the FPN top-down pathway. `skip` and `out` are `2w * 2h * C`.
/// Corner convention: edges replicate the border pixel.
pub fn bilinear_upsample2x_add_nhwc<const C: usize>(
    low: &[f32],
    w: usize,
    h: usize,
    skip: &[f32],
    out: &mut [f32],
) {
    assert!(C.is_multiple_of(8) && C <= 72);
    assert!(low.len() >= w * h * C);
    assert!(skip.len() >= w * 2 * h * 2 * C && out.len() >= w * 2 * h * 2 * C);
    incant!(
        upsample_impl(low, w, h, C, skip, out),
        [v4x(cfg(avx512)), v4(cfg(avx512)), v3, neon, wasm128, scalar]
    );
}

#[magetypes(v3, neon, wasm128, scalar)]
#[allow(clippy::too_many_arguments)]
fn tconv_impl(
    token: Token,
    in_a: &[f32],
    in_b: &[f32],
    w: usize,
    h: usize,
    ca: usize,
    cb: usize,
    out_c: usize,
    wts: &[f32],
    bias: &[f32],
    act: Activation,
    out: &mut [f32],
) {
    k8::conv_t2x2(token, in_a, in_b, w, h, ca, cb, out_c, wts, bias, act, out);
}
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
#[arcane]
#[allow(clippy::too_many_arguments)]
fn tconv_impl_v4x(
    token: X64V4xToken,
    in_a: &[f32],
    in_b: &[f32],
    w: usize,
    h: usize,
    ca: usize,
    cb: usize,
    out_c: usize,
    wts: &[f32],
    bias: &[f32],
    act: Activation,
    out: &mut [f32],
) {
    k8::conv_t2x2(token, in_a, in_b, w, h, ca, cb, out_c, wts, bias, act, out);
}
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
#[arcane]
#[allow(clippy::too_many_arguments)]
fn tconv_impl_v4(
    token: X64V4Token,
    in_a: &[f32],
    in_b: &[f32],
    w: usize,
    h: usize,
    ca: usize,
    cb: usize,
    out_c: usize,
    wts: &[f32],
    bias: &[f32],
    act: Activation,
    out: &mut [f32],
) {
    k8::conv_t2x2(token, in_a, in_b, w, h, ca, cb, out_c, wts, bias, act, out);
}

#[magetypes(v3, neon, wasm128, scalar)]
#[allow(clippy::too_many_arguments)]
fn head_impl(
    token: Token,
    in_a: &[f32],
    in_b: &[f32],
    npix: usize,
    ca: usize,
    cb: usize,
    wts: &[f32],
    bias0: f32,
    out: &mut [f32],
) {
    k8::conv1x1_cat_head(token, in_a, in_b, npix, ca, cb, wts, bias0, out);
}
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
#[arcane]
#[allow(clippy::too_many_arguments)]
fn head_impl_v4x(
    token: X64V4xToken,
    in_a: &[f32],
    in_b: &[f32],
    npix: usize,
    ca: usize,
    cb: usize,
    wts: &[f32],
    bias0: f32,
    out: &mut [f32],
) {
    k8::conv1x1_cat_head(token, in_a, in_b, npix, ca, cb, wts, bias0, out);
}
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
#[arcane]
#[allow(clippy::too_many_arguments)]
fn head_impl_v4(
    token: X64V4Token,
    in_a: &[f32],
    in_b: &[f32],
    npix: usize,
    ca: usize,
    cb: usize,
    wts: &[f32],
    bias0: f32,
    out: &mut [f32],
) {
    k8::conv1x1_cat_head(token, in_a, in_b, npix, ca, cb, wts, bias0, out);
}

/// Squeeze-excitation gate: spatial average pool -> `fc1` (ReLU) -> `fc2`
/// (Sigmoid) -> broadcast channel multiply, NHWC `w*h*C`.
/// `fc1_w` is `[MID_C][C]` raw, `fc2_w` is `[C][MID_C]` raw. `out` may not
/// alias `inp` (read the input through `inp` while writing `out`).
pub fn se_gate_nhwc<const C: usize, const MID_C: usize>(
    inp: &[f32],
    w: usize,
    h: usize,
    fc1_w: &[f32],
    fc1_b: &[f32],
    fc2_w: &[f32],
    fc2_b: &[f32],
    out: &mut [f32],
) {
    assert!(C.is_multiple_of(8) && C <= 160 && MID_C >= 1);
    assert!(inp.len() >= w * h * C && out.len() >= w * h * C);
    assert!(fc1_w.len() >= MID_C * C && fc1_b.len() >= MID_C);
    assert!(fc2_w.len() >= C * MID_C && fc2_b.len() >= C);
    incant!(
        se_gate_impl(inp, w, h, C, MID_C, fc1_w, fc1_b, fc2_w, fc2_b, out),
        [v4x(cfg(avx512)), v4(cfg(avx512)), v3, neon, wasm128, scalar]
    );
}

/// ConvTranspose 2x2 stride-2 pad-0 on a single `w*h*IN_C` NHWC input ->
/// `2w * 2h * OUT_C`. `packed_w` is [`pack_tconv_w`] output.
pub fn conv_transpose2x2_s2_nhwc<const IN_C: usize, const OUT_C: usize>(
    inp: &[f32],
    w: usize,
    h: usize,
    packed_w: &[f32],
    bias: &[f32],
    act: Activation,
    out: &mut [f32],
) {
    assert!(IN_C >= 1 && OUT_C.is_multiple_of(8) && OUT_C <= 160);
    assert!(inp.len() >= w * h * IN_C && bias.len() >= OUT_C);
    assert!(out.len() >= w * 2 * h * 2 * OUT_C);
    incant!(
        tconv_impl(inp, &[], w, h, IN_C, 0, OUT_C, packed_w, bias, act, out),
        [v4x(cfg(avx512)), v4(cfg(avx512)), v3, neon, wasm128, scalar]
    );
}

/// Concat-input ConvTranspose: logically `cat([in_a, in_b])` along channels
/// (`CA + CB` = `pack_tconv_w`'s `in_c`) without materializing the concat —
/// the two slices are read directly. `packed_w` covers `CA + CB` inputs.
pub fn conv_transpose2x2_s2_cat_nhwc<const CA: usize, const CB: usize, const OUT_C: usize>(
    in_a: &[f32],
    in_b: &[f32],
    w: usize,
    h: usize,
    packed_w: &[f32],
    bias: &[f32],
    act: Activation,
    out: &mut [f32],
) {
    assert!(CA >= 1 && CB >= 1 && OUT_C.is_multiple_of(8) && OUT_C <= 160);
    assert!(in_a.len() >= w * h * CA && in_b.len() >= w * h * CB);
    assert!(bias.len() >= OUT_C && out.len() >= w * 2 * h * 2 * OUT_C);
    incant!(
        tconv_impl(in_a, in_b, w, h, CA, CB, OUT_C, packed_w, bias, act, out),
        [v4x(cfg(avx512)), v4(cfg(avx512)), v3, neon, wasm128, scalar]
    );
}

/// Final saliency head on concat input: `sigmoid(bias + Wa·a + Wb·b)` per
/// pixel, `npix` NHWC pixels, output `npix` f32s in `[0, 1]`.
/// `w` is `CA + CB` raw weights (the `[1][CA+CB]` head conv flattened).
pub fn conv1x1_head_sigmoid_nhwc<const CA: usize, const CB: usize>(
    in_a: &[f32],
    in_b: &[f32],
    npix: usize,
    w: &[f32],
    bias0: f32,
    out: &mut [f32],
) {
    assert!(CA >= 1 && CB >= 1 && w.len() >= CA + CB);
    assert!(in_a.len() >= npix * CA && in_b.len() >= npix * CB);
    assert!(out.len() >= npix);
    incant!(
        head_impl(in_a, in_b, npix, CA, CB, w, bias0, out),
        [v4x(cfg(avx512)), v4(cfg(avx512)), v3, neon, wasm128, scalar]
    );
}

/// CenterNet-style peak decoder on a stride-8 NHWC head: a cell fires when
/// `val == max over its 3x3 window && val > threshold`. For each peak the
/// `(dl, dt, dr, db)` box regressions (channels 5..9) and the five keypoint
/// offsets (channels 9..14) are read from the same cell.
/// Anchor-free NMS — replaces prior boxes + O(N^2) IoU suppression.
///
/// `C` is the head channel count (14 for the zensally multi-task head).
/// Peaks are appended to `peaks` (callers may reuse a Vec across calls to
/// keep the warm pass allocation-free).
pub fn centernet_peaks_3x3<const C: usize>(
    head: &[f32],
    w: usize,
    h: usize,
    heat_channel: usize,
    threshold: f32,
    peaks: &mut Vec<Peak>,
) {
    assert!(C >= 14 && heat_channel < C && head.len() >= w * h * C);
    for y in 0..h {
        for x in 0..w {
            let cell = (y * w + x) * C;
            let v = head[cell + heat_channel];
            if v <= threshold {
                continue;
            }
            let mut is_max = true;
            'win: for dy in 0..3usize {
                for dx in 0..3usize {
                    if dy == 1 && dx == 1 {
                        continue;
                    }
                    let ny = y + dy;
                    let nx = x + dx;
                    if ny == 0 || ny > h || nx == 0 || nx > w {
                        continue; // outside the map = no constraint
                    }
                    if head[((ny - 1) * w + nx - 1) * C + heat_channel] > v {
                        is_max = false;
                        break 'win;
                    }
                }
            }
            if !is_max {
                continue;
            }
            let mut bbox = [0.0f32; 4];
            bbox.copy_from_slice(&head[cell + 5..cell + 9]);
            let mut kps = [0.0f32; 5];
            kps.copy_from_slice(&head[cell + 9..cell + 14]);
            peaks.push(Peak {
                x,
                y,
                score: v,
                bbox,
                kps,
            });
        }
    }
}

// ---------------------------------------------------------------------------
// Chained demo FPN forward — exercises every operator inside one
// ScratchArena with zero allocations (test parity oracle + bench target).
// ---------------------------------------------------------------------------

/// Weights for [`fpn_demo_forward`]: `(packed_w, bias)` pairs in graph order.
/// Fields map to the stage they feed; see the function docs for the chain.
pub struct FpnDemoWeights<'a> {
    /// Stem 3x3s2 (3 -> 16), packed via [`pack_stem_w`].
    pub stem: (&'a [f32], &'a [f32]),
    /// Stride-4 stage: depthwise 16 s2 + pointwise 16->24.
    pub s4_dw: (&'a [f32], &'a [f32]),
    pub s4_pw: (&'a [f32], &'a [f32]),
    /// Stride-8 stage: depthwise 24 s2 + pointwise 24->48.
    pub s8_dw: (&'a [f32], &'a [f32]),
    pub s8_pw: (&'a [f32], &'a [f32]),
    /// Stride-16 stage: depthwise 48 s2 + pointwise 48->48.
    pub s16_dw: (&'a [f32], &'a [f32]),
    pub s16_pw: (&'a [f32], &'a [f32]),
    /// Dilated bottleneck at stride 16: depthwise 48 s1 dil4 + pointwise
    /// residual 48->48.
    pub dil_dw: (&'a [f32], &'a [f32]),
    pub dil_pw: (&'a [f32], &'a [f32]),
    /// FPN fuse 1x1 48->48 after upsample+skip add.
    pub fuse_pw: (&'a [f32], &'a [f32]),
    /// Head projection 48->14 at stride 8 (tail channels run scalar).
    pub head_pw: (&'a [f32], &'a [f32]),
}

/// Chained FPN-shaped forward pass running entirely inside `arena`
/// (`F32` storage, dims `(any, any, 16, 48, 48)`; `w`,`h` multiples of 16):
///
/// `stem(3->16,s2)` -> `dw16 s2 + pw 16->24` -> `dw24 s2 + pw 24->48` ->
/// `dw48 s2 + pw 48->48` (+skip16 copy) -> `dw48 dil4 + pw48 res` ->
/// `up2x + skip8` -> `fuse pw 48->48` -> `head pw 48->14` ->
/// `centernet_peaks_3x3` on heat channels 3 (face) and 4 (body).
///
/// Zero heap allocations per call once `arena` and `peaks` carry capacity.
/// The 14-channel head ends up in [`BufId::B`].
pub fn fpn_demo_forward(
    input: &[f32],
    w: usize,
    h: usize,
    arena: &mut ScratchArena,
    wts: &FpnDemoWeights,
    threshold: f32,
    peaks: &mut Vec<Peak>,
) {
    assert!(
        w.is_multiple_of(16) && h.is_multiple_of(16),
        "demo FPN needs w,h multiples of 16"
    );
    assert!(
        arena.storage() == StorageMode::F32,
        "demo FPN runs on f32 slices"
    );
    let (_, _, stem_c, s8_c, s16_c) = arena.dims();
    assert!(stem_c == 16 && s8_c == 48 && s16_c == 48);
    assert!(input.len() >= w * h * 3);

    let (s2w, s2h) = (w / 2, h / 2);
    let (s8w, s8h) = (w / 8, h / 8);
    let (s16w, s16h) = (w / 16, h / 16);
    let s8_len = s8w * s8h * 48;
    let s16_len = s16w * s16h * 48;

    // stem: RGB -> A (stride-2, 16ch)
    {
        let a = arena.f32(BufId::A);
        conv3x3_s2_nhwc::<16>(input, w, h, wts.stem.0, wts.stem.1, Activation::ReLU6, a);
    }
    // stride-4 block: dw16 s2 (A->B), pw 16->24 (B->A)
    {
        let (a, b) = arena.two_f32(BufId::A, BufId::B);
        depthwise_conv3x3_nhwc::<16, 2, 1>(
            a,
            s2w,
            s2h,
            wts.s4_dw.0,
            wts.s4_dw.1,
            Activation::ReLU6,
            b,
        );
        pointwise_conv1x1_nhwc::<16, 24>(
            b,
            s2w / 2 * (s2h / 2),
            wts.s4_pw.0,
            wts.s4_pw.1,
            Activation::ReLU6,
            a,
        );
    }
    // stride-8 block: dw24 s2 (A->B), pw 24->48 (B->A), copy A -> SkipS8
    {
        let (a, b) = arena.two_f32(BufId::A, BufId::B);
        depthwise_conv3x3_nhwc::<24, 2, 1>(
            a,
            s2w / 2,
            s2h / 2,
            wts.s8_dw.0,
            wts.s8_dw.1,
            Activation::ReLU6,
            b,
        );
        pointwise_conv1x1_nhwc::<24, 48>(
            b,
            s8w * s8h,
            wts.s8_pw.0,
            wts.s8_pw.1,
            Activation::ReLU6,
            a,
        );
    }
    {
        let (s8, a) = arena.two_f32(BufId::SkipS8, BufId::A);
        s8[..s8_len].copy_from_slice(&a[..s8_len]);
    }
    // stride-16 block: dw48 s2 (A->B), pw 48->48 (B->A), copy A -> SkipS16
    {
        let (a, b) = arena.two_f32(BufId::A, BufId::B);
        depthwise_conv3x3_nhwc::<48, 2, 1>(
            a,
            s8w,
            s8h,
            wts.s16_dw.0,
            wts.s16_dw.1,
            Activation::ReLU6,
            b,
        );
        pointwise_conv1x1_nhwc::<48, 48>(
            b,
            s16w * s16h,
            wts.s16_pw.0,
            wts.s16_pw.1,
            Activation::ReLU6,
            a,
        );
    }
    {
        let (s16, a) = arena.two_f32(BufId::SkipS16, BufId::A);
        s16[..s16_len].copy_from_slice(&a[..s16_len]);
    }
    // dilated bottleneck at stride 16: dw48 dil4 (A->B), pw residual
    // (inp=B, res=SkipS16 copy of the block input, out=A)
    {
        let (a, b, _s8, s16) = arena.parts_f32();
        depthwise_conv3x3_nhwc::<48, 1, 4>(
            a,
            s16w,
            s16h,
            wts.dil_dw.0,
            wts.dil_dw.1,
            Activation::ReLU6,
            b,
        );
        pointwise_conv1x1_residual_nhwc::<48, 48>(
            b,
            s16w * s16h,
            wts.dil_pw.0,
            wts.dil_pw.1,
            s16,
            Activation::ReLU6,
            a,
        );
    }
    // top-down FPN: 2x upsample of the stride-16 tensor + SkipS8 add
    // (A -> B), fuse pw 48->48 (B -> A), head pw 48->14 (A -> B)
    {
        let (a, b, s8, _s16) = arena.parts_f32();
        bilinear_upsample2x_add_nhwc::<48>(a, s16w, s16h, s8, b);
        pointwise_conv1x1_nhwc::<48, 48>(
            b,
            s8w * s8h,
            wts.fuse_pw.0,
            wts.fuse_pw.1,
            Activation::ReLU6,
            a,
        );
        pointwise_conv1x1_nhwc::<48, 14>(
            a,
            s8w * s8h,
            wts.head_pw.0,
            wts.head_pw.1,
            Activation::Identity,
            b,
        );
        centernet_peaks_3x3::<14>(b, s8w, s8h, 3, threshold, peaks);
        centernet_peaks_3x3::<14>(b, s8w, s8h, 4, threshold, peaks);
    }
}
