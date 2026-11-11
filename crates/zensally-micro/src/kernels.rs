//! NHWC operator building blocks for the static micro-graph, with
//! `archmage` + `magetypes` multi-tier dispatch.
//!
//! Layout is NHWC everywhere: `buf[(y * w + x) * C + c]`. Internal channel
//! counts are multiples of 8, so one `f32x8` is one channel-vector and
//! depthwise taps are 9 contiguous FMAs with no shuffles or gathers.
//!
//! All tiers run the SAME `f32x8` kernel body (k8): on AVX2 it is one ymm,
//! on AVX-512 halves of a zmm, on NEON/WASM128 pairs of 128-bit lanes, and
//! lane-wise on scalar. Combined with `mul_add_portable` (one-rounding FMA
//! on every backend) each output element takes one FP accumulation order
//! whatever the dispatch tier — SIMD-vs-scalar parity by construction.

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
}

impl Activation {
    /// Scalar evaluation (test references, edge paths).
    pub fn scalar(self, x: f32) -> f32 {
        match self {
            Activation::Identity => x,
            Activation::ReLU => x.max(0.0),
            Activation::ReLU6 => x.clamp(0.0, 6.0),
            Activation::HardSwish => x * (x + 3.0).clamp(0.0, 6.0) / 6.0,
            Activation::Sigmoid => 1.0 / (1.0 + (-x).exp()),
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

    /// `exp(x)` — the magetypes `exp_midp` algorithm with every polynomial
    /// step as one-rounding `mul_add_portable`, so FMA and non-FMA backends
    /// produce the same bits. Ported from zensr-micro's verified version.
    #[inline(always)]
    pub(crate) fn exp_midp_portable<T: Backend>(token: T, v: V<T>) -> V<T> {
        const C0: f32 = 1.0;
        const C1: f32 = core::f32::consts::LN_2;
        const C2: f32 = 0.240_226_46;
        const C3: f32 = 0.055_504_545;
        const C4: f32 = 0.009_618_055;
        const C5: f32 = 0.001_333_37;
        const C6: f32 = 0.000_154_47;
        let sp = |c: f32| V::<T>::splat_t(token, c);
        let x = v * sp(core::f32::consts::LOG2_E);
        let lo = sp(-126.0);
        let hi = sp(128.0);
        let c = min_x86(max_x86(x, lo), hi);
        let xi = min_x86(c.round(), sp(127.0));
        let xf = c - xi;
        let poly = xf.mul_add_portable(sp(C6), sp(C5));
        let poly = poly.mul_add_portable(xf, sp(C4));
        let poly = poly.mul_add_portable(xf, sp(C3));
        let poly = poly.mul_add_portable(xf, sp(C2));
        let poly = poly.mul_add_portable(xf, sp(C1));
        let poly = poly.mul_add_portable(xf, sp(C0));
        let scale = (xi + sp(127.0))
            .to_i32_round()
            .shl_const::<23>()
            .bitcast_to_f32();
        let r = poly * scale;
        let r = V::<T>::blend(x.simd_lt(lo), sp(0.0), r);
        V::<T>::blend(x.simd_ge(hi), sp(f32::INFINITY), r)
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
                one / (one + exp_midp_portable(token, zero - v))
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
                                acc[cv] = xin.mul_add_portable(wv, acc[cv]);
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
                let mut acc = [V::<T>::zero_t(token); 9];
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
                            acc[cv] = v.mul_add_portable(wv, acc[cv]);
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
        for px in 0..npix {
            let mut acc = [V::<T>::zero_t(token); 9];
            for cv in 0..nv {
                acc[cv] = V::<T>::from_slice_t(token, &bias[cv * 8..]);
                if !res.is_empty() {
                    acc[cv] += V::<T>::from_slice_t(token, &res[px * out_c + cv * 8..]);
                }
            }
            for ic in 0..in_c {
                let xin = V::<T>::splat_t(token, inp[px * in_c + ic]);
                for cv in 0..nv {
                    let wv = V::<T>::from_slice_t(token, &wts[(cv * in_c + ic) * 8..]);
                    acc[cv] = xin.mul_add_portable(wv, acc[cv]);
                }
            }
            let ob = px * out_c;
            for cv in 0..nv {
                let dst: &mut [f32; 8] =
                    (&mut out[ob + cv * 8..ob + cv * 8 + 8]).try_into().unwrap();
                act(token, acc[cv], act_a).store(dst);
            }
            // Scalar tail for non-multiple-of-8 heads: same ic order as the
            // vector lanes, so the channel values match lane-wise math.
            for t in 0..tail {
                let oc = nv * 8 + t;
                let mut s = bias[oc] + if res.is_empty() { 0.0 } else { res[ob + oc] };
                for ic in 0..in_c {
                    s = inp[px * in_c + ic].mul_add(wts_tail(wts, ic, oc, in_c, out_c, nv), s);
                }
                out[ob + oc] = act_a.scalar(s);
            }
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
        let mut sums = [V::<T>::zero_t(token); 9];
        for px in 0..npix {
            for cv in 0..nv {
                sums[cv] += V::<T>::from_slice_t(token, &inp[px * c + cv * 8..]);
            }
        }
        let mut mean = [0.0f32; 96];
        for cv in 0..nv {
            let arr = sums[cv].to_array();
            for l in 0..8 {
                mean[cv * 8 + l] = arr[l] / npix as f32;
            }
        }
        // fc1 -> ReLU, fc2 -> Sigmoid on channel-vector chunks.
        let mut gate = [0.0f32; 96];
        for m in 0..mid_c {
            let mut s = fc1_b[m];
            for ch in 0..c {
                s = mean[ch].mul_add(fc1_w[m * c + ch], s);
            }
            gate[m] = s.max(0.0);
        }
        let mut scale = [0.0f32; 96];
        for ch in 0..c {
            let mut s = fc2_b[ch];
            for m in 0..mid_c {
                s = gate[m].mul_add(fc2_w[ch * mid_c + m], s);
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
    assert!(C.is_multiple_of(8) && C <= 72);
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
    assert!(IN_C >= 1 && OUT_C >= 1);
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
    assert!(C.is_multiple_of(8) && C <= 96 && MID_C >= 1);
    assert!(inp.len() >= w * h * C && out.len() >= w * h * C);
    assert!(fc1_w.len() >= MID_C * C && fc1_b.len() >= MID_C);
    assert!(fc2_w.len() >= C * MID_C && fc2_b.len() >= C);
    incant!(
        se_gate_impl(inp, w, h, C, MID_C, fc1_w, fc1_b, fc2_w, fc2_b, out),
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
