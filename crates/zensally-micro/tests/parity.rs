//! Tier-parity oracle: every dispatch tier must match a pure scalar
//! reference within `max_abs_diff <= 2e-6`, plus edge shapes and the
//! zero-allocation warm-pass gate.
#![allow(clippy::too_many_arguments)]

use archmage::testing::{CompileTimePolicy, for_each_token_permutation};
use zensally_micro::*;

const TOL: f32 = 2e-6;

/// Serializes tests in this binary: the counting allocator is process-global,
/// so a concurrent sibling test would pollute the zero-alloc measurement.
static SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());

// ---------------------------------------------------------------------------
// Fixture + scalar references (deliberately simple, RAW weight layout —
// the packing helpers are part of what is under test)
// ---------------------------------------------------------------------------

fn ramp(n: usize, seed: u32) -> Vec<f32> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ((s >> 8) as f32 / 8_388_608.0) - 1.0
        })
        .collect()
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

/// RAW layout reference for `conv3x3_s2_nhwc` (w: [oc][ic][k]).
fn ref_stem(
    inp: &[f32],
    w: usize,
    h: usize,
    w_raw: &[f32],
    bias: &[f32],
    act: Activation,
    out_c: usize,
) -> Vec<f32> {
    let (ow, oh) = (w.div_ceil(2), h.div_ceil(2));
    let mut out = vec![0.0; ow * oh * out_c];
    for oy in 0..oh {
        for ox in 0..ow {
            for oc in 0..out_c {
                let mut s = bias[oc];
                for ic in 0..3 {
                    for ky in 0..3usize {
                        let iy = oy * 2 + ky;
                        if iy == 0 || iy > h {
                            continue;
                        }
                        for kx in 0..3usize {
                            let ix = ox * 2 + kx;
                            if ix == 0 || ix > w {
                                continue;
                            }
                            s = inp[((iy - 1) * w + ix - 1) * 3 + ic]
                                .mul_add(w_raw[(oc * 3 + ic) * 9 + ky * 3 + kx], s);
                        }
                    }
                }
                out[(oy * ow + ox) * out_c + oc] = act.scalar(s);
            }
        }
    }
    out
}

/// RAW layout reference for `depthwise_conv3x3_nhwc` (w: [c][k]).
fn ref_depthwise(
    inp: &[f32],
    w: usize,
    h: usize,
    stride: usize,
    dil: usize,
    c: usize,
    w_raw: &[f32],
    bias: &[f32],
    act: Activation,
) -> Vec<f32> {
    let ow = if stride == 2 { w.div_ceil(2) } else { w };
    let oh = if stride == 2 { h.div_ceil(2) } else { h };
    let mut out = vec![0.0; ow * oh * c];
    for oy in 0..oh {
        for ox in 0..ow {
            for ch in 0..c {
                let mut s = bias[ch];
                for ky in 0..3usize {
                    let iy = oy * stride + ky * dil;
                    if iy < dil || iy >= dil + h {
                        continue;
                    }
                    for kx in 0..3usize {
                        let ix = ox * stride + kx * dil;
                        if ix < dil || ix >= dil + w {
                            continue;
                        }
                        s = inp[((iy - dil) * w + ix - dil) * c + ch]
                            .mul_add(w_raw[ch * 9 + ky * 3 + kx], s);
                    }
                }
                out[(oy * ow + ox) * c + ch] = act.scalar(s);
            }
        }
    }
    out
}

/// RAW layout reference for `pointwise_conv1x1_*` (w: [ic][oc]).
fn ref_pointwise(
    inp: &[f32],
    npix: usize,
    in_c: usize,
    out_c: usize,
    w_raw: &[f32],
    bias: &[f32],
    res: Option<&[f32]>,
    act: Activation,
) -> Vec<f32> {
    let mut out = vec![0.0; npix * out_c];
    for px in 0..npix {
        for oc in 0..out_c {
            let mut s = bias[oc] + res.map_or(0.0, |r| r[px * out_c + oc]);
            for ic in 0..in_c {
                s = inp[px * in_c + ic].mul_add(w_raw[ic * out_c + oc], s);
            }
            out[px * out_c + oc] = act.scalar(s);
        }
    }
    out
}

/// Reference for `bilinear_upsample2x_add_nhwc` (clamp-edge convention).
fn ref_upsample(low: &[f32], w: usize, h: usize, c: usize, skip: &[f32]) -> Vec<f32> {
    let (ow, oh) = (w * 2, h * 2);
    let mut out = vec![0.0; ow * oh * c];
    for oy2 in 0..oh {
        let iy = oy2 / 2;
        let iy1 = (iy + 1).min(h - 1);
        let fy = oy2 & 1;
        for ox2 in 0..ow {
            let ix = ox2 / 2;
            let ix1 = (ix + 1).min(w - 1);
            let fx = ox2 & 1;
            for ch in 0..c {
                let p00 = low[(iy * w + ix) * c + ch];
                let v = match (fy, fx) {
                    (0, 0) => p00,
                    (0, 1) => (p00 + low[(iy * w + ix1) * c + ch]) * 0.5,
                    (1, 0) => (p00 + low[(iy1 * w + ix) * c + ch]) * 0.5,
                    _ => {
                        (p00 + low[(iy * w + ix1) * c + ch]
                            + low[(iy1 * w + ix) * c + ch]
                            + low[(iy1 * w + ix1) * c + ch])
                            * 0.25
                    }
                };
                let o = (oy2 * ow + ox2) * c + ch;
                out[o] = v + skip[o];
            }
        }
    }
    out
}

/// Reference for `se_gate_nhwc`.
fn ref_se_gate(
    inp: &[f32],
    w: usize,
    h: usize,
    c: usize,
    mid_c: usize,
    fc1_w: &[f32],
    fc1_b: &[f32],
    fc2_w: &[f32],
    fc2_b: &[f32],
) -> Vec<f32> {
    let npix = w * h;
    let mut mean = vec![0.0f32; c];
    for px in 0..npix {
        for ch in 0..c {
            mean[ch] += inp[px * c + ch];
        }
    }
    for m in mean.iter_mut() {
        *m /= npix as f32;
    }
    let mut hidden = vec![0.0f32; mid_c];
    for m in 0..mid_c {
        let mut s = fc1_b[m];
        for ch in 0..c {
            s = mean[ch].mul_add(fc1_w[m * c + ch], s);
        }
        hidden[m] = s.max(0.0);
    }
    let mut gate = vec![0.0f32; c];
    for ch in 0..c {
        let mut s = fc2_b[ch];
        for m in 0..mid_c {
            s = hidden[m].mul_add(fc2_w[ch * mid_c + m], s);
        }
        gate[ch] = Activation::Sigmoid.scalar(s);
    }
    inp.iter()
        .enumerate()
        .map(|(i, &v)| v * gate[i % c])
        .collect()
}

/// Depthwise call with runtime-selected const generics (test matrix).
fn run_dw(
    c: usize,
    stride: usize,
    dil: usize,
    inp: &[f32],
    w: usize,
    h: usize,
    pw: &[f32],
    b: &[f32],
    act: Activation,
    out: &mut [f32],
) {
    match (c, stride, dil) {
        (24, 1, 1) => depthwise_conv3x3_nhwc::<24, 1, 1>(inp, w, h, pw, b, act, out),
        (24, 2, 1) => depthwise_conv3x3_nhwc::<24, 2, 1>(inp, w, h, pw, b, act, out),
        (48, 1, 2) => depthwise_conv3x3_nhwc::<48, 1, 2>(inp, w, h, pw, b, act, out),
        (48, 1, 4) => depthwise_conv3x3_nhwc::<48, 1, 4>(inp, w, h, pw, b, act, out),
        (96, 2, 1) => depthwise_conv3x3_nhwc::<96, 2, 1>(inp, w, h, pw, b, act, out),
        (144, 1, 1) => depthwise_conv3x3_nhwc::<144, 1, 1>(inp, w, h, pw, b, act, out),
        (144, 1, 8) => depthwise_conv3x3_nhwc::<144, 1, 8>(inp, w, h, pw, b, act, out),
        _ => unreachable!(),
    }
}

/// RAW IOHW layout reference for `conv_transpose2x2_s2_*` (w: `[ic][oc][2][2]`).
/// With kernel == stride == 2 each output pixel has exactly one tap; the cat
/// variant concatenates `in_a`/`in_b` channel-wise.
fn ref_tconv(
    in_a: &[f32],
    in_b: &[f32],
    w: usize,
    h: usize,
    ca: usize,
    cb: usize,
    oc: usize,
    w_raw: &[f32],
    bias: &[f32],
    act: Activation,
) -> Vec<f32> {
    let ow = w * 2;
    let mut out = vec![0.0; ow * h * 2 * oc];
    for y in 0..h {
        for x in 0..w {
            for ky in 0..2usize {
                for kx in 0..2usize {
                    let o = ((y * 2 + ky) * ow + x * 2 + kx) * oc;
                    for co in 0..oc {
                        let mut s = bias[co];
                        for i in 0..ca {
                            s = in_a[(y * w + x) * ca + i]
                                .mul_add(w_raw[(i * oc + co) * 4 + ky * 2 + kx], s);
                        }
                        for i in 0..cb {
                            s = in_b[(y * w + x) * cb + i]
                                .mul_add(w_raw[((ca + i) * oc + co) * 4 + ky * 2 + kx], s);
                        }
                        out[o + co] = act.scalar(s);
                    }
                }
            }
        }
    }
    out
}

/// Reference for `conv1x1_head_sigmoid_nhwc` (raw `[ca+cb]` weights).
fn ref_head(
    in_a: &[f32],
    in_b: &[f32],
    npix: usize,
    ca: usize,
    cb: usize,
    w: &[f32],
    bias0: f32,
) -> Vec<f32> {
    (0..npix)
        .map(|px| {
            let mut s = bias0;
            for i in 0..ca {
                s = in_a[px * ca + i].mul_add(w[i], s);
            }
            for i in 0..cb {
                s = in_b[px * cb + i].mul_add(w[ca + i], s);
            }
            Activation::Sigmoid.scalar(s)
        })
        .collect()
}

struct Fixture {
    w: usize,
    h: usize,
    input: Vec<f32>,
    stem_w: Vec<f32>,
    stem_b: Vec<f32>,
    dw_w: usize,
    dw_h: usize,
    dw_in: Vec<f32>,
    dw_w_raw: Vec<f32>,
    dw_b: Vec<f32>,
    npix: usize,
    pw_in: Vec<f32>,
    pw_w_raw: Vec<f32>,
    pwr_w_raw: Vec<f32>,
    pw_b: Vec<f32>,
    res: Vec<f32>,
    up_w: usize,
    up_h: usize,
    up_low: Vec<f32>,
    up_skip: Vec<f32>,
    se_w: usize,
    se_h: usize,
    se_in: Vec<f32>,
    fc1_w: Vec<f32>,
    fc1_b: Vec<f32>,
    fc2_w: Vec<f32>,
    fc2_b: Vec<f32>,
    se144_in: Vec<f32>,
    fc1_144: Vec<f32>,
    fc1_b144: Vec<f32>,
    fc2_144: Vec<f32>,
    fc2_b144: Vec<f32>,
    pw144_w: Vec<f32>,
    pw72_w: Vec<f32>,
    tc_a: Vec<f32>,
    tc_b: Vec<f32>,
    tc_w_raw: Vec<f32>,
    tc_b2: Vec<f32>,
    head_a: Vec<f32>,
    head_b: Vec<f32>,
    head_w: Vec<f32>,
    act: Activation,
}

fn build_fixture(w: usize, h: usize, dw_w: usize, dw_h: usize, act: Activation) -> Fixture {
    Fixture {
        w,
        h,
        input: ramp(w * h * 3, 3),
        stem_w: ramp(72 * 27, 5),
        stem_b: ramp(72, 7),
        dw_w,
        dw_h,
        dw_in: ramp(dw_w.max(1) * dw_h.max(1) * 144, 11),
        dw_w_raw: ramp(144 * 9, 13),
        dw_b: ramp(144, 17),
        npix: 128,
        pw_in: ramp(128 * 144, 19),
        pw_w_raw: ramp(24 * 48, 23),
        pwr_w_raw: ramp(48 * 24, 29),
        pw_b: ramp(144, 31),
        res: ramp(128 * 72, 37),
        up_w: dw_w.clamp(1, 32),
        up_h: dw_h.clamp(1, 32),
        up_low: ramp(32 * 32 * 24, 41),
        up_skip: ramp(64 * 64 * 24, 43),
        se_w: 16,
        se_h: 16,
        se_in: ramp(16 * 16 * 48, 47),
        fc1_w: ramp(12 * 48, 53),
        fc1_b: ramp(12, 59),
        fc2_w: ramp(48 * 12, 61),
        fc2_b: ramp(48, 67),
        se144_in: ramp(16 * 16 * 144, 103),
        fc1_144: ramp(36 * 144, 107),
        fc1_b144: ramp(36, 109),
        fc2_144: ramp(144 * 36, 113),
        fc2_b144: ramp(144, 127),
        pw144_w: ramp(144 * 72, 131),
        pw72_w: ramp(72 * 144, 137),
        tc_a: ramp(16 * 12 * 24, 71),
        tc_b: ramp(16 * 12 * 16, 73),
        tc_w_raw: ramp(40 * 24 * 4, 79),
        tc_b2: ramp(24, 83),
        head_a: ramp(128 * 24, 89),
        head_b: ramp(128 * 24, 97),
        head_w: ramp(48, 101),
        act,
    }
}

/// Runs every dispatched operator once on the current permutation state and
/// compares to the scalar references. Returns the worst diff seen.
fn check_all(fix: &Fixture) -> f32 {
    let mut worst = 0.0f32;
    let mut out = vec![0.0f32; 512 * 512 * 72];
    let (ow2, oh2) = (fix.w.div_ceil(2), fix.h.div_ceil(2));

    // stem 3x3s2 at OUT_C 16 and 24
    for &oc in &[16usize, 24] {
        let raw = &fix.stem_w[..oc * 27];
        let packed = pack_stem_w(raw, oc);
        let n = ow2 * oh2 * oc;
        if oc == 16 {
            conv3x3_s2_nhwc::<16>(
                &fix.input,
                fix.w,
                fix.h,
                &packed,
                &fix.stem_b,
                fix.act,
                &mut out[..n],
            );
        } else {
            conv3x3_s2_nhwc::<24>(
                &fix.input,
                fix.w,
                fix.h,
                &packed,
                &fix.stem_b,
                fix.act,
                &mut out[..n],
            );
        }
        let r = ref_stem(&fix.input, fix.w, fix.h, raw, &fix.stem_b, fix.act, oc);
        worst = worst.max(max_abs_diff(&out[..n], &r));
    }

    // depthwise variants (w24 uses 24/48/96 s1|s2 and 144 d1..8)
    for &(c, stride, dil) in &[
        (24usize, 1usize, 1usize),
        (24, 2, 1),
        (48, 1, 2),
        (48, 1, 4),
        (96, 2, 1),
        (144, 1, 1),
        (144, 1, 8),
    ] {
        let inp = &fix.dw_in[..fix.dw_w * fix.dw_h * c];
        let packed = pack_depthwise_w(&fix.dw_w_raw[..c * 9], c);
        let ow = if stride == 2 {
            fix.dw_w.div_ceil(2)
        } else {
            fix.dw_w
        };
        let oh = if stride == 2 {
            fix.dw_h.div_ceil(2)
        } else {
            fix.dw_h
        };
        let n = ow * oh * c;
        run_dw(
            c,
            stride,
            dil,
            inp,
            fix.dw_w,
            fix.dw_h,
            &packed,
            &fix.dw_b,
            fix.act,
            &mut out[..n],
        );
        let r = ref_depthwise(
            inp,
            fix.dw_w,
            fix.dw_h,
            stride,
            dil,
            c,
            &fix.dw_w_raw[..c * 9],
            &fix.dw_b,
            fix.act,
        );
        worst = worst.max(max_abs_diff(&out[..n], &r));
    }

    // pointwise + residual
    {
        let packed = pack_pointwise_w(&fix.pw_w_raw, 24, 48);
        pointwise_conv1x1_nhwc::<24, 48>(
            &fix.pw_in,
            fix.npix,
            &packed,
            &fix.pw_b,
            fix.act,
            &mut out[..fix.npix * 48],
        );
        let r = ref_pointwise(
            &fix.pw_in,
            fix.npix,
            24,
            48,
            &fix.pw_w_raw,
            &fix.pw_b,
            None,
            fix.act,
        );
        worst = worst.max(max_abs_diff(&out[..r.len()], &r));
    }
    {
        let packed = pack_pointwise_w(&fix.pwr_w_raw, 48, 24);
        pointwise_conv1x1_residual_nhwc::<48, 24>(
            &fix.pw_in,
            fix.npix,
            &packed,
            &fix.pw_b,
            &fix.res,
            fix.act,
            &mut out[..fix.npix * 24],
        );
        let r = ref_pointwise(
            &fix.pw_in,
            fix.npix,
            48,
            24,
            &fix.pwr_w_raw,
            &fix.pw_b,
            Some(&fix.res),
            fix.act,
        );
        worst = worst.max(max_abs_diff(&out[..r.len()], &r));
    }
    // w24-width pointwise: 144->72 projection and 72->144 expand
    {
        let packed = pack_pointwise_w(&fix.pw144_w, 144, 72);
        pointwise_conv1x1_nhwc::<144, 72>(
            &fix.pw_in,
            fix.npix,
            &packed,
            &fix.pw_b,
            fix.act,
            &mut out[..fix.npix * 72],
        );
        let r = ref_pointwise(
            &fix.pw_in,
            fix.npix,
            144,
            72,
            &fix.pw144_w,
            &fix.pw_b,
            None,
            fix.act,
        );
        worst = worst.max(max_abs_diff(&out[..r.len()], &r));
    }
    {
        let packed = pack_pointwise_w(&fix.pw72_w, 72, 144);
        pointwise_conv1x1_nhwc::<72, 144>(
            &fix.pw_in,
            fix.npix,
            &packed,
            &fix.pw_b,
            fix.act,
            &mut out[..fix.npix * 144],
        );
        let r = ref_pointwise(
            &fix.pw_in,
            fix.npix,
            72,
            144,
            &fix.pw72_w,
            &fix.pw_b,
            None,
            fix.act,
        );
        worst = worst.max(max_abs_diff(&out[..r.len()], &r));
    }
    {
        let packed = pack_pointwise_w(&fix.pw144_w, 144, 72);
        pointwise_conv1x1_residual_nhwc::<144, 72>(
            &fix.pw_in,
            fix.npix,
            &packed,
            &fix.pw_b,
            &fix.res,
            fix.act,
            &mut out[..fix.npix * 72],
        );
        let r = ref_pointwise(
            &fix.pw_in,
            fix.npix,
            144,
            72,
            &fix.pw144_w,
            &fix.pw_b,
            Some(&fix.res),
            fix.act,
        );
        worst = worst.max(max_abs_diff(&out[..r.len()], &r));
    }

    // upsample + add
    {
        let n = fix.up_w * 2 * fix.up_h * 2 * 24;
        bilinear_upsample2x_add_nhwc::<24>(
            &fix.up_low,
            fix.up_w,
            fix.up_h,
            &fix.up_skip,
            &mut out[..n],
        );
        let r = ref_upsample(&fix.up_low, fix.up_w, fix.up_h, 24, &fix.up_skip);
        worst = worst.max(max_abs_diff(&out[..n], &r));
    }

    // se gate
    {
        let n = fix.se_w * fix.se_h * 48;
        se_gate_nhwc::<48, 12>(
            &fix.se_in,
            fix.se_w,
            fix.se_h,
            &fix.fc1_w,
            &fix.fc1_b,
            &fix.fc2_w,
            &fix.fc2_b,
            &mut out[..n],
        );
        let r = ref_se_gate(
            &fix.se_in, fix.se_w, fix.se_h, 48, 12, &fix.fc1_w, &fix.fc1_b, &fix.fc2_w, &fix.fc2_b,
        );
        worst = worst.max(max_abs_diff(&out[..n], &r));
    }
    // w24-width SE gate: C=144, mid=36
    {
        let n = fix.se_w * fix.se_h * 144;
        se_gate_nhwc::<144, 36>(
            &fix.se144_in,
            fix.se_w,
            fix.se_h,
            &fix.fc1_144,
            &fix.fc1_b144,
            &fix.fc2_144,
            &fix.fc2_b144,
            &mut out[..n],
        );
        let r = ref_se_gate(
            &fix.se144_in,
            fix.se_w,
            fix.se_h,
            144,
            36,
            &fix.fc1_144,
            &fix.fc1_b144,
            &fix.fc2_144,
            &fix.fc2_b144,
        );
        worst = worst.max(max_abs_diff(&out[..n], &r));
    }

    // conv-transpose 2x2 s2: single input and concat input
    {
        let packed = pack_tconv_w(&fix.tc_w_raw[..24 * 24 * 4], 24, 24);
        let n = 32 * 24 * 24;
        conv_transpose2x2_s2_nhwc::<24, 24>(
            &fix.tc_a,
            16,
            12,
            &packed,
            &fix.tc_b2,
            fix.act,
            &mut out[..n],
        );
        let r = ref_tconv(
            &fix.tc_a,
            &[],
            16,
            12,
            24,
            0,
            24,
            &fix.tc_w_raw[..24 * 24 * 4],
            &fix.tc_b2,
            fix.act,
        );
        worst = worst.max(max_abs_diff(&out[..n], &r));
    }
    {
        let packed = pack_tconv_w(&fix.tc_w_raw, 40, 24);
        let n = 32 * 24 * 24;
        conv_transpose2x2_s2_cat_nhwc::<24, 16, 24>(
            &fix.tc_a,
            &fix.tc_b,
            16,
            12,
            &packed,
            &fix.tc_b2,
            fix.act,
            &mut out[..n],
        );
        let r = ref_tconv(
            &fix.tc_a,
            &fix.tc_b,
            16,
            12,
            24,
            16,
            24,
            &fix.tc_w_raw,
            &fix.tc_b2,
            fix.act,
        );
        worst = worst.max(max_abs_diff(&out[..n], &r));
    }

    // concat-input sigmoid head
    {
        let n = fix.npix;
        conv1x1_head_sigmoid_nhwc::<24, 24>(
            &fix.head_a,
            &fix.head_b,
            n,
            &fix.head_w,
            0.125,
            &mut out[..n],
        );
        let r = ref_head(&fix.head_a, &fix.head_b, n, 24, 24, &fix.head_w, 0.125);
        worst = worst.max(max_abs_diff(&out[..n], &r));
    }

    worst
}

#[test]
fn tier_parity_all_operators() {
    let _serial = SERIAL.lock().unwrap();
    for act in [
        Activation::ReLU6,
        Activation::HardSwish,
        Activation::Sigmoid,
        Activation::ReLU,
        Activation::HardSigmoid,
    ] {
        let fix = build_fixture(64, 48, 40, 30, act);
        let mut worst = 0.0f32;
        let report = for_each_token_permutation(CompileTimePolicy::Warn, |perm| {
            let d = check_all(&fix);
            worst = worst.max(d);
            assert!(d <= TOL, "{act:?}: max_abs_diff {d} at {perm}");
        });
        eprintln!("{act:?}: {report} worst_diff={worst:e}");
        assert!(report.permutations_run >= 2, "expected tier permutations");
    }
}

#[test]
fn edge_shapes_parity() {
    let _serial = SERIAL.lock().unwrap();
    // Odd widths (non-multiple-of-4/8), tiny maps, padded borders.
    for &(w, h) in &[(5usize, 3usize), (7, 9), (1, 1), (3, 33), (13, 1)] {
        let fix = build_fixture(w, h, w, h, Activation::ReLU6);
        let mut worst = 0.0f32;
        let report = for_each_token_permutation(CompileTimePolicy::Warn, |perm| {
            let d = check_all(&fix);
            worst = worst.max(d);
            assert!(d <= TOL, "{w}x{h}: max_abs_diff {d} at {perm}");
        });
        eprintln!("edge {w}x{h}: {report} worst_diff={worst:e}");
    }
}

#[test]
fn centernet_peaks_decoder() {
    let _serial = SERIAL.lock().unwrap();
    // 8x8 head, heat channel 3: peaks at (2,2)=0.9 and (6,5)=0.7 with a
    // plateau of 0.5 elsewhere; bbox/kps channels carry recognisable values.
    let (w, h, c) = (8usize, 8usize, 14usize);
    let mut head = vec![0.0f32; w * h * c];
    for y in 0..h {
        for x in 0..w {
            let cell = (y * w + x) * c;
            head[cell + 3] = 0.5;
            for i in 0..4 {
                head[cell + 5 + i] = (x * 10 + y + i) as f32 / 100.0;
            }
            for i in 0..5 {
                head[cell + 9 + i] = (x + y * 8 + i) as f32 / 1000.0;
            }
        }
    }
    head[(2 * w + 2) * c + 3] = 0.9;
    head[(5 * w + 6) * c + 3] = 0.7;
    let mut peaks = Vec::new();
    centernet_peaks_3x3::<14>(&head, w, h, 3, 0.6, &mut peaks);
    assert_eq!(peaks.len(), 2, "peaks: {peaks:?}");
    let p = &peaks[0];
    assert_eq!((p.x, p.y), (2, 2));
    assert!((p.score - 0.9).abs() < 1e-7);
    assert!((p.bbox[0] - (2 * 10 + 2) as f32 / 100.0).abs() < 1e-7);
    assert!((p.kps[4] - (2 + 2 * 8 + 4) as f32 / 1000.0).abs() < 1e-7);
    assert_eq!((peaks[1].x, peaks[1].y), (6, 5));
    // Below threshold: nothing fires.
    peaks.clear();
    centernet_peaks_3x3::<14>(&head, w, h, 3, 0.95, &mut peaks);
    assert!(peaks.is_empty());
    // Corner cell can still fire (3x3 window clipped at map edge).
    head[3] = 0.99;
    peaks.clear();
    centernet_peaks_3x3::<14>(&head, w, h, 3, 0.6, &mut peaks);
    assert!(peaks.iter().any(|p| p.x == 0 && p.y == 0));
}

/// Isolate the vector sigmoid's portable `exp`: a 1-channel pointwise conv
/// with `w = 1, b = 0` makes the pre-activation exactly `inp[px]`, so the
/// only error vs an f64 oracle is the sigmoid evaluation itself.
/// Gate: `<= 2e-6` relative over the wide pre-activation domain `[-24, 24]`.
#[test]
fn sigmoid_exp_precision() {
    let _serial = SERIAL.lock().unwrap();
    const NPIX: usize = 4096;
    let inp: Vec<f32> = (0..NPIX)
        .map(|i| (i as f32 / (NPIX - 1) as f32) * 48.0 - 24.0)
        .collect();
    let oracle: Vec<f32> = inp
        .iter()
        .map(|&x| (1.0f64 / (1.0f64 + (-(x as f64)).exp())) as f32)
        .collect();
    let packed = pack_pointwise_w(&[1.0f32; 8], 1, 8);
    let bias = [0.0f32; 8];
    let mut out = vec![0.0f32; NPIX * 8];
    let mut worst = 0.0f32;
    let report = for_each_token_permutation(CompileTimePolicy::Warn, |perm| {
        pointwise_conv1x1_nhwc::<1, 8>(&inp, NPIX, &packed, &bias, Activation::Sigmoid, &mut out);
        for px in 0..NPIX {
            for oc in 0..8 {
                let rel = (out[px * 8 + oc] - oracle[px]).abs() / oracle[px].abs().max(1e-30);
                worst = worst.max(rel);
                assert!(
                    rel <= 2e-6,
                    "sigmoid rel err {rel:e} at x={} ({perm})",
                    inp[px]
                );
            }
        }
    });
    eprintln!("sigmoid precision: {report} worst_rel={worst:e}");
}

// ---------------------------------------------------------------------------
// Chained FPN forward in ScratchArena + zero-allocation gate
// ---------------------------------------------------------------------------

fn chain_weights() -> FpnDemoWeights<'static> {
    // Raw -> packed once (model-load time); leaked for 'static test wiring.
    let raw = |n: usize, s: u32| ramp(n, s);
    let packed: Vec<Vec<f32>> = vec![
        pack_stem_w(&raw(16 * 27, 101), 16),
        pack_depthwise_w(&raw(16 * 9, 103), 16),
        pack_pointwise_w(&raw(16 * 24, 107), 16, 24),
        pack_depthwise_w(&raw(24 * 9, 109), 24),
        pack_pointwise_w(&raw(24 * 48, 113), 24, 48),
        pack_depthwise_w(&raw(48 * 9, 127), 48),
        pack_pointwise_w(&raw(48 * 48, 131), 48, 48),
        pack_depthwise_w(&raw(48 * 9, 137), 48),
        pack_pointwise_w(&raw(48 * 48, 139), 48, 48),
        pack_pointwise_w(&raw(48 * 48, 149), 48, 48),
        pack_pointwise_w(&raw(48 * 14, 151), 48, 14),
    ];
    let packed: &'static [Vec<f32>] = Box::leak(packed.into_boxed_slice());
    let bias: &'static [f32] = Box::leak(raw(512, 157).into_boxed_slice());
    let b = |off: usize, n: usize| -> &'static [f32] { &bias[off..off + n] };
    FpnDemoWeights {
        stem: (&packed[0], b(0, 16)),
        s4_dw: (&packed[1], b(16, 16)),
        s4_pw: (&packed[2], b(32, 24)),
        s8_dw: (&packed[3], b(56, 24)),
        s8_pw: (&packed[4], b(80, 48)),
        s16_dw: (&packed[5], b(128, 48)),
        s16_pw: (&packed[6], b(176, 48)),
        dil_dw: (&packed[7], b(224, 48)),
        dil_pw: (&packed[8], b(272, 48)),
        fuse_pw: (&packed[9], b(320, 48)),
        head_pw: (&packed[10], b(368, 14)),
    }
}

#[test]
fn chained_fpn_parity_and_peaks() {
    let _serial = SERIAL.lock().unwrap();
    let (w, h) = (96usize, 64usize);
    let input = ramp(w * h * 3, 3);
    let demo = chain_weights();

    // Reference result with all tiers enabled.
    let mut arena = ScratchArena::new(512, 512, 16, 48, 48);
    let mut ref_peaks = Vec::new();
    fpn_demo_forward(&input, w, h, &mut arena, &demo, -0.9, &mut ref_peaks);
    let head_len = (w / 8) * (h / 8) * 14;
    let reference = arena.f32(BufId::B)[..head_len].to_vec();

    let report = for_each_token_permutation(CompileTimePolicy::Warn, |perm| {
        let mut peaks = Vec::new();
        fpn_demo_forward(&input, w, h, &mut arena, &demo, -0.9, &mut peaks);
        let got = &arena.f32(BufId::B)[..head_len];
        let d = max_abs_diff(got, &reference);
        assert!(d <= TOL, "fpn head max_abs_diff {d} at {perm}");
        assert_eq!(peaks, ref_peaks, "peak list differs at {perm}");
    });
    eprintln!("fpn chain: {report}");
    assert!(!ref_peaks.is_empty(), "fixture should produce peaks");
}

/// Global allocator counting allocations for the zero-alloc gate.
mod alloc_count {
    use core::sync::atomic::{AtomicUsize, Ordering};
    use std::alloc::{GlobalAlloc, Layout, System};

    pub static ALLOCS: AtomicUsize = AtomicUsize::new(0);

    pub struct Counting;
    unsafe impl GlobalAlloc for Counting {
        unsafe fn alloc(&self, l: Layout) -> *mut u8 {
            ALLOCS.fetch_add(1, Ordering::SeqCst);
            unsafe { System.alloc(l) }
        }
        unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
            unsafe { System.dealloc(p, l) }
        }
    }
}

#[global_allocator]
static COUNTING: alloc_count::Counting = alloc_count::Counting;

#[test]
fn warm_fpn_pass_allocates_nothing() {
    use core::sync::atomic::Ordering;
    let _serial = SERIAL.lock().unwrap();
    let (w, h) = (96usize, 64usize);
    let input = ramp(w * h * 3, 3);
    let demo = chain_weights();
    let mut arena = ScratchArena::new(512, 512, 16, 48, 48);
    let mut peaks = Vec::with_capacity(64);

    // Warm pass: dispatch caches, token state, Vec capacity all settled.
    fpn_demo_forward(&input, w, h, &mut arena, &demo, 0.4, &mut peaks);
    peaks.clear();

    let before = alloc_count::ALLOCS.load(Ordering::SeqCst);
    fpn_demo_forward(&input, w, h, &mut arena, &demo, 0.4, &mut peaks);
    let allocs = alloc_count::ALLOCS.load(Ordering::SeqCst) - before;
    assert_eq!(
        allocs, 0,
        "warm FPN pass performed {allocs} heap allocations"
    );
}
