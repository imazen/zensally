//! Reuse each input broadcast for two output-channel groups. Spatial panels
//! bound rereads without changing the bias-first input-channel reduction order.
//! Based on this crate's pointwise kernel (owned zenjpegai origin b07f55c4).
use super::*;

pub(super) fn forward(kernel: &Pointwise, engine: &Engine, x: &BTensor) -> Result<BTensor> {
    let mut out = BTensor::scratch(kernel.cout, x.h, x.w, kernel.block)?;
    if matches!(engine.tier, Tier::Scalar) {
        tiled_scalar(ScalarToken, kernel, x, &mut out);
    } else {
        incant!(tiled(kernel, x, &mut out), [v3, neon, wasm128, scalar]);
    }
    Ok(out)
}

#[magetypes(v3, neon, wasm128, scalar)]
fn tiled(token: Token, kernel: &Pointwise, x: &BTensor, out: &mut BTensor) {
    if x.v == 8 {
        packed::<_, 8>(token, kernel, x, out);
    } else {
        packed::<_, 16>(token, kernel, x, out);
    }
}

#[inline(always)]
fn packed<T: F32x8Backend, const V: usize>(
    token: T,
    k: &Pointwise,
    x: &BTensor,
    out: &mut BTensor,
) {
    const B: usize = if cfg!(feature = "experimental-load-ahead") {
        4
    } else {
        6
    };
    let panel_size = if k.cin >= 96 { 48 } else { 96 };
    let area = x.h * x.w;
    let groups = out.c.div_ceil(V) * (V / 8);
    let (src, _) = x.data.as_chunks::<V>();
    let (dst, _) = out.data.as_chunks_mut::<V>();
    let (weight, _) = k.weight.as_chunks::<V>();
    let (bias, _) = k.bias.as_chunks::<V>();
    for panel in (0..area).step_by(panel_size) {
        let end = (panel + panel_size).min(area);
        for group in (0..groups).step_by(2) {
            let mut pos = panel;
            if group + 1 < groups {
                while pos + B <= end {
                    pair::<T, V, B>(token, k.cin, area, src, dst, weight, bias, group, pos);
                    pos += B;
                }
                while pos < end {
                    pair::<T, V, 1>(token, k.cin, area, src, dst, weight, bias, group, pos);
                    pos += 1;
                }
            } else {
                // An odd number of physical eight-lane groups has one tail.
                while pos + 12 <= end {
                    single::<T, V, 12>(token, k.cin, area, src, dst, weight, bias, group, pos);
                    pos += 12;
                }
                while pos < end {
                    single::<T, V, 1>(token, k.cin, area, src, dst, weight, bias, group, pos);
                    pos += 1;
                }
            }
        }
    }
}

struct Channel<T: F32x8Backend, const B: usize> {
    w0: f32x8<T>,
    w1: f32x8<T>,
    input: [f32; B],
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn load<T: F32x8Backend, const V: usize, const B: usize>(
    token: T,
    cin: usize,
    area: usize,
    src: &[[f32; V]],
    weight: &[[f32; V]],
    group: usize,
    pos: usize,
    ic: usize,
) -> Channel<T, B> {
    let (b0, l0) = (group / (V / 8), group % (V / 8) * 8);
    let (b1, l1) = ((group + 1) / (V / 8), (group + 1) % (V / 8) * 8);
    let row: &[[f32; V]; B] = src[ic / V * area + pos..][..B].try_into().unwrap();
    Channel {
        w0: f32x8::load_t(token, weight[b0 * cin + ic][l0..][..8].try_into().unwrap()),
        w1: f32x8::load_t(token, weight[b1 * cin + ic][l1..][..8].try_into().unwrap()),
        input: core::array::from_fn(|j| row[j][ic % V]),
    }
}

#[inline(always)]
fn accumulate<T: F32x8Backend, const B: usize>(
    token: T,
    channel: Channel<T, B>,
    a0: &mut [f32x8<T>; B],
    a1: &mut [f32x8<T>; B],
) {
    for j in 0..B {
        let a = f32x8::splat_t(token, channel.input[j]);
        a0[j] = pointwise_madd::<T, false>(token, channel.w0, a, a0[j]);
        a1[j] = pointwise_madd::<T, false>(token, channel.w1, a, a1[j]);
    }
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn pair<T: F32x8Backend, const V: usize, const B: usize>(
    token: T,
    cin: usize,
    area: usize,
    src: &[[f32; V]],
    dst: &mut [[f32; V]],
    weight: &[[f32; V]],
    bias: &[[f32; V]],
    group: usize,
    pos: usize,
) {
    let (b0, l0) = (group / (V / 8), group % (V / 8) * 8);
    let (b1, l1) = ((group + 1) / (V / 8), (group + 1) % (V / 8) * 8);
    let mut a0 = [f32x8::load_t(token, bias[b0][l0..][..8].try_into().unwrap()); B];
    let mut a1 = [f32x8::load_t(token, bias[b1][l1..][..8].try_into().unwrap()); B];
    #[cfg(feature = "experimental-load-ahead")]
    if cin > 0 {
        let mut current = load::<T, V, B>(token, cin, area, src, weight, group, pos, 0);
        for ic in 1..cin {
            let next = load::<T, V, B>(token, cin, area, src, weight, group, pos, ic);
            accumulate(token, current, &mut a0, &mut a1);
            current = next;
        }
        accumulate(token, current, &mut a0, &mut a1);
    }
    #[cfg(not(feature = "experimental-load-ahead"))]
    for ic in 0..cin {
        let channel = load::<T, V, B>(token, cin, area, src, weight, group, pos, ic);
        accumulate(token, channel, &mut a0, &mut a1);
    }
    for j in 0..B {
        a0[j].store(
            (&mut dst[b0 * area + pos + j][l0..][..8])
                .try_into()
                .unwrap(),
        );
        a1[j].store(
            (&mut dst[b1 * area + pos + j][l1..][..8])
                .try_into()
                .unwrap(),
        );
    }
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn single<T: F32x8Backend, const V: usize, const B: usize>(
    token: T,
    cin: usize,
    area: usize,
    src: &[[f32; V]],
    dst: &mut [[f32; V]],
    weight: &[[f32; V]],
    bias: &[[f32; V]],
    group: usize,
    pos: usize,
) {
    let (block, lane) = (group / (V / 8), group % (V / 8) * 8);
    let mut acc = [f32x8::load_t(token, bias[block][lane..][..8].try_into().unwrap()); B];
    for ic in 0..cin {
        let w = f32x8::load_t(
            token,
            weight[block * cin + ic][lane..][..8].try_into().unwrap(),
        );
        let row: &[[f32; V]; B] = src[ic / V * area + pos..][..B].try_into().unwrap();
        for j in 0..B {
            let a = f32x8::splat_t(token, row[j][ic % V]);
            acc[j] = pointwise_madd::<T, false>(token, w, a, acc[j]);
        }
    }
    for j in 0..B {
        acc[j].store(
            (&mut dst[block * area + pos + j][lane..][..8])
                .try_into()
                .unwrap(),
        );
    }
}
