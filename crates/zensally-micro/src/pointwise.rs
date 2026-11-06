//! Flat spatial-tile specialization of zenjpegai's packed convolution algorithm.
//! Origin: imazen/zenjpegai b07f55c4, nn/fast/conv.rs (block/tap_loop_n).
//! The 1x1 case needs neither im2col nor per-row boundary handling. Input
//! channels still accumulate in their original order, with the bias first.
use super::Result;
use alloc::{vec, vec::Vec};
use archmage::prelude::*;
use magetypes::simd::{backends::F32x8Backend, generic::f32x8};
use zenjpegai::nn::fast::{BTensor, Engine, Tier};

pub(super) struct Pointwise {
    cin: usize,
    cout: usize,
    block: usize,
    weight: Vec<f32>,
    bias: Vec<f32>,
}
impl Pointwise {
    pub(super) fn new(
        cin: usize,
        cout: usize,
        block: usize,
        weight: Vec<f32>,
        bias: Vec<f32>,
    ) -> Self {
        let mut packed = vec![0.0; cout.div_ceil(block) * cin * block];
        for oc in 0..cout {
            for ic in 0..cin {
                packed[(oc / block * cin + ic) * block + oc % block] = weight[oc * cin + ic];
            }
        }
        let mut biases = vec![0.0; cout.div_ceil(block) * block];
        biases[..cout].copy_from_slice(&bias);
        Self {
            cin,
            cout,
            block,
            weight: packed,
            bias: biases,
        }
    }
    pub(super) fn forward(&self, engine: &Engine, x: &BTensor) -> Result<BTensor> {
        let mut out = BTensor::scratch(self.cout, x.h, x.w, self.block)?;
        if matches!(engine.tier, Tier::Scalar) {
            pointwise_scalar(ScalarToken, self, x, &mut out);
        } else {
            incant!(pointwise(self, x, &mut out), [v3, neon, wasm128, scalar]);
        }
        Ok(out)
    }

    #[cfg(feature = "experimental-reassociate")]
    pub(super) fn forward_reassociated(&self, engine: &Engine, x: &BTensor) -> Result<BTensor> {
        let mut out = BTensor::scratch(self.cout, x.h, x.w, self.block)?;
        if matches!(engine.tier, Tier::Scalar) {
            pointwise_reassociated_scalar(ScalarToken, self, x, &mut out);
        } else {
            incant!(
                pointwise_reassociated(self, x, &mut out),
                [v3, neon, wasm128, scalar]
            );
        }
        Ok(out)
    }
}

#[cfg(feature = "experimental-reassociate")]
#[magetypes(v3, neon, wasm128, scalar)]
fn pointwise_reassociated(token: Token, kernel: &Pointwise, x: &BTensor, out: &mut BTensor) {
    if x.v == 8 {
        reassociated::<_, 8>(token, kernel, x, out);
    } else {
        reassociated::<_, 16>(token, kernel, x, out);
    }
}

#[cfg(feature = "experimental-reassociate")]
#[inline(always)]
fn reassociated<T: F32x8Backend, const V: usize>(
    token: T,
    kernel: &Pointwise,
    x: &BTensor,
    out: &mut BTensor,
) {
    // Six spatial outputs keep live vectors within the AVX2 register budget.
    // Split reductions have twelve accumulators; algebraic contraction may
    // keep both old accumulators and newly contracted FMA results live.
    const R: usize = if cfg!(feature = "experimental-algebraic") {
        1
    } else {
        2
    };
    const B: usize = 6;
    let area = x.h * x.w;
    let (dst, _) = out.data.as_chunks_mut::<V>();
    let (src, _) = x.data.as_chunks::<V>();
    let (weight, _) = kernel.weight.as_chunks::<V>();
    let (bias, _) = kernel.bias.as_chunks::<V>();
    for ocb in 0..out.c.div_ceil(V) {
        for sub in 0..V / 8 {
            let lane = sub * 8;
            let bias = f32x8::load_t(token, bias[ocb][lane..][..8].try_into().unwrap());
            let mut pos = 0;
            while pos + B <= area {
                reduction_tile::<T, V, B, R>(
                    token, kernel, src, dst, weight, bias, ocb, lane, pos, area,
                );
                pos += B;
            }
            while pos < area {
                reduction_tile::<T, V, 1, R>(
                    token, kernel, src, dst, weight, bias, ocb, lane, pos, area,
                );
                pos += 1;
            }
        }
    }
}

#[cfg(feature = "experimental-reassociate")]
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn reduction_tile<T: F32x8Backend, const V: usize, const B: usize, const R: usize>(
    token: T,
    kernel: &Pointwise,
    src: &[[f32; V]],
    dst: &mut [[f32; V]],
    weight: &[[f32; V]],
    bias: f32x8<T>,
    ocb: usize,
    lane: usize,
    pos: usize,
    area: usize,
) {
    let zero = f32x8::zero_t(token);
    let mut acc = [[zero; B]; R];
    acc[0].fill(bias);
    for base in (0..kernel.cin).step_by(R) {
        for (r, sums) in acc.iter_mut().enumerate() {
            let ic = base + r;
            if ic >= kernel.cin {
                break;
            }
            let w = f32x8::load_t(
                token,
                weight[ocb * kernel.cin + ic][lane..][..8]
                    .try_into()
                    .unwrap(),
            );
            let row: &[[f32; V]; B] = src[ic / V * area + pos..][..B].try_into().unwrap();
            for j in 0..B {
                let a = f32x8::splat_t(token, row[j][ic % V]);
                sums[j] = super::reduction::madd(token, w, a, sums[j]);
            }
        }
    }
    let target: &mut [[f32; V]; B] = (&mut dst[ocb * area + pos..][..B]).try_into().unwrap();
    for j in 0..B {
        let mut sum = acc[0][j];
        for sums in acc.iter().skip(1) {
            sum += sums[j];
        }
        sum.store((&mut target[j][lane..][..8]).try_into().unwrap());
    }
}

#[magetypes(v3, neon, wasm128, scalar)]
fn pointwise(token: Token, kernel: &Pointwise, x: &BTensor, out: &mut BTensor) {
    if x.v == 8 {
        packed::<_, 8>(token, kernel, x, out);
    } else {
        packed::<_, 16>(token, kernel, x, out);
    }
}

#[inline(always)]
fn packed<T: F32x8Backend, const V: usize>(
    token: T,
    kernel: &Pointwise,
    x: &BTensor,
    out: &mut BTensor,
) {
    const B: usize = 12;
    let area = x.h * x.w;
    let (dst, _) = out.data.as_chunks_mut::<V>();
    let (src, _) = x.data.as_chunks::<V>();
    let (weight, _) = kernel.weight.as_chunks::<V>();
    let (bias, _) = kernel.bias.as_chunks::<V>();
    for ocb in 0..out.c.div_ceil(V) {
        for sub in 0..V / 8 {
            let lane = sub * 8;
            let bias = f32x8::load_t(token, bias[ocb][lane..][..8].try_into().unwrap());
            let mut pos = 0;
            while pos + B <= area {
                let mut acc = [bias; B];
                for ic in 0..kernel.cin {
                    let row = &src[ic / V * area + pos..][..B];
                    let w = f32x8::load_t(
                        token,
                        weight[ocb * kernel.cin + ic][lane..][..8]
                            .try_into()
                            .unwrap(),
                    );
                    for j in 0..B {
                        let a = f32x8::splat_t(token, row[j][ic % V]);
                        #[cfg(not(target_arch = "wasm32"))]
                        {
                            acc[j] = w.mul_add_fused(a, acc[j]);
                        }
                        #[cfg(target_arch = "wasm32")]
                        {
                            acc[j] = w * a + acc[j];
                        }
                    }
                }
                for j in 0..B {
                    acc[j].store(
                        (&mut dst[ocb * area + pos + j][lane..][..8])
                            .try_into()
                            .unwrap(),
                    );
                }
                pos += B;
            }
            for p in pos..area {
                let mut acc = bias;
                for ic in 0..kernel.cin {
                    let a = f32x8::splat_t(token, src[ic / V * area + p][ic % V]);
                    let w = f32x8::load_t(
                        token,
                        weight[ocb * kernel.cin + ic][lane..][..8]
                            .try_into()
                            .unwrap(),
                    );
                    #[cfg(not(target_arch = "wasm32"))]
                    {
                        acc = w.mul_add_fused(a, acc);
                    }
                    #[cfg(target_arch = "wasm32")]
                    {
                        acc = w * a + acc;
                    }
                }
                acc.store((&mut dst[ocb * area + p][lane..][..8]).try_into().unwrap());
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use zenjpegai::{
        nn::{Conv2d, reference},
        tensor::Tensor,
    };
    #[test]
    fn flat_tiles_match_independent_planar_convolution() {
        for (cin, cout) in [(3, 24), (24, 48), (36, 96), (144, 72)] {
            for (h, w) in [(1, 1), (1, 13), (3, 7)] {
                let data: Vec<f32> = (0..cin * h * w)
                    .map(|i| ((i * 137 % 509) as f32 - 254.0) / 127.0)
                    .collect();
                let weights: Vec<f32> = (0..cin * cout)
                    .map(|i| ((i * 83 % 127) as f32 - 63.0) / 64.0)
                    .collect();
                let bias: Vec<f32> = (0..cout).map(|i| i as f32 / 31.0 - 0.5).collect();
                let input = Tensor::from_vec(cin, h, w, data).unwrap();
                let conv = Conv2d::new(
                    cin,
                    cout,
                    (1, 1),
                    1,
                    (0, 0),
                    1,
                    weights.clone(),
                    Some(bias.clone()),
                )
                .unwrap();
                let expected = reference::conv2d(&conv, &input).unwrap();
                for tier in Tier::available() {
                    let engine = Engine::with(tier, false);
                    let kernel =
                        Pointwise::new(cin, cout, tier.block(), weights.clone(), bias.clone());
                    let blocked = BTensor::from_planar(&input, tier.block()).unwrap();
                    let actual = kernel
                        .forward(&engine, &blocked)
                        .unwrap()
                        .to_planar()
                        .unwrap();
                    for (i, (&a, &b)) in actual.data.iter().zip(&expected.data).enumerate() {
                        assert_eq!(
                            a.to_bits(),
                            b.to_bits(),
                            "pointwise {tier:?} cin={cin} cout={cout} {h}x{w} at {i}"
                        );
                    }
                }
            }
        }
    }

    #[cfg(feature = "experimental-reassociate")]
    #[test]
    fn experimental_reductions_match_f64_dot_products_at_channel_and_tile_tails() {
        for (cin, cout, area) in [(3, 9, 1), (25, 17, 13), (144, 24, 19)] {
            let input: Vec<f32> = (0..cin * area)
                .map(|i| ((i * 137 % 509) as f32 - 254.0) / 127.0)
                .collect();
            let weights: Vec<f32> = (0..cin * cout)
                .map(|i| ((i * 83 % 127) as f32 - 63.0) / 64.0)
                .collect();
            let bias: Vec<f32> = (0..cout).map(|i| i as f32 / 31.0 - 0.5).collect();
            let planar = Tensor::from_vec(cin, 1, area, input.clone()).unwrap();
            for tier in Tier::available() {
                let engine = Engine::with(tier, false);
                let kernel = Pointwise::new(cin, cout, tier.block(), weights.clone(), bias.clone());
                let packed = BTensor::from_planar(&planar, tier.block()).unwrap();
                let actual = kernel
                    .forward_reassociated(&engine, &packed)
                    .unwrap()
                    .to_planar()
                    .unwrap();
                for oc in 0..cout {
                    for p in 0..area {
                        let mut expected = f64::from(bias[oc]);
                        let mut magnitude = expected.abs();
                        for ic in 0..cin {
                            let product =
                                f64::from(weights[oc * cin + ic]) * f64::from(input[ic * area + p]);
                            expected += product;
                            magnitude += product.abs();
                        }
                        // Forward error scaled by sum of absolute terms, including
                        // cancellation. This new approximate path has no bit oracle.
                        let bound = 2.0 * cin as f64 * f64::from(f32::EPSILON) * magnitude;
                        let error = (f64::from(actual.data[oc * area + p]) - expected).abs();
                        assert!(
                            error <= bound,
                            "{tier:?} cin={cin} oc={oc} p={p} error={error} bound={bound}"
                        );
                    }
                }
            }
        }
    }
}
