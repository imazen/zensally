//! Strided/dilated extension of zenjpegai's channel-blocked depthwise algorithm.
//! Origin: imazen/zenjpegai b07f55c4, src/nn/fast/depthwise.rs. Packing and
//! per-output tap order are preserved; spatial addressing admits stride/dilation.
use super::Result;
use alloc::{vec, vec::Vec};
use archmage::prelude::*;
use zenjpegai::nn::fast::{BTensor, Engine, Tier};

pub(super) struct Depthwise {
    pub(super) relu: bool,
    channels: usize,
    kh: usize,
    kw: usize,
    stride: usize,
    pad: usize,
    dilation: usize,
    block: usize,
    weight: Vec<f32>,
    bias: Vec<f32>,
}
impl Depthwise {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn new(
        channels: usize,
        kh: usize,
        kw: usize,
        stride: usize,
        pad: usize,
        dilation: usize,
        block: usize,
        weight: Vec<f32>,
        bias: Vec<f32>,
    ) -> Self {
        let mut packed = vec![0.0; channels.div_ceil(block) * kh * kw * block];
        for c in 0..channels {
            for tap in 0..kh * kw {
                packed[((c / block) * kh * kw + tap) * block + c % block] =
                    weight[c * kh * kw + tap];
            }
        }
        let mut b = vec![0.0; channels.div_ceil(block) * block];
        b[..channels].copy_from_slice(&bias);
        Self {
            relu: false,
            channels,
            kh,
            kw,
            stride,
            pad,
            dilation,
            block,
            weight: packed,
            bias: b,
        }
    }
    pub(super) fn forward(&self, engine: &Engine, x: &BTensor) -> Result<BTensor> {
        self.forward_mode::<false>(engine, x)
    }
    #[cfg(feature = "experimental-depthwise")]
    pub(super) fn forward_reassociated(&self, engine: &Engine, x: &BTensor) -> Result<BTensor> {
        self.forward_mode::<true>(engine, x)
    }
    fn forward_mode<const REASSOCIATE: bool>(
        &self,
        engine: &Engine,
        x: &BTensor,
    ) -> Result<BTensor> {
        let padded = x.pad(self.pad, self.pad, self.pad, self.pad)?;
        let (oh, ow) = (
            (padded.h - (self.kh - 1) * self.dilation - 1) / self.stride + 1,
            (padded.w - (self.kw - 1) * self.dilation - 1) / self.stride + 1,
        );
        let mut out = BTensor::scratch(self.channels, oh, ow, self.block)?;
        #[cfg(feature = "experimental-depthwise")]
        if REASSOCIATE {
            if matches!(engine.tier, Tier::Scalar) {
                depthwise_reassociated_scalar(ScalarToken, self, &padded, &mut out);
            } else {
                incant!(
                    depthwise_reassociated(self, &padded, &mut out),
                    [v3, neon, wasm128, scalar]
                );
            }
            return Ok(out);
        }
        if matches!(engine.tier, Tier::Scalar) {
            depthwise_scalar(ScalarToken, self, &padded, &mut out);
        } else {
            incant!(
                depthwise(self, &padded, &mut out),
                [v3, neon, wasm128, scalar]
            );
        }
        Ok(out)
    }
}

#[magetypes(v3, neon, wasm128, scalar)]
fn depthwise(token: Token, kernel: &Depthwise, x: &BTensor, out: &mut BTensor) {
    if x.v == 8 {
        packed::<_, 8, false>(token, kernel, x, out);
    } else {
        packed::<_, 16, false>(token, kernel, x, out);
    }
}

#[cfg(feature = "experimental-depthwise")]
#[magetypes(v3, neon, wasm128, scalar)]
fn depthwise_reassociated(token: Token, kernel: &Depthwise, x: &BTensor, out: &mut BTensor) {
    if x.v == 8 {
        packed::<_, 8, true>(token, kernel, x, out);
    } else {
        packed::<_, 16, true>(token, kernel, x, out);
    }
}

#[inline(always)]
fn packed<T: magetypes::simd::backends::F32x8Backend, const V: usize, const REASSOCIATE: bool>(
    token: T,
    kernel: &Depthwise,
    x: &BTensor,
    out: &mut BTensor,
) {
    use magetypes::simd::generic::f32x8;
    let (src, _) = x.data.as_chunks::<V>();
    let (dst, _) = out.data.as_chunks_mut::<V>();
    let (weights, _) = kernel.weight.as_chunks::<V>();
    let (bias, _) = kernel.bias.as_chunks::<V>();
    let reach = (out.w - 1) * kernel.stride + 1;
    for cb in 0..x.blocks() {
        for sub in 0..V / 8 {
            let lane = sub * 8;
            let bias = f32x8::load_t(token, bias[cb][lane..][..8].try_into().unwrap());
            let weights: [f32x8<T>; 9] = core::array::from_fn(|tap| {
                f32x8::load_t(
                    token,
                    weights[cb * 9 + tap][lane..][..8].try_into().unwrap(),
                )
            });
            for oy in 0..out.h {
                let rows: [&[[f32; V]]; 9] = core::array::from_fn(|tap| {
                    let iy = oy * kernel.stride + tap / 3 * kernel.dilation;
                    let ix = tap % 3 * kernel.dilation;
                    &src[(cb * x.h + iy) * x.w + ix..][..reach]
                });
                let row = &mut dst[(cb * out.h + oy) * out.w..][..out.w];
                for (ox, target) in row.iter_mut().enumerate() {
                    let mut acc = bias;
                    #[cfg(feature = "experimental-depthwise")]
                    if REASSOCIATE {
                        let zero = f32x8::zero_t(token);
                        let mut partial = [bias, zero, zero];
                        for group in 0..3 {
                            for (r, sum) in partial.iter_mut().enumerate() {
                                let tap = group * 3 + r;
                                let a = f32x8::load_t(
                                    token,
                                    rows[tap][ox * kernel.stride][lane..][..8]
                                        .try_into()
                                        .unwrap(),
                                );
                                *sum = super::reduction::madd(token, weights[tap], a, *sum);
                            }
                        }
                        acc = (partial[0] + partial[1]) + partial[2];
                        if kernel.relu {
                            acc = acc.max(zero);
                        }
                        acc.store((&mut target[lane..][..8]).try_into().unwrap());
                        continue;
                    }
                    for tap in 0..9 {
                        let a = f32x8::load_t(
                            token,
                            rows[tap][ox * kernel.stride][lane..][..8]
                                .try_into()
                                .unwrap(),
                        );
                        #[cfg(not(target_arch = "wasm32"))]
                        {
                            acc = weights[tap].mul_add_fused(a, acc);
                        }
                        #[cfg(target_arch = "wasm32")]
                        {
                            acc = weights[tap] * a + acc;
                        }
                    }
                    if kernel.relu {
                        acc = acc.max(f32x8::splat_t(token, 0.0));
                    }
                    acc.store((&mut target[lane..][..8]).try_into().unwrap());
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use zenjpegai::tensor::Tensor;

    #[test]
    fn strided_dilated_edges_and_fused_relu_match_planar_reference() {
        let (c, h, w) = (9, 17, 19);
        let input = Tensor::from_vec(
            c,
            h,
            w,
            (0..c * h * w)
                .map(|i| (i * 83 % 127) as f32 / 31.0 - 2.0)
                .collect(),
        )
        .unwrap();
        let weight: Vec<f32> = (0..c * 9)
            .map(|i| (i * 37 % 61) as f32 / 31.0 - 1.0)
            .collect();
        let bias: Vec<f32> = (0..c).map(|i| i as f32 / 7.0 - 1.0).collect();
        for stride in [1, 2] {
            for dilation in [1, 2, 4, 8] {
                for relu in [false, true] {
                    let (oh, ow) = (h.div_ceil(stride), w.div_ceil(stride));
                    let mut expected = vec![0.0; c * oh * ow];
                    for ch in 0..c {
                        for oy in 0..oh {
                            for ox in 0..ow {
                                let mut acc = bias[ch];
                                for ky in 0..3 {
                                    for kx in 0..3 {
                                        let iy = (oy * stride + ky * dilation) as isize
                                            - dilation as isize;
                                        let ix = (ox * stride + kx * dilation) as isize
                                            - dilation as isize;
                                        let value = if iy >= 0
                                            && ix >= 0
                                            && iy < h as isize
                                            && ix < w as isize
                                        {
                                            input.data[(ch * h + iy as usize) * w + ix as usize]
                                        } else {
                                            0.0
                                        };
                                        acc = zenjpegai::nn::fmadd(
                                            weight[ch * 9 + ky * 3 + kx],
                                            value,
                                            acc,
                                        );
                                    }
                                }
                                expected[(ch * oh + oy) * ow + ox] =
                                    if relu { acc.max(0.0) } else { acc };
                            }
                        }
                    }
                    for tier in Tier::available() {
                        let engine = Engine::with(tier, false);
                        let mut kernel = Depthwise::new(
                            c,
                            3,
                            3,
                            stride,
                            dilation,
                            dilation,
                            tier.block(),
                            weight.clone(),
                            bias.clone(),
                        );
                        kernel.relu = relu;
                        let actual = kernel
                            .forward(
                                &engine,
                                &BTensor::from_planar(&input, tier.block()).unwrap(),
                            )
                            .unwrap()
                            .to_planar()
                            .unwrap();
                        for (i, (&a, &b)) in actual.data.iter().zip(&expected).enumerate() {
                            assert_eq!(
                                a.to_bits(),
                                b.to_bits(),
                                "depthwise {tier:?} stride={stride} dilation={dilation} relu={relu} at {i}"
                            );
                        }
                        #[cfg(feature = "experimental-depthwise")]
                        {
                            let approximate = kernel
                                .forward_reassociated(
                                    &engine,
                                    &BTensor::from_planar(&input, tier.block()).unwrap(),
                                )
                                .unwrap()
                                .to_planar()
                                .unwrap();
                            for (&a, &b) in approximate.data.iter().zip(&expected) {
                                assert!(a.is_finite());
                                assert!(
                                    (a - b).abs() <= 1e-5 * b.abs().max(1.0),
                                    "reassociated depthwise {tier:?} stride={stride} dilation={dilation} relu={relu}: {a} / {b}"
                                );
                            }
                        }
                    }
                }
            }
        }
    }
}
