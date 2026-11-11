//! Per-kernel dispatched-tier vs forced-scalar for zensally-micro.
//!
//! The dispatched arm carries the `#[arcane]` boundary; forced scalar drops
//! every x86 tier so `incant!` lands on `ScalarToken`. ~1.00x already means
//! the SIMD path wins nothing — below ~0.95x is a finding.
//!
//! Run: `cargo bench -p zensally-micro --bench kernel_tiers`
#![allow(clippy::too_many_arguments)]

use zenbench::prelude::*;
use zensally_micro::*;

#[cfg(target_arch = "x86_64")]
fn tier_name() -> &'static str {
    use archmage::SimdToken as _;
    // Ladder order, strongest first — must match the `incant!` lists in
    // src/kernels.rs or this label lies.
    if archmage::X64V4xToken::summon().is_some() {
        return "v4x(avx512)";
    }
    if archmage::X64V4Token::summon().is_some() {
        return "v4(avx512)";
    }
    if archmage::X64V3Token::summon().is_some() {
        return "v3(avx2)";
    }
    "scalar(no simd token)"
}

#[cfg(target_arch = "x86_64")]
fn set_simd(enabled: bool) -> bool {
    // Disable every x86 tier, not just V3: V4x is a superset, so leaving it
    // live would make the "scalar" arm run AVX-512 (silent ~1.00x pass).
    let a = archmage::X64V3Token::dangerously_disable_token_process_wide(!enabled).is_ok();
    let b = archmage::X64V4Token::dangerously_disable_token_process_wide(!enabled).is_ok();
    let c = archmage::X64V4xToken::dangerously_disable_token_process_wide(!enabled).is_ok();
    a && b && c
}

#[cfg(target_arch = "aarch64")]
fn set_simd(enabled: bool) -> bool {
    archmage::NeonToken::dangerously_disable_token_process_wide(!enabled).is_ok()
}

#[cfg(target_arch = "aarch64")]
fn tier_name() -> &'static str {
    "neon"
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
fn set_simd(_e: bool) -> bool {
    false
}
#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
fn tier_name() -> &'static str {
    "unknown"
}

fn ramp(n: usize, seed: u32) -> Vec<f32> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ((s >> 8) as f32 / 8_388_608.0) - 1.0
        })
        .collect()
}

fn chain_weights() -> FpnDemoWeights<'static> {
    let packed: Vec<Vec<f32>> = vec![
        pack_stem_w(&ramp(16 * 27, 101), 16),
        pack_depthwise_w(&ramp(16 * 9, 103), 16),
        pack_pointwise_w(&ramp(16 * 24, 107), 16, 24),
        pack_depthwise_w(&ramp(24 * 9, 109), 24),
        pack_pointwise_w(&ramp(24 * 48, 113), 24, 48),
        pack_depthwise_w(&ramp(48 * 9, 127), 48),
        pack_pointwise_w(&ramp(48 * 48, 131), 48, 48),
        pack_depthwise_w(&ramp(48 * 9, 137), 48),
        pack_pointwise_w(&ramp(48 * 48, 139), 48, 48),
        pack_pointwise_w(&ramp(48 * 48, 149), 48, 48),
        pack_pointwise_w(&ramp(48 * 14, 151), 48, 14),
    ];
    let packed: &'static [Vec<f32>] = Box::leak(packed.into_boxed_slice());
    let bias: &'static [f32] = Box::leak(ramp(512, 157).into_boxed_slice());
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

fn bench_kernels(suite: &mut Suite) {
    if !set_simd(true) || !set_simd(false) {
        eprintln!("[kernel_tiers] SIMD tier not toggleable here. Skipping.");
        return;
    }
    set_simd(true);
    let tier = tier_name();
    eprintln!("[kernel_tiers] comparing {tier} vs forced scalar");

    // Stem 3x3s2 on a 512x512 RGB input -> 256x256x16.
    {
        let (w, h) = (512usize, 512usize);
        let inp: &'static [f32] = Box::leak(ramp(w * h * 3, 3).into_boxed_slice());
        let wts: &'static [f32] = Box::leak(pack_stem_w(&ramp(16 * 27, 5), 16).into_boxed_slice());
        let bias: &'static [f32] = Box::leak(ramp(16, 7).into_boxed_slice());
        let n = (w / 2) * (h / 2) * 16;
        suite.compare("conv3x3_s2/512x512x3->16", |g| {
            g.throughput(Throughput::Elements(n as u64));
            for (arm, simd) in [(tier, true), ("scalar", false)] {
                g.bench(arm, move |b| {
                    b.with_input(move || {
                        set_simd(simd);
                        vec![0f32; n]
                    })
                    .run(move |mut out| {
                        conv3x3_s2_nhwc::<16>(inp, w, h, wts, bias, Activation::ReLU6, &mut out);
                        out
                    })
                });
            }
        });
    }

    // Depthwise: standard s1 (64x64x24), strided s2 (64x64x24),
    // dilated d2/d4 bottleneck (32x32x48).
    for &(name, c, stride, dil, w, h) in &[
        (
            "dw3x3_s1/64x64x24",
            24usize,
            1usize,
            1usize,
            64usize,
            64usize,
        ),
        ("dw3x3_s2/64x64x24", 24, 2, 1, 64, 64),
        ("dw3x3_d2/32x32x48", 48, 1, 2, 32, 32),
        ("dw3x3_d4/32x32x48", 48, 1, 4, 32, 32),
    ] {
        let inp: &'static [f32] = Box::leak(ramp(w * h * c, 11).into_boxed_slice());
        let wts: &'static [f32] =
            Box::leak(pack_depthwise_w(&ramp(c * 9, 13), c).into_boxed_slice());
        let bias: &'static [f32] = Box::leak(ramp(c, 17).into_boxed_slice());
        let ow = if stride == 2 { w.div_ceil(2) } else { w };
        let oh = if stride == 2 { h.div_ceil(2) } else { h };
        let n = ow * oh * c;
        let name: &'static str = Box::leak(name.to_string().into_boxed_str());
        suite.compare(name, move |g| {
            g.throughput(Throughput::Elements(n as u64));
            for (arm, simd) in [(tier, true), ("scalar", false)] {
                g.bench(arm, move |b| {
                    b.with_input(move || {
                        set_simd(simd);
                        vec![0f32; n]
                    })
                    .run(move |mut out| {
                        run_dw(c, stride, dil, inp, w, h, wts, bias, &mut out);
                        out
                    })
                });
            }
        });
    }

    // Pointwise 1x1 (24->48) and residual (48->24) over a 64x64 map.
    for &(name, res) in &[
        ("pw1x1/64x64_24->48", false),
        ("pw1x1_res/64x64_48->24", true),
    ] {
        let npix = 64 * 64usize;
        let (ic, oc) = if res { (48usize, 24usize) } else { (24, 48) };
        let inp: &'static [f32] = Box::leak(ramp(npix * ic, 19).into_boxed_slice());
        let wts: &'static [f32] =
            Box::leak(pack_pointwise_w(&ramp(ic * oc, 23), ic, oc).into_boxed_slice());
        let bias: &'static [f32] = Box::leak(ramp(oc, 31).into_boxed_slice());
        let r: &'static [f32] = Box::leak(ramp(npix * oc, 37).into_boxed_slice());
        let n = npix * oc;
        let name: &'static str = Box::leak(name.to_string().into_boxed_str());
        suite.compare(name, move |g| {
            g.throughput(Throughput::Elements(n as u64));
            for (arm, simd) in [(tier, true), ("scalar", false)] {
                g.bench(arm, move |b| {
                    b.with_input(move || {
                        set_simd(simd);
                        vec![0f32; n]
                    })
                    .run(move |mut out| {
                        if res {
                            pointwise_conv1x1_residual_nhwc::<48, 24>(
                                inp,
                                npix,
                                wts,
                                bias,
                                r,
                                Activation::ReLU6,
                                &mut out,
                            );
                        } else {
                            pointwise_conv1x1_nhwc::<24, 48>(
                                inp,
                                npix,
                                wts,
                                bias,
                                Activation::ReLU6,
                                &mut out,
                            );
                        }
                        out
                    })
                });
            }
        });
    }

    // FPN upsample 2x + skip add: 32x32 -> 64x64, 48ch.
    {
        let (w, h, c) = (32usize, 32usize, 48usize);
        let low: &'static [f32] = Box::leak(ramp(w * h * c, 41).into_boxed_slice());
        let skip: &'static [f32] = Box::leak(ramp(w * 2 * h * 2 * c, 43).into_boxed_slice());
        let n = w * 2 * h * 2 * c;
        suite.compare("up2x_add/32->64x48", |g| {
            g.throughput(Throughput::Elements(n as u64));
            for (arm, simd) in [(tier, true), ("scalar", false)] {
                g.bench(arm, move |b| {
                    b.with_input(move || {
                        set_simd(simd);
                        vec![0f32; n]
                    })
                    .run(move |mut out| {
                        bilinear_upsample2x_add_nhwc::<48>(low, w, h, skip, &mut out);
                        out
                    })
                });
            }
        });
    }

    // SE gate on a 32x32x48 map.
    {
        let (w, h, c, mid) = (32usize, 32usize, 48usize, 12usize);
        let inp: &'static [f32] = Box::leak(ramp(w * h * c, 47).into_boxed_slice());
        let f1w: &'static [f32] = Box::leak(ramp(mid * c, 53).into_boxed_slice());
        let f1b: &'static [f32] = Box::leak(ramp(mid, 59).into_boxed_slice());
        let f2w: &'static [f32] = Box::leak(ramp(c * mid, 61).into_boxed_slice());
        let f2b: &'static [f32] = Box::leak(ramp(c, 67).into_boxed_slice());
        let n = w * h * c;
        suite.compare("se_gate/32x32x48", |g| {
            g.throughput(Throughput::Elements(n as u64));
            for (arm, simd) in [(tier, true), ("scalar", false)] {
                g.bench(arm, move |b| {
                    b.with_input(move || {
                        set_simd(simd);
                        vec![0f32; n]
                    })
                    .run(move |mut out| {
                        se_gate_nhwc::<48, 12>(inp, w, h, f1w, f1b, f2w, f2b, &mut out);
                        out
                    })
                });
            }
        });
    }

    // CenterNet peaks on a 64x64x14 stride-8 head (scalar-only code, shown
    // for completeness — same arm twice is expected ~1.00x).
    {
        let (w, h, c) = (64usize, 64usize, 14usize);
        let head: &'static [f32] = Box::leak(ramp(w * h * c, 71).into_boxed_slice());
        suite.compare("peaks3x3/64x64x14", |g| {
            g.throughput(Throughput::Elements((w * h) as u64));
            for (arm, simd) in [(tier, true), ("scalar", false)] {
                g.bench(arm, move |b| {
                    b.with_input(move || {
                        set_simd(simd);
                        Vec::with_capacity(64)
                    })
                    .run(move |mut peaks| {
                        centernet_peaks_3x3::<14>(head, w, h, 3, -0.2, &mut peaks);
                        peaks
                    })
                });
            }
        });
    }

    // Chained FPN forward on a 256x256 input inside a pre-built arena.
    {
        let (w, h) = (256usize, 256usize);
        let inp: &'static [f32] = Box::leak(ramp(w * h * 3, 3).into_boxed_slice());
        let demo: &'static FpnDemoWeights = Box::leak(Box::new(chain_weights()));
        suite.compare("fpn_chain/256x256", |g| {
            g.throughput(Throughput::Elements((w * h) as u64));
            for (arm, simd) in [(tier, true), ("scalar", false)] {
                g.bench(arm, move |b| {
                    b.with_input(move || {
                        set_simd(simd);
                        (
                            ScratchArena::new(512, 512, 16, 48, 48),
                            Vec::with_capacity(64),
                        )
                    })
                    .run(move |(mut arena, mut peaks)| {
                        fpn_demo_forward(inp, w, h, &mut arena, demo, 0.4, &mut peaks);
                        (arena, peaks)
                    })
                });
            }
        });
    }
}

fn run_dw(
    c: usize,
    stride: usize,
    dil: usize,
    inp: &[f32],
    w: usize,
    h: usize,
    pw: &[f32],
    b: &[f32],
    out: &mut [f32],
) {
    match (c, stride, dil) {
        (24, 1, 1) => depthwise_conv3x3_nhwc::<24, 1, 1>(inp, w, h, pw, b, Activation::ReLU6, out),
        (24, 2, 1) => depthwise_conv3x3_nhwc::<24, 2, 1>(inp, w, h, pw, b, Activation::ReLU6, out),
        (48, 1, 2) => depthwise_conv3x3_nhwc::<48, 1, 2>(inp, w, h, pw, b, Activation::ReLU6, out),
        (48, 1, 4) => depthwise_conv3x3_nhwc::<48, 1, 4>(inp, w, h, pw, b, Activation::ReLU6, out),
        _ => unreachable!(),
    }
}

zenbench::main!(bench_kernels);
