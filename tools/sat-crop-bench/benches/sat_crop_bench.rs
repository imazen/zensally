//! Bench harness for the zensally v0.2 Phase-1 crop path.
//!
//!   * `prep/*`: fused `preprocess_nhwc_strided` 1920×1080 → 512×384 with
//!     simultaneous OKLab/Sobel classical energy + horizon extraction.
//!   * `sat/build/*`: `DualIntegralMap::build` on the 64×48 stride-8 grid.
//!   * `crop/{minimal,maximal}/{w:h}`: `solve_crop_sat` for six aspect
//!     ratios in both modes. Every solve must stay well under the 80 µs
//!     search budget.
//!
//! `--quick` is a harness-local flag: zenbench rejects unknown args, so we
//! re-exec this binary with it stripped and a 3-round plan.

use zenbench::prelude::*;

use sat_crop_bench::PixelFormat;
use sat_crop_bench::preprocess::{Normalization, PreprocessConfig, ResizeMode};
use sat_crop_bench::sat_crop::{
    DualIntegralMap, IntegralChannels, SatCropConfig, SatCropMode, TargetToSrc, solve_crop_sat,
};
use sat_crop_bench::stream_prep::{PrepOutputs, prep_scratch_len, preprocess_nhwc_strided};

fn lcg_bytes(n: usize) -> Vec<u8> {
    let mut s = 0x9e37_79b9u32;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (s >> 24) as u8
        })
        .collect()
}

/// Deterministic stride-8 maps: noisy saliency floor, a subject matte blob
/// with a contour, a landmark peak, classical texture energy.
fn channels(gw: usize, gh: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut s = 0x1357_9bdfu32;
    let mut next = || {
        s = s.wrapping_mul(2_246_822_519).wrapping_add(1);
        ((s >> 24) as f32) / 255.0
    };
    let n = gw * gh;
    let mut sal = vec![0.0f32; n];
    let mut matte = vec![0.0f32; n];
    let mut lm = vec![0.0f32; n];
    let mut cls = vec![0.0f32; n];
    let mut cut = vec![0.0f32; n];
    for y in 0..gh {
        for x in 0..gw {
            let i = y * gw + x;
            sal[i] = next() * 0.2;
            cls[i] = next() * 0.4;
            // Subject torso: cells (24..40) x (14..38) — center frame.
            if (24..40).contains(&x) && (14..38).contains(&y) {
                matte[i] = 0.85;
                sal[i] += 0.5;
            }
            // Face core box → explicit cut penalty ring.
            if (28..36).contains(&x) && (14..24).contains(&y) {
                lm[i] = 0.9;
                cut[i] = 30.0;
            }
        }
    }
    (sal, matte, lm, cls, cut)
}

fn bench_prep(suite: &mut Suite) {
    let (sw, sh, tw, th) = (1920usize, 1080usize, 512usize, 384usize);
    let px: &'static [u8] = Box::leak(lcg_bytes(sw * sh * 3).into_boxed_slice());
    let stride = sw * 3;
    let (gw, gh) = (tw / 8, th / 8);
    let cfg = PreprocessConfig {
        target_w: tw,
        target_h: th,
        mode: ResizeMode::Stretch,
        norm: Normalization::UnitScale,
    };

    suite.compare("prep/1920x1080->512x384", |g| {
        g.throughput(Throughput::Elements((sw * sh) as u64));
        g.bench("nhwc_fused", move |b| {
            let mut out = vec![0.0f32; 3 * tw * th];
            let mut energy = vec![0.0f32; gw * gh];
            let mut horizon = vec![0.0f32; gh];
            let mut scratch = vec![0.0f32; prep_scratch_len(tw, gw, gh)];
            b.iter(move || {
                let mut outs = PrepOutputs {
                    output_nhwc: &mut out,
                    classical_energy: &mut energy,
                    horizon_profile: &mut horizon,
                    scratch: &mut scratch,
                };
                preprocess_nhwc_strided(
                    px,
                    sw as u32,
                    sh as u32,
                    stride,
                    PixelFormat::Rgb,
                    &cfg,
                    &mut outs,
                )
            })
        });
    });
}

fn bench_sat(suite: &mut Suite) {
    let (gw, gh) = (64usize, 48usize);
    let (sal, matte, lm, cls, cut) = channels(gw, gh);
    // Bench arms must be 'static: leak the input maps, channels struct,
    // built map and config.
    let ch: &'static IntegralChannels = Box::leak(Box::new(IntegralChannels {
        saliency: Some(Box::leak(sal.into_boxed_slice())),
        matte: Some(Box::leak(matte.into_boxed_slice())),
        landmark_weights: Some(Box::leak(lm.into_boxed_slice())),
        classical_energy: Some(Box::leak(cls.into_boxed_slice())),
        cut_penalty: Some(Box::leak(cut.into_boxed_slice())),
        horizon_profile: None,
    }));

    suite.compare("sat/build/64x48", |g| {
        g.bench("dual_sat", move |b| {
            b.iter(|| DualIntegralMap::build(ch, gw, gh, 0.25))
        });
    });

    let map: &'static DualIntegralMap =
        Box::leak(Box::new(DualIntegralMap::build(ch, gw, gh, 0.25)));
    let to_src = TargetToSrc::stretch(1920, 1080, 512, 384);
    let cfg: &'static SatCropConfig = Box::leak(Box::new(SatCropConfig::default()));
    let aspects: [(u32, u32); 6] = [(1, 1), (16, 9), (9, 16), (4, 3), (3, 4), (4, 5)];
    for (aw, ah) in aspects {
        for (mname, mode) in [
            ("minimal", SatCropMode::Minimal),
            ("maximal", SatCropMode::Maximal),
        ] {
            suite.compare(format!("crop/{mname}/{aw}x{ah}"), |g| {
                g.bench("solve_sat", move |b| {
                    b.iter(|| solve_crop_sat(map, &to_src, (aw, ah), mode, cfg))
                });
            });
        }
    }
}

fn main() {
    // Exploratory-only harness: on asymmetric (P/E hybrid) hosts zenbench
    // refuses to run without an explicit policy. Default to unrestricted;
    // `--cpu-policy=` / env still override.
    if std::env::var_os("ZENBENCH_CPU_POLICY").is_none() {
        // SAFETY: single-threaded before zenbench spawns workers.
        unsafe { std::env::set_var("ZENBENCH_CPU_POLICY", "unrestricted") };
    }

    // zenbench rejects unknown harness args; `--quick` re-execs this binary
    // without it plus a 3-round plan.
    let args: Vec<String> = std::env::args().collect();
    if args.iter().any(|a| a == "--quick") {
        let status = std::process::Command::new(&args[0])
            .args(
                args[1..]
                    .iter()
                    .filter(|a| a.as_str() != "--quick")
                    .map(String::as_str)
                    .chain(std::iter::once("--rounds=3")),
            )
            .status()
            .expect("re-exec bench without --quick");
        std::process::exit(status.code().unwrap_or(1));
    }

    let result = zenbench::run(|suite| {
        bench_prep(suite);
        bench_sat(suite);
    });
    zenbench::postprocess_result(&result);
}
