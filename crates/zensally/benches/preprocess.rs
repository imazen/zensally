//! Cost of `preprocess_nchw` at the model input sizes zensally actually uses.
//!
//! This runs once per image before ONNX inference. It is fully scalar: a
//! bilinear resample plus normalize plus NCHW planar scatter, per pixel. The
//! question this answers is not "can it be made faster" (it can) but "is it a
//! meaningful share of a detect() call" — inference on these models is tens of
//! milliseconds, so a preprocessing win only matters if preprocessing is a
//! non-trivial fraction of that.
//!
//! Run: `cargo bench -p zensally --bench preprocess`

use zenbench::prelude::*;
use zensally::PixelFormat;
use zensally::preprocess::{Normalization, ResizeMode, preprocess_nchw};

fn src(w: usize, h: usize, bpp: usize) -> Vec<u8> {
    let mut s = 0x9e37_79b9u32;
    (0..w * h * bpp)
        .map(|_| {
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (s >> 24) as u8
        })
        .collect()
}

fn bench_preprocess(suite: &mut Suite) {
    // Source sizes span a phone photo down to a thumbnail; targets are the
    // real model inputs (UltraFace RFB-320 is 320x240, saliency nets 320x320).
    for &(slabel, sw, sh) in &[("4032x3024", 4032usize, 3024usize), ("1920x1080", 1920, 1080)] {
        let px: &'static [u8] = Box::leak(src(sw, sh, 4).into_boxed_slice());
        for &(tlabel, tw, th) in &[("320x240", 320usize, 240usize), ("320x320", 320, 320)] {
            suite.compare(format!("preprocess/{slabel}->{tlabel}"), |g| {
                g.throughput(Throughput::Elements((tw * th) as u64));
                g.bench("nchw", move |b| {
                    let mut out = vec![0.0f32; 3 * tw * th];
                    b.iter(move || {
                        preprocess_nchw(
                            px,
                            sw as u32,
                            sh as u32,
                            PixelFormat::Rgba,
                            tw,
                            th,
                            ResizeMode::Letterbox,
                            Normalization::UnitScale,
                            &mut out,
                        )
                    })
                });
            });
        }
    }
}

zenbench::main!(bench_preprocess);
