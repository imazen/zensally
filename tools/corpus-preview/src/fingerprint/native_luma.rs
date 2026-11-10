//! Reused from imazen/zensimx ed956176e02e335d4a6efb3c31724e2aaf361d34; original kernel and tests.
use super::dhash_bits;
use anyhow::{Context, Result, ensure};
use zenpixels_convert::PixelDescriptor;

pub(super) fn fingerprint_luma(luma: &mut [f32], width: u32, height: u32) -> Result<(u64, f32)> {
    ensure!(
        width >= 9 && height >= 8 && luma.len() == width as usize * height as usize,
        "luma geometry"
    );
    ensure!(luma.iter().all(|v| v.is_finite()), "nonfinite luminance");
    let peak = luma.iter().copied().fold(0.0f32, f32::max);
    for value in &mut *luma {
        *value = if peak > 0.0 {
            (255.0 * (value.max(0.0) / peak)).ln_1p()
        } else {
            0.0
        };
    }
    let config = zenresize::ResizeConfig::builder(width, height, 9, 8)
        .filter(zenresize::Filter::Lanczos)
        .format(PixelDescriptor::GRAYF32_LINEAR)
        .build();
    let small = zenresize::Resizer::new(&config).resize_f32(luma);
    ensure!(small.iter().all(|v| v.is_finite()), "nonfinite resample");
    Ok((
        dhash_bits::from_luma9x8(small.as_slice().try_into().context("linear 9x8")?),
        peak,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exposure_invariance_and_unrelated_control() {
        let base: Vec<f32> = (0..64 * 64)
            .map(|i| (((i / 64) * 13 + (i % 64) * 7) % 257) as f32)
            .collect();
        let expected = fingerprint_luma(&mut base.clone(), 64, 64).unwrap().0;
        for exposure in [0.125, 1.0, 8.0, 1024.0] {
            let mut scaled: Vec<_> = base.iter().map(|v| v * exposure).collect();
            assert_eq!(fingerprint_luma(&mut scaled, 64, 64).unwrap().0, expected);
        }
        let mut other: Vec<_> = base.iter().rev().copied().collect();
        assert!((fingerprint_luma(&mut other, 64, 64).unwrap().0 ^ expected).count_ones() > 16);
    }

    #[test]
    fn black_is_defined_and_nonfinite_or_wrong_geometry_fails() {
        assert_eq!(fingerprint_luma(&mut [0.0; 72], 9, 8).unwrap(), (0, 0.0));
        assert!(fingerprint_luma(&mut [f32::NAN; 72], 9, 8).is_err());
        assert!(fingerprint_luma(&mut [f32::INFINITY; 72], 9, 8).is_err());
        assert!(fingerprint_luma(&mut [0.0; 72], 8, 9).is_err());
    }
}
