//! Study-only padding. Running sums adapted from owned zenmetrics
//! a3329089, crates/margarine/box_blur.rs; cropped, renormalized boundaries.
use super::geometry::{View, Window};
use zensally::{
    PixelFormat,
    preprocess::{Normalization, ResizeMode, preprocess_nchw},
};

fn box_plane(src: &[f32], w: usize, h: usize, stride: usize, radius: usize) -> Vec<f32> {
    let mut horizontal = vec![0.; w * h];
    for y in 0..h {
        let row = &src[y * stride..][..w];
        let mut sum: f64 = row[..(radius + 1).min(w)]
            .iter()
            .map(|&v| f64::from(v))
            .sum();
        for x in 0..w {
            let lo = x.saturating_sub(radius);
            let hi = (x + radius + 1).min(w);
            horizontal[y * w + x] = (sum / (hi - lo) as f64) as f32;
            if x >= radius {
                sum -= f64::from(row[x - radius]);
            }
            if x + radius + 1 < w {
                sum += f64::from(row[x + radius + 1]);
            }
        }
    }
    let mut output = vec![0.; w * h];
    let mut sums = vec![0f64; w];
    for y in 0..(radius + 1).min(h) {
        for (sum, &v) in sums.iter_mut().zip(&horizontal[y * w..][..w]) {
            *sum += f64::from(v);
        }
    }
    for y in 0..h {
        let count = (y + radius + 1).min(h) - y.saturating_sub(radius);
        for (out, &sum) in output[y * w..][..w].iter_mut().zip(&sums) {
            *out = (sum / count as f64) as f32;
        }
        if y >= radius {
            for (sum, &v) in sums.iter_mut().zip(&horizontal[(y - radius) * w..][..w]) {
                *sum -= f64::from(v);
            }
        }
        if y + radius + 1 < h {
            for (sum, &v) in sums
                .iter_mut()
                .zip(&horizontal[(y + radius + 1) * w..][..w])
            {
                *sum += f64::from(v);
            }
        }
    }
    output
}

pub fn fill(view: View<'_>, input: &mut [f32], fitted: Window, cover: bool) {
    const SIZE: usize = 256;
    const N: usize = SIZE * SIZE;
    let mut background = vec![0.; 3 * N];
    if cover {
        let side = view.w.min(view.h);
        let square = view.crop(Window {
            x: (view.w - side) / 2,
            y: (view.h - side) / 2,
            w: side,
            h: side,
        });
        preprocess_nchw(
            &square.packed(),
            side as u32,
            side as u32,
            PixelFormat::Rgb,
            SIZE,
            SIZE,
            ResizeMode::Stretch,
            Normalization::UnitScale,
            &mut background,
        );
    } else {
        for c in 0..3 {
            for y in 0..SIZE {
                for x in 0..SIZE {
                    let sx = x.clamp(fitted.x, fitted.x + fitted.w - 1);
                    let sy = y.clamp(fitted.y, fitted.y + fitted.h - 1);
                    background[c * N + y * SIZE + x] = input[c * N + sy * SIZE + sx];
                }
            }
        }
    }
    for c in 0..3 {
        let blurred = box_plane(&background[c * N..][..N], SIZE, SIZE, SIZE, 16);
        for y in 0..SIZE {
            for x in 0..SIZE {
                if x < fitted.x
                    || x >= fitted.x + fitted.w
                    || y < fitted.y
                    || y >= fitted.y + fitted.h
                {
                    input[c * N + y * SIZE + x] = blurred[y * SIZE + x];
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn strided_running_sum_matches_direct_clipped_box() {
        for (w, h, r) in [(1, 1, 7), (3, 5, 9), (17, 19, 3), (7, 9, 0)] {
            let stride = w + 5;
            let mut input = vec![f32::NAN; stride * h];
            for y in 0..h {
                for x in 0..w {
                    input[y * stride + x] = ((x * 17 + y * 3) % 23) as f32;
                }
            }
            let got = box_plane(&input, w, h, stride, r);
            for y in 0..h {
                for x in 0..w {
                    let mut sum = 0f64;
                    let mut count = 0;
                    for yy in y.saturating_sub(r)..(y + r + 1).min(h) {
                        for xx in x.saturating_sub(r)..(x + r + 1).min(w) {
                            sum += f64::from(input[yy * stride + xx]);
                            count += 1;
                        }
                    }
                    assert!((got[y * w + x] as f64 - sum / count as f64).abs() < 2e-6);
                }
            }
        }
    }
    #[test]
    fn padding_preserves_every_fitted_foreground_value() {
        let image = [90u8; 7 * 9 * 3];
        let view = View {
            data: &image,
            w: 7,
            h: 9,
            stride: 21,
        };
        let fitted = Window {
            x: 30,
            y: 0,
            w: 196,
            h: 256,
        };
        for cover in [false, true] {
            let mut input = vec![0.; 3 * 65536];
            for c in 0..3 {
                for y in 0..256 {
                    for x in fitted.x..fitted.x + fitted.w {
                        input[c * 65536 + y * 256 + x] = (x + y + c) as f32 / 1024.;
                    }
                }
            }
            let original = input.clone();
            fill(view, &mut input, fitted, cover);
            for c in 0..3 {
                for y in 0..256 {
                    for x in fitted.x..fitted.x + fitted.w {
                        assert_eq!(
                            input[c * 65536 + y * 256 + x].to_bits(),
                            original[c * 65536 + y * 256 + x].to_bits()
                        );
                    }
                }
            }
            assert!(
                input
                    .iter()
                    .all(|v| v.is_finite() && (0.0..=1.0).contains(v))
            );
        }
    }
}
