//! Study-only image-space view projection; production model API stays fixed.
use super::Result;
use std::borrow::Cow;
use zensally::{
    PixelFormat,
    preprocess::{Normalization, ResizeMode, preprocess_nchw},
};
use zensally_micro::{Input, MicroSalNetW24};

pub const GRID: usize = 256;
#[derive(Clone)]
pub struct Image {
    pub rgb: Vec<u8>,
    pub w: usize,
    pub h: usize,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Window {
    pub x: usize,
    pub y: usize,
    pub w: usize,
    pub h: usize,
}
#[derive(Clone, Copy)]
pub struct View<'a> {
    pub data: &'a [u8],
    pub w: usize,
    pub h: usize,
    pub stride: usize,
}
impl Image {
    pub fn view(&self) -> View<'_> {
        View {
            data: &self.rgb,
            w: self.w,
            h: self.h,
            stride: self.w * 3,
        }
    }
    pub fn whole(&self) -> Window {
        Window {
            x: 0,
            y: 0,
            w: self.w,
            h: self.h,
        }
    }
}
impl<'a> View<'a> {
    pub fn crop(self, r: Window) -> Self {
        assert!(r.w > 0 && r.h > 0 && r.x + r.w <= self.w && r.y + r.h <= self.h);
        Self {
            data: &self.data[r.y * self.stride + r.x * 3..],
            w: r.w,
            h: r.h,
            stride: self.stride,
        }
    }
    pub fn packed(self) -> Cow<'a, [u8]> {
        if self.stride == self.w * 3 {
            return Cow::Borrowed(&self.data[..self.w * self.h * 3]);
        }
        let mut p = Vec::with_capacity(self.w * self.h * 3);
        for row in 0..self.h {
            p.extend_from_slice(&self.data[row * self.stride..][..self.w * 3]);
        }
        Cow::Owned(p)
    }
    pub fn resized(self, w: usize, h: usize) -> Image {
        let mut planar = vec![0.; 3 * w * h];
        preprocess_nchw(
            &self.packed(),
            self.w as u32,
            self.h as u32,
            PixelFormat::Rgb,
            w,
            h,
            ResizeMode::Stretch,
            Normalization::UnitScale,
            &mut planar,
        );
        let mut rgb = vec![0; w * h * 3];
        for i in 0..w * h {
            for c in 0..3 {
                rgb[i * 3 + c] = (planar[c * w * h + i] * 255.).round().clamp(0., 255.) as u8;
            }
        }
        Image { rgb, w, h }
    }
}

// Bilinear point projection follows the owned core preprocessor's four-sample
// interpolation. This maps coordinates; RGB resize itself uses that preprocessor.
pub fn sample(data: &[f32], w: usize, h: usize, x: f32, y: f32) -> f32 {
    let x = x.clamp(0., (w - 1) as f32);
    let y = y.clamp(0., (h - 1) as f32);
    let x0 = x.floor() as usize;
    let y0 = y.floor() as usize;
    let x1 = (x0 + 1).min(w - 1);
    let y1 = (y0 + 1).min(h - 1);
    let fx = x - x0 as f32;
    let fy = y - y0 as f32;
    (data[y0 * w + x0] * (1. - fx) + data[y0 * w + x1] * fx) * (1. - fy)
        + (data[y1 * w + x0] * (1. - fx) + data[y1 * w + x1] * fx) * fy
}
#[derive(Clone)]
pub struct Prediction {
    pub data: Vec<f32>,
    pub views: Vec<Window>,
}
pub struct Runner {
    model: MicroSalNetW24,
    input: Vec<f32>,
}
impl Runner {
    pub fn new() -> Result<Self> {
        Ok(Self {
            model: MicroSalNetW24::new()?.with_fast_sigmoid(),
            input: vec![0.; 3 * 256 * 256],
        })
    }
    fn view(&mut self, v: View<'_>, mode: u8) -> Result<Vec<f32>> {
        let info = preprocess_nchw(
            &v.packed(),
            v.w as u32,
            v.h as u32,
            PixelFormat::Rgb,
            256,
            256,
            if mode == 0 {
                ResizeMode::Stretch
            } else {
                ResizeMode::Letterbox
            },
            Normalization::UnitScale,
            &mut self.input,
        );
        let (rw, rh) = if mode == 0 {
            (256, 256)
        } else {
            (
                ((v.w as f32 * info.ratio).round() as usize).max(1),
                ((v.h as f32 * info.ratio).round() as usize).max(1),
            )
        };
        if mode == 2 {
            let packed = v.packed();
            let mut mean = [0.; 3];
            for row in packed.as_chunks::<3>().0 {
                for c in 0..3 {
                    mean[c] += row[c] as f32;
                }
            }
            for m in &mut mean {
                *m /= (v.w * v.h * 255) as f32;
            }
            for (c, &value) in mean.iter().enumerate() {
                for y in 0..256 {
                    for x in 0..256 {
                        if x < info.pad_left as usize
                            || x >= info.pad_left as usize + rw
                            || y < info.pad_top as usize
                            || y >= info.pad_top as usize + rh
                        {
                            self.input[c * 65536 + y * 256 + x] = value;
                        }
                    }
                }
            }
        }
        let map = self.model.infer(Input::contiguous(&self.input)?)?;
        let mut out = vec![0.; GRID * GRID];
        for y in 0..GRID {
            for x in 0..GRID {
                // Align-corners source resize: undo the actual rounded content size,
                // rather than the unrounded scale in LetterboxInfo.
                let px = info.pad_left + x as f32 / (GRID - 1) as f32 * (rw - 1) as f32;
                let py = info.pad_top + y as f32 / (GRID - 1) as f32 * (rh - 1) as f32;
                out[y * GRID + x] =
                    sample(&map.data, 128, 128, px * 127. / 255., py * 127. / 255.).clamp(0., 1.);
            }
        }
        Ok(out)
    }
    pub fn predict(&mut self, image: &Image) -> Result<Vec<(&'static str, Prediction)>> {
        let global = self.view(image.view(), 0)?;
        let mut result = vec![
            (
                "stretch",
                Prediction {
                    data: global.clone(),
                    views: vec![image.whole()],
                },
            ),
            (
                "letterbox",
                Prediction {
                    data: self.view(image.view(), 1)?,
                    views: vec![image.whole()],
                },
            ),
            (
                "letterbox-mean",
                Prediction {
                    data: self.view(image.view(), 2)?,
                    views: vec![image.whole()],
                },
            ),
        ];
        let mut quadrants = Vec::new();
        for (y, h) in [(0, image.h / 2), (image.h / 2, image.h - image.h / 2)] {
            for (x, w) in [(0, image.w / 2), (image.w / 2, image.w - image.w / 2)] {
                if w > 0 && h > 0 {
                    quadrants.push(Window { x, y, w, h });
                }
            }
        }
        result.push((
            "quadrants",
            self.combine(image, &global, quadrants, false, false)?,
        ));
        let views = selected_views(image.w, image.h, &global);
        result.push((
            "overlap-refine",
            self.combine(image, &global, views, true, true)?,
        ));
        let views = cover_views(image.w, image.h);
        result.push((
            "aspect-refine",
            self.combine(image, &global, views, true, true)?,
        ));
        Ok(result)
    }
    fn combine(
        &mut self,
        image: &Image,
        global: &[f32],
        views: Vec<Window>,
        taper: bool,
        guided: bool,
    ) -> Result<Prediction> {
        let mut sums = vec![0.; GRID * GRID];
        let mut weights = vec![0.; GRID * GRID];
        for &r in &views {
            let local = self.view(image.view().crop(r), if r.w == r.h { 0 } else { 1 })?;
            project_add(&local, image.w, image.h, r, taper, &mut sums, &mut weights);
        }
        let data = fuse(global, &sums, &weights, guided);
        Ok(Prediction { data, views })
    }
}
pub fn fuse(global: &[f32], sums: &[f32], weights: &[f32], guided: bool) -> Vec<f32> {
    global
        .iter()
        .zip(sums)
        .zip(weights)
        .map(|((&g, &s), &w)| {
            if w == 0. {
                g
            } else {
                let l = s / w;
                if guided { g + 0.5 * g * (l - g) } else { l }
            }
        })
        .collect()
}
fn project_add(
    local: &[f32],
    w: usize,
    h: usize,
    r: Window,
    taper: bool,
    sums: &mut [f32],
    weights: &mut [f32],
) {
    for gy in 0..GRID {
        let sy = gy as f32 * (h - 1) as f32 / (GRID - 1) as f32;
        if sy < r.y as f32 - 0.5 || sy >= (r.y + r.h) as f32 - 0.5 {
            continue;
        }
        let v = ((sy - r.y as f32) / (r.h - 1).max(1) as f32).clamp(0., 1.);
        for gx in 0..GRID {
            let sx = gx as f32 * (w - 1) as f32 / (GRID - 1) as f32;
            if sx < r.x as f32 - 0.5 || sx >= (r.x + r.w) as f32 - 0.5 {
                continue;
            }
            let u = ((sx - r.x as f32) / (r.w - 1).max(1) as f32).clamp(0., 1.);
            let weight = if taper {
                (4. * u.min(1. - u)).clamp(0.05, 1.) * (4. * v.min(1. - v)).clamp(0.05, 1.)
            } else {
                1.
            };
            let i = gy * GRID + gx;
            sums[i] += weight
                * sample(
                    local,
                    GRID,
                    GRID,
                    u * (GRID - 1) as f32,
                    v * (GRID - 1) as f32,
                );
            weights[i] += weight;
        }
    }
}
fn starts(length: usize, side: usize) -> Vec<usize> {
    if length <= side {
        return vec![0];
    }
    let n = (length - side).div_ceil((side / 2).max(1)) + 1;
    (0..n).map(|i| i * (length - side) / (n - 1)).collect()
}
pub fn cover_views(w: usize, h: usize) -> Vec<Window> {
    let (long, short) = (w.max(h), w.min(h));
    let side = short.max((2 * long + 15).div_ceil(17)).min(long);
    starts(long, side)
        .into_iter()
        .map(|p| {
            if w >= h {
                Window {
                    x: p,
                    y: 0,
                    w: side,
                    h,
                }
            } else {
                Window {
                    x: 0,
                    y: p,
                    w,
                    h: side,
                }
            }
        })
        .collect()
}
fn iou(a: Window, b: Window) -> f32 {
    let ix = (a.x + a.w).min(b.x + b.w).saturating_sub(a.x.max(b.x));
    let iy = (a.y + a.h).min(b.y + b.h).saturating_sub(a.y.max(b.y));
    let overlap = (ix * iy) as f32;
    overlap / ((a.w * a.h + b.w * b.h) as f32 - overlap).max(1.)
}
pub fn selected_views(w: usize, h: usize, global: &[f32]) -> Vec<Window> {
    let side = ((w.min(h) * 5) / 8).max(1);
    let mut candidates = Vec::new();
    for y in starts(h, side) {
        for x in starts(w, side) {
            let r = Window {
                x,
                y,
                w: side,
                h: side,
            };
            let mut sum = 0.;
            let mut n = 0;
            for gy in 0..32 {
                for gx in 0..32 {
                    let px = (x as f32 + (gx as f32 + 0.5) * side as f32 / 32.) / w as f32;
                    let py = (y as f32 + (gy as f32 + 0.5) * side as f32 / 32.) / h as f32;
                    let g = sample(
                        global,
                        GRID,
                        GRID,
                        px * GRID as f32 - 0.5,
                        py * GRID as f32 - 0.5,
                    );
                    sum += g + 4. * g * (1. - g);
                    n += 1;
                }
            }
            candidates.push((r, sum / n as f32));
        }
    }
    let mut chosen = Vec::new();
    while !candidates.is_empty() && chosen.len() < 4 {
        let index = candidates
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| {
                let score = |v: &(Window, f32)| {
                    chosen
                        .iter()
                        .fold(v.1, |s, &r| s * (1. - 0.5 * iou(v.0, r)))
                };
                score(a).total_cmp(&score(b))
            })
            .unwrap()
            .0;
        chosen.push(candidates.swap_remove(index).0);
    }
    chosen
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn covering_views_have_no_gaps_and_respect_budget() {
        for (w, h) in [
            (512, 512),
            (1024, 128),
            (128, 1024),
            (8192, 128),
            (17, 3),
            (1, 1),
        ] {
            let views = cover_views(w, h);
            assert!(views.len() <= 16);
            for y in 0..h {
                for x in 0..w {
                    assert!(
                        views
                            .iter()
                            .any(|r| x >= r.x && x < r.x + r.w && y >= r.y && y < r.y + r.h)
                    );
                }
            }
        }
    }
    #[test]
    fn translated_view_projects_ramp_in_source_coordinates() {
        let local: Vec<f32> = (0..GRID * GRID)
            .map(|i| (i % GRID) as f32 / (GRID - 1) as f32)
            .collect();
        let mut sums = vec![0.; GRID * GRID];
        let mut weights = sums.clone();
        let r = Window {
            x: 256,
            y: 0,
            w: 512,
            h: 512,
        };
        project_add(&local, 1024, 512, r, true, &mut sums, &mut weights);
        for x in 65..190 {
            let source_x = x as f32 * 1023. / 255.;
            let expected = (source_x - 256.) / 511.;
            let i = 100 * GRID + x;
            assert!((sums[i] / weights[i] - expected).abs() < 1e-5);
        }
        assert_eq!(weights[100 * GRID], 0.);
        assert_eq!(weights[100 * GRID + 255], 0.);
    }
    #[test]
    fn odd_cover_windows_keep_the_sixteen_view_bound() {
        for long in 1..1024 {
            for short in 1..=long.min(32) {
                assert!(cover_views(long, short).len() <= 16, "{long}x{short}");
            }
        }
    }
    #[test]
    fn coordinate_projection_and_overlap_preserve_constant_fields() {
        let r = Window {
            x: 0,
            y: 0,
            w: 512,
            h: 512,
        };
        let mut sums = vec![0.; GRID * GRID];
        let mut weights = sums.clone();
        for _ in 0..4 {
            project_add(
                &vec![0.75; GRID * GRID],
                512,
                512,
                r,
                true,
                &mut sums,
                &mut weights,
            );
        }
        let actual = fuse(&vec![0.25; GRID * GRID], &sums, &weights, false);
        assert!(actual.iter().all(|&v| (v - 0.75).abs() < 1e-6));
        assert_eq!(
            fuse(&[0., 0.25], &[1., 0.], &[1., 0.], true),
            vec![0., 0.25]
        );
    }
    #[test]
    fn strided_crop_reads_only_selected_rows() {
        let rgb: Vec<u8> = (0..6 * 5 * 3).map(|i| i as u8).collect();
        let image = Image { rgb, w: 6, h: 5 };
        let crop = image
            .view()
            .crop(Window {
                x: 2,
                y: 1,
                w: 2,
                h: 3,
            })
            .packed();
        let expected: Vec<u8> = [24..30, 42..48, 60..66].into_iter().flatten().collect();
        assert_eq!(&*crop, expected);
    }
    #[test]
    fn bilinear_projection_reproduces_a_ramp() {
        assert_eq!(sample(&[0., 1., 2., 3.], 2, 2, 0.25, 0.5), 1.25);
    }
}
