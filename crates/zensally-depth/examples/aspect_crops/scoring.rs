use super::{
    Result,
    geometry::{GRID, Image, Prediction},
};
use rgb::FromSlice;
use std::{fs, io::Write, path::Path};
use zenlayout::{
    Rect,
    smart_crop::{AspectRatio, CropConfig, CropMode, HeatMap, SmartCropInput},
};

pub fn read_image(path: &Path) -> Result<Image> {
    let bytes = fs::read(path)?;
    let decoded = zenjpeg::decoder::Decoder::new()
        .auto_orient(false)
        .output_format(zenjpeg::decoder::PixelFormat::Rgb)
        .decode(&bytes, enough::Unstoppable)?;
    Ok(Image {
        rgb: decoded.pixels_u8().ok_or("RGB8 JPEG required")?.to_vec(),
        w: decoded.width as usize,
        h: decoded.height as usize,
    })
}
pub fn read_mask(path: &Path, w: usize, h: usize) -> Result<Vec<u8>> {
    let bytes = fs::read(path)?;
    let mut config = zenpng::PngDecodeConfig::strict();
    config.max_pixels = Some(64 * 1024 * 1024);
    config.max_memory_bytes = Some(512 * 1024 * 1024);
    let decoded = zenpng::decode(&bytes, &config, &enough::Unstoppable)?;
    let pixels = decoded.pixels.as_slice();
    if pixels.width() as usize != w || pixels.rows() as usize != h {
        return Err("mask/image dimension mismatch".into());
    }
    if pixels.descriptor().channel_type() != zenpixels::ChannelType::U8 {
        return Err("8-bit mask required".into());
    }
    let channels = pixels.descriptor().channels();
    let mut result = vec![0; w * h];
    for y in 0..h {
        let row = pixels.row(y as u32);
        for x in 0..w {
            // Same BT.601 integer luma convention used by our JPEG gray
            // conversion (zenjpeg colorspace_paths), checked against Pillow
            // convert("L") below. A few frozen DUTS masks contain RGB noise.
            let luma = if channels >= 3 {
                let p = &row[x * channels..][..3];
                (19595 * u32::from(p[0]) + 38470 * u32::from(p[1]) + 7471 * u32::from(p[2]) + 32768)
                    >> 16
            } else {
                u32::from(row[x * channels])
            };
            result[y * w + x] = if luma >= 128 { 255 } else { 0 };
        }
    }
    if !result.iter().any(|&x| x != 0) {
        return Err("empty human foreground mask".into());
    }
    Ok(result)
}
pub fn grid_mask(mask: &[u8], w: usize, h: usize) -> Vec<f32> {
    (0..GRID * GRID)
        .map(|i| {
            let x = ((i % GRID) * (w - 1) + (GRID - 1) / 2) / (GRID - 1);
            let y = ((i / GRID) * (h - 1) + (GRID - 1) / 2) / (GRID - 1);
            if mask[y * w + x] > 0 { 1. } else { 0. }
        })
        .collect()
}
pub fn write_png(path: &Path, image: &Image) -> Result<()> {
    let mut config = zenpng::EncodeConfig::default().with_compression(zenpng::Compression::Fastest);
    config.parallel = false;
    let img = imgref::ImgRef::new(image.rgb.as_rgb(), image.w, image.h);
    fs::write(
        path,
        zenpng::encode_rgb8(
            img,
            None,
            &config,
            &enough::Unstoppable,
            &enough::Unstoppable,
        )?,
    )?;
    Ok(())
}
pub fn heat_image(map: &[f32]) -> Image {
    let mut rgb = Vec::with_capacity(GRID * GRID * 3);
    for &p in map {
        let q = p.clamp(0., 1.);
        rgb.extend_from_slice(&[
            (255. * q) as u8,
            (180. * (1. - (2. * q - 1.).abs())) as u8,
            (255. * (1. - q)) as u8,
        ]);
    }
    Image {
        rgb,
        w: GRID,
        h: GRID,
    }
}
struct Integral {
    data: Vec<u32>,
}
impl Integral {
    fn new(mask: &[f32]) -> Self {
        let stride = GRID + 1;
        let mut data = vec![0; stride * stride];
        for y in 0..GRID {
            let mut row = 0;
            for x in 0..GRID {
                row += u32::from(mask[y * GRID + x] >= 0.5);
                data[(y + 1) * stride + x + 1] = data[y * stride + x + 1] + row;
            }
        }
        Self { data }
    }
    fn sum(&self, x: usize, y: usize, w: usize, h: usize) -> u32 {
        let s = GRID + 1;
        self.data[(y + h) * s + x + w] + self.data[y * s + x]
            - self.data[y * s + x + w]
            - self.data[(y + h) * s + x]
    }
    fn best(&self, w: usize, h: usize) -> u32 {
        let mut best = 0;
        for y in 0..=GRID - h {
            for x in 0..=GRID - w {
                best = best.max(self.sum(x, y, w, h));
            }
        }
        best
    }
}
fn grid_rect(r: Rect, w: usize, h: usize) -> (usize, usize, usize, usize) {
    let x = (r.x as usize * GRID / w).min(GRID - 1);
    let y = (r.y as usize * GRID / h).min(GRID - 1);
    let x1 = ((r.x + r.width) as usize * GRID / w).clamp(x + 1, GRID);
    let y1 = ((r.y + r.height) as usize * GRID / h).clamp(y + 1, GRID);
    (x, y, x1 - x, y1 - y)
}
pub fn score(
    image: &Image,
    mask: &[f32],
    pred: &Prediction,
    prefix: &str,
    method: &str,
    reports: (&mut dyn Write, &mut dyn Write),
    gallery: Option<&Path>,
) -> Result<()> {
    let (metrics, crops) = reports;
    if pred
        .data
        .iter()
        .any(|x| !x.is_finite() || !(0.0..=1.0).contains(x))
    {
        return Err("invalid saliency probability".into());
    }
    let mut mae = 0.;
    let (mut tp, mut fp, mut fn_) = (0usize, 0usize, 0usize);
    for (&p, &m) in pred.data.iter().zip(mask) {
        mae += (p - m).abs() as f64;
        match (p >= 0.5, m >= 0.5) {
            (true, true) => tp += 1,
            (true, false) => fp += 1,
            (false, true) => fn_ += 1,
            _ => {}
        }
    }
    let iou = tp as f64 / (tp + fp + fn_).max(1) as f64;
    let fbeta = 1.3 * tp as f64 / (1.3 * tp as f64 + 0.3 * fn_ as f64 + fp as f64).max(1.);
    let foreground = mask.iter().sum::<f32>() as f64;
    let integral = Integral::new(mask);
    let input = SmartCropInput {
        focus_regions: vec![],
        heatmap: Some(HeatMap {
            data: pred.data.clone(),
            width: GRID as u32,
            height: GRID as u32,
        }),
    };
    let mut retained = 0.;
    let mut regret = 0.;
    let mut worst = 1f64;
    let mut n = 0;
    if let Some(dir) = gallery {
        write_png(
            &dir.join(format!("{method}-map.png")),
            &heat_image(&pred.data),
        )?;
    }
    for (rw, rh) in [(1, 1), (16, 9), (9, 16), (4, 1), (1, 4)] {
        for mode in [CropMode::Minimal, CropMode::Maximal] {
            let config = CropConfig {
                target_aspect: AspectRatio { w: rw, h: rh },
                mode,
                ..Default::default()
            };
            let crop = input
                .compute_crop(image.w as u32, image.h as u32, &config)
                .ok_or("crop solver returned no crop")?;
            if crop.width == 0
                || crop.height == 0
                || crop.x + crop.width > image.w as u32
                || crop.y + crop.height > image.h as u32
            {
                return Err(format!("crop outside source: {prefix} method={method} target={rw}:{rh} mode={mode:?} rectangle={crop:?}").into());
            }
            let (x, y, w, h) = grid_rect(crop, image.w, image.h);
            let captured = integral.sum(x, y, w, h) as f64;
            let oracle = integral.best(w, h) as f64;
            let retention = captured / foreground.max(1.);
            let loss = (oracle - captured).max(0.) / foreground.max(1.);
            retained += retention;
            regret += loss;
            worst = worst.min(retention);
            n += 1;
            let mode = if mode == CropMode::Minimal {
                "minimal"
            } else {
                "maximal"
            };
            writeln!(
                crops,
                "{prefix}\t{method}\t{rw}:{rh}\t{mode}\t{}\t{}\t{}\t{}\t{retention:.9}\t{:.9}\t{loss:.9}",
                crop.x,
                crop.y,
                crop.width,
                crop.height,
                oracle / foreground.max(1.)
            )?;
            if let Some(dir) = gallery
                && ((rw, rh) == (1, 1) || (rw, rh) == (16, 9) || (rw, rh) == (9, 16))
            {
                let view = image.view().crop(super::geometry::Window {
                    x: crop.x as usize,
                    y: crop.y as usize,
                    w: crop.width as usize,
                    h: crop.height as usize,
                });
                let scale = 320f32 / view.w.max(view.h) as f32;
                let output = view.resized(
                    (view.w as f32 * scale).round().max(1.) as usize,
                    (view.h as f32 * scale).round().max(1.) as usize,
                );
                write_png(&dir.join(format!("{method}-{rw}x{rh}-{mode}.png")), &output)?;
            }
        }
    }
    let calls = if ["overlap-refine", "aspect-refine"].contains(&method) {
        1 + pred.views.len()
    } else {
        pred.views.len()
    };
    writeln!(
        metrics,
        "{prefix}\t{method}\t{calls}\t{:.9}\t{iou:.9}\t{fbeta:.9}\t{:.9}\t{:.9}\t{worst:.9}",
        mae / (GRID * GRID) as f64,
        retained / n as f64,
        regret / n as f64
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn mask_oracle_finds_best_feasible_position() {
        let mut mask = vec![0.; GRID * GRID];
        for y in 40..80 {
            for x in 190..230 {
                mask[y * GRID + x] = 1.;
            }
        }
        let a = Integral::new(&mask);
        assert_eq!(a.sum(190, 40, 40, 40), 1600);
        assert_eq!(a.best(40, 40), 1600);
        assert_eq!(a.best(20, 20), 400);
        assert_eq!(a.sum(0, 0, 40, 40), 0);
    }
}
