//! v0.1-vs-v0.2 real-image crop evaluation.
//!
//! Compares the shipping v0.1 path (`UltraFace` + `MicroSalNet v1` +
//! `zenlayout::smart_crop::compute_crop`) against the v0.2 pipeline
//! (`YuNet+5kps` + `MicroSalNet v3ds w24` + `SelfieSeg` + `stream_prep` +
//! `DualIntegralMap::from_crop_inputs` + `solve_crop_sat`) on the repo's
//! committed test photos.
//!
//! Prints a markdown comparison table and writes optional crop overlays to
//! `$ZENSALLY_EVAL_OUT` (default `/home/lilith/tmp/sally-crop-v2/eval_out`).
//!
//! Usage:
//!   cargo run --manifest-path tools/sat-crop-bench/Cargo.toml \
//!       --example eval_crop_v2 --release

use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::time::Instant;

use sat_crop_bench::sat_crop::{
    CropInputSources, DualIntegralMap, FaceLandmarkInput, SatCropConfig, SatCropMode, TargetToSrc,
    solve_crop_sat,
};
use sat_crop_bench::stream_prep::{self, PrepOutputs};
use zenlayout::Rect;
use zenlayout::smart_crop::{AspectRatio, CropConfig, CropMode};
use zensally::preprocess::{Normalization, PreprocessConfig, ResizeMode};
use zensally::{
    CropRect, FaceDetector, FaceRect, ImageRef, PixelFormat, SaliencyDetector, SaliencyMap,
};
use zensally_tract::{MicroSalNet, UltraFaceDetector};

use tract_onnx::prelude::*;

// ---------------------------------------------------------------------------
// Model loading (gz → ONNX → optimized runnable)
// ---------------------------------------------------------------------------

fn gunzip(compressed: &[u8]) -> Vec<u8> {
    let len = compressed.len();
    let orig = u32::from_le_bytes([
        compressed[len - 4],
        compressed[len - 3],
        compressed[len - 2],
        compressed[len - 1],
    ]) as usize;
    let mut out = vec![0u8; orig];
    let mut dz = zenflate::Decompressor::new();
    let r = dz
        .gzip_decompress(compressed, &mut out, enough::Unstoppable)
        .expect("gz decompress failed");
    assert_eq!(r.output_written, orig);
    out
}

fn load_model_gz(path: &Path, input_w: usize, input_h: usize) -> Arc<TypedRunnableModel> {
    let gz = std::fs::read(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    let bytes = gunzip(&gz);
    tract_onnx::onnx()
        .model_for_read(&mut std::io::Cursor::new(&bytes))
        .and_then(|m| {
            m.with_input_fact(
                0,
                InferenceFact::dt_shape(DatumType::F32, [1, 3, input_h, input_w]),
            )
        })
        .and_then(|m| m.into_optimized())
        .and_then(|m| m.into_runnable())
        .unwrap_or_else(|e| panic!("load {}: {e}", path.display()))
}

fn plain_f32(t: &Tensor) -> &[f32] {
    t.as_plain_ram()
        .and_then(|v| v.as_slice::<f32>().ok())
        .expect("model output not plain f32")
}

// ---------------------------------------------------------------------------
// YuNet + 5-point landmarks (the crate's YuNetDetector drops `kps_*`)
// ---------------------------------------------------------------------------

struct YuNetKps {
    model: Arc<TypedRunnableModel>,
    /// Output indices: [cls8,cls16,cls32, obj8,obj16,obj32, bbox8,bbox16,bbox32,
    /// kps8,kps16,kps32] in model output order.
    out_idx: [usize; 12],
    buf: Vec<f32>,
}

const YUNET_SIZE: usize = 640;
const YUNET_STRIDES: [usize; 3] = [8, 16, 32];
const YUNET_SCORE_THR: f32 = 0.5;
const YUNET_NMS_IOU: f32 = 0.3;

impl YuNetKps {
    fn new(models_dir: &Path) -> Self {
        let model = load_model_gz(
            &models_dir.join("yunet_2023mar.onnx.gz"),
            YUNET_SIZE,
            YUNET_SIZE,
        );
        // Resolve output-name → index (must happen on the proto graph; we
        // re-load labels from a fresh parse since load_model_gz consumed it).
        let gz = std::fs::read(models_dir.join("yunet_2023mar.onnx.gz")).unwrap();
        let bytes = gunzip(&gz);
        let proto = tract_onnx::onnx()
            .model_for_read(&mut std::io::Cursor::new(&bytes))
            .unwrap();
        let labels: Vec<(usize, String)> = proto
            .output_outlets()
            .unwrap()
            .iter()
            .enumerate()
            .filter_map(|(i, o)| proto.outlet_label(*o).map(|n| (i, n.to_string())))
            .collect();
        let find = |prefix: &str, stride: usize| -> usize {
            let name = format!("{prefix}_{stride}");
            *labels
                .iter()
                .find(|(_, n)| n == &name)
                .map(|(i, _)| i)
                .unwrap_or_else(|| panic!("yunet output {name} missing: {labels:?}"))
        };
        let out_idx = [
            find("cls", 8),
            find("cls", 16),
            find("cls", 32),
            find("obj", 8),
            find("obj", 16),
            find("obj", 32),
            find("bbox", 8),
            find("bbox", 16),
            find("bbox", 32),
            find("kps", 8),
            find("kps", 16),
            find("kps", 32),
        ];
        Self {
            model,
            out_idx,
            buf: vec![0.0; 3 * YUNET_SIZE * YUNET_SIZE],
        }
    }

    /// Bilinear-resize letterbox to 640², NCHW **BGR** [0,255].
    fn preprocess(&mut self, image: &ImageRef<'_>) -> (f32, f32, f32) {
        let (src_w, src_h) = (image.width as usize, image.height as usize);
        let bpp = image.format.bytes_per_pixel();
        let (ri, gi, bi) = match image.format {
            PixelFormat::Bgra => (2, 1, 0),
            PixelFormat::Rgba | PixelFormat::Rgb => (0, 1, 2),
        };
        let ratio = (YUNET_SIZE as f32 / src_w as f32).min(YUNET_SIZE as f32 / src_h as f32);
        let rw = ((src_w as f32 * ratio).round() as usize).max(1);
        let rh = ((src_h as f32 * ratio).round() as usize).max(1);
        let (pad_l, pad_t) = ((YUNET_SIZE - rw) / 2, (YUNET_SIZE - rh) / 2);
        self.buf.fill(0.0);
        let plane = YUNET_SIZE * YUNET_SIZE;
        let stride = src_w * bpp;
        for dy in 0..rh {
            let syf = dy as f32 * (src_h - 1) as f32 / (rh - 1).max(1) as f32;
            let (y0, y1) = (syf as usize, ((syf as usize) + 1).min(src_h - 1));
            let fy = syf - y0 as f32;
            for dx in 0..rw {
                let sxf = dx as f32 * (src_w - 1) as f32 / (rw - 1).max(1) as f32;
                let (x0, x1) = (sxf as usize, ((sxf as usize) + 1).min(src_w - 1));
                let fx = sxf - x0 as f32;
                let (w00, w10) = ((1.0 - fx) * (1.0 - fy), fx * (1.0 - fy));
                let (w01, w11) = ((1.0 - fx) * fy, fx * fy);
                let (o00, o10) = (y0 * stride + x0 * bpp, y0 * stride + x1 * bpp);
                let (o01, o11) = (y1 * stride + x0 * bpp, y1 * stride + x1 * bpp);
                let px = image.pixels;
                let r = px[o00 + ri] as f32 * w00
                    + px[o10 + ri] as f32 * w10
                    + px[o01 + ri] as f32 * w01
                    + px[o11 + ri] as f32 * w11;
                let g = px[o00 + gi] as f32 * w00
                    + px[o10 + gi] as f32 * w10
                    + px[o01 + gi] as f32 * w01
                    + px[o11 + gi] as f32 * w11;
                let b = px[o00 + bi] as f32 * w00
                    + px[o10 + bi] as f32 * w10
                    + px[o01 + bi] as f32 * w01
                    + px[o11 + bi] as f32 * w11;
                let p = (dy + pad_t) * YUNET_SIZE + dx + pad_l;
                self.buf[p] = b;
                self.buf[plane + p] = g;
                self.buf[2 * plane + p] = r;
            }
        }
        (pad_l as f32, pad_t as f32, ratio)
    }

    /// Detect faces with 5-point landmarks; coords as 0–100% of src dims.
    fn detect(&mut self, image: &ImageRef<'_>) -> Vec<FaceLandmarkInput> {
        let (pad_l, pad_t, ratio) = self.preprocess(image);
        let input = Tensor::from_shape(&[1, 3, YUNET_SIZE, YUNET_SIZE], &self.buf).unwrap();
        let outputs = self.model.run(tvec!(input.into())).unwrap();

        /// (x, y, w, h, score, landmarks)
        type Det = (f32, f32, f32, f32, f32, [[f32; 2]; 5]);
        let mut dets: Vec<Det> = Vec::new();
        for (si, &stride) in YUNET_STRIDES.iter().enumerate() {
            let cls = plain_f32(&outputs[self.out_idx[si]]);
            let obj = plain_f32(&outputs[self.out_idx[3 + si]]);
            let bbox = plain_f32(&outputs[self.out_idx[6 + si]]);
            let kps = plain_f32(&outputs[self.out_idx[9 + si]]);
            let (cols, rows) = (YUNET_SIZE / stride, YUNET_SIZE / stride);
            assert_eq!(kps.len(), rows * cols * 10, "kps_{stride} layout");
            for row in 0..rows {
                for col in 0..cols {
                    let idx = row * cols + col;
                    let score = (cls[idx].clamp(0.0, 1.0) * obj[idx].clamp(0.0, 1.0)).sqrt();
                    if score < YUNET_SCORE_THR {
                        continue;
                    }
                    let bi = idx * 4;
                    let cx = (col as f32 + bbox[bi]) * stride as f32;
                    let cy = (row as f32 + bbox[bi + 1]) * stride as f32;
                    let w = bbox[bi + 2].exp() * stride as f32;
                    let h = bbox[bi + 3].exp() * stride as f32;
                    let mut kp = [[0.0f32; 2]; 5];
                    for (k, p) in kp.iter_mut().enumerate() {
                        let kx = (col as f32 + kps[idx * 10 + 2 * k]) * stride as f32;
                        let ky = (row as f32 + kps[idx * 10 + 2 * k + 1]) * stride as f32;
                        *p = [(kx - pad_l) / ratio, (ky - pad_t) / ratio];
                    }
                    dets.push((
                        (cx - w * 0.5 - pad_l) / ratio,
                        (cy - h * 0.5 - pad_t) / ratio,
                        w / ratio,
                        h / ratio,
                        score,
                        kp,
                    ));
                }
            }
        }

        // NMS (higher-score first, suppress IoU ≥ 0.3).
        dets.sort_by(|a, b| b.4.partial_cmp(&a.4).unwrap_or(std::cmp::Ordering::Equal));
        let mut keep: Vec<bool> = vec![true; dets.len()];
        for i in 0..dets.len() {
            if !keep[i] {
                continue;
            }
            for j in (i + 1)..dets.len() {
                if !keep[j] {
                    continue;
                }
                let (a, b) = (&dets[i], &dets[j]);
                let ox = (a.0 + a.2).min(b.0 + b.2) - a.0.max(b.0);
                let oy = (a.1 + a.3).min(b.1 + b.3) - a.1.max(b.1);
                let inter = ox.max(0.0) * oy.max(0.0);
                let union = a.2 * a.3 + b.2 * b.3 - inter;
                if union > 0.0 && inter / union >= YUNET_NMS_IOU {
                    keep[j] = false;
                }
            }
        }

        let (iw, ih) = (image.width as f32, image.height as f32);
        dets.iter()
            .zip(keep)
            .filter(|(_, k)| *k)
            .map(|(d, _)| FaceLandmarkInput {
                rect: FaceRect {
                    x1: (d.0 / iw * 100.0).clamp(0.0, 100.0),
                    y1: (d.1 / ih * 100.0).clamp(0.0, 100.0),
                    x2: ((d.0 + d.2) / iw * 100.0).clamp(0.0, 100.0),
                    y2: ((d.1 + d.3) / ih * 100.0).clamp(0.0, 100.0),
                    confidence: d.4,
                },
                landmarks: Some([
                    [d.5[0][0] / iw * 100.0, d.5[0][1] / ih * 100.0],
                    [d.5[1][0] / iw * 100.0, d.5[1][1] / ih * 100.0],
                    [d.5[2][0] / iw * 100.0, d.5[2][1] / ih * 100.0],
                    [d.5[3][0] / iw * 100.0, d.5[3][1] / ih * 100.0],
                    [d.5[4][0] / iw * 100.0, d.5[4][1] / ih * 100.0],
                ]),
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Generic 256²-stretch NCHW model (saliency w24 / selfie_seg matte)
// ---------------------------------------------------------------------------

/// Bilinear-stretch to `size`² NCHW RGB [0,1], run `model`, return the
/// `[1,1,oh,ow]` f32 plane.
fn run_square_model(
    model: &Arc<TypedRunnableModel>,
    image: &ImageRef<'_>,
    size: usize,
) -> (Vec<f32>, usize, usize) {
    let (src_w, src_h) = (image.width as usize, image.height as usize);
    let bpp = image.format.bytes_per_pixel();
    let (ri, gi, bi) = match image.format {
        PixelFormat::Bgra => (2, 1, 0),
        PixelFormat::Rgba | PixelFormat::Rgb => (0, 1, 2),
    };
    let mut buf = vec![0.0f32; 3 * size * size];
    let plane = size * size;
    let stride = src_w * bpp;
    for dy in 0..size {
        let syf = dy as f32 * (src_h - 1) as f32 / (size - 1) as f32;
        let (y0, y1) = (syf as usize, ((syf as usize) + 1).min(src_h - 1));
        let fy = syf - y0 as f32;
        for dx in 0..size {
            let sxf = dx as f32 * (src_w - 1) as f32 / (size - 1) as f32;
            let (x0, x1) = (sxf as usize, ((sxf as usize) + 1).min(src_w - 1));
            let fx = sxf - x0 as f32;
            let (w00, w10) = ((1.0 - fx) * (1.0 - fy), fx * (1.0 - fy));
            let (w01, w11) = ((1.0 - fx) * fy, fx * fy);
            let (o00, o10) = (y0 * stride + x0 * bpp, y0 * stride + x1 * bpp);
            let (o01, o11) = (y1 * stride + x0 * bpp, y1 * stride + x1 * bpp);
            let px = image.pixels;
            let r = px[o00 + ri] as f32 * w00
                + px[o10 + ri] as f32 * w10
                + px[o01 + ri] as f32 * w01
                + px[o11 + ri] as f32 * w11;
            let g = px[o00 + gi] as f32 * w00
                + px[o10 + gi] as f32 * w10
                + px[o01 + gi] as f32 * w01
                + px[o11 + gi] as f32 * w11;
            let b = px[o00 + bi] as f32 * w00
                + px[o10 + bi] as f32 * w10
                + px[o01 + bi] as f32 * w01
                + px[o11 + bi] as f32 * w11;
            let p = dy * size + dx;
            buf[p] = r / 255.0;
            buf[plane + p] = g / 255.0;
            buf[2 * plane + p] = b / 255.0;
        }
    }
    let input = Tensor::from_shape(&[1, 3, size, size], &buf).unwrap();
    let outputs = model.run(tvec!(input.into())).unwrap();
    let out = &outputs[0];
    let shape = out.shape().to_vec();
    assert_eq!(shape.len(), 4, "expected NCHW output, got {shape:?}");
    let (ow, oh) = (shape[3], shape[2]);
    let data = plain_f32(out);
    assert_eq!(data.len(), ow * oh, "expected single-channel map");
    (data.to_vec(), ow, oh)
}

// ---------------------------------------------------------------------------
// Metrics helpers
// ---------------------------------------------------------------------------

fn face_rect_px(f: &FaceRect, w: u32, h: u32) -> (f32, f32, f32, f32) {
    (
        f.x1 / 100.0 * w as f32,
        f.y1 / 100.0 * h as f32,
        f.x2 / 100.0 * w as f32,
        f.y2 / 100.0 * h as f32,
    )
}

/// Protected face *core* — the eye→mouth interval the crop must not
/// slice. With landmarks: the landmark bbox padded by 25% of its width
/// (outer eye corners) horizontally and `[lo_y − 0.16·fh, hi_y +
/// 0.06·fh]` vertically (brow line to lower lip), so the metric catches
/// an edge grazing an eyebrow, eye corner, or lip. Without landmarks:
/// the classic estimate (inner 76% width, [top+15%, bottom+15%]).
/// Forehead/hair and cheek margins are expendable headroom by design.
fn face_core_px(
    r: (f32, f32, f32, f32),
    landmarks: Option<&[[f32; 2]; 5]>,
    iw: u32,
    ih: u32,
) -> (f32, f32, f32, f32) {
    let (fx1, fy1, fx2, fy2) = r;
    let (fw, fh) = (fx2 - fx1, fy2 - fy1);
    if let Some(lm) = landmarks {
        let lx = |i: usize| lm[i][0] / 100.0 * iw as f32;
        let ly = |i: usize| lm[i][1] / 100.0 * ih as f32;
        let lo_x = (0..5).map(lx).fold(f32::INFINITY, f32::min);
        let hi_x = (0..5).map(lx).fold(f32::NEG_INFINITY, f32::max);
        let lo_y = (0..5).map(ly).fold(f32::INFINITY, f32::min);
        let hi_y = (0..5).map(ly).fold(f32::NEG_INFINITY, f32::max);
        let pad = 0.25 * (hi_x - lo_x);
        (
            (lo_x - pad).max(fx1 + 0.06 * fw),
            (lo_y - 0.16 * fh).max(0.0),
            (hi_x + pad).min(fx2 - 0.06 * fw),
            hi_y + 0.06 * fh,
        )
    } else {
        (
            fx1 + 0.12 * fw,
            fy1 + 0.15 * fh,
            fx2 - 0.12 * fw,
            fy2 + 0.15 * fh,
        )
    }
}

/// The inviolable feature band: the raw landmark bbox (eyes, nose, mouth
/// corners) when landmarks exist — the pads in [`face_core_px`] are trim
/// margin, this is what an edge must never cross. Without landmarks the
/// protected core doubles as the band.
fn face_band_px(
    core: (f32, f32, f32, f32),
    landmarks: Option<&[[f32; 2]; 5]>,
    iw: u32,
    ih: u32,
) -> (f32, f32, f32, f32) {
    match landmarks {
        Some(lm) => {
            let lx = |i: usize| lm[i][0] / 100.0 * iw as f32;
            let ly = |i: usize| lm[i][1] / 100.0 * ih as f32;
            (
                (0..5).map(lx).fold(f32::INFINITY, f32::min),
                (0..5).map(ly).fold(f32::INFINITY, f32::min),
                (0..5).map(lx).fold(f32::NEG_INFINITY, f32::max),
                (0..5).map(ly).fold(f32::NEG_INFINITY, f32::max),
            )
        }
        None => core,
    }
}

/// Crop boundary slices through a protected region, per axis: where the
/// padded `core` fits inside the crop on that axis it must be contained;
/// where the core is taller/wider than the crop (uncontainable by
/// construction — pads are designated trim margin), the feature `band`
/// must be covered on that axis instead.
fn face_is_cut(
    core: &(f32, f32, f32, f32),
    band: &(f32, f32, f32, f32),
    c: &(u32, u32, u32, u32),
) -> bool {
    let (cx1, cy1, cx2, cy2) = (
        c.0 as f32,
        c.1 as f32,
        (c.0 + c.2) as f32,
        (c.1 + c.3) as f32,
    );
    let overlap = cx2 > core.0 && cx1 < core.2 && cy2 > core.1 && cy1 < core.3;
    if !overlap {
        return false;
    }
    let x_ok = if core.2 - core.0 <= cx2 - cx1 {
        cx1 <= core.0 && cx2 >= core.2
    } else {
        cx1 <= band.0 && cx2 >= band.2
    };
    let y_ok = if core.3 - core.1 <= cy2 - cy1 {
        cy1 <= core.1 && cy2 >= core.3
    } else {
        cy1 <= band.1 && cy2 >= band.3
    };
    !(x_ok && y_ok)
}

/// Fraction of the face's *retention region* inside the crop, per the
/// same axis rule as [`face_is_cut`]: the padded `core` on axes where it
/// can fit inside the crop, else the landmark `band` (pads are trim
/// margin when the zoom makes the padded core uncontainable).
fn face_retained_frac(
    core: &(f32, f32, f32, f32),
    band: &(f32, f32, f32, f32),
    c: &(u32, u32, u32, u32),
) -> f32 {
    let (cx1, cy1, cx2, cy2) = (
        c.0 as f32,
        c.1 as f32,
        (c.0 + c.2) as f32,
        (c.1 + c.3) as f32,
    );
    let (rx1, rx2) = if core.2 - core.0 <= cx2 - cx1 {
        (core.0, core.2)
    } else {
        (band.0, band.2)
    };
    let (ry1, ry2) = if core.3 - core.1 <= cy2 - cy1 {
        (core.1, core.3)
    } else {
        (band.1, band.3)
    };
    let ox = (cx2.min(rx2) - cx1.max(rx1)).max(0.0);
    let oy = (cy2.min(ry2) - cy1.max(ry1)).max(0.0);
    let ra = (rx2 - rx1).max(0.001) * (ry2 - ry1).max(0.001);
    (ox * oy / ra).min(1.0)
}

/// Mean of `map` (mw×mh, row-major over src dims) inside the crop rect.
fn mean_in_crop(
    map: &[f32],
    mw: usize,
    mh: usize,
    c: &(u32, u32, u32, u32),
    sw: u32,
    sh: u32,
) -> f32 {
    let (kx, ky) = (mw as f64 / sw as f64, mh as f64 / sh as f64);
    let (x1, y1) = ((c.0 as f64 * kx) as usize, (c.1 as f64 * ky) as usize);
    let (x2, y2) = (
        ((c.0 + c.2) as f64 * kx).ceil() as usize,
        ((c.1 + c.3) as f64 * ky).ceil() as usize,
    );
    let (mut s, mut n) = (0.0f64, 0u64);
    for y in y1.min(mh - 1)..y2.min(mh) {
        for x in x1.min(mw - 1)..x2.min(mw) {
            s += map[y * mw + x] as f64;
            n += 1;
        }
    }
    if n == 0 { 0.0 } else { (s / n as f64) as f32 }
}

/// Write a float map as a min-max normalized grayscale PNG (debug only).
fn save_heat(path: &Path, data: &[f32], w: usize, h: usize) {
    let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
    for &v in data {
        lo = lo.min(v);
        hi = hi.max(v);
    }
    let span = (hi - lo).max(1e-6);
    let px: Vec<u8> = data
        .iter()
        .map(|&v| ((v - lo) / span * 255.0) as u8)
        .collect();
    image::save_buffer(path, &px, w as u32, h as u32, image::ExtendedColorType::L8)
        .expect("save heat");
}

fn draw_rect(rgb: &mut [u8], w: usize, r: &(u32, u32, u32, u32), col: [u8; 3]) {
    let (x, y, rw, rh) = (r.0 as usize, r.1 as usize, r.2 as usize, r.3 as usize);
    let h = rgb.len() / (w * 3);
    let put = |rgb: &mut [u8], px: usize, py: usize| {
        if px < w && py < h {
            let o = (py * w + px) * 3;
            rgb[o] = col[0];
            rgb[o + 1] = col[1];
            rgb[o + 2] = col[2];
        }
    };
    for t in 0..2usize {
        for px in x..x + rw {
            put(rgb, px, y + t.min(rh.saturating_sub(1)));
            put(rgb, px, (y + rh).saturating_sub(1 + t));
        }
        for py in y..y + rh {
            put(rgb, x + t.min(rw.saturating_sub(1)), py);
            put(rgb, (x + rw).saturating_sub(1 + t), py);
        }
    }
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

fn main() {
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    let repo = manifest.parent().unwrap().parent().unwrap();
    let models_dir = repo.join("crates/zensally-tract/models");
    let test_data = repo.join("test_data");
    let out_dir = PathBuf::from(
        std::env::var("ZENSALLY_EVAL_OUT")
            .unwrap_or_else(|_| "/home/lilith/tmp/sally-crop-v2/eval_out".into()),
    );
    std::fs::create_dir_all(&out_dir).expect("create eval_out dir");

    let mut images: Vec<(String, PathBuf)> = Vec::new();
    let portrait = test_data.join("portrait.jpg");
    if portrait.exists() {
        images.push(("portrait".into(), portrait));
    }
    for name in [
        "group",
        "person_dark",
        "dog_beach",
        "lighthouse",
        "food_plate",
        "neon_signs",
        "product_shoes",
    ] {
        let p = test_data.join("saliency").join(format!("{name}.jpg"));
        if p.exists() {
            images.push((name.into(), p));
        }
    }
    let wider = repo.join("data/wider_face/WIDER_val/images");
    if wider.is_dir() {
        let mut picks: Vec<PathBuf> = Vec::new();
        for e in std::fs::read_dir(&wider).unwrap().take(6).flatten() {
            picks.push(e.path());
        }
        picks.sort();
        for (i, p) in picks.into_iter().enumerate() {
            images.push((format!("wider_{i}"), p));
        }
    }
    if images.is_empty() {
        eprintln!("no test images found");
        std::process::exit(1);
    }

    eprintln!("loading detectors…");
    let t = Instant::now();
    let mut ultra = UltraFaceDetector::new().expect("ultraface");
    let mut sal_v1 = MicroSalNet::new().expect("microsalnet v1");
    let mut yunet = YuNetKps::new(&models_dir);
    let sal_w24 = load_model_gz(&models_dir.join("microsalnet_v3ds_w24.onnx.gz"), 256, 256);
    let seg = load_model_gz(&models_dir.join("selfie_seg.onnx.gz"), 256, 256);
    eprintln!("models ready in {:.0}ms", t.elapsed().as_secs_f64() * 1e3);

    // (label, AspectRatio, zenlayout CropMode, SatCropMode)
    let configs: &[(&str, AspectRatio, CropMode, SatCropMode)] = &[
        (
            "9:16_min",
            AspectRatio { w: 9, h: 16 },
            CropMode::Minimal,
            SatCropMode::Minimal,
        ),
        (
            "9:16_max",
            AspectRatio { w: 9, h: 16 },
            CropMode::Maximal,
            SatCropMode::Maximal,
        ),
        (
            "3:4_min",
            AspectRatio { w: 3, h: 4 },
            CropMode::Minimal,
            SatCropMode::Minimal,
        ),
        (
            "3:4_max",
            AspectRatio { w: 3, h: 4 },
            CropMode::Maximal,
            SatCropMode::Maximal,
        ),
        (
            "1:1_min",
            AspectRatio { w: 1, h: 1 },
            CropMode::Minimal,
            SatCropMode::Minimal,
        ),
        (
            "1:1_max",
            AspectRatio { w: 1, h: 1 },
            CropMode::Maximal,
            SatCropMode::Maximal,
        ),
        (
            "16:9_min",
            AspectRatio { w: 16, h: 9 },
            CropMode::Minimal,
            SatCropMode::Minimal,
        ),
        (
            "16:9_max",
            AspectRatio { w: 16, h: 9 },
            CropMode::Maximal,
            SatCropMode::Maximal,
        ),
    ];

    // Analysis target: 512×384 → 64×48 stride-8 grid (matches the v0.2 design).
    let (tw, th, gw, gh) = (512usize, 384usize, 64usize, 48usize);
    let prep_cfg = PreprocessConfig {
        target_w: tw,
        target_h: th,
        mode: ResizeMode::Stretch,
        norm: Normalization::UnitScale,
    };
    let mut nhwc = vec![0.0f32; 3 * tw * th];
    let mut classical = vec![0.0f32; gw * gh];
    let mut horizon = vec![0.0f32; gh];
    let mut scratch = vec![0.0f32; stream_prep::prep_scratch_len(tw, gw, gh)];

    println!();
    println!(
        "| image | cfg | v0.1 crop (x,y,w,h) | v0.2 crop (x,y,w,h) | cuts v1 | cuts v2 | face% v2 | conc v1 | conc v2 | max/min |"
    );
    println!("|---|---|---|---|---|---|---|---|---|---|");

    let mut totals = (0u64, 0u64); // cuts v1, cuts v2
    let mut fail_flags: Vec<String> = Vec::new();
    let mut time_acc = [0.0f64; 3]; // prep, map-build, solve×8

    for (name, path) in &images {
        let img = image::open(path).expect("open image").to_rgb8();
        let (iw, ih) = (img.width(), img.height());
        let image = ImageRef {
            pixels: img.as_raw(),
            width: iw,
            height: ih,
            format: PixelFormat::Rgb,
        };

        // ---- v0.1 detections ----
        let faces_v1 = ultra.detect(&image);
        let sal1 = sal_v1.saliency_map(&image);

        // ---- v0.2 detections ----
        let faces_v2 = yunet.detect(&image);
        let (sal2_data, sal2_w, sal2_h) = run_square_model(&sal_w24, &image, 256);
        let (matte_data, matte_w, matte_h) = run_square_model(&seg, &image, 256);
        let sal2 = SaliencyMap {
            data: sal2_data,
            width: sal2_w as u32,
            height: sal2_h as u32,
        };
        let matte = SaliencyMap {
            data: matte_data,
            width: matte_w as u32,
            height: matte_h as u32,
        };

        // ---- v0.2 fused prep + map ----
        let t0 = Instant::now();
        let mut outputs = PrepOutputs {
            output_nhwc: &mut nhwc,
            classical_energy: &mut classical,
            horizon_profile: &mut horizon,
            scratch: &mut scratch,
        };
        stream_prep::preprocess_nhwc_strided(
            img.as_raw(),
            iw,
            ih,
            iw as usize * 3,
            PixelFormat::Rgb,
            &prep_cfg,
            &mut outputs,
        );
        let prep_ms = t0.elapsed().as_secs_f64() * 1e3;

        let t0 = Instant::now();
        let to_src = TargetToSrc::stretch(iw, ih, tw as u32, th as u32);
        let sources = CropInputSources {
            saliency: Some(&sal2),
            matte: Some(&matte),
            faces: &faces_v2,
            manual_focus: &[],
            classical_energy: Some(&classical),
            horizon_profile: Some(&horizon),
        };
        let map = DualIntegralMap::from_crop_inputs(&sources, gw, gh, &to_src, 0.25);
        let map_ms = t0.elapsed().as_secs_f64() * 1e3;

        if std::env::var_os("EVAL_DEBUG").is_some() {
            save_heat(
                &out_dir.join(format!("{name}_sal.png")),
                &sal2.data,
                sal2.width as usize,
                sal2.height as usize,
            );
            save_heat(
                &out_dir.join(format!("{name}_matte.png")),
                &matte.data,
                matte.width as usize,
                matte.height as usize,
            );
            save_heat(
                &out_dir.join(format!("{name}_classical.png")),
                &classical,
                gw,
                gh,
            );
            eprintln!("  [dbg] {name}: subject={:?}", map.subject);
            for f in &map.faces {
                eprintln!(
                    "  [dbg]   face cells ({:.1},{:.1})-({:.1},{:.1}) hr={:.1} chin={:.1} conf={:.2}",
                    f.fx1, f.fy1, f.fx2, f.fy2, f.headroom_y1, f.chin_y2, f.confidence
                );
            }
            for f in &faces_v2 {
                let r = face_rect_px(&f.rect, iw, ih);
                eprintln!(
                    "  [dbg]   face px ({:.0},{:.0})-({:.0},{:.0}) conf={:.2} lm={:?}",
                    r.0,
                    r.1,
                    r.2,
                    r.3,
                    f.rect.confidence,
                    f.landmarks
                        .map(|l| l.map(|p| [p[0] * iw as f32 / 100.0, p[1] * ih as f32 / 100.0]))
                );
            }
        }

        // ---- crops ----
        let t0 = Instant::now();
        let smart_in =
            zensally::bridge::build_smart_crop_input_raw(faces_v1.clone(), Some(sal1), &[]);
        let mut rows: Vec<(String, Option<Rect>, Option<CropRect>)> = Vec::new();
        for (label, aspect, zmode, smode) in configs {
            let zcfg = CropConfig {
                target_aspect: *aspect,
                mode: *zmode,
                ..CropConfig::default()
            };
            let c1 = smart_in.compute_crop(iw, ih, &zcfg);
            let c2 = solve_crop_sat(
                &map,
                &to_src,
                (aspect.w, aspect.h),
                *smode,
                &SatCropConfig::default(),
            );
            rows.push(((*label).to_string(), c1, c2));
        }
        let solve_ms = t0.elapsed().as_secs_f64() * 1e3;
        if std::env::var_os("EVAL_DEBUG").is_some() {
            for (label, _, c2) in &rows {
                if let Some(c) = c2 {
                    // Invert src px back to cell coords for inspection.
                    let (sx, sy) = (iw as f64 / tw as f64 * 8.0, ih as f64 / th as f64 * 8.0);
                    eprintln!(
                        "  [dbg]   {label}: cells ({:.1},{:.1})-({:.1},{:.1})",
                        c.x as f64 / sx,
                        c.y as f64 / sy,
                        (c.x + c.w) as f64 / sx,
                        (c.y + c.h) as f64 / sy
                    );
                    {
                        // Score terms at the winning (w,h) across x so a
                        // stubborn edge can be attributed to energy vs cut.
                        let (cw, chh) = (
                            (c.w as f64 / sx).round() as usize,
                            (c.h as f64 / sy).round() as usize,
                        );
                        let cy = (c.y as f64 / sy).round() as usize;
                        let mut line = String::from("     x-scan e/c:");
                        for xx in (0..=gw.saturating_sub(cw)).step_by(2) {
                            let e = map.rect_energy(xx, cy, xx + cw, cy + chh);
                            let k = map.rect_edge_cut_penalty(xx, cy, xx + cw, cy + chh, 1);
                            line.push_str(&format!(" {xx}:{e}/{k}"));
                        }
                        eprintln!("  [dbg]{line}");
                    }
                }
            }
        }
        time_acc[0] += prep_ms;
        time_acc[1] += map_ms;
        time_acc[2] += solve_ms;

        // ---- metrics ----
        // "Primary" face = max conf × √area — the dominant face, matching
        // the solver's per-face weighting (raw confidence picks marginal
        // background detections).
        let sal_full_mean: f64 =
            sal2.data.iter().map(|&v| v as f64).sum::<f64>() / sal2.data.len() as f64;
        let dom = |f: &FaceRect| {
            let r = face_rect_px(f, iw, ih);
            f.confidence * ((r.2 - r.0) * (r.3 - r.1)).sqrt()
        };
        let primary_v2 = faces_v2
            .iter()
            .max_by(|a, b| {
                dom(&a.rect)
                    .partial_cmp(&dom(&b.rect))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|f| {
                let core =
                    face_core_px(face_rect_px(&f.rect, iw, ih), f.landmarks.as_ref(), iw, ih);
                (core, face_band_px(core, f.landmarks.as_ref(), iw, ih))
            });
        let primary_v1 = faces_v1
            .iter()
            .max_by(|a, b| {
                dom(a)
                    .partial_cmp(&dom(b))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|f| face_core_px(face_rect_px(f, iw, ih), None, iw, ih));

        eprintln!(
            "\n{name} {iw}x{ih}: faces v1={} v2={} (kps={}) | prep {prep_ms:.1}ms map {map_ms:.2}ms solve8 {solve_ms:.2}ms",
            faces_v1.len(),
            faces_v2.len(),
            faces_v2.iter().filter(|f| f.landmarks.is_some()).count()
        );

        let mut area_ratio: Option<f64> = None;
        for (label, c1, c2) in &rows {
            let rc1 = c1.map(|r| (r.x, r.y, r.width, r.height));
            let rc2 = c2.map(|r| (r.x, r.y, r.w, r.h));
            let cuts1 = rc1.map_or(0, |c| {
                faces_v1
                    .iter()
                    .filter(|f| {
                        let core = face_core_px(face_rect_px(f, iw, ih), None, iw, ih);
                        face_is_cut(&core, &core, &c)
                    })
                    .count()
            });
            let cuts2 = rc2.map_or(0, |c| {
                faces_v2
                    .iter()
                    .filter(|f| {
                        let core = face_core_px(
                            face_rect_px(&f.rect, iw, ih),
                            f.landmarks.as_ref(),
                            iw,
                            ih,
                        );
                        let band = face_band_px(core, f.landmarks.as_ref(), iw, ih);
                        face_is_cut(&core, &band, &c)
                    })
                    .count()
            });
            totals.0 += cuts1 as u64;
            totals.1 += cuts2 as u64;
            let face_ret = match (rc2, primary_v2) {
                (Some(c), Some((core, band))) => face_retained_frac(&core, &band, &c) * 100.0,
                _ => f32::NAN,
            };
            let conc1 = rc1.map_or(f32::NAN, |c| {
                mean_in_crop(
                    &sal2.data,
                    sal2.width as usize,
                    sal2.height as usize,
                    &c,
                    iw,
                    ih,
                ) / sal_full_mean.max(1e-6) as f32
            });
            let conc2 = rc2.map_or(f32::NAN, |c| {
                mean_in_crop(
                    &sal2.data,
                    sal2.width as usize,
                    sal2.height as usize,
                    &c,
                    iw,
                    ih,
                ) / sal_full_mean.max(1e-6) as f32
            });
            if let (Some(_), Some(b)) = (rc1, rc2)
                && label.ends_with("_max")
            {
                // Match the _min crop of the same aspect for the ratio.
                let min_label = label.replace("_max", "_min");
                if let Some((_, _, Some(cmin))) = rows.iter().find(|(l, _, _)| *l == min_label) {
                    let amin = (cmin.w as f64) * (cmin.h as f64);
                    let amax = (b.2 as f64) * (b.3 as f64);
                    area_ratio = Some(amax / amin);
                }
            }
            println!(
                "| {name} | {label} | {} | {} | {} | {} | {} | {:.2} | {:.2} | {} |",
                rc1.map_or("-".into(), |c| format!("{},{},{},{}", c.0, c.1, c.2, c.3)),
                rc2.map_or("-".into(), |c| format!("{},{},{},{}", c.0, c.1, c.2, c.3)),
                cuts1,
                cuts2,
                if face_ret.is_nan() {
                    "-".into()
                } else {
                    format!("{face_ret:.0}")
                },
                conc1,
                conc2,
                area_ratio.map_or("-".into(), |r| format!("{r:.2}")),
            );
            if cuts2 > 0 {
                fail_flags.push(format!("{name}/{label}: {cuts2} v0.2 face cuts"));
            }
            if !face_ret.is_nan() && face_ret < 99.0 {
                fail_flags.push(format!(
                    "{name}/{label}: primary face {face_ret:.0}% retained"
                ));
            }
            if let Some(r) = area_ratio
                && r > 0.85
            {
                fail_flags.push(format!("{name}/{label}: max/min area {r:.2} > 0.85"));
            }
            area_ratio = None;
        }

        // ---- overlays ----
        let mut vis = img.clone().into_raw();
        for f in &faces_v2 {
            let (fx1, fy1, fx2, fy2) = face_rect_px(&f.rect, iw, ih);
            draw_rect(
                &mut vis,
                iw as usize,
                &(
                    fx1 as u32,
                    fy1 as u32,
                    (fx2 - fx1).max(1.0) as u32,
                    (fy2 - fy1).max(1.0) as u32,
                ),
                [255, 255, 0],
            );
        }
        for (i, (_, _, c2)) in rows.iter().enumerate() {
            if let Some(r) = c2 {
                draw_rect(
                    &mut vis,
                    iw as usize,
                    &(r.x, r.y, r.w, r.h),
                    [
                        [0, 255, 0],
                        [0, 200, 255],
                        [255, 0, 255],
                        [128, 255, 0],
                        [0, 128, 255],
                        [255, 128, 0],
                        [180, 255, 180],
                        [255, 255, 255],
                    ][i],
                );
            }
        }
        let mut v1vis = img.clone().into_raw();
        for f in &faces_v1 {
            let (fx1, fy1, fx2, fy2) = face_rect_px(f, iw, ih);
            draw_rect(
                &mut v1vis,
                iw as usize,
                &(
                    fx1 as u32,
                    fy1 as u32,
                    (fx2 - fx1).max(1.0) as u32,
                    (fy2 - fy1).max(1.0) as u32,
                ),
                [255, 255, 0],
            );
        }
        for (i, (_, c1, _)) in rows.iter().enumerate() {
            if let Some(r) = c1 {
                draw_rect(
                    &mut v1vis,
                    iw as usize,
                    &(r.x, r.y, r.width, r.height),
                    [
                        [255, 0, 0],
                        [255, 128, 0],
                        [255, 0, 128],
                        [255, 64, 64],
                        [200, 0, 0],
                        [255, 180, 0],
                        [180, 0, 0],
                        [255, 200, 200],
                    ][i],
                );
            }
        }
        image::save_buffer(
            out_dir.join(format!("{name}_v2.png")),
            &vis,
            iw,
            ih,
            image::ColorType::Rgb8,
        )
        .ok();
        image::save_buffer(
            out_dir.join(format!("{name}_v1.png")),
            &v1vis,
            iw,
            ih,
            image::ColorType::Rgb8,
        )
        .ok();
        if primary_v1.is_none() && primary_v2.is_none() {
            eprintln!("  (no faces detected by either pipeline)");
        }
    }

    println!();
    println!("## Summary");
    println!();
    println!("- images: {}", images.len());
    println!(
        "- total face cuts: v0.1 = {}, v0.2 = {}",
        totals.0, totals.1
    );
    println!(
        "- mean stage timing: stream_prep {:.1}ms, from_crop_inputs {:.2}ms, solve×8 {:.2}ms",
        time_acc[0] / images.len() as f64,
        time_acc[1] / images.len() as f64,
        time_acc[2] / images.len() as f64
    );
    if fail_flags.is_empty() {
        println!("- ALL GATES PASSED (0 v0.2 face cuts, 100% primary retention, max/min ≤ 0.85)");
    } else {
        println!("- GATE FAILURES:");
        let mut s = String::new();
        for f in &fail_flags {
            let _ = writeln!(s, "  - {f}");
        }
        println!("{s}");
    }
}
