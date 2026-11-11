//! v0.1-vs-v0.2 large-corpus crop evaluation (741 photos, 8 cells each).
//!
//! Parallel (rayon) evaluator over the frozen split corpus manifest:
//! every image runs the shipping v0.1 path (`UltraFace` + `MicroSalNet v1`
//! with `zenlayout::smart_crop::compute_crop`) and the v0.2 pipeline
//! (`YuNet+5kps` + `MicroSalNet v3ds w24` + `SelfieSeg` + `stream_prep`
//! with `DualIntegralMap::from_crop_inputs` + `solve_crop_sat`) for all
//! 4 aspects × {Minimal, Maximal}, emitting:
//!
//! - `corpus_results.json` — full per-image/per-cell metrics
//! - `corpus_data.js` — `window.ZENSALLY_CORPUS_DATA = {...}` for the
//!   split-switching viewer (zero CORS issues)
//! - `SUMMARY.md` — aggregate markdown tables per split + corpus
//!
//! No per-image PNG artifacts are written — all overlays render
//! client-side in `index.html` from `images/<safe_id>.<ext>` + metadata.
//!
//! Usage:
//!   cargo run --manifest-path tools/sat-crop-bench/Cargo.toml \
//!       --example eval_corpus_v2 --release \
//!       -- [--manifest <corpus_manifest.json>] [--out-dir <dir>]

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use rayon::prelude::*;
use serde::Serialize;

use sat_crop_bench::sat_crop::{
    CropInputSources, DualIntegralMap, FaceLandmarkInput, SatCropConfig, SatCropMode, TargetToSrc,
    solve_crop_sat,
};
use sat_crop_bench::stream_prep::{self, PrepOutputs};
use zenlayout::smart_crop::{AspectRatio, CropConfig, CropMode};
use zensally::preprocess::{Normalization, PreprocessConfig, ResizeMode};
use zensally::{
    CropRect, FaceDetector, FaceRect, ImageRef, PixelFormat, SaliencyDetector, SaliencyMap,
};
use zensally_tract::{MicroSalNet, UltraFaceDetector};

use tract_onnx::prelude::*;

const DEFAULT_MANIFEST: &str =
    "/mnt/v/output/zensally/crop-v2-corpus-2026-11-11/corpus_manifest.json";
const DEFAULT_OUT_DIR: &str = "/mnt/v/output/zensally/crop-v2-corpus-2026-11-11";
const CELL_LABELS: [&str; 8] = [
    "9:16_min", "9:16_max", "3:4_min", "3:4_max", "1:1_min", "1:1_max", "16:9_min", "16:9_max",
];

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
// Metrics helpers (same semantics as eval_crop_v2)
// ---------------------------------------------------------------------------

fn face_rect_px(f: &FaceRect, w: u32, h: u32) -> (f32, f32, f32, f32) {
    (
        f.x1 / 100.0 * w as f32,
        f.y1 / 100.0 * h as f32,
        f.x2 / 100.0 * w as f32,
        f.y2 / 100.0 * h as f32,
    )
}

/// Clamp a px-space (x1, y1, x2, y2) region to the image bounds. Landmark
/// predictions can extend past the frame on a clipped face; the metric
/// only protects features that actually exist inside the image (the
/// solver's grid-space clamp is equivalent).
fn clamp_px(r: (f32, f32, f32, f32), iw: u32, ih: u32) -> (f32, f32, f32, f32) {
    (
        r.0.clamp(0.0, iw as f32),
        r.1.clamp(0.0, ih as f32),
        r.2.clamp(0.0, iw as f32),
        r.3.clamp(0.0, ih as f32),
    )
}

/// Protected face *core* — see `eval_crop_v2.rs` for the derivation.
fn face_core_px(
    r: (f32, f32, f32, f32),
    landmarks: Option<&[[f32; 2]; 5]>,
    iw: u32,
    ih: u32,
) -> (f32, f32, f32, f32) {
    let (fx1, fy1, fx2, fy2) = r;
    let (fw, fh) = (fx2 - fx1, fy2 - fy1);
    let core = if let Some(lm) = landmarks {
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
    };
    clamp_px(core, iw, ih)
}

/// The inviolable feature band: the raw landmark bbox (see `eval_crop_v2`).
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
            clamp_px(
                (
                    (0..5).map(lx).fold(f32::INFINITY, f32::min),
                    (0..5).map(ly).fold(f32::INFINITY, f32::min),
                    (0..5).map(lx).fold(f32::NEG_INFINITY, f32::max),
                    (0..5).map(ly).fold(f32::NEG_INFINITY, f32::max),
                ),
                iw,
                ih,
            )
        }
        None => core,
    }
}

/// Crop boundary slices through a protected region, per axis.
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
/// same axis rule as [`face_is_cut`].
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

// ---------------------------------------------------------------------------
// Serializable output
// ---------------------------------------------------------------------------

#[derive(Serialize)]
struct FaceOut1 {
    rect: [f32; 4],
    core: [f32; 4],
    conf: f32,
}

#[derive(Serialize)]
struct FaceOut2 {
    rect: [f32; 4],
    core: [f32; 4],
    band: [f32; 4],
    kps: Option<[[f32; 2]; 5]>,
    conf: f32,
}

#[derive(Serialize)]
struct CellV1 {
    rect: [u32; 4],
    cuts: usize,
    v2_ref_cuts: usize,
    face_ret: Option<f32>,
    conc: f32,
}

#[derive(Serialize)]
struct CellV2 {
    rect: [u32; 4],
    cuts: usize,
    face_ret: Option<f32>,
    conc: f32,
}

#[derive(Serialize)]
struct CellOut {
    v1: Option<CellV1>,
    v2: Option<CellV2>,
    area_ratio_v1: Option<f32>,
    area_ratio_v2: Option<f32>,
}

#[derive(Serialize)]
struct TimingMs {
    prep: f64,
    map: f64,
    solve8: f64,
}

#[derive(Serialize)]
struct ImageOut {
    id: String,
    safe_id: String,
    corpus: String,
    split: String,
    raw_role: String,
    rel_image: String,
    width: u32,
    height: u32,
    faces_v1: Vec<FaceOut1>,
    faces_v2: Vec<FaceOut2>,
    subject_bbox: Option<[u32; 4]>,
    timing_ms: TimingMs,
    cells: BTreeMap<String, CellOut>,
}

#[derive(Default)]
struct Agg {
    n_images: u64,
    n_faces_v1_images: u64,
    n_faces_v2_images: u64,
    total_faces_v2: u64,
    v1_own_face_cuts: u64,
    v1_yunet_face_cuts: u64,
    v2_face_cuts: u64,
    ret1: f64,
    ret1_n: u64,
    ret2: f64,
    ret2_n: u64,
    conc1: f64,
    conc2: f64,
    conc_max1: f64,
    conc_max2: f64,
    n_max: u64,
    ar1: f64,
    ar1_n: u64,
    ar2: f64,
    ar2_n: u64,
    prep_ms: f64,
    map_ms: f64,
    solve8_ms: f64,
}

impl Agg {
    fn add(&mut self, o: &ImageOut) {
        self.n_images += 1;
        self.n_faces_v1_images += u64::from(!o.faces_v1.is_empty());
        self.n_faces_v2_images += u64::from(!o.faces_v2.is_empty());
        self.total_faces_v2 += o.faces_v2.len() as u64;
        self.prep_ms += o.timing_ms.prep;
        self.map_ms += o.timing_ms.map;
        self.solve8_ms += o.timing_ms.solve8;
        for (label, cell) in &o.cells {
            if let Some(v1) = &cell.v1 {
                self.v1_own_face_cuts += v1.cuts as u64;
                self.v1_yunet_face_cuts += v1.v2_ref_cuts as u64;
                self.conc1 += f64::from(v1.conc);
                if let Some(r) = v1.face_ret {
                    self.ret1 += f64::from(r);
                    self.ret1_n += 1;
                }
                if label.ends_with("_max") {
                    self.conc_max1 += f64::from(v1.conc);
                }
            }
            if let Some(v2) = &cell.v2 {
                self.v2_face_cuts += v2.cuts as u64;
                self.conc2 += f64::from(v2.conc);
                if let Some(r) = v2.face_ret {
                    self.ret2 += f64::from(r);
                    self.ret2_n += 1;
                }
                if label.ends_with("_max") {
                    self.conc_max2 += f64::from(v2.conc);
                    self.n_max += 1;
                }
            }
            if let Some(r) = cell.area_ratio_v1 {
                self.ar1 += f64::from(r);
                self.ar1_n += 1;
            }
            if let Some(r) = cell.area_ratio_v2 {
                self.ar2 += f64::from(r);
                self.ar2_n += 1;
            }
        }
    }

    fn mean(&self, s: f64, n: u64) -> Option<f64> {
        (n > 0).then_some(s / n as f64)
    }
}

#[derive(Serialize)]
struct SummaryOut {
    n_images: u64,
    n_faces_v1_images: u64,
    n_faces_v2_images: u64,
    total_faces_v2: u64,
    v1_own_face_cuts: u64,
    v1_yunet_face_cuts: u64,
    v2_face_cuts: u64,
    mean_primary_face_ret_v1: Option<f64>,
    mean_primary_face_ret_v2: Option<f64>,
    mean_conc_v1: Option<f64>,
    mean_conc_v2: Option<f64>,
    mean_conc_max_v1: Option<f64>,
    mean_conc_max_v2: Option<f64>,
    mean_area_ratio_v1: Option<f64>,
    mean_area_ratio_v2: Option<f64>,
    mean_prep_ms: f64,
    mean_map_ms: f64,
    mean_solve8_ms: f64,
}

impl From<&Agg> for SummaryOut {
    fn from(a: &Agg) -> Self {
        let n_cells = a.n_images * 8;
        Self {
            n_images: a.n_images,
            n_faces_v1_images: a.n_faces_v1_images,
            n_faces_v2_images: a.n_faces_v2_images,
            total_faces_v2: a.total_faces_v2,
            v1_own_face_cuts: a.v1_own_face_cuts,
            v1_yunet_face_cuts: a.v1_yunet_face_cuts,
            v2_face_cuts: a.v2_face_cuts,
            mean_primary_face_ret_v1: a.mean(a.ret1, a.ret1_n),
            mean_primary_face_ret_v2: a.mean(a.ret2, a.ret2_n),
            mean_conc_v1: a.mean(a.conc1, n_cells),
            mean_conc_v2: a.mean(a.conc2, n_cells),
            mean_conc_max_v1: a.mean(a.conc_max1, a.n_max),
            mean_conc_max_v2: a.mean(a.conc_max2, a.n_max),
            mean_area_ratio_v1: a.mean(a.ar1, a.ar1_n),
            mean_area_ratio_v2: a.mean(a.ar2, a.ar2_n),
            mean_prep_ms: a.prep_ms / a.n_images.max(1) as f64,
            mean_map_ms: a.map_ms / a.n_images.max(1) as f64,
            mean_solve8_ms: a.solve8_ms / a.n_images.max(1) as f64,
        }
    }
}

#[derive(Serialize)]
struct ResultsOut {
    schema: &'static str,
    generated_at: String,
    manifest_path: String,
    n_images: usize,
    summaries: BTreeMap<String, SummaryOut>,
    images: Vec<ImageOut>,
}

// ---------------------------------------------------------------------------
// Per-worker model bundle
// ---------------------------------------------------------------------------

struct Worker {
    ultra: UltraFaceDetector,
    sal_v1: MicroSalNet,
    yunet: YuNetKps,
    sal_w24: Arc<TypedRunnableModel>,
    seg: Arc<TypedRunnableModel>,
    nhwc: Vec<f32>,
    classical: Vec<f32>,
    horizon: Vec<f32>,
    scratch: Vec<f32>,
    prep_cfg: PreprocessConfig,
}

impl Worker {
    fn new(models_dir: &Path) -> Self {
        let (tw, th, gw, gh) = (512usize, 384usize, 64usize, 48usize);
        Self {
            ultra: UltraFaceDetector::new().expect("ultraface"),
            sal_v1: MicroSalNet::new().expect("microsalnet v1"),
            yunet: YuNetKps::new(models_dir),
            sal_w24: load_model_gz(&models_dir.join("microsalnet_v3ds_w24.onnx.gz"), 256, 256),
            seg: load_model_gz(&models_dir.join("selfie_seg.onnx.gz"), 256, 256),
            nhwc: vec![0.0; 3 * tw * th],
            classical: vec![0.0; gw * gh],
            horizon: vec![0.0; gh],
            scratch: vec![0.0; stream_prep::prep_scratch_len(tw, gw, gh)],
            prep_cfg: PreprocessConfig {
                target_w: tw,
                target_h: th,
                mode: ResizeMode::Stretch,
                norm: Normalization::UnitScale,
            },
        }
    }
}

fn r1(v: f32) -> f32 {
    (v * 10.0).round() / 10.0
}

// ---------------------------------------------------------------------------
// Per-image evaluation
// ---------------------------------------------------------------------------

fn eval_image(
    w: &mut Worker,
    item: &serde_json::Value,
    image_bytes: &[u8],
    iw: u32,
    ih: u32,
    configs: &[(String, AspectRatio, CropMode, SatCropMode)],
) -> ImageOut {
    let image = ImageRef {
        pixels: image_bytes,
        width: iw,
        height: ih,
        format: PixelFormat::Rgb,
    };
    let (tw, gw, gh) = (512usize, 64usize, 48usize);

    // ---- v0.1 detections ----
    let faces_v1 = w.ultra.detect(&image);
    let sal1 = w.sal_v1.saliency_map(&image);

    // ---- v0.2 detections ----
    let faces_v2 = w.yunet.detect(&image);
    let (sal2_data, sal2_w, sal2_h) = run_square_model(&w.sal_w24, &image, 256);
    let (matte_data, matte_w, matte_h) = run_square_model(&w.seg, &image, 256);
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
        output_nhwc: &mut w.nhwc,
        classical_energy: &mut w.classical,
        horizon_profile: &mut w.horizon,
        scratch: &mut w.scratch,
    };
    stream_prep::preprocess_nhwc_strided(
        image_bytes,
        iw,
        ih,
        iw as usize * 3,
        PixelFormat::Rgb,
        &w.prep_cfg,
        &mut outputs,
    );
    let prep_ms = t0.elapsed().as_secs_f64() * 1e3;

    let t0 = Instant::now();
    let to_src = TargetToSrc::stretch(iw, ih, tw as u32, 384);
    let sources = CropInputSources {
        saliency: Some(&sal2),
        matte: Some(&matte),
        faces: &faces_v2,
        manual_focus: &[],
        classical_energy: Some(&w.classical),
        horizon_profile: Some(&w.horizon),
    };
    let map = DualIntegralMap::from_crop_inputs(&sources, gw, gh, &to_src, 0.25);
    let map_ms = t0.elapsed().as_secs_f64() * 1e3;

    // ---- crops ----
    let t0 = Instant::now();
    let smart_in = zensally::bridge::build_smart_crop_input_raw(faces_v1.clone(), Some(sal1), &[]);
    type CropPair = (Option<(u32, u32, u32, u32)>, Option<CropRect>);
    let mut crops: Vec<CropPair> = Vec::new();
    for (_, aspect, zmode, smode) in configs {
        let zcfg = CropConfig {
            target_aspect: *aspect,
            mode: *zmode,
            ..CropConfig::default()
        };
        let c1 = smart_in
            .compute_crop(iw, ih, &zcfg)
            .map(|r| (r.x, r.y, r.width, r.height));
        let c2 = solve_crop_sat(
            &map,
            &to_src,
            (aspect.w, aspect.h),
            *smode,
            &SatCropConfig::default(),
        );
        crops.push((c1, c2));
    }
    let solve_ms = t0.elapsed().as_secs_f64() * 1e3;

    // ---- metrics ----
    let sal_full_mean: f64 =
        sal2.data.iter().map(|&v| v as f64).sum::<f64>() / sal2.data.len() as f64;
    let dom = |f: &FaceRect| {
        let r = face_rect_px(f, iw, ih);
        f.confidence * ((r.2 - r.0) * (r.3 - r.1)).sqrt()
    };
    // Primary = dominant YuNet face — the shared landmark-accurate oracle
    // both pipelines are measured against.
    let primary = faces_v2
        .iter()
        .max_by(|a, b| {
            dom(&a.rect)
                .partial_cmp(&dom(&b.rect))
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|f| {
            let core = face_core_px(face_rect_px(&f.rect, iw, ih), f.landmarks.as_ref(), iw, ih);
            (core, face_band_px(core, f.landmarks.as_ref(), iw, ih))
        });

    let v2_core_band = |f: &FaceLandmarkInput| {
        let core = face_core_px(face_rect_px(&f.rect, iw, ih), f.landmarks.as_ref(), iw, ih);
        (core, face_band_px(core, f.landmarks.as_ref(), iw, ih))
    };
    let conc_of = |c: &(u32, u32, u32, u32)| -> f32 {
        mean_in_crop(
            &sal2.data,
            sal2.width as usize,
            sal2.height as usize,
            c,
            iw,
            ih,
        ) / sal_full_mean.max(1e-6) as f32
    };

    let mut cells: BTreeMap<String, CellOut> = BTreeMap::new();
    for (i, (label, ..)) in configs.iter().enumerate() {
        let (c1, c2) = &crops[i];
        let cell_v1 = c1.map(|c| CellV1 {
            rect: [c.0, c.1, c.2, c.3],
            cuts: faces_v1
                .iter()
                .filter(|f| {
                    let core = face_core_px(face_rect_px(f, iw, ih), None, iw, ih);
                    face_is_cut(&core, &core, &c)
                })
                .count(),
            v2_ref_cuts: faces_v2
                .iter()
                .filter(|f| {
                    let (core, band) = v2_core_band(f);
                    face_is_cut(&core, &band, &c)
                })
                .count(),
            face_ret: primary.map(|(core, band)| face_retained_frac(&core, &band, &c) * 100.0),
            conc: conc_of(&c),
        });
        let cell_v2 = c2.map(|r| {
            let c = (r.x, r.y, r.w, r.h);
            CellV2 {
                rect: [c.0, c.1, c.2, c.3],
                cuts: faces_v2
                    .iter()
                    .filter(|f| {
                        let (core, band) = v2_core_band(f);
                        face_is_cut(&core, &band, &c)
                    })
                    .count(),
                face_ret: primary.map(|(core, band)| face_retained_frac(&core, &band, &c) * 100.0),
                conc: conc_of(&c),
            }
        });
        // max/min area ratio vs the same aspect's _min cell (the previous
        // config entry — labels alternate min,max).
        let (ar1, ar2) = if label.ends_with("_max") {
            let amin1 = crops[i - 1].0.map(|c| f64::from(c.2) * f64::from(c.3));
            let amin2 = crops[i - 1].1.map(|c| f64::from(c.w) * f64::from(c.h));
            let ar = |amax: Option<f64>, amin: Option<f64>| match (amax, amin) {
                (Some(a), Some(b)) if b > 0.0 => Some((a / b) as f32),
                _ => None,
            };
            (
                ar(c1.map(|c| f64::from(c.2) * f64::from(c.3)), amin1),
                ar(c2.map(|c| f64::from(c.w) * f64::from(c.h)), amin2),
            )
        } else {
            (None, None)
        };
        cells.insert(
            label.clone(),
            CellOut {
                v1: cell_v1,
                v2: cell_v2,
                area_ratio_v1: ar1,
                area_ratio_v2: ar2,
            },
        );
    }

    // subject bbox: cell coords → src px (grid is stride-8 over the
    // stretch target; px per cell = iw/gw, ih/gh).
    let subject_bbox = map.subject.map(|(x1, y1, x2, y2)| {
        let (sx, sy) = (iw as f64 / gw as f64, ih as f64 / gh as f64);
        let (px1, py1) = (
            (x1 as f64 * sx).round() as u32,
            (y1 as f64 * sy).round() as u32,
        );
        let (px2, py2) = (
            ((x2 as f64 * sx).round() as u32).min(iw),
            ((y2 as f64 * sy).round() as u32).min(ih),
        );
        [px1, py1, px2.saturating_sub(px1), py2.saturating_sub(py1)]
    });

    let faces_out1 = faces_v1
        .iter()
        .map(|f| {
            let r = face_rect_px(f, iw, ih);
            let core = face_core_px(r, None, iw, ih);
            FaceOut1 {
                rect: [r1(r.0), r1(r.1), r1(r.2 - r.0), r1(r.3 - r.1)],
                core: [r1(core.0), r1(core.1), r1(core.2), r1(core.3)],
                conf: f.confidence,
            }
        })
        .collect();
    let faces_out2 = faces_v2
        .iter()
        .map(|f| {
            let r = face_rect_px(&f.rect, iw, ih);
            let (core, band) = v2_core_band(f);
            FaceOut2 {
                rect: [r1(r.0), r1(r.1), r1(r.2 - r.0), r1(r.3 - r.1)],
                core: [r1(core.0), r1(core.1), r1(core.2), r1(core.3)],
                band: [r1(band.0), r1(band.1), r1(band.2), r1(band.3)],
                kps: f.landmarks.map(|lm| {
                    lm.map(|p| [r1(p[0] / 100.0 * iw as f32), r1(p[1] / 100.0 * ih as f32)])
                }),
                conf: f.rect.confidence,
            }
        })
        .collect();

    ImageOut {
        id: item["id"].as_str().unwrap_or("").to_string(),
        safe_id: item["safe_id"].as_str().unwrap_or("").to_string(),
        corpus: item["corpus"].as_str().unwrap_or("").to_string(),
        split: item["split"].as_str().unwrap_or("").to_string(),
        raw_role: item["raw_role"].as_str().unwrap_or("").to_string(),
        rel_image: item["rel_image"].as_str().unwrap_or("").to_string(),
        width: iw,
        height: ih,
        faces_v1: faces_out1,
        faces_v2: faces_out2,
        subject_bbox,
        timing_ms: TimingMs {
            prep: prep_ms,
            map: map_ms,
            solve8: solve_ms,
        },
        cells,
    }
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

fn main() {
    let mut manifest_path = PathBuf::from(DEFAULT_MANIFEST);
    let mut out_dir = PathBuf::from(DEFAULT_OUT_DIR);
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--manifest" => manifest_path = PathBuf::from(args.next().expect("--manifest value")),
            "--out-dir" => out_dir = PathBuf::from(args.next().expect("--out-dir value")),
            other => panic!("unknown arg {other} (usage: [--manifest P] [--out-dir D])"),
        }
    }
    let manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&manifest_path).expect("read corpus_manifest.json"))
            .expect("parse corpus_manifest.json");
    let items = manifest["items"]
        .as_array()
        .expect("manifest.items must be an array");
    eprintln!("{} images in {}", items.len(), manifest_path.display());

    let manifest_dir = manifest_path
        .parent()
        .unwrap_or(Path::new("."))
        .to_path_buf();
    let tool_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let models_dir = tool_dir
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("crates/zensally-tract/models");

    // (label, AspectRatio, zenlayout CropMode, SatCropMode)
    let configs: Vec<(String, AspectRatio, CropMode, SatCropMode)> = CELL_LABELS
        .iter()
        .map(|l| {
            let aspect = match l.split('_').next().unwrap() {
                "9:16" => AspectRatio { w: 9, h: 16 },
                "3:4" => AspectRatio { w: 3, h: 4 },
                "1:1" => AspectRatio { w: 1, h: 1 },
                "16:9" => AspectRatio { w: 16, h: 9 },
                other => panic!("bad cell label {other}"),
            };
            let (zmode, smode) = if l.ends_with("_min") {
                (CropMode::Minimal, SatCropMode::Minimal)
            } else {
                (CropMode::Maximal, SatCropMode::Maximal)
            };
            ((*l).to_string(), aspect, zmode, smode)
        })
        .collect();

    let t_all = Instant::now();
    let done = AtomicUsize::new(0);
    let total = items.len();
    let results: Vec<ImageOut> = items
        .par_iter()
        .map_init(
            || Worker::new(&models_dir),
            |w, item| {
                let rel = item["rel_image"].as_str().expect("item.rel_image");
                let img_path = manifest_dir.join(rel);
                let img = image::open(&img_path)
                    .unwrap_or_else(|e| panic!("open {}: {e}", img_path.display()))
                    .to_rgb8();
                let (iw, ih) = (img.width(), img.height());
                let out = eval_image(w, item, img.as_raw(), iw, ih, &configs);
                let n = done.fetch_add(1, Ordering::Relaxed) + 1;
                if n.is_multiple_of(50) || n == total {
                    eprintln!("  [{n}/{total}] {:.0}s", t_all.elapsed().as_secs_f64());
                }
                out
            },
        )
        .collect();
    eprintln!(
        "evaluated {} images in {:.1}s",
        results.len(),
        t_all.elapsed().as_secs_f64()
    );

    // ---- aggregate summaries: split + (corpus, split) ----
    let mut by_split: BTreeMap<String, Agg> = BTreeMap::new();
    let mut by_corpus: BTreeMap<(String, String), Agg> = BTreeMap::new();
    let mut all = Agg::default();
    for o in &results {
        all.add(o);
        by_split.entry(o.split.clone()).or_default().add(o);
        by_corpus
            .entry((o.corpus.clone(), o.split.clone()))
            .or_default()
            .add(o);
    }
    let mut ordered: Vec<(String, Agg)> = Vec::new();
    ordered.push(("all".into(), all));
    for k in ["train", "eval", "test"] {
        if let Some(a) = by_split.remove(k) {
            ordered.push((k.to_string(), a));
        }
    }
    for ((c, s), a) in by_corpus {
        ordered.push((format!("{c}/{s}"), a));
    }
    let summaries: BTreeMap<String, SummaryOut> =
        ordered.iter().map(|(k, a)| (k.clone(), a.into())).collect();

    // ---- SUMMARY.md ----
    let mut md = String::new();
    let _ = writeln!(md, "# zensally v0.2 corpus evaluation — summary\n");
    let _ = writeln!(md, "- images: {}", results.len());
    let _ = writeln!(
        md,
        "- v0.1 = UltraFace + MicroSalNet v1 + zenlayout::smart_crop; v0.2 = YuNet(+5kps) + MicroSalNet v3ds w24 + SelfieSeg + stream_prep + DualIntegralMap::from_crop_inputs + solve_crop_sat"
    );
    let _ = writeln!(
        md,
        "- `cuts` = crops slicing a protected face core (per-axis core/band rule); `v1_yunet` = v0.1 crops scored against the YuNet landmark oracle; `ret` = % of the primary YuNet retention region inside the crop; `conc` = mean w24 saliency inside crop ÷ global mean; `max/min` = area(Maximal)/area(Minimal)\n"
    );
    let _ = writeln!(
        md,
        "| split | n | faces v1 img | faces v2 img | faces v2 | cuts v1 own | cuts v1 yunet | cuts v2 | ret v1 % | ret v2 % | conc v1 | conc v2 | conc max v1 | conc max v2 | max/min v1 | max/min v2 | prep ms | map ms | solve8 ms |"
    );
    let _ = writeln!(
        md,
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"
    );
    let f = |o: Option<f64>| o.map_or("-".into(), |v| format!("{v:.2}"));
    for (name, a) in &ordered {
        let s = SummaryOut::from(a);
        let _ = writeln!(
            md,
            "| {name} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {:.1} | {:.2} | {:.2} |",
            s.n_images,
            s.n_faces_v1_images,
            s.n_faces_v2_images,
            s.total_faces_v2,
            s.v1_own_face_cuts,
            s.v1_yunet_face_cuts,
            s.v2_face_cuts,
            f(s.mean_primary_face_ret_v1),
            f(s.mean_primary_face_ret_v2),
            f(s.mean_conc_v1),
            f(s.mean_conc_v2),
            f(s.mean_conc_max_v1),
            f(s.mean_conc_max_v2),
            f(s.mean_area_ratio_v1),
            f(s.mean_area_ratio_v2),
            s.mean_prep_ms,
            s.mean_map_ms,
            s.mean_solve8_ms,
        );
    }

    let out = ResultsOut {
        schema: "zensally-crop-v2-corpus-results-v1",
        generated_at: "2026-11-11".to_string(),
        manifest_path: manifest_path.display().to_string(),
        n_images: results.len(),
        summaries,
        images: results,
    };

    std::fs::create_dir_all(&out_dir).expect("create out dir");
    let json = serde_json::to_string(&out).expect("serialize results");
    std::fs::write(
        out_dir.join("corpus_data.js"),
        format!("window.ZENSALLY_CORPUS_DATA = {json};\n"),
    )
    .expect("write corpus_data.js");
    let pretty = serde_json::to_string_pretty(&out).expect("serialize results pretty");
    std::fs::write(out_dir.join("corpus_results.json"), pretty).expect("write corpus_results.json");
    std::fs::write(out_dir.join("SUMMARY.md"), &md).expect("write SUMMARY.md");
    eprintln!("wrote {}", out_dir.join("corpus_results.json").display());
    eprintln!("wrote {}", out_dir.join("corpus_data.js").display());
    eprintln!("wrote {}", out_dir.join("SUMMARY.md").display());
    println!("{md}");
}
