//! Per-stage CPU latency benchmark for the zensally v0.1 vs v0.2 crop
//! pipelines (bench brief sally-cpu-gpu-bench-20261111).
//!
//! RAYON_NUM_THREADS=1, 5 warmup + 50 timed iterations per stage, median /
//! p95 / mean reported per stage on two inputs:
//!   * test_data/portrait.jpg (1024×1024)
//!   * a 768×576 corpus-shape image (path arg or default scratch file)
//!
//! Emits JSON to stdout: {"image": {..., "stages": {name: {median_ms, p95_ms,
//! mean_ms}}}} so the driver script can merge it into benchmarks.json.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use sat_crop_bench::sat_crop::{
    CropInputSources, DualIntegralMap, FaceLandmarkInput, SatCropConfig, SatCropMode, TargetToSrc,
    solve_crop_sat,
};
use sat_crop_bench::stream_prep::{self, PrepOutputs};
use zenlayout::smart_crop::{AspectRatio, CropConfig, CropMode as ZlCropMode};
use zensally::preprocess::{Normalization, PreprocessConfig, ResizeMode};
use zensally::{FaceDetector, FaceRect, ImageRef, PixelFormat, SaliencyDetector, SaliencyMap};
use zensally_micro::w24::{W24MicroModel, W24Workspace};
use zensally_tract::{MicroSalNet, UltraFaceDetector};

use tract_onnx::prelude::*;

const MODELS: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../crates/zensally-tract/models"
);

fn gunzip(gz: &[u8]) -> Vec<u8> {
    let n = u32::from_le_bytes(gz[gz.len() - 4..].try_into().unwrap()) as usize;
    let mut out = vec![0u8; n];
    let r = zenflate::Decompressor::new()
        .gzip_decompress(gz, &mut out, enough::Unstoppable)
        .expect("gz decompress failed");
    assert_eq!(r.output_written, n);
    out
}

fn load_model_gz(
    path: &Path,
    input_w: usize,
    input_h: usize,
) -> Arc<tract_onnx::prelude::TypedRunnableModel> {
    let bytes = gunzip(&std::fs::read(path).expect("model read"));
    tract_onnx::onnx()
        .model_for_read(&mut std::io::Cursor::new(bytes))
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
// YuNet + 5-point landmarks (production decode, copied from eval_corpus_v2)
// ---------------------------------------------------------------------------

struct YuNetKps {
    model: Arc<TypedRunnableModel>,
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

    fn detect(&mut self, image: &ImageRef<'_>) -> Vec<FaceLandmarkInput> {
        let (pad_l, pad_t, ratio) = self.preprocess(image);
        let input = Tensor::from_shape(&[1, 3, YUNET_SIZE, YUNET_SIZE], &self.buf).unwrap();
        let outputs = self.model.run(tvec!(input.into())).unwrap();

        type Det = (f32, f32, f32, f32, f32, [[f32; 2]; 5]);
        let mut dets: Vec<Det> = Vec::new();
        for (si, &stride) in YUNET_STRIDES.iter().enumerate() {
            let cls = plain_f32(&outputs[self.out_idx[si]]);
            let obj = plain_f32(&outputs[self.out_idx[3 + si]]);
            let bbox = plain_f32(&outputs[self.out_idx[6 + si]]);
            let kps = plain_f32(&outputs[self.out_idx[9 + si]]);
            let (cols, rows) = (YUNET_SIZE / stride, YUNET_SIZE / stride);
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
// Generic 256²-stretch NCHW model (tract w24 / selfie_seg)
// ---------------------------------------------------------------------------

/// Bilinear-stretch to `size`², run `model`, return the `[1,1,oh,ow]` plane.
fn run_square_model(
    model: &Arc<TypedRunnableModel>,
    buf: &mut Vec<f32>,
    image: &ImageRef<'_>,
    size: usize,
) -> (Vec<f32>, usize, usize) {
    run_square_model_nchw(model, buf, image, size);
    let input = Tensor::from_shape(&[1, 3, size, size], buf.as_slice()).unwrap();
    let outputs = model.run(tvec!(input.into())).unwrap();
    let out = &outputs[0];
    let shape = out.shape().to_vec();
    let (ow, oh) = (shape[3], shape[2]);
    let plane = plain_f32(out);
    (plane.to_vec(), ow, oh)
}

fn run_square_model_nchw(
    model: &Arc<TypedRunnableModel>,
    buf: &mut Vec<f32>,
    image: &ImageRef<'_>,
    size: usize,
) {
    let (src_w, src_h) = (image.width as usize, image.height as usize);
    let bpp = image.format.bytes_per_pixel();
    let (ri, gi, bi) = match image.format {
        PixelFormat::Bgra => (2, 1, 0),
        PixelFormat::Rgba | PixelFormat::Rgb => (0, 1, 2),
    };
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
    let input = Tensor::from_shape(&[1, 3, size, size], buf.as_slice()).unwrap();
    let _ = model.run(tvec!(input.into())).unwrap();
}

// ---------------------------------------------------------------------------
// Timing harness
// ---------------------------------------------------------------------------

fn bench<T, F: FnMut() -> T>(times: &mut Vec<f64>, mut f: F) {
    for _ in 0..5 {
        f();
    }
    for _ in 0..50 {
        let t = Instant::now();
        f();
        times.push(t.elapsed().as_secs_f64() * 1e3);
    }
}

fn stats(times: &mut [f64]) -> String {
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = times.len();
    let median = times[n / 2];
    let p95 = times[((n as f64) * 0.95) as usize - 1];
    let mean: f64 = times.iter().sum::<f64>() / n as f64;
    format!("{median:.3},{p95:.3},{mean:.3}")
}

fn decode_rgb(path: &Path) -> (Vec<u8>, u32, u32) {
    let img = image::open(path).expect("decode").to_rgb8();
    let (w, h) = img.dimensions();
    (img.into_raw(), w, h)
}

#[allow(clippy::too_many_lines)]
fn bench_image(name: &str, pixels: &[u8], iw: u32, ih: u32, w: &mut Worker, out: &mut Vec<String>) {
    let image = ImageRef {
        pixels,
        width: iw,
        height: ih,
        format: PixelFormat::Rgb,
    };
    let (tw, th, gw, gh) = (512usize, 384usize, 64usize, 48usize);

    let emit = |out: &mut Vec<String>, key: &str, times: &mut Vec<f64>| {
        out.push(format!("\"{name}.{key}\":[{}]", stats(times)));
    };

    // v0.1 stages -----------------------------------------------------------
    emit(out, "v01_ultraface", &mut {
        let mut t = Vec::new();
        bench(&mut t, || std::hint::black_box(w.ultra.detect(&image)));
        t
    });
    emit(out, "v01_microsalnet_v1", &mut {
        let mut t = Vec::new();
        bench(&mut t, || {
            std::hint::black_box(w.sal_v1.saliency_map(&image))
        });
        t
    });

    let faces_v1 = w.ultra.detect(&image);
    let sal1 = w.sal_v1.saliency_map(&image);
    let smart_in = zensally::bridge::build_smart_crop_input_raw(faces_v1.clone(), Some(sal1), &[]);
    let zcfg = CropConfig {
        target_aspect: AspectRatio { w: 3, h: 4 },
        mode: ZlCropMode::Minimal,
        ..CropConfig::default()
    };
    emit(out, "v01_smart_crop_1x", &mut {
        let mut t = Vec::new();
        bench(&mut t, || {
            std::hint::black_box(smart_in.compute_crop(iw, ih, &zcfg))
        });
        t
    });
    let configs8: [(u32, u32, ZlCropMode); 8] = [
        (9, 16, ZlCropMode::Minimal),
        (9, 16, ZlCropMode::Maximal),
        (3, 4, ZlCropMode::Minimal),
        (3, 4, ZlCropMode::Maximal),
        (1, 1, ZlCropMode::Minimal),
        (1, 1, ZlCropMode::Maximal),
        (16, 9, ZlCropMode::Minimal),
        (16, 9, ZlCropMode::Maximal),
    ];
    emit(out, "v01_smart_crop_8x", &mut {
        let mut t = Vec::new();
        bench(&mut t, || {
            for &(aw, ah, m) in &configs8 {
                let c = CropConfig {
                    target_aspect: AspectRatio { w: aw, h: ah },
                    mode: m,
                    ..CropConfig::default()
                };
                std::hint::black_box(smart_in.compute_crop(iw, ih, &c));
            }
        });
        t
    });
    emit(out, "v01_total", &mut {
        let mut t = Vec::new();
        bench(&mut t, || {
            let f = w.ultra.detect(&image);
            let s = w.sal_v1.saliency_map(&image);
            let si = zensally::bridge::build_smart_crop_input_raw(f, Some(s), &[]);
            for &(aw, ah, m) in &configs8 {
                let c = CropConfig {
                    target_aspect: AspectRatio { w: aw, h: ah },
                    mode: m,
                    ..CropConfig::default()
                };
                std::hint::black_box(si.compute_crop(iw, ih, &c));
            }
        });
        t
    });

    // v0.2 stages -----------------------------------------------------------
    let mut outputs = PrepOutputs {
        output_nhwc: &mut w.nhwc,
        classical_energy: &mut w.classical,
        horizon_profile: &mut w.horizon,
        scratch: &mut w.scratch,
    };
    emit(out, "v02_stream_prep", &mut {
        let mut t = Vec::new();
        bench(&mut t, || {
            stream_prep::preprocess_nhwc_strided(
                pixels,
                iw,
                ih,
                iw as usize * 3,
                PixelFormat::Rgb,
                &w.prep_cfg,
                &mut outputs,
            );
        });
        t
    });

    // NHWC model input at 256² (bilinear stretch, same as tract path).
    let size = 256usize;
    let mut nhwc256 = vec![0.0f32; 3 * size * size];
    {
        let (src_w, src_h) = (iw as usize, ih as usize);
        let stride = src_w * 3;
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
                let (o00, o10) = (y0 * stride + x0 * 3, y0 * stride + x1 * 3);
                let (o01, o11) = (y1 * stride + x0 * 3, y1 * stride + x1 * 3);
                for c in 0..3 {
                    nhwc256[(dy * size + dx) * 3 + c] = (pixels[o00 + c] as f32 * w00
                        + pixels[o10 + c] as f32 * w10
                        + pixels[o01 + c] as f32 * w01
                        + pixels[o11 + c] as f32 * w11)
                        / 255.0;
                }
            }
        }
    }
    emit(out, "v02_w24_tract_nchw", &mut {
        let mut t = Vec::new();
        bench(&mut t, || {
            run_square_model_nchw(&w.sal_w24_tract, &mut w.model_buf, &image, 256)
        });
        t
    });
    emit(out, "v02_w24_micro_nhwc", &mut {
        let mut t = Vec::new();
        bench(&mut t, || {
            std::hint::black_box(nhwc256.as_slice());
            w.w24micro
                .forward_nhwc(&nhwc256, &mut w.w24ws, &mut w.w24out);
        });
        t
    });
    emit(out, "v02_selfie_seg_tract", &mut {
        let mut t = Vec::new();
        bench(&mut t, || {
            run_square_model_nchw(&w.seg_tract, &mut w.model_buf, &image, 256)
        });
        t
    });
    emit(out, "v02_yunet_kps", &mut {
        let mut t = Vec::new();
        bench(&mut t, || std::hint::black_box(w.yunet.detect(&image)));
        t
    });

    // Detect + run models once so the map/solve benches consume real inputs.
    let faces_v2 = w.yunet.detect(&image);
    let (sal2_data, sal2_w, sal2_h) =
        run_square_model(&w.sal_w24_tract, &mut w.model_buf, &image, 256);
    let (matte_data, matte_w, matte_h) =
        run_square_model(&w.seg_tract, &mut w.model_buf, &image, 256);
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
    let to_src = TargetToSrc::stretch(iw, ih, tw as u32, th as u32);
    let map = DualIntegralMap::from_crop_inputs(
        &mk_crop_sources(w, &sal2, &matte, &faces_v2),
        gw,
        gh,
        &to_src,
        0.25,
    );
    emit(out, "v02_dual_integral_map", &mut {
        let mut t = Vec::new();
        bench(&mut t, || {
            std::hint::black_box(DualIntegralMap::from_crop_inputs(
                &mk_crop_sources(w, &sal2, &matte, &faces_v2),
                gw,
                gh,
                &to_src,
                0.25,
            ))
        });
        t
    });
    emit(out, "v02_solve_crop_sat_1x", &mut {
        let mut t = Vec::new();
        bench(&mut t, || {
            std::hint::black_box(solve_crop_sat(
                &map,
                &to_src,
                (3, 4),
                SatCropMode::Minimal,
                &SatCropConfig::default(),
            ))
        });
        t
    });
    let s8: [((u32, u32), SatCropMode); 8] = [
        ((9, 16), SatCropMode::Minimal),
        ((9, 16), SatCropMode::Maximal),
        ((3, 4), SatCropMode::Minimal),
        ((3, 4), SatCropMode::Maximal),
        ((1, 1), SatCropMode::Minimal),
        ((1, 1), SatCropMode::Maximal),
        ((16, 9), SatCropMode::Minimal),
        ((16, 9), SatCropMode::Maximal),
    ];
    emit(out, "v02_solve_crop_sat_8x", &mut {
        let mut t = Vec::new();
        bench(&mut t, || {
            for &(a, m) in &s8 {
                std::hint::black_box(solve_crop_sat(
                    &map,
                    &to_src,
                    a,
                    m,
                    &SatCropConfig::default(),
                ));
            }
        });
        t
    });

    // Totals: fast path = prep + w24micro + map + 8 solves; full = prep +
    // w24micro + seg + yunet + map + 8 solves. (NHWC output of preprocess is
    // 512×384; the w24 model input is stretched to 256² separately — the
    // stream prep NHWC feeds the classical path.)
    emit(out, "v02_total_fast", &mut {
        let mut t = Vec::new();
        bench(&mut t, || {
            {
                let mut outputs = PrepOutputs {
                    output_nhwc: &mut w.nhwc,
                    classical_energy: &mut w.classical,
                    horizon_profile: &mut w.horizon,
                    scratch: &mut w.scratch,
                };
                stream_prep::preprocess_nhwc_strided(
                    pixels,
                    iw,
                    ih,
                    iw as usize * 3,
                    PixelFormat::Rgb,
                    &w.prep_cfg,
                    &mut outputs,
                );
            }
            w.w24micro
                .forward_nhwc(&nhwc256, &mut w.w24ws, &mut w.w24out);
            let m = DualIntegralMap::from_crop_inputs(
                &mk_crop_sources(w, &sal2, &matte, &faces_v2),
                gw,
                gh,
                &to_src,
                0.25,
            );
            for &(a, md) in &s8 {
                std::hint::black_box(solve_crop_sat(
                    &m,
                    &to_src,
                    a,
                    md,
                    &SatCropConfig::default(),
                ));
            }
        });
        t
    });
    emit(out, "v02_total_full", &mut {
        let mut t = Vec::new();
        bench(&mut t, || {
            {
                let mut outputs = PrepOutputs {
                    output_nhwc: &mut w.nhwc,
                    classical_energy: &mut w.classical,
                    horizon_profile: &mut w.horizon,
                    scratch: &mut w.scratch,
                };
                stream_prep::preprocess_nhwc_strided(
                    pixels,
                    iw,
                    ih,
                    iw as usize * 3,
                    PixelFormat::Rgb,
                    &w.prep_cfg,
                    &mut outputs,
                );
            }
            w.w24micro
                .forward_nhwc(&nhwc256, &mut w.w24ws, &mut w.w24out);
            run_square_model_nchw(&w.seg_tract, &mut w.model_buf, &image, 256);
            std::hint::black_box(w.yunet.detect(&image));
            let m = DualIntegralMap::from_crop_inputs(
                &mk_crop_sources(w, &sal2, &matte, &faces_v2),
                gw,
                gh,
                &to_src,
                0.25,
            );
            for &(a, md) in &s8 {
                std::hint::black_box(solve_crop_sat(
                    &m,
                    &to_src,
                    a,
                    md,
                    &SatCropConfig::default(),
                ));
            }
        });
        t
    });
}

fn mk_crop_sources<'a>(
    w: &'a Worker,
    sal2: &'a SaliencyMap,
    matte: &'a SaliencyMap,
    faces: &'a [FaceLandmarkInput],
) -> CropInputSources<'a> {
    CropInputSources {
        saliency: Some(sal2),
        matte: Some(matte),
        faces,
        manual_focus: &[],
        classical_energy: Some(&w.classical),
        horizon_profile: Some(&w.horizon),
    }
}

struct Worker {
    ultra: UltraFaceDetector,
    sal_v1: MicroSalNet,
    yunet: YuNetKps,
    sal_w24_tract: Arc<TypedRunnableModel>,
    seg_tract: Arc<TypedRunnableModel>,
    w24micro: W24MicroModel,
    w24ws: W24Workspace,
    w24out: Vec<f32>,
    nhwc: Vec<f32>,
    classical: Vec<f32>,
    horizon: Vec<f32>,
    scratch: Vec<f32>,
    model_buf: Vec<f32>,
    prep_cfg: PreprocessConfig,
}

fn main() {
    let corpus_img = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/home/lilith/tmp/sally-crop-v2/corpus_768x576.jpg"));
    let models_dir = PathBuf::from(MODELS);
    let (gw, gh) = (64usize, 48usize);
    let mut w = Worker {
        ultra: UltraFaceDetector::new().expect("ultraface"),
        sal_v1: MicroSalNet::new().expect("microsalnet v1"),
        yunet: YuNetKps::new(&models_dir),
        sal_w24_tract: load_model_gz(&models_dir.join("microsalnet_v3ds_w24.onnx.gz"), 256, 256),
        seg_tract: load_model_gz(&models_dir.join("selfie_seg.onnx.gz"), 256, 256),
        w24micro: W24MicroModel::new().expect("w24 micro"),
        w24ws: W24Workspace::new(),
        w24out: vec![0.0; 128 * 128],
        nhwc: vec![0.0; 3 * 512 * 384],
        classical: vec![0.0; gw * gh],
        horizon: vec![0.0; gh],
        scratch: vec![0.0; stream_prep::prep_scratch_len(512, gw, gh)],
        model_buf: vec![0.0; 3 * 256 * 256],
        prep_cfg: PreprocessConfig {
            target_w: 512,
            target_h: 384,
            mode: ResizeMode::Stretch,
            norm: Normalization::UnitScale,
        },
    };
    let mut out: Vec<String> = Vec::new();
    for (name, path) in [
        (
            "portrait_1024x1024",
            concat!(env!("CARGO_MANIFEST_DIR"), "/../../test_data/portrait.jpg"),
        ),
        ("corpus_768x576", corpus_img.to_str().unwrap()),
    ] {
        let (px, iw, ih) = decode_rgb(Path::new(path));
        eprintln!("[bench] {name} {iw}x{ih}");
        bench_image(name, &px, iw, ih, &mut w, &mut out);
    }
    println!("{{{}}}", out.join(","));
}
