//! Compare source and 128x128 map-pixel rectangle edges on frozen inputs. No timings.
use codec_corpus::Corpus;
use sha2::{Digest, Sha256};
use std::{
    fs::OpenOptions,
    io::{Cursor, Write},
    path::Path,
};
use tract_onnx::prelude::*;
use zenlayout::smart_crop::{AspectRatio, CropConfig, CropMode, HeatMap, SmartCropInput};
use zenpixels::ChannelType;
use zensally::{
    PixelFormat,
    preprocess::{Normalization, ResizeMode, preprocess_nchw},
};
use zensally_micro::{Input, MicroSalNetW24};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

fn png_rgb(path: &Path) -> Result<(Vec<u8>, u32, u32)> {
    let bytes = std::fs::read(path)?;
    let mut config = zenpng::PngDecodeConfig::strict();
    config.max_pixels = Some(64 * 1024 * 1024);
    config.max_memory_bytes = Some(512 * 1024 * 1024);
    let image = zenpng::decode(&bytes, &config, &enough::Unstoppable)?;
    let pixels = image.pixels.as_slice();
    let (w, h) = (pixels.width(), pixels.rows());
    let channels = pixels.descriptor().channels();
    let gray = image.pixels.is_grayscale();
    if !(1..=4).contains(&channels) {
        return Err("unsupported PNG channel count".into());
    }
    let mut rgb = vec![0; w as usize * h as usize * 3];
    for y in 0..h {
        let row = pixels.row(y);
        for x in 0..w as usize {
            for c in 0..3 {
                let i = x * channels + if gray { 0 } else { c };
                rgb[(y as usize * w as usize + x) * 3 + c] =
                    match pixels.descriptor().channel_type() {
                        ChannelType::U8 => row[i],
                        ChannelType::U16 => {
                            ((u32::from(u16::from_ne_bytes([row[i * 2], row[i * 2 + 1]])) * 255
                                + 32767)
                                / 65535) as u8
                        }
                        _ => return Err("unsupported PNG channel type".into()),
                    };
            }
        }
    }
    Ok((rgb, w, h))
}

fn resized_rgb(rgb: &[u8], w: u32, h: u32, dw: u32, dh: u32) -> Vec<u8> {
    let area = dw as usize * dh as usize;
    let mut planar = vec![0.0; area * 3];
    preprocess_nchw(
        rgb,
        w,
        h,
        PixelFormat::Rgb,
        dw as usize,
        dh as usize,
        ResizeMode::Stretch,
        Normalization::UnitScale,
        &mut planar,
    );
    let mut out = vec![0; area * 3];
    for i in 0..area {
        for c in 0..3 {
            out[i * 3 + c] = (planar[c * area + i] * 255.0).round().clamp(0.0, 255.0) as u8;
        }
    }
    out
}

// Owning source: zenlayout 0.2.2 smart_crop::heatmap_bbox. Kept private in
// this audit because production geometry continues to use the shared solver.
fn bbox(hm: &HeatMap, threshold: f64, w: u32, h: u32) -> [f64; 4] {
    let max = hm.data.iter().copied().fold(0.0f32, f32::max) as f64;
    if max < 1e-10 {
        return [
            0.25 * w as f64,
            0.25 * h as f64,
            0.75 * w as f64,
            0.75 * h as f64,
        ];
    }
    let (mut x0, mut y0, mut x1, mut y1) = (hm.width, hm.height, 0, 0);
    for y in 0..hm.height {
        for x in 0..hm.width {
            if hm.data[(y * hm.width + x) as usize] as f64 >= max * threshold {
                x0 = x0.min(x);
                y0 = y0.min(y);
                x1 = x1.max(x);
                y1 = y1.max(y);
            }
        }
    }
    [
        x0 as f64 / hm.width as f64 * w as f64,
        y0 as f64 / hm.height as f64 * h as f64,
        (x1 + 1) as f64 / hm.width as f64 * w as f64,
        (y1 + 1) as f64 / hm.height as f64 * h as f64,
    ]
}
fn movement(a: [f64; 4], b: [f64; 4], scale: [f64; 4]) -> f64 {
    a.into_iter()
        .zip(b)
        .zip(scale)
        .map(|((a, b), scale)| (a - b).abs() * scale)
        .fold(0.0, f64::max)
}
fn compare_rectangles(
    a: &HeatMap,
    b: &HeatMap,
    w: u32,
    h: u32,
    map_pixels: bool,
) -> (f64, f64, bool) {
    let scale = if map_pixels {
        [
            128.0 / w as f64,
            128.0 / h as f64,
            128.0 / w as f64,
            128.0 / h as f64,
        ]
    } else {
        [1.0; 4]
    };
    let roi = [0.3, 0.5]
        .into_iter()
        .map(|t| movement(bbox(a, t, w, h), bbox(b, t, w, h), scale))
        .fold(0.0, f64::max);
    let a = SmartCropInput {
        focus_regions: vec![],
        heatmap: Some(a.clone()),
    };
    let b = SmartCropInput {
        focus_regions: vec![],
        heatmap: Some(b.clone()),
    };
    let mut crop: f64 = 0.0;
    let mut presence_mismatch = false;
    for (rw, rh) in [
        (1, 1),
        (16, 9),
        (9, 16),
        (4, 3),
        (3, 4),
        (4, 5),
        (5, 4),
        (2, 1),
        (1, 2),
    ] {
        for mode in [CropMode::Minimal, CropMode::Maximal] {
            let config = CropConfig {
                target_aspect: AspectRatio { w: rw, h: rh },
                mode,
                ..Default::default()
            };
            match (a.compute_crop(w, h, &config), b.compute_crop(w, h, &config)) {
                (Some(a), Some(b)) => {
                    crop = crop.max(movement(
                        [
                            a.x as f64,
                            a.y as f64,
                            (a.x as u64 + a.width as u64) as f64,
                            (a.y as u64 + a.height as u64) as f64,
                        ],
                        [
                            b.x as f64,
                            b.y as f64,
                            (b.x as u64 + b.width as u64) as f64,
                            (b.y as u64 + b.height as u64) as f64,
                        ],
                        scale,
                    ));
                }
                (None, None) => {}
                _ => presence_mismatch = true,
            }
        }
    }
    (roi, crop, presence_mismatch)
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 5 {
        return Err(
            "usage: eval_rectangles CORPUS_DIR INPUTS.tsv REPORT.tsv MAX_EDGE_MAP_PIXELS".into(),
        );
    }
    let corpus = Corpus::new()?.local_path(&args[1])?;
    let budget: f64 = args[4].parse()?;
    if !budget.is_finite() || budget < 0.0 {
        return Err("invalid rectangle budget".into());
    }
    let manifest = std::fs::read_to_string(&args[2])?;
    let gz = include_bytes!("../../zensally-tract/models/microsalnet_v3ds_w24.onnx.gz");
    let mut bytes = vec![0; 769236];
    zenflate::Decompressor::new()
        .gzip_decompress(gz, &mut bytes, enough::Unstoppable)
        .map_err(|e| format!("{e:?}"))?;
    let tract = tract_onnx::onnx()
        .model_for_read(&mut Cursor::new(bytes))?
        .with_input_fact(0, InferenceFact::dt_shape(DatumType::F32, [1, 3, 256, 256]))?
        .into_optimized()?
        .into_runnable()?;
    let mut models = [
        ("fixed-f32", MicroSalNetW24::new()?),
        ("fast-sigmoid", MicroSalNetW24::new()?.with_fast_sigmoid()),
    ];
    let mut report = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&args[3])?;
    writeln!(
        report,
        "id\tclass\tsource_width\tsource_height\tcandidate\tmax_map_abs_delta\tmean_map_abs_delta\troi_edge_source_px\tcrop_edge_source_px\troi_edge_map_px\tcrop_edge_map_px\tpresence_mismatch\twithin_budget"
    )?;
    let (mut images, mut cases, mut failed) = (0, 0, 0);
    let (mut max_roi, mut max_crop) = (0.0f64, 0.0f64);
    for line in manifest.lines().skip(1) {
        let fields: Vec<&str> = line.split('\t').collect();
        if fields.len() != 6 {
            return Err("invalid frozen input row".into());
        }
        let path = corpus.join(fields[2]);
        let encoded = std::fs::read(&path)?;
        let actual: String = Sha256::digest(&encoded)
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect();
        if actual != fields[3] {
            return Err(format!("input hash changed for {}", fields[0]).into());
        }
        let (rgb, w, h) = png_rgb(&path)?;
        let longest = w.max(h);
        let mut sizes: Vec<(u32, u32)> = [64, 256, 1024, 4096]
            .into_iter()
            .filter(|&side| side < longest)
            .map(|side| {
                (
                    ((w as u64 * side as u64 + longest as u64 / 2) / longest as u64).max(1) as u32,
                    ((h as u64 * side as u64 + longest as u64 / 2) / longest as u64).max(1) as u32,
                )
            })
            .collect();
        sizes.push((w, h));
        for (sw, sh) in sizes {
            let resized = if (sw, sh) == (w, h) {
                rgb.clone()
            } else {
                resized_rgb(&rgb, w, h, sw, sh)
            };
            let mut input = vec![0.0; 3 * 65536];
            preprocess_nchw(
                &resized,
                sw,
                sh,
                PixelFormat::Rgb,
                256,
                256,
                ResizeMode::Stretch,
                Normalization::UnitScale,
                &mut input,
            );
            let tensor = Tensor::from_shape(&[1, 3, 256, 256], &input)?;
            let out = tract.run(tvec!(tensor.into()))?;
            let expected = out[0]
                .as_plain_ram()
                .ok_or("nonplain tract output")?
                .as_slice::<f32>()?;
            if expected.len() != 16384 || expected.iter().any(|v| !v.is_finite()) {
                return Err("invalid tract map".into());
            }
            let reference = HeatMap {
                data: expected.to_vec(),
                width: 128,
                height: 128,
            };
            for (name, model) in &mut models {
                let actual = model.infer(Input::contiguous(&input)?)?;
                if actual.data.iter().any(|v| !v.is_finite()) {
                    return Err("invalid custom map".into());
                }
                let max_delta = actual
                    .data
                    .iter()
                    .zip(expected)
                    .map(|(&a, &b)| (a - b).abs())
                    .fold(0.0f32, f32::max);
                let mean_delta = actual
                    .data
                    .iter()
                    .zip(expected)
                    .map(|(&a, &b)| (a - b).abs() as f64)
                    .sum::<f64>()
                    / expected.len() as f64;
                let candidate = HeatMap {
                    data: actual.data,
                    width: 128,
                    height: 128,
                };
                let (roi, crop, presence) =
                    compare_rectangles(&reference, &candidate, sw, sh, true);
                let (source_roi, source_crop, _) =
                    compare_rectangles(&reference, &candidate, sw, sh, false);
                let pass = roi <= budget && crop <= budget && !presence;
                writeln!(
                    report,
                    "{}\t{}\t{sw}\t{sh}\t{name}\t{max_delta:.9e}\t{mean_delta:.9e}\t{source_roi:.9}\t{source_crop:.9}\t{roi:.9}\t{crop:.9}\t{presence}\t{pass}",
                    fields[0], fields[1]
                )?;
                cases += 1;
                failed += usize::from(!pass);
                max_roi = max_roi.max(roi);
                max_crop = max_crop.max(crop);
            }
        }
        report.flush()?;
        images += 1;
        println!(
            "images={images} cases={cases} failed={failed} max_roi_map_px={max_roi:.9} max_crop_map_px={max_crop:.9}"
        );
    }
    if images == 0 {
        return Err("empty corpus".into());
    }
    println!(
        "complete images={images} cases={cases} failed={failed} budget_map_px={budget} max_roi_map_px={max_roi:.9} max_crop_map_px={max_crop:.9}"
    );
    if failed != 0 {
        return Err("rectangle movement budget exceeded; see per-case report".into());
    }
    Ok(())
}
