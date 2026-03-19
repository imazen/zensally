/// Evaluate saliency models on DUTS-TE benchmark.
///
/// Computes MAE (Mean Absolute Error) and adaptive F-measure against
/// ground truth saliency masks. Lower MAE = better. Higher F = better.
///
/// Usage:
///   cargo run --example eval_duts --features u2netp --release [-- LIMIT]
///   cargo run --example eval_duts --features selfie_seg --release [-- LIMIT]
///   cargo run --example eval_duts --features microsalnet --release [-- LIMIT]
///   cargo run --example eval_duts --features "u2netp,microsalnet" --release [-- LIMIT]
///
/// Expects DUTS-TE dataset at: data/DUTS-TE/DUTS-TE-Image/ and data/DUTS-TE/DUTS-TE-Mask/
fn main() {
    #[cfg(not(any(feature = "u2netp", feature = "selfie_seg", feature = "microsalnet")))]
    {
        eprintln!("This example requires the 'u2netp', 'selfie_seg', or 'microsalnet' feature.");
        return;
    }

    #[cfg(any(feature = "u2netp", feature = "selfie_seg", feature = "microsalnet"))]
    run();
}

#[cfg(any(feature = "u2netp", feature = "selfie_seg", feature = "microsalnet"))]
fn run() {
    use std::path::Path;
    use std::time::Instant;
    use zensally::{ImageRef, PixelFormat, SaliencyDetector};

    let workspace_root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap();

    let image_dir = workspace_root.join("data/DUTS-TE/DUTS-TE-Image");
    let mask_dir = workspace_root.join("data/DUTS-TE/DUTS-TE-Mask");

    if !image_dir.exists() || !mask_dir.exists() {
        eprintln!("DUTS-TE dataset not found.");
        eprintln!("Expected: data/DUTS-TE/DUTS-TE-Image/ and data/DUTS-TE/DUTS-TE-Mask/");
        eprintln!("Download: https://saliencydetection.net/duts/download/DUTS-TE.zip");
        return;
    }

    let mut image_paths: Vec<_> = std::fs::read_dir(&image_dir)
        .expect("read image dir")
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.path()
                .extension()
                .is_some_and(|ext| ext == "jpg" || ext == "png")
        })
        .map(|e| e.path())
        .collect();
    image_paths.sort();

    let total = image_paths.len();
    let limit: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(total);
    let image_paths = &image_paths[..limit.min(total)];

    // Build list of detectors to evaluate
    let mut detectors: Vec<(String, Box<dyn SaliencyDetector>)> = Vec::new();

    #[cfg(feature = "u2netp")]
    {
        let t0 = Instant::now();
        let d = zensally_tract::U2NetpDetector::new().expect("U2-Netp load failed");
        println!("U2-Netp load: {:.0}ms", t0.elapsed().as_secs_f64() * 1000.0);
        detectors.push(("U2-Netp".into(), Box::new(d)));
    }

    #[cfg(feature = "selfie_seg")]
    {
        let t0 = Instant::now();
        let d = zensally_tract::SelfieSeg::new().expect("SelfieSeg load failed");
        println!(
            "SelfieSeg load: {:.0}ms",
            t0.elapsed().as_secs_f64() * 1000.0
        );
        detectors.push(("SelfieSeg".into(), Box::new(d)));
    }

    #[cfg(feature = "microsalnet")]
    {
        let t0 = Instant::now();
        let d = zensally_tract::MicroSalNet::new().expect("MicroSalNet load failed");
        println!(
            "MicroSalNet load: {:.0}ms",
            t0.elapsed().as_secs_f64() * 1000.0
        );
        detectors.push(("MicroSalNet".into(), Box::new(d)));
    }

    println!(
        "\n=== DUTS-TE Evaluation: {} images, {} models ===\n",
        image_paths.len(),
        detectors.len()
    );

    for (model_name, detector) in &mut detectors {
        let mut total_mae = 0.0f64;
        let mut total_precision = 0.0f64;
        let mut total_recall = 0.0f64;
        let mut total_f = 0.0f64;
        let mut total_ms = 0.0f64;
        let mut count = 0usize;
        let mut skipped = 0usize;

        let t_start = Instant::now();

        for (i, img_path) in image_paths.iter().enumerate() {
            let stem = img_path.file_stem().unwrap().to_string_lossy();
            let mask_path = mask_dir.join(format!("{}.png", stem));
            if !mask_path.exists() {
                skipped += 1;
                continue;
            }

            let img = match image::open(img_path) {
                Ok(img) => img,
                Err(_) => {
                    skipped += 1;
                    continue;
                }
            };
            let rgb = img.to_rgb8();
            let (w, h) = rgb.dimensions();
            let pixels = rgb.as_raw();

            let mask_img = match image::open(&mask_path) {
                Ok(m) => m.into_luma8(),
                Err(_) => {
                    skipped += 1;
                    continue;
                }
            };

            let image_ref = ImageRef::new(pixels, w, h, PixelFormat::Rgb).unwrap();
            let t0 = Instant::now();
            let map = detector.saliency_map(&image_ref);
            let ms = t0.elapsed().as_secs_f64() * 1000.0;
            total_ms += ms;

            let (gt_w, gt_h) = mask_img.dimensions();
            let pred_resized = resize_saliency(&map.data, map.width, map.height, gt_w, gt_h);
            let gt: Vec<f32> = mask_img.as_raw().iter().map(|&v| v as f32 / 255.0).collect();

            let mae: f64 = pred_resized
                .iter()
                .zip(gt.iter())
                .map(|(&p, &g)| (p - g).abs() as f64)
                .sum::<f64>()
                / gt.len() as f64;
            total_mae += mae;

            let pred_mean: f32 = pred_resized.iter().sum::<f32>() / pred_resized.len() as f32;
            let threshold = (2.0 * pred_mean).min(1.0);

            let mut tp = 0u64;
            let mut fp = 0u64;
            let mut fn_ = 0u64;
            for (&p, &g) in pred_resized.iter().zip(gt.iter()) {
                let pred_pos = p >= threshold;
                let gt_pos = g >= 0.5;
                if pred_pos && gt_pos {
                    tp += 1;
                } else if pred_pos && !gt_pos {
                    fp += 1;
                } else if !pred_pos && gt_pos {
                    fn_ += 1;
                }
            }

            let precision = if tp + fp > 0 {
                tp as f64 / (tp + fp) as f64
            } else {
                0.0
            };
            let recall = if tp + fn_ > 0 {
                tp as f64 / (tp + fn_) as f64
            } else {
                0.0
            };
            let beta2 = 0.3;
            let f_measure = if precision + recall > 0.0 {
                (1.0 + beta2) * precision * recall / (beta2 * precision + recall)
            } else {
                0.0
            };

            total_precision += precision;
            total_recall += recall;
            total_f += f_measure;
            count += 1;

            if (i + 1) % 200 == 0 || i + 1 == image_paths.len() {
                let elapsed = t_start.elapsed().as_secs_f64();
                let rate = (i + 1) as f64 / elapsed;
                let eta = (image_paths.len() - i - 1) as f64 / rate;
                println!(
                    "  {}: [{:5}/{}] MAE={:.4} F={:.4} {:.1}ms/img  ETA {:.0}s",
                    model_name,
                    i + 1,
                    image_paths.len(),
                    total_mae / count as f64,
                    total_f / count as f64,
                    total_ms / count as f64,
                    eta,
                );
            }
        }

        println!(
            "\n--- {} ({} images, {} skipped) ---",
            model_name, count, skipped
        );
        println!("  MAE:       {:.4}", total_mae / count as f64);
        println!("  Precision: {:.4}", total_precision / count as f64);
        println!("  Recall:    {:.4}", total_recall / count as f64);
        println!(
            "  F_beta:    {:.4}  (beta^2=0.3)",
            total_f / count as f64
        );
        println!("  Avg time:  {:.1}ms/image", total_ms / count as f64);
        println!("  Total:     {:.1}s\n", t_start.elapsed().as_secs_f64());
    }
}

/// Bilinear resize of a flat saliency map to target dimensions.
#[cfg(any(feature = "u2netp", feature = "selfie_seg", feature = "microsalnet"))]
fn resize_saliency(data: &[f32], src_w: u32, src_h: u32, dst_w: u32, dst_h: u32) -> Vec<f32> {
    let mut out = vec![0.0f32; (dst_w * dst_h) as usize];
    let x_ratio = if dst_w > 1 {
        (src_w as f32 - 1.0) / (dst_w as f32 - 1.0)
    } else {
        0.0
    };
    let y_ratio = if dst_h > 1 {
        (src_h as f32 - 1.0) / (dst_h as f32 - 1.0)
    } else {
        0.0
    };

    for dy in 0..dst_h as usize {
        let sy = dy as f32 * y_ratio;
        let y0 = sy as usize;
        let y1 = (y0 + 1).min(src_h as usize - 1);
        let fy = sy - y0 as f32;

        for dx in 0..dst_w as usize {
            let sx = dx as f32 * x_ratio;
            let x0 = sx as usize;
            let x1 = (x0 + 1).min(src_w as usize - 1);
            let fx = sx - x0 as f32;

            let v00 = data[y0 * src_w as usize + x0];
            let v10 = data[y0 * src_w as usize + x1];
            let v01 = data[y1 * src_w as usize + x0];
            let v11 = data[y1 * src_w as usize + x1];

            let v = v00 * (1.0 - fx) * (1.0 - fy)
                + v10 * fx * (1.0 - fy)
                + v01 * (1.0 - fx) * fy
                + v11 * fx * fy;
            out[dy * dst_w as usize + dx] = v;
        }
    }
    out
}
