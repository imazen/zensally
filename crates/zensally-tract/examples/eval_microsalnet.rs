/// Visual evaluation of MicroSalNet saliency on test images.
///
/// Usage:
///   cargo run --example eval_microsalnet --features microsalnet --release

fn main() {
    #[cfg(not(feature = "microsalnet"))]
    {
        eprintln!("This example requires the 'microsalnet' feature.");
        eprintln!("Run: cargo run --example eval_microsalnet --features microsalnet --release");
        return;
    }

    #[cfg(feature = "microsalnet")]
    run();
}

#[cfg(feature = "microsalnet")]
fn run() {
    use std::path::{Path, PathBuf};
    use std::time::Instant;
    use zensally::{ImageRef, PixelFormat, SaliencyDetector};
    use zensally_tract::MicroSalNet;

    let test_data = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("test_data");

    let output_dir = PathBuf::from(
        std::env::var("ZENSALLY_OUTPUT_DIR")
            .unwrap_or_else(|_| "/mnt/v/output/zensally".into()),
    )
    .join("microsalnet_eval");
    std::fs::create_dir_all(output_dir).expect("create output dir");

    // Collect all test images
    let mut images: Vec<(String, std::path::PathBuf)> = Vec::new();

    let portrait = test_data.join("portrait.jpg");
    if portrait.exists() {
        images.push(("portrait".into(), portrait));
    }

    let saliency_dir = test_data.join("saliency");
    if saliency_dir.is_dir() {
        let mut entries: Vec<_> = std::fs::read_dir(&saliency_dir)
            .expect("read saliency dir")
            .filter_map(|e| e.ok())
            .filter(|e| {
                e.path()
                    .extension()
                    .is_some_and(|ext| ext == "jpg" || ext == "png")
            })
            .collect();
        entries.sort_by_key(|e| e.file_name());
        for entry in entries {
            let stem = entry
                .path()
                .file_stem()
                .unwrap()
                .to_string_lossy()
                .to_string();
            images.push((stem, entry.path()));
        }
    }

    if images.is_empty() {
        eprintln!("No test images found in {}", test_data.display());
        return;
    }

    println!("=== MicroSalNet Saliency Evaluation ===\n");
    println!("Found {} test images", images.len());
    println!("Output: {}\n", output_dir.display());

    let t0 = Instant::now();
    let mut detector = MicroSalNet::new().expect("failed to create detector");
    println!(
        "Model load: {:.0}ms\n",
        t0.elapsed().as_secs_f64() * 1000.0
    );

    for (name, path) in &images {
        let img = image::open(path).expect("open image");
        let rgb = img.to_rgb8();
        let (w, h) = rgb.dimensions();
        let pixels = rgb.as_raw();

        let image_ref = ImageRef::new(pixels, w, h, PixelFormat::Rgb).unwrap();

        let t0 = Instant::now();
        let map = detector.saliency_map(&image_ref);
        let ms = t0.elapsed().as_secs_f64() * 1000.0;

        let min = map.data.iter().cloned().fold(f32::MAX, f32::min);
        let max = map.data.iter().cloned().fold(f32::MIN, f32::max);
        let mean = map.data.iter().sum::<f32>() / map.data.len() as f32;
        let threshold = max * 0.5;
        let salient_frac =
            map.data.iter().filter(|&&v| v >= threshold).count() as f32 / map.data.len() as f32;

        println!(
            "{:20} {:4}x{:<4} {:7.1}ms  min={:.3} max={:.3} mean={:.3}  salient={:.1}%",
            name, w, h, ms, min, max, mean,
            salient_frac * 100.0
        );

        // Save side-by-side: original | saliency | overlay
        let resized = image::imageops::resize(
            &rgb,
            map.width,
            map.height,
            image::imageops::FilterType::Lanczos3,
        );
        let side_w = map.width * 3;
        let mut sidebyside = image::RgbImage::new(side_w, map.height);
        for y in 0..map.height {
            for x in 0..map.width {
                let idx = y as usize * map.width as usize + x as usize;
                let sal = map.data[idx];

                // Left: resized original
                let px = resized.get_pixel(x, y);
                sidebyside.put_pixel(x, y, *px);

                // Middle: saliency as grayscale
                let v = (sal * 255.0).clamp(0.0, 255.0) as u8;
                sidebyside.put_pixel(map.width + x, y, image::Rgb([v, v, v]));

                // Right: overlay (red tint)
                let r = (px[0] as f32 * (1.0 - sal * 0.5) + 255.0 * sal * 0.5).clamp(0.0, 255.0);
                let g = (px[1] as f32 * (1.0 - sal * 0.5)).clamp(0.0, 255.0);
                let b = (px[2] as f32 * (1.0 - sal * 0.5)).clamp(0.0, 255.0);
                sidebyside.put_pixel(
                    map.width * 2 + x,
                    y,
                    image::Rgb([r as u8, g as u8, b as u8]),
                );
            }
        }
        let sbs_path = output_dir.join(format!("{}_compare.png", name));
        sidebyside.save(&sbs_path).expect("save comparison");
    }

    println!("\nDone. Open {} to view results.", output_dir.display());
}
