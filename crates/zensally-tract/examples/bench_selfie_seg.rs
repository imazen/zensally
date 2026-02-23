fn main() {
    #[cfg(not(feature = "selfie_seg"))]
    {
        eprintln!("This example requires the 'selfie_seg' feature.");
        eprintln!("Run: cargo run --example bench_selfie_seg --features selfie_seg --release");
        return;
    }

    #[cfg(feature = "selfie_seg")]
    run();
}

#[cfg(feature = "selfie_seg")]
fn run() {
    use std::path::Path;
    use std::time::Instant;
    use zensally::{ImageRef, PixelFormat, SaliencyDetector};
    use zensally_tract::SelfieSeg;

    let img_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("test_data/portrait.jpg");

    let img = image::open(&img_path).expect("failed to open test image");
    let rgb = img.to_rgb8();
    let (w, h) = rgb.dimensions();
    let pixels = rgb.as_raw();

    println!("=== MediaPipe Selfie Segmentation ===\n");
    println!("Source image: {}x{}", w, h);

    let t0 = Instant::now();
    let mut detector = SelfieSeg::new().expect("failed to create detector");
    println!(
        "Model load + optimize: {:.1}ms",
        t0.elapsed().as_secs_f64() * 1000.0
    );

    let image_ref = ImageRef::new(pixels, w, h, PixelFormat::Rgb).unwrap();

    // Warmup
    let t0 = Instant::now();
    let map = detector.saliency_map(&image_ref);
    println!(
        "First inference: {:.1}ms",
        t0.elapsed().as_secs_f64() * 1000.0
    );
    println!(
        "  Output: {}x{}, mean={:.3}, max={:.3}",
        map.width,
        map.height,
        map.data.iter().sum::<f32>() / map.data.len() as f32,
        map.data.iter().cloned().fold(0.0f32, f32::max),
    );

    // Timed runs
    let iters = 100;
    let t0 = Instant::now();
    for _ in 0..iters {
        let _ = detector.saliency_map(&image_ref);
    }
    let per_ms = t0.elapsed().as_secs_f64() * 1000.0 / iters as f64;
    println!("\nInference: {:.1}ms/iter ({} iters)", per_ms, iters);
}
