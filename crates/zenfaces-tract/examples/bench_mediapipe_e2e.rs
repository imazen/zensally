use std::path::Path;
use std::time::Instant;
use zenfaces::{FaceDetector, ImageRef, PixelFormat};
use zenfaces_tract::MediaPipeBlazeFaceDetector;

fn main() {
    let img_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent().unwrap().parent().unwrap()
        .join("test_data/portrait.jpg");

    let img = image::open(&img_path).expect("failed to open test image");
    let rgb = img.to_rgb8();
    let (w, h) = rgb.dimensions();
    let pixels = rgb.as_raw();

    println!("=== MediaPipe BlazeFace 128x128 End-to-End ===\n");
    println!("Source image: {}x{}", w, h);

    // Time model loading
    let t0 = Instant::now();
    let mut detector = MediaPipeBlazeFaceDetector::new().expect("failed to create detector");
    println!("Model load: {:.1}ms", t0.elapsed().as_secs_f64() * 1000.0);

    let image_ref = ImageRef::new(pixels, w, h, PixelFormat::Rgb).unwrap();

    // Warm up
    for _ in 0..5 {
        let _ = detector.detect(&image_ref);
    }

    // Benchmark at native size (1024x1024)
    let iters = 200;
    let t0 = Instant::now();
    let mut last = Vec::new();
    for _ in 0..iters {
        last = detector.detect(&image_ref);
    }
    let per_ms = t0.elapsed().as_secs_f64() * 1000.0 / iters as f64;
    println!("\n1024x1024: {:.2}ms/iter ({} iters)", per_ms, iters);
    println!("  Faces: {}", last.len());
    for (i, f) in last.iter().enumerate() {
        println!("  [{i}] ({:.1}%, {:.1}%) - ({:.1}%, {:.1}%) conf={:.3}",
            f.x1, f.y1, f.x2, f.y2, f.confidence);
    }

    // Test at different resolutions
    for (tw, th) in [(800, 600), (1920, 1080), (2048, 1536), (640, 480)] {
        let resized = image::imageops::resize(&rgb, tw, th, image::imageops::FilterType::Lanczos3);
        let px = resized.as_raw();
        let ir = ImageRef::new(px, tw, th, PixelFormat::Rgb).unwrap();

        let _ = detector.detect(&ir); // warmup
        let iters = 100;
        let t0 = Instant::now();
        let mut r = Vec::new();
        for _ in 0..iters {
            r = detector.detect(&ir);
        }
        let per_ms = t0.elapsed().as_secs_f64() * 1000.0 / iters as f64;
        println!("\n{}x{}: {:.2}ms/iter, {} faces", tw, th, per_ms, r.len());
    }
}
