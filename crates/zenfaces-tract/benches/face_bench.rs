#![forbid(unsafe_code)]

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use zenfaces::{FaceDetector, ImageRef, PixelFormat};
use zenfaces_tract::MediaPipeBlazeFaceDetector;

/// Load portrait.jpg and resize to the given dimensions using nearest-neighbor.
/// Returns RGB pixel data.
fn load_test_image(width: u32, height: u32) -> Vec<u8> {
    let img_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("test_data/portrait.jpg");

    let img = image::open(img_path).expect("failed to open test image");
    let resized = img.resize_exact(width, height, image::imageops::FilterType::Triangle);
    resized.to_rgb8().into_raw()
}

fn bench_model_load(c: &mut Criterion) {
    c.bench_function("mediapipe/model_load", |b| {
        b.iter(|| {
            MediaPipeBlazeFaceDetector::new().expect("failed to create detector")
        });
    });
}

fn bench_detect(c: &mut Criterion) {
    let resolutions: &[(u32, u32)] = &[
        (640, 480),
        (800, 600),
        (1024, 1024),
        (1920, 1080),
    ];

    let mut detector = MediaPipeBlazeFaceDetector::new().expect("failed to create detector");

    let mut group = c.benchmark_group("mediapipe/detect");
    for &(w, h) in resolutions {
        let pixels = load_test_image(w, h);
        let label = format!("{w}x{h}");
        group.bench_with_input(BenchmarkId::from_parameter(&label), &pixels, |b, pixels| {
            b.iter(|| {
                let img = ImageRef::new(pixels, w, h, PixelFormat::Rgb).unwrap();
                detector.detect(&img)
            });
        });
    }
    group.finish();
}

#[cfg(feature = "blazeface320")]
fn bench_blazeface320_load(c: &mut Criterion) {
    use zenfaces_tract::BlazeFaceDetector;
    c.bench_function("blazeface320/model_load", |b| {
        b.iter(|| {
            BlazeFaceDetector::new().expect("failed to create detector")
        });
    });
}

#[cfg(feature = "blazeface320")]
fn bench_blazeface320_detect(c: &mut Criterion) {
    use zenfaces_tract::BlazeFaceDetector;

    let resolutions: &[(u32, u32)] = &[
        (640, 480),
        (800, 600),
        (1024, 1024),
        (1920, 1080),
    ];

    let mut detector = BlazeFaceDetector::new().expect("failed to create detector");

    let mut group = c.benchmark_group("blazeface320/detect");
    for &(w, h) in resolutions {
        let pixels = load_test_image(w, h);
        let label = format!("{w}x{h}");
        group.bench_with_input(BenchmarkId::from_parameter(&label), &pixels, |b, pixels| {
            b.iter(|| {
                let img = ImageRef::new(pixels, w, h, PixelFormat::Rgb).unwrap();
                detector.detect(&img)
            });
        });
    }
    group.finish();
}

#[cfg(not(feature = "blazeface320"))]
criterion_group!(benches, bench_model_load, bench_detect);

#[cfg(feature = "blazeface320")]
criterion_group!(
    benches,
    bench_model_load,
    bench_detect,
    bench_blazeface320_load,
    bench_blazeface320_detect
);

criterion_main!(benches);
