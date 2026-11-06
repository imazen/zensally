#![forbid(unsafe_code)]

use zenbench::prelude::*;
use zensally::{FaceDetector, ImageRef, PixelFormat};
use zensally_tract::MediaPipeBlazeFaceDetector;

/// Load portrait.jpg and resize to the given dimensions using the triangle filter.
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

fn bench_model_load(c: &mut Suite) {
    c.bench_fn("mediapipe/model_load", || {
        MediaPipeBlazeFaceDetector::new().expect("failed to create detector")
    });
}

fn bench_detect(c: &mut Suite) {
    bench_resolutions(c, "mediapipe", || {
        MediaPipeBlazeFaceDetector::new().unwrap()
    });
}

fn bench_resolutions<D: FaceDetector + Send + 'static>(
    c: &mut Suite,
    name: &str,
    mut make_detector: impl FnMut() -> D,
) {
    for (w, h) in [(640, 480), (800, 600), (1024, 1024), (1920, 1080)] {
        let pixels = load_test_image(w, h);
        let mut detector = make_detector();
        c.bench_fn(format!("{name}/detect/{w}x{h}"), move || {
            let img = ImageRef::new(&pixels, w, h, PixelFormat::Rgb).unwrap();
            detector.detect(&img)
        });
    }
}

fn face_benches(c: &mut Suite) {
    bench_model_load(c);
    bench_detect(c);
    #[cfg(feature = "blazeface320")]
    {
        use zensally_tract::BlazeFaceDetector;
        c.bench_fn("blazeface320/model_load", || {
            BlazeFaceDetector::new().expect("failed to create detector")
        });
        bench_resolutions(c, "blazeface320", || BlazeFaceDetector::new().unwrap());
    }
}

zenbench::main!(face_benches);
