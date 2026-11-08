//! Evaluation-only runner; decoded RGB and model lifetime are outside warm timing.
#![forbid(unsafe_code)]
use std::{fs, hint::black_box, path::Path, time::Instant};
use zensally::{FaceDetector, ImageRef, PixelFormat, SaliencyDetector};
use zensally_tract::{
    BlazeFaceDetector, MediaPipeBlazeFaceDetector, MicroSalNet, SelfieSeg, U2NetpDetector,
    UltraFaceDetector, YuNetDetector,
};

enum Detector {
    Face(Box<dyn FaceDetector>),
    Map(Box<dyn SaliencyDetector>),
}
fn make(name: &str) -> Result<Detector, anyhow::Error> {
    Ok(match name {
        "ultraface" => Detector::Face(Box::new(UltraFaceDetector::new()?)),
        "front" => Detector::Face(Box::new(MediaPipeBlazeFaceDetector::new()?)),
        "blazeface320" => Detector::Face(Box::new(BlazeFaceDetector::new()?)),
        "yunet" => Detector::Face(Box::new(YuNetDetector::new()?)),
        "selfie" => Detector::Map(Box::new(SelfieSeg::new()?)),
        "microsalnet" => Detector::Map(Box::new(MicroSalNet::new()?)),
        "u2netp" => Detector::Map(Box::new(U2NetpDetector::new()?)),
        _ => anyhow::bail!("unknown model"),
    })
}
fn call(d: &mut Detector, pixels: &[u8], width: u32, height: u32) {
    let image = ImageRef::new(pixels, width, height, PixelFormat::Rgb).unwrap();
    match d {
        Detector::Face(d) => {
            black_box(d.detect(&image));
        }
        Detector::Map(d) => {
            black_box(d.saliency_map(&image));
        }
    }
}
fn main() -> Result<(), anyhow::Error> {
    let args: Vec<_> = std::env::args().collect();
    anyhow::ensure!(args.len() == 6, "mode model inputs.tsv output repeats");
    let mut detector = make(&args[2])?;
    let output = Path::new(&args[4]);
    fs::create_dir(output)?;
    let repeats: usize = args[5].parse()?;
    anyhow::ensure!((1..=10000).contains(&repeats), "bounded repeats");
    let mut inputs = Vec::new();
    for line in fs::read_to_string(&args[3])?.lines() {
        let fields: Vec<_> = line.split('\t').collect();
        anyhow::ensure!(fields.len() == 5, "invalid input TSV");
        let (width, height): (u32, u32) = (fields[1].parse()?, fields[2].parse()?);
        let pixels = fs::read(fields[3])?;
        anyhow::ensure!(
            pixels.len() == width as usize * height as usize * 3,
            "invalid RGB length"
        );
        inputs.push((
            fields[0].to_string(),
            width,
            height,
            pixels,
            fields[4] == "1",
        ));
    }
    if args[1] == "bench" || args[1] == "profile" {
        let chosen: Vec<_> = inputs.iter().filter(|x| x.4).collect();
        anyhow::ensure!(!chosen.is_empty(), "empty benchmark");
        for x in &chosen {
            call(&mut detector, &x.3, x.1, x.2);
        }
        let start = Instant::now();
        for _ in 0..repeats {
            for x in &chosen {
                call(&mut detector, &x.3, x.1, x.2);
            }
        }
        let ns = start.elapsed().as_nanos() as f64 / (repeats * chosen.len()) as f64;
        println!(
            "RESULT {{\"inference_ns\":{ns},\"calls\":{},\"images\":{}}}",
            repeats * chosen.len(),
            chosen.len()
        );
    } else {
        anyhow::ensure!(args[1] == "evaluate", "unknown mode");
        let mut rows = String::from("[");
        for (index, (id, width, height, pixels, _)) in inputs.iter().enumerate() {
            if index > 0 {
                rows.push(',');
            }
            anyhow::ensure!(
                id.chars()
                    .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_'),
                "unsafe source id"
            );
            rows.push_str(&format!("{{\"id\":\"{id}\",\"result\":{{"));
            let image = ImageRef::new(pixels, *width, *height, PixelFormat::Rgb)
                .map_err(anyhow::Error::msg)?;
            match &mut detector {
                Detector::Face(d) => {
                    rows.push_str("\"faces\":[");
                    for (i, f) in d.detect(&image).iter().enumerate() {
                        if i > 0 {
                            rows.push(',');
                        }
                        anyhow::ensure!(
                            [f.x1, f.y1, f.x2, f.y2, f.confidence]
                                .iter()
                                .all(|x| x.is_finite()),
                            "nonfinite detection"
                        );
                        rows.push_str(&format!(
                            "{{\"box_percent\":[{},{},{},{}],\"confidence\":{}}}",
                            f.x1, f.y1, f.x2, f.y2, f.confidence
                        ));
                    }
                    rows.push(']');
                }
                Detector::Map(d) => {
                    let map = d.saliency_map(&image);
                    anyhow::ensure!(map.data.iter().all(|x| x.is_finite()), "nonfinite map");
                    let mut pgm = format!("P5\n{} {}\n255\n", map.width, map.height).into_bytes();
                    pgm.extend(map.data.iter().map(|v| (v.clamp(0., 1.) * 255.) as u8));
                    fs::write(output.join(format!("{id}-mask.pgm")), pgm)?;
                    let min = map.data.iter().copied().fold(f32::INFINITY, f32::min);
                    let max = map.data.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                    let mean = map.data.iter().sum::<f32>() / map.data.len() as f32;
                    rows.push_str(&format!("\"map\":{{\"width\":{},\"height\":{},\"min\":{min},\"max\":{max},\"mean\":{mean}}}",map.width,map.height));
                }
            }
            rows.push_str("}}");
        }
        rows.push(']');
        fs::write(output.join("results.json"), rows)?;
        println!("evaluated {} sources", inputs.len());
    }
    Ok(())
}
