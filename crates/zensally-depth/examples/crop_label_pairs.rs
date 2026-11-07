//! Render annotation candidates using owned pixels/codecs. These are unfiltered
//! negative/positive teaching examples, never deployment crop recommendations.
#![forbid(unsafe_code)]
#[allow(dead_code)]
#[path = "aspect_crops/blur.rs"]
mod blur;
#[allow(dead_code)]
#[path = "aspect_crops/geometry.rs"]
mod geometry;
#[allow(dead_code)]
#[path = "aspect_crops/real.rs"]
mod real;
#[allow(dead_code)]
#[path = "aspect_crops/scoring.rs"]
mod scoring;
use geometry::{Image, Window};
use sha2::{Digest, Sha256};
use std::{fs, io::Write, path::Path};
type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
fn hash(data: &[u8]) -> String {
    Sha256::digest(data)
        .iter()
        .map(|x| format!("{x:02x}"))
        .collect()
}
fn preview(image: &Image, rect: Window, max_side: usize) -> Image {
    let view = image.view().crop(rect);
    let scale = (max_side as f32 / view.w.max(view.h) as f32).min(1.);
    view.resized(
        (view.w as f32 * scale).round().max(1.) as usize,
        (view.h as f32 * scale).round().max(1.) as usize,
    )
}
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 4 {
        return Err("usage: crop_label_pairs SOURCE_PNG NEW_OUTPUT SOURCE_ID".into());
    }
    let image = real::read_png(Path::new(&args[1]))?;
    let out = Path::new(&args[2]);
    fs::create_dir(out)?;
    scoring::write_png(
        &out.join("source.png"),
        &preview(&image, image.whole(), 1024),
    )?;
    let mut table = fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(out.join("candidates.tsv"))?;
    writeln!(
        table,
        "task_id\tsource_id\tmode\tratio_w\tratio_h\tsource_w\tsource_h\tcandidate\tx\ty\tw\th\tfile"
    )?;
    for (mode, rw, rh) in [("minimal", 9, 16), ("maximal", 1, 1)] {
        let ratio = rw as f64 / rh as f64;
        let base_w = (image.w as f64).min(image.h as f64 * ratio);
        let base_h = base_w / ratio;
        for (letter, scale, placement) in if mode == "minimal" {
            [("A", 1., 0.), ("B", 1., 1.)]
        } else {
            [("A", 0.75, 0.5), ("B", 0.5, 0.5)]
        } {
            let w = (base_w * scale).floor().max(1.) as usize;
            let h = (base_h * scale).floor().max(1.) as usize;
            let rect = Window {
                x: ((image.w - w) as f64 * placement).round() as usize,
                y: ((image.h - h) as f64 * placement).round() as usize,
                w,
                h,
            };
            let file = format!("{mode}-{letter}.png");
            scoring::write_png(&out.join(&file), &preview(&image, rect, 512))?;
            writeln!(
                table,
                "{}-{mode}\t{}\t{mode}\t{rw}\t{rh}\t{}\t{}\t{letter}\t{}\t{}\t{}\t{}\t{file}",
                args[3], args[3], image.w, image.h, rect.x, rect.y, rect.w, rect.h
            )?;
        }
    }
    Ok(())
}
