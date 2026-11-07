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
    if args.len() != 5 {
        return Err("usage: crop_label_pairs SOURCE_PNG NEW_OUTPUT SOURCE_ID PROPOSALS_TSV".into());
    }
    let image = real::read_png(Path::new(&args[1]))?;
    let proposals = fs::read_to_string(&args[4])?;
    let mut lines = proposals.lines();
    let header = lines.next().ok_or("empty proposal table")?;
    let mut rows = Vec::new();
    let mut ids = std::collections::BTreeSet::new();
    for line in lines {
        let cells: Vec<_> = line.split('\t').collect();
        if cells.len() < 13 || cells[1] != args[3] {
            return Err("invalid source/proposal table".into());
        }
        let number = |i: usize| -> Result<usize> { Ok(cells[i].parse()?) };
        let (rw, rh) = (number(3)?, number(4)?);
        if rw == 0 || rh == 0 || number(5)? != image.w || number(6)? != image.h {
            return Err("invalid proposal aspect/source dimensions".into());
        }
        let rect = Window {
            x: number(8)?,
            y: number(9)?,
            w: number(10)?,
            h: number(11)?,
        };
        let units = (image.w / rw).min(image.h / rh);
        if rect.w == 0
            || rect.h == 0
            || rect.w > image.w
            || rect.h > image.h
            || rect.x > image.w - rect.w
            || rect.y > image.h - rect.h
            || rect.w.checked_mul(rh).is_none()
            || rect.w.checked_mul(rh) != rect.h.checked_mul(rw)
            || (cells[2] == "minimal" && (rect.w != units * rw || rect.h != units * rh))
            || !matches!(cells[2], "minimal" | "maximal")
        {
            return Err("proposal violates source bounds, aspect or Minimal area".into());
        }
        let file = cells[12];
        if !file.ends_with(".png")
            || file.contains('/')
            || file.contains('\\')
            || !ids.insert(file.to_owned())
        {
            return Err("invalid/duplicate proposal filename".into());
        }
        rows.push((line.to_owned(), file.to_owned(), rect));
    }
    if rows.is_empty() {
        return Err("no proposals".into());
    }
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
    writeln!(table, "{header}")?;
    for (line, file, rect) in rows {
        scoring::write_png(&out.join(file), &preview(&image, rect, 512))?;
        writeln!(table, "{line}")?;
    }
    Ok(())
}
