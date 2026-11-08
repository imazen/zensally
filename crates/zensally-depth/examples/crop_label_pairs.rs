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
fn previews(
    image: &Image,
    rects: &[Window],
    source_max_side: usize,
    candidate_pixels: usize,
) -> Result<Vec<Image>> {
    let mut specs = Vec::new();
    let mut shapes = Vec::new();
    for (i, rect) in rects.iter().enumerate() {
        let (side, pixels) = if i == 0 {
            (source_max_side, source_max_side * source_max_side)
        } else {
            (1536, candidate_pixels)
        };
        let scale = (side as f64 / rect.w.max(rect.h) as f64)
            .min((pixels as f64 / (rect.w * rect.h) as f64).sqrt())
            .min(1.);
        let w = (rect.w as f64 * scale).floor().max(1.) as usize;
        let h = (rect.h as f64 * scale).floor().max(1.) as usize;
        shapes.push((w, h));
        specs.push(format!(
            "{},{},{},{},{},{}",
            rect.x, rect.y, rect.w, rect.h, w, h
        ));
    }
    let executable = std::env::var_os("ZENSALLY_PREVIEW_RESIZER")
        .ok_or("ZENSALLY_PREVIEW_RESIZER is required; build tools/crop-preview")?;
    let mut child = std::process::Command::new(executable)
        .args([
            image.w.to_string(),
            image.h.to_string(),
            (image.w * 3).to_string(),
            specs.join(";"),
        ])
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .spawn()?;
    // Worker consumes the complete source once, then emits ordered raw previews.
    // Dropping stdin delivers EOF before waiting for its bounded output.
    child
        .stdin
        .take()
        .ok_or("preview stdin missing")?
        .write_all(&image.rgb)?;
    let output = child.wait_with_output()?;
    if !output.status.success() {
        return Err("zenresize preview worker failed".into());
    }
    let expected: usize = shapes.iter().map(|(w, h)| w * h * 3).sum();
    if output.stdout.len() != expected {
        return Err("preview output length differs".into());
    }
    let mut offset = 0;
    Ok(shapes
        .into_iter()
        .map(|(w, h)| {
            let size = w * h * 3;
            let rgb = output.stdout[offset..offset + size].to_vec();
            offset += size;
            Image { rgb, w, h }
        })
        .collect())
}

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() == 6 && args[1] == "--from-rgb" {
        let w: usize = args[2].parse()?;
        let h: usize = args[3].parse()?;
        if w == 0 || h == 0 || w.checked_mul(h).is_none_or(|n| n > 32_000_000) {
            return Err("invalid/bounded RGB8 dimensions".into());
        }
        let rgb = fs::read(&args[4])?;
        if rgb.len() != w * h * 3 || Path::new(&args[5]).exists() {
            return Err("RGB8 length mismatch or output already exists".into());
        }
        scoring::write_png(Path::new(&args[5]), &Image { rgb, w, h })?;
        return Ok(());
    }
    if args.len() == 3 && args[1] == "--inspect" {
        let image = real::read_png(Path::new(&args[2]))?;
        println!(
            "{{\"width\":{},\"height\":{},\"rgb_sha256\":\"{}\"}}",
            image.w,
            image.h,
            hash(&image.rgb)
        );
        return Ok(());
    }
    if !(5..=7).contains(&args.len()) {
        return Err("usage: crop_label_pairs SOURCE_PNG NEW_OUTPUT SOURCE_ID PROPOSALS_TSV [SOURCE_MAX_SIDE [CANDIDATE_PIXELS]]".into());
    }
    let source_max_side = if let Some(value) = args.get(5) {
        value.parse::<usize>()?
    } else {
        1024
    };
    if !(128..=2048).contains(&source_max_side) {
        return Err("source preview limit must be 128..2048".into());
    }
    let candidate_pixels = args
        .get(6)
        .map(|s| s.parse::<usize>())
        .transpose()?
        .unwrap_or(262_144);
    if !(4096..=262_144).contains(&candidate_pixels) {
        return Err("candidate pixel cap must be 4096..262144".into());
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
    let rects: Vec<_> = std::iter::once(image.whole())
        .chain(rows.iter().map(|(_, _, r)| *r))
        .collect();
    let rendered = previews(&image, &rects, source_max_side, candidate_pixels)?;
    scoring::write_png(&out.join("source.png"), &rendered[0])?;
    let mut table = fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(out.join("candidates.tsv"))?;
    writeln!(table, "{header}")?;
    for ((line, file, _), preview) in rows.into_iter().zip(&rendered[1..]) {
        scoring::write_png(&out.join(file), preview)?;
        writeln!(table, "{line}")?;
    }
    Ok(())
}
