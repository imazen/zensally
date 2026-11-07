//! Frozen aspect/preprocessing experiment; all policy functions are tool-private.
#![forbid(unsafe_code)]
#[path = "aspect_crops/geometry.rs"]
mod geometry;
#[path = "aspect_crops/scoring.rs"]
mod scoring;
use geometry::{GRID, Image, Runner};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeSet,
    fs,
    io::{BufWriter, Write},
    path::Path,
};
type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
fn hash(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|x| format!("{x:02x}"))
        .collect()
}
fn stress(image: &Image, mask: &[u8], rw: usize, rh: usize, position: usize) -> (Image, Vec<u8>) {
    let (w, h) = (128 * rw, 128 * rh);
    let scale = 120f32 / image.w.max(image.h) as f32;
    let (iw, ih) = (
        (image.w as f32 * scale).round().max(1.) as usize,
        (image.h as f32 * scale).round().max(1.) as usize,
    );
    let small = image.view().resized(iw, ih);
    let mut bg = [0u32; 3];
    for i in [
        0,
        image.w - 1,
        (image.h - 1) * image.w,
        image.w * image.h - 1,
    ] {
        for (c, v) in bg.iter_mut().enumerate() {
            *v += image.rgb[i * 3 + c] as u32;
        }
    }
    let mut rgb = Vec::with_capacity(w * h * 3);
    for _ in 0..w * h {
        rgb.extend(bg.map(|x| (x / 4) as u8));
    }
    let mut target = vec![0; w * h];
    let x = if w >= h {
        (w - iw) * [1, 5, 9][position] / 10
    } else {
        (w - iw) / 2
    };
    let y = if h > w {
        (h - ih) * [1, 5, 9][position] / 10
    } else {
        (h - ih) / 2
    };
    for iy in 0..ih {
        rgb[((y + iy) * w + x) * 3..][..iw * 3]
            .copy_from_slice(&small.rgb[iy * iw * 3..][..iw * 3]);
        for ix in 0..iw {
            let sx = (ix * image.w / iw).min(image.w - 1);
            let sy = (iy * image.h / ih).min(image.h - 1);
            target[(y + iy) * w + x + ix] = mask[sy * image.w + sx];
        }
    }
    (Image { rgb, w, h }, target)
}
fn run_case(
    image: &Image,
    mask: &[u8],
    prefix: &str,
    runner: &mut Runner,
    metrics: &mut dyn Write,
    crops: &mut dyn Write,
    gallery: Option<&Path>,
) -> Result<()> {
    let gt = scoring::grid_mask(mask, image.w, image.h);
    if let Some(dir) = gallery {
        fs::create_dir_all(dir)?;
        let scale = (768f32 / image.w.max(image.h) as f32).min(1.);
        scoring::write_png(
            &dir.join("source.png"),
            &image.view().resized(
                (image.w as f32 * scale).round().max(1.) as usize,
                (image.h as f32 * scale).round().max(1.) as usize,
            ),
        )?;
        scoring::write_png(&dir.join("mask.png"), &scoring::heat_image(&gt))?;
    }
    for (method, prediction) in runner.predict(image)? {
        scoring::score(
            image,
            &gt,
            &prediction,
            prefix,
            method,
            (&mut *metrics, &mut *crops),
            gallery,
        )?;
    }
    Ok(())
}
fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 7 {
        return Err(
            "usage: aspect_crops DATASET MANIFEST OUTPUT natural|stress MAX_SOURCES GALLERY_IDS|-"
                .into(),
        );
    }
    let cache = codec_corpus::Corpus::with_cache_root(Path::new(&args[3]).join("corpus-cache"))?;
    let dataset = cache.local_path(&args[1])?;
    let output = Path::new(&args[3]);
    fs::create_dir_all(output)?;
    let kind = &args[4];
    if kind != "natural" && kind != "stress" {
        return Err("natural|stress required".into());
    }
    let limit: usize = args[5].parse()?;
    let selected: BTreeSet<String> = if args[6] == "-" {
        BTreeSet::new()
    } else {
        fs::read_to_string(&args[6])?
            .lines()
            .map(str::to_owned)
            .collect()
    };
    let manifest = fs::read_to_string(&args[2])?;
    let mut metrics = BufWriter::new(
        fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(output.join("metrics.tsv"))?,
    );
    let mut crops = BufWriter::new(
        fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(output.join("crops.tsv"))?,
    );
    writeln!(
        metrics,
        "id\tsplit\tcase\twidth\theight\tmethod\tpasses\tmae\tiou\tf_beta\tretention_mean\tregret_mean\tretention_min"
    )?;
    writeln!(
        crops,
        "id\tsplit\tcase\twidth\theight\tmethod\taspect\tmode\tx\ty\tw\th\tretention\toracle_retention\tregret"
    )?;
    let mut runner = Runner::new()?;
    let mut count = 0;
    let mut cases = 0;
    for line in manifest.lines().skip(1) {
        let f: Vec<_> = line.split('\t').collect();
        if f.len() != 8 {
            return Err("invalid manifest row".into());
        }
        if kind == "stress" && f[7] != "true" {
            continue;
        }
        if count == limit {
            break;
        }
        for (name, expected) in [(f[2], f[4]), (f[3], f[5])] {
            if hash(&fs::read(dataset.join(name))?) != expected {
                return Err(format!("source hash changed {}", f[0]).into());
            }
        }
        let image = scoring::read_image(&dataset.join(f[2]))?;
        let mask = scoring::read_mask(&dataset.join(f[3]), image.w, image.h)?;
        if kind == "natural" {
            let prefix = format!("{}\t{}\tnatural\t{}\t{}", f[0], f[1], image.w, image.h);
            let gallery = selected
                .contains(f[0])
                .then(|| output.join("gallery").join(format!("{}-natural", f[0])));
            run_case(
                &image,
                &mask,
                &prefix,
                &mut runner,
                &mut metrics,
                &mut crops,
                gallery.as_deref(),
            )?;
            cases += 1;
        } else {
            for (rw, rh) in [(1, 1), (2, 1), (4, 1), (8, 1), (1, 2), (1, 4), (1, 8)] {
                for position in 0..3 {
                    let (image, mask) = stress(&image, &mask, rw, rh, position);
                    let name = format!("canvas-{rw}x{rh}-p{position}");
                    let prefix = format!("{}\t{}\t{name}\t{}\t{}", f[0], f[1], image.w, image.h);
                    let gallery = (selected.contains(f[0]) && position == 0)
                        .then(|| output.join("gallery").join(format!("{}-{name}", f[0])));
                    run_case(
                        &image,
                        &mask,
                        &prefix,
                        &mut runner,
                        &mut metrics,
                        &mut crops,
                        gallery.as_deref(),
                    )?;
                    cases += 1;
                }
            }
        }
        count += 1;
        metrics.flush()?;
        crops.flush()?;
        eprintln!("sources={count} cases={cases} kind={kind}");
    }
    if count == 0 {
        return Err("no selected sources".into());
    }
    metrics.flush()?;
    crops.flush()?;
    fs::write(
        output.join("complete.txt"),
        format!("sources={count} cases={cases} grid={GRID}\n"),
    )?;
    Ok(())
}
