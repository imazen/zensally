//! Qualitative real-aspect cases. No fabricated mask or quality score.
use super::{
    Result,
    geometry::{GRID, Image, Runner, Window},
    hash, scoring,
};
use std::{
    fs,
    io::{BufWriter, Write},
    path::Path,
};
use zenlayout::smart_crop::{AspectRatio, CropConfig, CropMode, HeatMap, SmartCropInput};
fn read_png(path: &Path) -> Result<Image> {
    // Owned reader convention: zensally-micro/examples/eval_rectangles.rs.
    let data = fs::read(path)?;
    let mut config = zenpng::PngDecodeConfig::strict();
    config.max_pixels = Some(64 * 1024 * 1024);
    config.max_memory_bytes = Some(512 * 1024 * 1024);
    let decoded = zenpng::decode(&data, &config, &enough::Unstoppable)?;
    let pixels = decoded.pixels.as_slice();
    let (w, h) = (pixels.width() as usize, pixels.rows() as usize);
    let channels = pixels.descriptor().channels();
    let gray = decoded.pixels.is_grayscale();
    let mut rgb = vec![0; w * h * 3];
    if pixels.descriptor().channel_type() != zenpixels::ChannelType::U8 {
        return Err("RGB8 qualitative source required".into());
    }
    for y in 0..h {
        let row = pixels.row(y as u32);
        for x in 0..w {
            for c in 0..3 {
                rgb[(y * w + x) * 3 + c] = row[x * channels + if gray { 0 } else { c }];
            }
        }
    }
    Ok(Image { rgb, w, h })
}
pub fn run(dataset: &Path, manifest: &Path, output: &Path, limit: usize) -> Result<()> {
    let corpus = codec_corpus::Corpus::with_cache_root(output.join("corpus-cache"))?;
    let dataset = corpus.local_path(dataset)?;
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
        "id\tsplit\tcase\twidth\theight\tmethod\tpasses\tmae"
    )?;
    writeln!(
        crops,
        "id\tsplit\tcase\twidth\theight\tmethod\taspect\tmode\tx\ty\tw\th\tretention\toracle_retention\tregret"
    )?;
    let mut runner = Runner::new()?;
    let mut count = 0;
    for line in fs::read_to_string(manifest)?.lines().skip(1).take(limit) {
        let f: Vec<_> = line.split('\t').collect();
        if f.len() != 6 {
            return Err("invalid real-image manifest".into());
        }
        let path = dataset.join(f[2]);
        if hash(&fs::read(&path)?) != f[3] {
            return Err("qualitative source hash changed".into());
        }
        let image = read_png(&path)?;
        let dir = output.join("gallery").join(format!("{}-real", f[0]));
        fs::create_dir_all(&dir)?;
        let scale = (768f32 / image.w.max(image.h) as f32).min(1.);
        scoring::write_png(
            &dir.join("source.png"),
            &image.view().resized(
                (image.w as f32 * scale).round().max(1.) as usize,
                (image.h as f32 * scale).round().max(1.) as usize,
            ),
        )?;
        for (method, pred) in runner.predict(&image)? {
            scoring::write_png(
                &dir.join(format!("{method}-map.png")),
                &scoring::heat_image(&pred.data),
            )?;
            let passes = if ["overlap-refine", "aspect-refine"].contains(&method) {
                1 + pred.views.len()
            } else {
                pred.views.len()
            };
            writeln!(
                metrics,
                "{}\tunlabelled\treal\t{}\t{}\t{method}\t{passes}\t",
                f[0], image.w, image.h
            )?;
            let input = SmartCropInput {
                focus_regions: vec![],
                heatmap: Some(HeatMap {
                    data: pred.data,
                    width: GRID as u32,
                    height: GRID as u32,
                }),
            };
            for (rw, rh) in [(1, 1), (16, 9), (9, 16)] {
                for mode in [CropMode::Minimal, CropMode::Maximal] {
                    let config = CropConfig {
                        target_aspect: AspectRatio { w: rw, h: rh },
                        mode,
                        ..Default::default()
                    };
                    let r = input
                        .compute_crop(image.w as u32, image.h as u32, &config)
                        .ok_or("missing qualitative crop")?;
                    if r.width == 0
                        || r.height == 0
                        || r.x + r.width > image.w as u32
                        || r.y + r.height > image.h as u32
                    {
                        return Err("invalid qualitative crop".into());
                    }
                    let mode = if mode == CropMode::Minimal {
                        "minimal"
                    } else {
                        "maximal"
                    };
                    writeln!(
                        crops,
                        "{}\tunlabelled\treal\t{}\t{}\t{method}\t{rw}:{rh}\t{mode}\t{}\t{}\t{}\t{}\t\t\t",
                        f[0], image.w, image.h, r.x, r.y, r.width, r.height
                    )?;
                    let view = image.view().crop(Window {
                        x: r.x as usize,
                        y: r.y as usize,
                        w: r.width as usize,
                        h: r.height as usize,
                    });
                    let scale = 320f32 / view.w.max(view.h) as f32;
                    scoring::write_png(
                        &dir.join(format!("{method}-{rw}x{rh}-{mode}.png")),
                        &view.resized(
                            (view.w as f32 * scale).round().max(1.) as usize,
                            (view.h as f32 * scale).round().max(1.) as usize,
                        ),
                    )?;
                }
            }
        }
        count += 1;
        eprintln!("unlabelled sources={count} id={}", f[0]);
    }
    if count == 0 {
        return Err("no qualitative sources".into());
    }
    crops.flush()?;
    metrics.flush()?;
    fs::write(
        output.join("complete.txt"),
        format!("unlabelled_sources={count}\n"),
    )?;
    Ok(())
}
