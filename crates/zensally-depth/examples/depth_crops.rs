//! Compare RGB saliency with experimental geometry fusion, without fitting.
use std::{fs::OpenOptions, io::Write};
use zencodec::{ImageFormat, ResourceLimits};
use zenlayout::smart_crop::{AspectRatio, CropConfig, CropMode, HeatMap, SmartCropInput};
use zensally::{
    PixelFormat,
    preprocess::{Normalization, ResizeMode, preprocess_nchw},
};
use zensally_depth::{Options, decode_embedded_depth, depth_cue, fuse};
use zensally_micro::{Input, MicroSalNetW24};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 4 {
        return Err("usage: depth_crops jpeg|heic INPUT REPORT.tsv".into());
    }
    let format = match args[1].as_str() {
        "jpeg" => ImageFormat::Jpeg,
        "heic" => ImageFormat::Heic,
        _ => return Err("select jpeg or heic".into()),
    };
    let bytes = std::fs::read(&args[2])?;
    let depth = decode_embedded_depth(&bytes, format, &ResourceLimits::default())?
        .ok_or("input has no embedded depth")?;
    let (rgb, w, h) = match format {
        ImageFormat::Jpeg => {
            let image = zenjpeg::decoder::Decoder::new()
                .output_format(zenjpeg::decoder::PixelFormat::Rgb)
                .decode(&bytes, enough::Unstoppable)?;
            (
                image
                    .pixels_u8()
                    .ok_or("8-bit RGB output required")?
                    .to_vec(),
                image.width,
                image.height,
            )
        }
        ImageFormat::Heic => {
            let image = heic::DecoderConfig::new().decode(&bytes, heic::PixelLayout::Rgb8)?;
            (image.data, image.width, image.height)
        }
        _ => unreachable!(),
    };
    if (w, h) != (depth.primary_width, depth.primary_height) {
        return Err("primary/depth orientation dimensions disagree".into());
    }
    let cue = depth_cue(depth.view()?, Options::default())?;
    let mut input = vec![0.0; 3 * 256 * 256];
    preprocess_nchw(
        &rgb,
        w,
        h,
        PixelFormat::Rgb,
        256,
        256,
        ResizeMode::Stretch,
        Normalization::UnitScale,
        &mut input,
    );
    let rgb_saliency = MicroSalNetW24::new()?
        .with_fast_sigmoid()
        .infer(Input::contiguous(&input)?)?;
    let mut report = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&args[3])?;
    writeln!(report, "depth_weight\taspect\tmode\tx\ty\twidth\theight")?;
    for amount in [0.0, 0.25, 0.5, 1.0] {
        let map = fuse(&rgb_saliency, &cue, amount)?;
        let crop = SmartCropInput {
            focus_regions: vec![],
            heatmap: Some(HeatMap {
                data: map.data,
                width: map.width,
                height: map.height,
            }),
        };
        for (rw, rh) in [
            (1, 1),
            (16, 9),
            (9, 16),
            (4, 3),
            (3, 4),
            (4, 5),
            (5, 4),
            (2, 1),
            (1, 2),
        ] {
            for mode in [CropMode::Minimal, CropMode::Maximal] {
                let config = CropConfig {
                    target_aspect: AspectRatio { w: rw, h: rh },
                    mode,
                    ..Default::default()
                };
                let rect = crop
                    .compute_crop(w, h, &config)
                    .ok_or("crop solver returned no rectangle")?;
                writeln!(
                    report,
                    "{amount}\t{rw}:{rh}\t{mode:?}\t{}\t{}\t{}\t{}",
                    rect.x, rect.y, rect.width, rect.height
                )?;
            }
        }
    }
    println!(
        "primary={w}x{h} depth={}x{} cue={}x{} orientation={:?} mean_depth_confidence={:.6}",
        depth.width,
        depth.height,
        cue.map.width,
        cue.map.height,
        depth.orientation,
        cue.confidence.iter().map(|&v| v as f64).sum::<f64>() / cue.confidence.len() as f64
    );
    println!("72 crops written; weights are unfitted experiments, not recommended defaults");
    Ok(())
}
