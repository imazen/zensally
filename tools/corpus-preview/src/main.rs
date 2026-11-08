#![forbid(unsafe_code)]

use std::error::Error;
use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::time::Duration;
use zenpipe::Limits;
use zenpipe::job::{CmsMode, ImageJob, MetadataPolicy};

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 4 {
        return Err("usage: zensally-corpus-preview INPUT.jpg NEW_OUTPUT.png LONG_EDGE".into());
    }
    let requested: u32 = args[3].parse()?;
    if !(128..=2048).contains(&requested) {
        return Err("long edge must be 128..=2048".into());
    }
    let file = File::open(&args[1])?;
    let encoded_len = usize::try_from(file.metadata()?.len())?;
    if encoded_len > 64 * 1024 * 1024 {
        return Err("encoded input exceeds 64MiB".into());
    }
    let mut input = Vec::new();
    input.try_reserve_exact(encoded_len + 1)?;
    file.take((encoded_len + 1) as u64)
        .read_to_end(&mut input)?;
    if input.len() != encoded_len {
        return Err("input length changed".into());
    }
    // Explicit per-request allowance for the screened 160MP source. Library
    // defaults remain unchanged; run-heavy supplies the process memory cap.
    let limits = Limits {
        max_pixels: Some(200_000_000),
        max_memory_bytes: Some(4 * 1024 * 1024 * 1024),
        max_width: Some(32768),
        max_height: Some(32768),
        max_frames: Some(1),
        max_total_pixels: Some(200_000_000),
        max_output_bytes: Some(16 * 1024 * 1024),
        max_duration: Some(Duration::from_secs(120)),
    };
    let job = ImageJob::new()
        .with_limits(limits)
        .with_cms(CmsMode::SceneReferred)
        .with_metadata_policy(MetadataPolicy::WebDefault)
        .add_input_ref(0, &input)
        .add_output(1);
    let info = job.probe()?;
    if info.mime_type != "image/jpeg" || info.has_animation {
        return Err("single-frame JPEG required".into());
    }
    let edge = requested.min(info.width.max(info.height));
    let dimension = if info.width >= info.height { "w" } else { "h" };
    let query = format!("autorotate=true&{dimension}={edge}&down.filter=lanczos3");
    let (query, warnings) = zenpipe::riapi::preprocess_querystring(&query);
    if !warnings.is_empty() {
        return Err(format!("query preprocessing warnings: {warnings:?}").into());
    }
    let mut parsed = zenpipe::full_registry().from_querystring(&query);
    if !parsed.warnings.is_empty() {
        return Err(format!("query warnings: {:?}", parsed.warnings).into());
    }
    zenpipe::riapi::riapi_order(&mut parsed.instances);
    let result = job
        .with_nodes(&parsed.instances)
        .with_output_extension("png")
        .with_lossless(true)
        .run()?;
    let output = result
        .encode_results
        .iter()
        .find(|encoded| encoded.io_id == 1)
        .ok_or("output missing")?;
    if output.mime_type != "image/png"
        || output.width.max(output.height) != edge
        || output.width == 0
        || output.height == 0
        || (i64::from(output.width) * i64::from(info.height)
            - i64::from(output.height) * i64::from(info.width))
        .unsigned_abs()
            > u64::from(info.width.max(info.height))
    {
        return Err("output format/aspect/extent differs".into());
    }
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&args[2])?;
    file.write_all(&output.bytes)?;
    file.sync_all()?;
    println!(
        "{{\"source_width\":{},\"source_height\":{},\"width\":{},\"height\":{},\"encoded_bytes\":{},\"cms\":\"strict-sRGB\",\"filter\":\"lanczos3\"}}",
        info.width,
        info.height,
        output.width,
        output.height,
        output.bytes.len()
    );
    Ok(())
}
