//! Native sample-code audit. Writes only to an explicit, previously absent path.
use std::{fs::OpenOptions, io::Write};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 {
        return Err("usage: inspect_heic_depth INPUT.heic OUTPUT.u16le".into());
    }
    let bytes = std::fs::read(&args[1])?;
    if std::env::var_os("HEIC_DEBLOCK_TRACE_PATH").is_some() {
        heic::enable_deblock_trace();
    }
    let depth = heic::DecoderConfig::new().decode_depth_with_limits(
        &bytes,
        &heic::Limits::default(),
        &enough::Unstoppable,
    )?;
    println!(
        "{}x{} bits={} metadata={:?}",
        depth.width, depth.height, depth.bit_depth, depth.depth_info
    );
    let mut output = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&args[2])?;
    for sample in depth.data {
        output.write_all(&sample.to_le_bytes())?;
    }
    Ok(())
}
