//! Completed preprocessing operations, full f32-bit validation outside timing.
#![forbid(unsafe_code)]
use std::{
    hint::black_box,
    io::Write,
    process::{Command, Stdio},
    time::{Duration, Instant},
};
use zensally::{
    PixelFormat,
    preprocess::{Normalization, ResizeMode, preprocess_nchw},
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    coz::thread_init();
    let args: Vec<String> = std::env::args().skip(1).collect();
    assert!(
        (8..=9).contains(&args.len()),
        "src_w src_h dst_w dst_h rgb|rgba|bgra stretch|letterbox unit|center|mean iterations [dump]"
    );
    let sw: u32 = args[0].parse()?;
    let sh: u32 = args[1].parse()?;
    let dw: usize = args[2].parse()?;
    let dh: usize = args[3].parse()?;
    assert!(sw > 0 && sh > 0 && dw > 0 && dh > 0);
    let format = match args[4].as_str() {
        "rgb" => PixelFormat::Rgb,
        "rgba" => PixelFormat::Rgba,
        "bgra" => PixelFormat::Bgra,
        _ => panic!("format"),
    };
    let mode = match args[5].as_str() {
        "stretch" => ResizeMode::Stretch,
        "letterbox" => ResizeMode::Letterbox,
        _ => panic!("resize mode"),
    };
    let norm = match args[6].as_str() {
        "unit" => Normalization::UnitScale,
        "center" => Normalization::CenterScale,
        "mean" => Normalization::MeanSubtract {
            r: 123.0,
            g: 117.0,
            b: 104.0,
        },
        _ => panic!("normalization"),
    };
    let iterations: usize = args[7].parse()?;
    assert!(iterations > 0);
    let mut state = 0x19b4_7e93u32;
    let pixels: Vec<u8> = (0..sw as usize * sh as usize * format.bytes_per_pixel())
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (state >> 24) as u8
        })
        .collect();
    let mut output = vec![0.0; 3 * dw * dh];
    let mut run = || {
        let info = preprocess_nchw(
            black_box(&pixels),
            sw,
            sh,
            format,
            dw,
            dh,
            mode,
            norm,
            black_box(&mut output),
        );
        black_box(info);
        black_box(&output);
        coz::progress!("preprocess-completed");
    };
    for _ in 0..5 {
        run();
    }
    let warmup: u64 = std::env::var("ZENSALLY_COZ_WARMUP_MS").map_or(Ok(0), |s| s.parse())?;
    let start = Instant::now();
    while start.elapsed() < Duration::from_millis(warmup) {
        run();
    }
    for _ in 0..3 {
        let start = Instant::now();
        for _ in 0..iterations {
            run();
        }
        println!("elapsed_ns={}", start.elapsed().as_nanos());
    }
    let info = preprocess_nchw(&pixels, sw, sh, format, dw, dh, mode, norm, &mut output);
    // Include metadata and every output bit; no float tolerance or lossy fold.
    let bytes: Vec<u8> = [info.ratio, info.pad_left, info.pad_top]
        .into_iter()
        .chain(output)
        .flat_map(|x| x.to_bits().to_le_bytes())
        .collect();
    let mut child = Command::new("sha256sum")
        .env_remove("LD_PRELOAD")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()?;
    let written = child.stdin.take().expect("stdin").write_all(&bytes);
    let hash = child.wait_with_output()?;
    written?;
    assert!(hash.status.success());
    let text = String::from_utf8(hash.stdout)?;
    let digest = text.split_whitespace().next().expect("digest");
    assert!(digest.len() == 64 && digest.bytes().all(|b| b.is_ascii_hexdigit()));
    println!(
        "result={{\"sha256\":\"{digest}\",\"bytes\":{},\"width\":{dw},\"height\":{dh}}}",
        bytes.len()
    );
    if let Some(path) = args.get(8) {
        std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)?
            .write_all(&bytes)?;
    }
    Ok(())
}
