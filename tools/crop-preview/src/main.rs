//! Offline strided RGB8 crop/resample worker. No codec implementation.
#![forbid(unsafe_code)]
use std::io::{Read, Write};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 5 {
        return Err(
            "usage: preview WIDTH HEIGHT STRIDE x,y,w,h,ow,oh[;...] < RGB8 > previews".into(),
        );
    }
    let w: usize = args[1].parse()?;
    let h: usize = args[2].parse()?;
    let stride: usize = args[3].parse()?;
    if w == 0 || h == 0 || w.checked_mul(h).is_none_or(|v| v > 64 * 1024 * 1024) || stride < w * 3 {
        return Err("invalid/bounded RGB8 dimensions or stride".into());
    }
    let len = stride
        .checked_mul(h - 1)
        .and_then(|v| v.checked_add(w * 3))
        .ok_or("RGB8 input overflow")?;
    if len > 512 * 1024 * 1024 {
        return Err("RGB8 input exceeds 512 MiB".into());
    }
    let mut configs = Vec::new();
    for text in args[4].split(';') {
        let v: Vec<usize> = text
            .split(',')
            .map(str::parse)
            .collect::<std::result::Result<_, _>>()?;
        if v.len() != 6
            || v[2] == 0
            || v[3] == 0
            || v[2] > w
            || v[3] > h
            || v[0] > w - v[2]
            || v[1] > h - v[3]
            || v[4] == 0
            || v[5] == 0
            || v[4].checked_mul(v[5]).is_none_or(|n| n > 2048 * 2048)
        {
            return Err("invalid/bounded crop or output dimensions".into());
        }
        configs.push(
            zenresize::ResizeConfig::builder(w as u32, h as u32, v[4] as u32, v[5] as u32)
                .in_stride(stride)
                .format(zenresize::PixelDescriptor::RGB8_SRGB)
                .filter(zenresize::Filter::Lanczos)
                .crop(v[0] as u32, v[1] as u32, v[2] as u32, v[3] as u32)
                .build(),
        );
    }
    if configs.len() > 256 {
        return Err("preview batch exceeds 256 views".into());
    }
    let mut rgb = Vec::new();
    rgb.try_reserve_exact(len)?;
    std::io::stdin()
        .take(len as u64 + 1)
        .read_to_end(&mut rgb)?;
    if rgb.len() != len {
        return Err("RGB8 input length differs from dimensions/stride".into());
    }
    let mut out = std::io::stdout().lock();
    for config in configs {
        out.write_all(&zenresize::Resizer::new(&config).resize(&rgb))?;
    }
    Ok(())
}
