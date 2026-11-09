#![forbid(unsafe_code)]
use std::{
    error::Error,
    fs::{File, OpenOptions},
    hint::black_box,
    io::{BufReader, BufWriter, Read, Write},
    time::Instant,
};
use zenpredict::{Model, Predictor};
type Result<T> = std::result::Result<T, Box<dyn Error>>;
const BASE: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/base.znpr"));
const RESIDUAL: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/residual.znpr"));

fn load(path: &str) -> Result<Vec<[f32; 96]>> {
    let file = File::open(path)?;
    if file.metadata()?.len() > 128 * 1024 * 1024 {
        return Err("fixture exceeds128MiB".into());
    }
    let mut reader = BufReader::new(file);
    let mut magic = [0; 8];
    reader.read_exact(&mut magic)?;
    if &magic != b"ZSRH001\0" {
        return Err("fixture format differs".into());
    }
    let mut bytes = [0; 4];
    reader.read_exact(&mut bytes)?;
    let count = u32::from_le_bytes(bytes) as usize;
    if count == 0 || count > 4096 {
        return Err("bounded query count required".into());
    }
    let mut rows = Vec::new();
    rows.try_reserve_exact(count * 128)?;
    for _ in 0..count * 128 {
        let mut row = [0.; 96];
        for value in &mut row {
            reader.read_exact(&mut bytes)?;
            *value = f32::from_le_bytes(bytes);
            if !value.is_finite() {
                return Err("finite hidden values required".into());
            }
        }
        rows.push(row);
    }
    if reader.read(&mut [0])? != 0 {
        return Err("trailing fixture bytes".into());
    }
    Ok(rows)
}

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() < 4 {
        return Err(
            "gate FIXTURE OUTPUT | bench FIXTURE baseline|residual QUERIES REPEATS OFFSET".into(),
        );
    }
    let rows = load(&args[2])?;
    let base = Model::from_bytes(BASE)?;
    let residual = Model::from_bytes(RESIDUAL)?;
    let mut bp = Predictor::new(&base);
    let mut rp = Predictor::new(&residual);
    if args[1] == "gate" {
        let mut output = BufWriter::new(
            OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&args[3])?,
        );
        for row in &rows {
            let score = bp.predict(row)?[0] + rp.predict(row)?[0];
            if !score.is_finite() {
                return Err("nonfinite native output".into());
            }
            output.write_all(&score.to_le_bytes())?;
        }
        output.flush()?;
        println!("CHECK {{\"queries\":{},\"slots\":128}}", rows.len() / 128);
        return Ok(());
    }
    if args[1] != "bench" || args.len() != 7 {
        return Err("invalid benchmark arguments".into());
    }
    let active = match args[3].as_str() {
        "baseline" => false,
        "residual" => true,
        _ => return Err("invalid arm".into()),
    };
    let queries: usize = args[4].parse()?;
    let repeats: usize = args[5].parse()?;
    let offset: usize = args[6].parse()?;
    if !(1..=22).contains(&queries)
        || !(1..=20000).contains(&repeats)
        || offset
            .checked_add(queries)
            .is_none_or(|end| end > rows.len() / 128)
    {
        return Err("bounded benchmark range required".into());
    }
    let slice = &rows[offset * 128..(offset + queries) * 128];
    let mut checksum = 0.;
    for row in slice {
        checksum += bp.predict(row)?[0];
        if active {
            checksum += rp.predict(row)?[0];
        }
    }
    black_box(checksum);
    let start = Instant::now();
    for _ in 0..repeats {
        for row in slice {
            let mut score = bp.predict(black_box(row))?[0];
            if active {
                score += rp.predict(black_box(row))?[0];
            }
            checksum += black_box(score);
        }
    }
    let ms = start.elapsed().as_secs_f64() * 1000. / (repeats * queries) as f64;
    black_box(checksum);
    println!("RESULT {{\"query_ms\":{ms}}}");
    println!("CHECK {{\"queries\":{queries},\"slots\":128}}");
    Ok(())
}
