//! Full-model profiler driver; no wall-clock instrumentation.
#![forbid(unsafe_code)]
use std::hint::black_box;
mod workers;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let arm = args.get(1).map_or("compare", String::as_str);
    let repeats = args.get(2).map_or(Ok(1), |s| s.parse::<usize>())?;
    if arm == "batch" {
        let mut workers = workers::Workers::new(&workers::cpus());
        for _ in 0..repeats {
            black_box(workers.batch());
        }
        return Ok(());
    }
    let input: Vec<f32> = (0..3 * 65536)
        .map(|i| ((i * 137 + i / 256 * 59) % 1024) as f32 / 1023.0)
        .collect();
    let mut base = baseline::MicroSalNetW24::new()?.with_fast_sigmoid();
    let mut candidate = candidate::MicroSalNetW24::new()?.with_fast_sigmoid();
    match arm {
        "baseline" => {
            for _ in 0..repeats {
                black_box(base.infer(baseline::Input::contiguous(black_box(&input))?)?);
            }
        }
        "candidate" => {
            for _ in 0..repeats {
                black_box(candidate.infer(candidate::Input::contiguous(black_box(&input))?)?);
            }
        }
        "compare" => {
            let a = base.infer(baseline::Input::contiguous(&input)?)?;
            let b = candidate.infer(candidate::Input::contiguous(&input)?)?;
            assert_eq!(a.data.len(), b.data.len());
            assert!(a.data.iter().chain(&b.data).all(|x| x.is_finite()));
            let error = a
                .data
                .iter()
                .zip(&b.data)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            println!("max_abs={error:.9e}");
        }
        _ => return Err("expected baseline, candidate or compare".into()),
    }
    Ok(())
}
