//! Numerical audit and profiler workload. Outputs measurements, never timing
//! probes. Arguments: `compare` (default), `micro N`, or `tract N`.
use std::{hint::black_box, io::Cursor};
use tract_onnx::prelude::*;
use zensally_micro::{Input, MicroSalNetW24};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let mode = args.get(1).map_or("compare", String::as_str);
    let repeats: usize = args.get(2).map_or(Ok(1), |n| n.parse())?;
    let mut input: Vec<f32> = (0..3 * 65536)
        .map(|i| ((i * 137 + i / 256 * 59) % 1024) as f32 / 1023.0)
        .collect();
    let mut micro = MicroSalNetW24::new()?;
    if mode == "micro-fast" {
        micro = micro.with_fast_sigmoid();
    }
    if mode == "micro" || mode == "micro-fast" {
        for _ in 0..repeats {
            black_box(micro.infer(Input::contiguous(black_box(&input))?)?);
        }
        return Ok(());
    }
    let gz = include_bytes!("../../zensally-tract/models/microsalnet_v3ds_w24.onnx.gz");
    let mut model = vec![0; 769_236];
    zenflate::Decompressor::new()
        .gzip_decompress(gz, &mut model, enough::Unstoppable)
        .map_err(|e| format!("{e:?}"))?;
    let tract = tract_onnx::onnx()
        .model_for_read(&mut Cursor::new(model))?
        .with_input_fact(0, InferenceFact::dt_shape(DatumType::F32, [1, 3, 256, 256]))?
        .into_optimized()?
        .into_runnable()?;
    let tensor = Tensor::from_shape(&[1, 3, 256, 256], &input)?;
    if mode == "tract" {
        for _ in 0..repeats {
            black_box(tract.run(tvec!(tensor.clone().into()))?);
        }
        return Ok(());
    }
    if mode != "compare" {
        return Err("expected compare, micro or tract".into());
    }
    for kind in ["pattern", "zero", "one"] {
        match kind {
            "zero" => input.fill(0.0),
            "one" => input.fill(1.0),
            _ => {}
        }
        let custom = micro.infer(Input::contiguous(&input)?)?;
        let tensor = Tensor::from_shape(&[1, 3, 256, 256], &input)?;
        let result = tract.run(tvec!(tensor.into()))?;
        let expected = result[0]
            .as_plain_ram()
            .ok_or("non-plain tract output")?
            .as_slice::<f32>()?;
        let mut mismatches = 0;
        let mut max_abs = 0.0f32;
        let mut sum_abs = 0.0f64;
        assert_eq!(result[0].shape(), &[1, 1, 128, 128]);
        for (&a, &b) in custom.data.iter().zip(expected.iter()) {
            if !a.is_finite() || !b.is_finite() {
                return Err("nonfinite model result".into());
            }
            if a.to_bits() != b.to_bits() {
                mismatches += 1;
            }
            let delta = (a - b).abs();
            max_abs = max_abs.max(delta);
            sum_abs += delta as f64;
        }
        println!(
            "{kind}: mismatches={mismatches}/{} max_abs={max_abs:.9e} mean_abs={:.9e}",
            custom.data.len(),
            sum_abs / custom.data.len() as f64
        );
    }
    // Owner-approved contract: exact scalar/SIMD agreement internally, with
    // tract audited numerically and downstream rectangles checked separately.
    Ok(())
}
