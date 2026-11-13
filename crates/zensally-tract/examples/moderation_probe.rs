//! Development-only graph loading and native-output probe. No detector API.
#![forbid(unsafe_code)]

use anyhow::{Context, Result, ensure};
use std::path::{Path, PathBuf};
use std::time::Instant;
use tract_onnx::prelude::*;
use zensally_micro::swiftformer::{SwiftFormerModel, SwiftFormerWorkspace};

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    if args.first().is_some_and(|v| v == "--bench") {
        return benchmark(&args[1..]);
    }
    ensure!(
        args.len() == 4,
        "usage: moderation_probe MODEL INPUT.f32 SIDE OUTPUT.f32"
    );
    let path = PathBuf::from(&args[0]);
    let side: usize = args[2].to_string_lossy().parse()?;
    let bytes = std::fs::read(&args[1])?;
    ensure!(
        bytes.len() == 4 * 3 * side * side,
        "input byte count differs from model shape"
    );
    let input: Vec<f32> = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .copied()
        .map(f32::from_le_bytes)
        .collect();
    let force_tract = std::env::var_os("ZENSALLY_MODERATION_FORCE_TRACT").is_some();
    let values: Vec<f32> = if !force_tract
        && side == 224
        && let Ok(onnx_bytes) = std::fs::read(&path)
        && let Ok(sf_model) = SwiftFormerModel::from_onnx_bytes(&onnx_bytes)
    {
        let mut ws = SwiftFormerWorkspace::new();
        sf_model.forward_nchw(&input, &mut ws).to_vec()
    } else {
        let model = tract_onnx::onnx()
            .model_for_path(&path)
            .context("parse ONNX")?
            .with_input_fact(
                0,
                InferenceFact::dt_shape(DatumType::F32, [1, 3, side, side]),
            )?
            .into_optimized()
            .context("optimize ONNX")?
            .into_runnable()
            .context("plan ONNX")?;
        let input = Tensor::from_shape(&[1, 3, side, side], &input)?;
        let result = model.run(tvec!(input.into()))?;
        let view = result[0].as_plain_ram().context("plain output tensor")?;
        view.as_slice::<f32>()?.to_vec()
    };
    ensure!(values.iter().all(|v| v.is_finite()), "non-finite output");
    let output: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    let output_path = PathBuf::from(&args[3]);
    ensure!(!output_path.exists(), "refusing to replace existing output");
    std::fs::write(output_path, output)?;
    println!("logits={values:?}");
    Ok(())
}

fn benchmark(args: &[std::ffi::OsString]) -> Result<()> {
    ensure!(
        args.len() == 6,
        "--bench MODEL INPUTS.f32 SIDE OUTPUT_DIR MODEL_KEY MANIFEST_SHA"
    );
    let side: usize = args[2].to_string_lossy().parse()?;
    let key = args[4].to_string_lossy();
    let sha = args[5].to_string_lossy();
    ensure!(
        key.chars().all(|c| c.is_ascii_alphanumeric() || c == '-'),
        "invalid model key"
    );
    ensure!(
        sha.len() == 64 && sha.chars().all(|c| c.is_ascii_hexdigit()),
        "invalid manifest hash"
    );
    let bytes = std::fs::read(&args[1])?;
    let image_bytes = 4 * 3 * side * side;
    ensure!(
        !bytes.is_empty() && bytes.len().is_multiple_of(image_bytes),
        "input shape mismatch"
    );
    let raw_inputs: Vec<Vec<f32>> = bytes
        .chunks_exact(image_bytes)
        .map(|chunk| {
            chunk
                .as_chunks::<4>()
                .0
                .iter()
                .copied()
                .map(f32::from_le_bytes)
                .collect()
        })
        .collect();
    ensure!(
        raw_inputs.len() == 8,
        "benchmark requires the eight-image manifest"
    );
    let force_tract = std::env::var_os("ZENSALLY_MODERATION_FORCE_TRACT").is_some();
    let mut samples = Vec::new();
    let mut logits = Vec::new();

    let backend = if !force_tract
        && side == 224
        && let Ok(onnx_bytes) = std::fs::read(Path::new(&args[0]))
        && let Ok(sf_model) = SwiftFormerModel::from_onnx_bytes(&onnx_bytes)
    {
        let mut ws = SwiftFormerWorkspace::new();
        for i in 0..4 {
            let out = sf_model.forward_nchw(&raw_inputs[i % raw_inputs.len()], &mut ws);
            std::hint::black_box(out);
        }
        for i in 0..16 {
            let inp = &raw_inputs[i % raw_inputs.len()];
            let start = Instant::now();
            let out = sf_model.forward_nchw(inp, &mut ws);
            samples.push(start.elapsed().as_secs_f64() * 1000.0);
            ensure!(out.iter().all(|v| v.is_finite()), "non-finite output");
            logits.push(vec![out.to_vec()]);
        }
        "zensally-micro"
    } else {
        let inputs: Vec<Tensor> = raw_inputs
            .iter()
            .map(|values| Tensor::from_shape(&[1, 3, side, side], values))
            .collect::<TractResult<_>>()?;
        let model = tract_onnx::onnx()
            .model_for_path(Path::new(&args[0]))?
            .with_input_fact(
                0,
                InferenceFact::dt_shape(DatumType::F32, [1, 3, side, side]),
            )?
            .into_optimized()?
            .into_runnable()?;
        for i in 0..4 {
            let result = model.run(tvec!(inputs[i % inputs.len()].clone().into()))?;
            std::hint::black_box(result);
        }
        for i in 0..16 {
            let input = inputs[i % inputs.len()].clone().into();
            let start = Instant::now();
            let result = model.run(tvec!(input))?;
            samples.push(start.elapsed().as_secs_f64() * 1000.0);
            let view = result[0].as_plain_ram().context("plain output")?;
            let values = view.as_slice::<f32>()?.to_vec();
            ensure!(values.iter().all(|v| v.is_finite()), "non-finite output");
            logits.push(vec![values]);
        }
        "tract"
    };
    let directory = PathBuf::from(&args[3]);
    std::fs::create_dir_all(&directory)?;
    let output = directory.join("worker.json");
    ensure!(!output.exists(), "refusing to replace benchmark output");
    std::fs::write(
        output,
        format!(
            "{{\"model\":\"{key}\",\"stage\":\"inference\",\"threads\":1,\"count\":16,\"warmup\":4,\"input_manifest_sha256\":\"{sha}\",\"backend\":\"{backend}\",\"samples_ms\":{samples:?},\"logits\":{logits:?}}}\n"
        ),
    )?;
    for sample in samples {
        println!("elapsed_ms={sample:.9}");
    }
    Ok(())
}
