//! Import/inference feasibility and raw graph timing; no crop/face quality claim.
#![forbid(unsafe_code)]
use tract_tflite::prelude::*;
fn main() -> anyhow::Result<()> {
    let path = std::env::args()
        .nth(1)
        .ok_or_else(|| anyhow::anyhow!("model.tflite required"))?;
    let model = tract_tflite::tflite().model_for_path(path)?;
    let shape = model
        .input_fact(0)?
        .shape
        .as_concrete()
        .ok_or_else(|| anyhow::anyhow!("symbolic shape"))?
        .to_vec();
    anyhow::ensure!(
        model.input_fact(0)?.datum_type == DatumType::F32,
        "this feasibility probe requires f32 input"
    );
    let input = Tensor::from_shape(&shape, &vec![0f32; shape.iter().product()])?;
    let runnable = model.into_optimized()?.into_runnable()?;
    let inputs = tvec!(input.into());
    let outputs = runnable.run(inputs.clone())?;
    for (i, output) in outputs.iter().enumerate() {
        let data = output
            .as_plain_ram()
            .ok_or_else(|| anyhow::anyhow!("non-plain output"))?
            .as_slice::<f32>()?;
        anyhow::ensure!(data.iter().all(|v| v.is_finite()), "nonfinite output");
        println!("output={i} shape={:?} finite=true", output.shape());
    }
    if let Some(raw) = std::env::args().nth(2) {
        let repeats: usize = raw.parse()?;
        anyhow::ensure!((1..=10000).contains(&repeats), "bounded repeat count");
        let start = std::time::Instant::now();
        for _ in 0..repeats {
            std::hint::black_box(runnable.run(inputs.clone())?);
        }
        let inference_ns = start.elapsed().as_nanos() as f64 / repeats as f64;
        println!("RESULT {{\"inference_ns\":{inference_ns},\"calls\":{repeats}}}");
    }
    Ok(())
}
