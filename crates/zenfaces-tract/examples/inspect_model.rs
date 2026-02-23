use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("models/blazeface-320.onnx");
    println!("Loading model from: {}", model_path.display());

    let model = tract_onnx::onnx()
        .model_for_path(&model_path)?;

    println!("\n=== Inputs ===");
    for (ix, input) in model.input_outlets()?.iter().enumerate() {
        let fact = model.outlet_fact(*input)?;
        let name = &model.node(input.node).name;
        println!("  Input {ix}: name={name:?}, fact={fact:?}");
    }

    println!("\n=== Outputs ===");
    for (ix, output) in model.output_outlets()?.iter().enumerate() {
        let fact = model.outlet_fact(*output)?;
        let name = &model.node(output.node).name;
        println!("  Output {ix}: name={name:?}, fact={fact:?}");
    }

    // Try with concrete input shape
    println!("\n=== With concrete 1x3x320x320 input ===");
    let model = tract_onnx::onnx()
        .model_for_path(&model_path)?
        .with_input_fact(0, tract_onnx::prelude::InferenceFact::dt_shape(
            tract_onnx::prelude::DatumType::F32,
            &[1, 3, 320, 320],
        ))?
        .into_optimized()?
        .into_runnable()?;

    // Run with zeros to see output shapes
    use tract_onnx::prelude::*;
    let input = Tensor::zero::<f32>(&[1, 3, 320, 320])?;
    let outputs = model.run(tvec!(input.into()))?;

    println!("  Number of outputs: {}", outputs.len());
    for (ix, output) in outputs.iter().enumerate() {
        println!("  Output {ix}: shape={:?}, dtype={:?}", output.shape(), output.datum_type());
    }

    Ok(())
}
