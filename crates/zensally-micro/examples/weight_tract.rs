//! Link only the general ONNX runtime, with the identical w24 model and input.
use std::io::Cursor;
use tract_onnx::prelude::*;
fn main() {
    let input: Vec<f32> = (0..3 * 65536)
        .map(|i| ((i * 137 + i / 256 * 59) % 1024) as f32 / 1023.0)
        .collect();
    let gzip = include_bytes!("../../zensally-tract/models/microsalnet_v3ds_w24.onnx.gz");
    let mut bytes = vec![0; 769236];
    zenflate::Decompressor::new()
        .gzip_decompress(gzip, &mut bytes, enough::Unstoppable)
        .unwrap();
    let model = tract_onnx::onnx()
        .model_for_read(&mut Cursor::new(bytes))
        .unwrap()
        .with_input_fact(0, InferenceFact::dt_shape(DatumType::F32, [1, 3, 256, 256]))
        .unwrap()
        .into_optimized()
        .unwrap()
        .into_runnable()
        .unwrap();
    let tensor = Tensor::from_shape(&[1, 3, 256, 256], &input).unwrap();
    let output = model.run(tvec!(tensor.into())).unwrap();
    let values = std::hint::black_box(
        output[0]
            .as_plain_ram()
            .unwrap()
            .as_slice::<f32>()
            .unwrap()
            .to_vec(),
    );
    println!("{:.9e}", values.iter().copied().sum::<f32>());
}
