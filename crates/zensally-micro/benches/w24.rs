use std::{hint::black_box, io::Cursor};
use tract_onnx::prelude::*;
use zenbench::prelude::*;
use zensally_micro::{Input, MicroSalNetW24};

fn load_tract() -> std::sync::Arc<TypedRunnableModel> {
    let gz = include_bytes!("../../zensally-tract/models/microsalnet_v3ds_w24.onnx.gz");
    let mut bytes = vec![0; 769_236];
    zenflate::Decompressor::new()
        .gzip_decompress(gz, &mut bytes, enough::Unstoppable)
        .unwrap();
    tract_onnx::onnx()
        .model_for_read(&mut Cursor::new(bytes))
        .unwrap()
        .with_input_fact(0, InferenceFact::dt_shape(DatumType::F32, [1, 3, 256, 256]))
        .unwrap()
        .into_optimized()
        .unwrap()
        .into_runnable()
        .unwrap()
}

fn bench(suite: &mut Suite) {
    // Both arms include input packing and output materialization. No resize:
    // the exact same already-preprocessed 256x256 RGB tensor enters each.
    let input: Vec<f32> = (0..3 * 65536)
        .map(|i| ((i * 137 + i / 256 * 59) % 1024) as f32 / 1023.0)
        .collect();
    let tensor = Tensor::from_shape(&[1, 3, 256, 256], &input).unwrap();
    let tract = load_tract();
    let mut custom = MicroSalNetW24::new().unwrap();
    let mut fast = MicroSalNetW24::new().unwrap().with_fast_sigmoid();
    suite.group("w24/warm-single-thread", |g| {
        g.bench("tract", move |b| {
            b.iter(|| {
                let out = tract.run(tvec!(black_box(tensor.clone()).into())).unwrap();
                black_box(
                    out[0]
                        .as_plain_ram()
                        .unwrap()
                        .as_slice::<f32>()
                        .unwrap()
                        .to_vec(),
                )
            })
        });
        g.bench("micro", move |b| {
            b.iter(|| {
                black_box(
                    custom
                        .infer(Input::contiguous(black_box(&input)).unwrap())
                        .unwrap()
                        .data,
                )
            })
        });
        let fast_input: Vec<f32> = (0..3 * 65536)
            .map(|i| ((i * 137 + i / 256 * 59) % 1024) as f32 / 1023.0)
            .collect();
        g.bench("micro-fast-sigmoid", move |b| {
            b.iter(|| {
                black_box(
                    fast.infer(Input::contiguous(black_box(&fast_input)).unwrap())
                        .unwrap()
                        .data,
                )
            })
        });
    });
    suite.group("w24/load-and-pack", |g| {
        g.bench("tract", |b| b.iter(|| black_box(load_tract())));
        g.bench("micro", move |b| {
            b.iter(|| black_box(MicroSalNetW24::new().unwrap()))
        });
    });
}
zenbench::main!(bench);
