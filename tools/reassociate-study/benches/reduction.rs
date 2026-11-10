#![forbid(unsafe_code)]
use std::hint::black_box;
use zenbench::prelude::*;

fn bench(suite: &mut Suite) {
    let input: Vec<f32> = (0..3 * 65536)
        .map(|i| ((i * 137 + i / 256 * 59) % 1024) as f32 / 1023.0)
        .collect();
    let other = input.clone();
    let mut base = baseline::MicroSalNetW24::new().unwrap().with_fast_sigmoid();
    let mut candidate = candidate::MicroSalNetW24::new()
        .unwrap()
        .with_fast_sigmoid();
    suite.group("w24/reduction", |g| {
        g.bench("ordered-fast-sigmoid", move |b| {
            b.iter(|| {
                black_box(
                    base.infer(baseline::Input::contiguous(black_box(&input)).unwrap())
                        .unwrap()
                        .data,
                )
            })
        });
        g.bench("candidate-fast-sigmoid", move |b| {
            b.iter(|| {
                black_box(
                    candidate
                        .infer(candidate::Input::contiguous(black_box(&other)).unwrap())
                        .unwrap()
                        .data,
                )
            })
        });
    });
}
zenbench::main!(bench);
