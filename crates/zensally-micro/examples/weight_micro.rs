//! Link only the fixed runtime; pair with weight_tract for deployment probes.
use zensally_micro::{Input, MicroSalNetW24};
fn main() {
    let input: Vec<f32> = (0..3 * 65536)
        .map(|i| ((i * 137 + i / 256 * 59) % 1024) as f32 / 1023.0)
        .collect();
    let output = MicroSalNetW24::new()
        .unwrap()
        .with_fast_sigmoid()
        .infer(Input::contiguous(&input).unwrap())
        .unwrap();
    println!(
        "{:.9e}",
        std::hint::black_box(output.data)
            .iter()
            .copied()
            .sum::<f32>()
    );
}
