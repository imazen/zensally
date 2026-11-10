fn main() -> Result<(), Box<dyn std::error::Error>> {
    let repeats = std::env::args().nth(1).map_or(Ok(1), |s| s.parse::<usize>())?;
    let input: Vec<f32> = (0..3 * 65536)
        .map(|i| ((i * 137 + i / 256 * 59) % 1024) as f32 / 1023.0)
        .collect();
    let mut model = runtime::MicroSalNetW24::new()?.with_fast_sigmoid();
    for _ in 0..repeats {
        std::hint::black_box(model.infer(runtime::Input::contiguous(std::hint::black_box(&input))?)?);
    }
    Ok(())
}
