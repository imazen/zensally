use zensally_micro::{Input, MicroSalNetW24};

#[test]
fn rejects_invalid_strides_and_overflow() {
    assert!(Input::new(&[], usize::MAX, usize::MAX).is_err());
    assert!(Input::new(&[0.0; 10], 256, 65536).is_err());
    assert!(Input::new(&vec![0.0; 3 * 65536], 255, 65536).is_err());
}

#[test]
fn complete_model_matches_scalar_and_strided_input() {
    let mut seed = 0x97619a53u32;
    let input: Vec<f32> = (0..3 * 256 * 256)
        .map(|_| {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            (seed & 65535) as f32 / 65535.0
        })
        .collect();
    let mut packed = MicroSalNetW24::new().unwrap();
    let mut scalar = MicroSalNetW24::scalar().unwrap();
    let fast = packed.infer(Input::contiguous(&input).unwrap()).unwrap();
    let reference = scalar.infer(Input::contiguous(&input).unwrap()).unwrap();
    assert_eq!((fast.width, fast.height), (128, 128));
    assert!(
        fast.data
            .iter()
            .all(|&x| x.is_finite() && (0.0..=1.0).contains(&x))
    );
    for (i, (&a, &b)) in fast.data.iter().zip(&reference.data).enumerate() {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "scalar/SIMD mismatch at {i}: {a} / {b}"
        );
    }
    let row_stride = 263;
    let channel_stride = row_stride * 256 + 17;
    let mut strided = vec![f32::NAN; channel_stride * 3];
    for c in 0..3 {
        for y in 0..256 {
            strided[c * channel_stride + y * row_stride..][..256]
                .copy_from_slice(&input[c * 65536 + y * 256..][..256]);
        }
    }
    let padded = packed
        .infer(Input::new(&strided, row_stride, channel_stride).unwrap())
        .unwrap();
    assert_eq!(
        fast.data.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
        padded.data.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
    );
}

#[test]
fn fast_sigmoid_keeps_its_scalar_simd_contract() {
    let input: Vec<f32> = (0..3 * 65536)
        .map(|i| ((i * 137 + i / 256 * 59) % 1024) as f32 / 1023.0)
        .collect();
    let mut simd = MicroSalNetW24::new().unwrap().with_fast_sigmoid();
    let mut scalar = MicroSalNetW24::scalar().unwrap().with_fast_sigmoid();
    let a = simd.infer(Input::contiguous(&input).unwrap()).unwrap();
    let b = scalar.infer(Input::contiguous(&input).unwrap()).unwrap();
    for (i, (&a, &b)) in a.data.iter().zip(&b.data).enumerate() {
        assert_eq!(a.to_bits(), b.to_bits(), "fast sigmoid at {i}");
    }
    let row_stride = 261;
    let channel_stride = row_stride * 256 + 11;
    let mut strided = vec![f32::NAN; channel_stride * 3];
    for c in 0..3 {
        for y in 0..256 {
            strided[c * channel_stride + y * row_stride..][..256]
                .copy_from_slice(&input[c * 65536 + y * 256..][..256]);
        }
    }
    let padded = simd
        .infer(Input::new(&strided, row_stride, channel_stride).unwrap())
        .unwrap();
    assert!(
        a.data
            .iter()
            .zip(&padded.data)
            .all(|(a, b)| a.to_bits() == b.to_bits())
    );
}

#[test]
fn selected_tier_obeys_requested_validation_target() {
    use zenjpegai::nn::fast::Tier;
    let name = |tier: Tier| match tier {
        #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
        Tier::V4(_) => "v4",
        #[cfg(target_arch = "x86_64")]
        Tier::V3(_) => "v3",
        #[cfg(target_arch = "aarch64")]
        Tier::Neon(_) => "neon",
        #[cfg(target_arch = "wasm32")]
        Tier::Wasm128(_) => "wasm128",
        Tier::Scalar => "scalar",
    };
    let selected = name(Tier::detect());
    assert!(
        Tier::available()
            .into_iter()
            .any(|tier| name(tier) == selected)
    );
    if let Ok(required) = std::env::var("ZENSALLY_REQUIRED_TIER") {
        assert_eq!(
            selected, required,
            "the requested SIMD tier must actually execute"
        );
    }
    println!("selected runtime tier: {selected}");
}
