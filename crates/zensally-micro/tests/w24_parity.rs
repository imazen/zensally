//! End-to-end parity of the fixed w24 executor (`W24MicroModel`) against the
//! tract-onnx oracle running the same embedded model on real images, plus
//! the warm zero-allocation gate and stride-8 feature extraction checks.
#![allow(clippy::too_many_arguments)]

use std::sync::{Arc, OnceLock};

use archmage::testing::{CompileTimePolicy, for_each_token_permutation};
use tract_onnx::prelude::*;
use zensally_micro::{W24MicroModel, W24Workspace, w24_plan};

const MODEL_GZ: &[u8] = include_bytes!("../../zensally-tract/models/microsalnet_v3ds_w24.onnx.gz");

const TEST_IMAGES: &[&str] = &[
    "test_data/portrait.jpg",
    "test_data/saliency/dog_beach.jpg",
    "test_data/saliency/group.jpg",
    "test_data/saliency/person_dark.jpg",
];

/// Cross-implementation gates from the Phase-2 brief: identical weights,
/// different accumulation orders (tract's planner vs our FMA chains) allow
/// a small diff on the sigmoid-bounded output.
const MAX_ABS_TOL: f32 = 1.5e-5;
const MEAN_ABS_TOL: f32 = 2e-6;

fn raw_onnx() -> Vec<u8> {
    let mut out = vec![0u8; w24_plan::MODEL_BYTES];
    let o = zenflate::Decompressor::new()
        .gzip_decompress(MODEL_GZ, &mut out, enough::Unstoppable)
        .expect("model gzip failed");
    assert_eq!(o.output_written, w24_plan::MODEL_BYTES);
    out
}

/// tract oracle, loaded once (model optimize is the slow part).
fn tract_model() -> &'static Arc<TypedRunnableModel> {
    static M: OnceLock<Arc<TypedRunnableModel>> = OnceLock::new();
    M.get_or_init(|| {
        let bytes = raw_onnx();
        tract_onnx::onnx()
            .model_for_read(&mut std::io::Cursor::new(&bytes))
            .expect("onnx parse")
            .with_input_fact(0, InferenceFact::dt_shape(DatumType::F32, [1, 3, 256, 256]))
            .expect("input fact")
            .into_optimized()
            .expect("optimize")
            .into_runnable()
            .expect("runnable")
    })
}

/// tract output `[1, 1, 128, 128]` for an NHWC `256*256*3` input.
fn tract_forward(nhwc: &[f32]) -> Vec<f32> {
    let mut nchw = vec![0.0f32; 3 * 256 * 256];
    for y in 0..256 {
        for x in 0..256 {
            for c in 0..3 {
                nchw[c * 256 * 256 + y * 256 + x] = nhwc[(y * 256 + x) * 3 + c];
            }
        }
    }
    let input = Tensor::from_shape(&[1, 3, 256, 256], &nchw).expect("tensor");
    let out = tract_model().run(tvec!(input.into())).expect("tract run");
    out[0]
        .as_plain_ram()
        .and_then(|t| t.as_slice::<f32>().ok())
        .expect("plain f32 output")
        .to_vec()
}

/// Decode + stretch-resize to the model's `256x256` NHWC `[0,1]` input.
fn load_input(path: &str) -> Vec<f32> {
    let full = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join(path);
    let img = image::open(&full)
        .unwrap_or_else(|e| panic!("test image {}: {e}", full.display()))
        .resize_exact(256, 256, image::imageops::FilterType::Triangle)
        .to_rgb8();
    img.pixels()
        .flat_map(|p| p.0)
        .map(|v| v as f32 / 255.0)
        .collect()
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

fn mean_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs() as f64)
        .sum::<f64>() as f32
        / a.len() as f32
}

/// Deterministic synthetic 256x256 pattern (smooth gradients + edges).
fn synthetic_input() -> Vec<f32> {
    let mut v = vec![0.0f32; 256 * 256 * 3];
    for y in 0..256 {
        for x in 0..256 {
            let p = (y * 256 + x) * 3;
            v[p] = (x as f32 / 255.0) * ((y as f32 / 255.0) * 0.5 + 0.5);
            v[p + 1] = ((x + y) % 64) as f32 / 63.0;
            v[p + 2] = (((x as i32 - 128).pow(2) + (y as i32 - 128).pow(2)) as f32).sqrt() / 181.0;
        }
    }
    v
}

/// Serializes tests in this binary: the counting allocator is process-global,
/// so a concurrent sibling test would pollute the zero-alloc measurement.
static SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[test]
fn w24_forward_matches_tract_on_real_images() {
    let _serial = SERIAL.lock().unwrap();
    let model = W24MicroModel::new().expect("model load");
    let mut ws = W24Workspace::new();
    let mut out = vec![0.0f32; 128 * 128];

    let mut worst_max = 0.0f32;
    let mut worst_mean = 0.0f32;
    for (name, input) in TEST_IMAGES
        .iter()
        .map(|p| (*p, load_input(p)))
        .chain([("synthetic", synthetic_input())])
    {
        let oracle = tract_forward(&input);
        model.forward_nhwc(&input, &mut ws, &mut out);
        let d = max_abs_diff(&out, &oracle);
        let m = mean_abs_diff(&out, &oracle);
        worst_max = worst_max.max(d);
        worst_mean = worst_mean.max(m);
        eprintln!("{name}: vs tract max_abs_diff={d:e} mean_abs_diff={m:e}");
        assert!(d < MAX_ABS_TOL, "{name}: w24 vs tract max_abs_diff {d}");
        assert!(m < MEAN_ABS_TOL, "{name}: w24 vs tract mean_abs_diff {m}");

        // Sigmoid-bounded output with real dynamic range on real input.
        assert!(out.iter().all(|v| (0.0..=1.0).contains(v)));
        let (mut lo, mut hi) = (f32::MAX, f32::MIN);
        for &v in &out {
            lo = lo.min(v);
            hi = hi.max(v);
        }
        assert!(
            hi - lo > 0.1,
            "{name}: degenerate output range [{lo}, {hi}]"
        );
    }
    eprintln!("w24 vs tract: worst max_abs_diff={worst_max:e} mean_abs_diff={worst_mean:e}");
}

#[test]
fn w24_tier_parity_on_real_image() {
    let _serial = SERIAL.lock().unwrap();
    let model = W24MicroModel::new().expect("model load");
    let input = load_input(TEST_IMAGES[0]);
    let mut ws = W24Workspace::new();
    let mut reference = vec![0.0f32; 128 * 128];
    model.forward_nhwc(&input, &mut ws, &mut reference);

    let mut out = vec![0.0f32; 128 * 128];
    let mut worst = 0.0f32;
    let report = for_each_token_permutation(CompileTimePolicy::Warn, |perm| {
        model.forward_nhwc(&input, &mut ws, &mut out);
        let d = max_abs_diff(&out, &reference);
        worst = worst.max(d);
        assert!(d <= 2e-6, "tier parity diff {d} at {perm}");
    });
    eprintln!("w24 tier parity: {report} worst={worst:e}");
}

#[test]
fn w24_stride8_features() {
    let _serial = SERIAL.lock().unwrap();
    let model = W24MicroModel::new().expect("model load");
    let input = load_input(TEST_IMAGES[0]);
    let mut ws = W24Workspace::new();
    let mut out = vec![0.0f32; 128 * 128];
    model.forward_nhwc(&input, &mut ws, &mut out);
    let f = model.extract_stride8_features(&ws, &out);

    for i in 0..32 * 32 {
        assert!((0.0..=1.0).contains(&f.saliency_32x32[i]));
        assert!((0.0..=1.0).contains(&f.semantic_energy_32x32[i]));
        assert_eq!(
            f.matte_32x32[i],
            f.saliency_32x32[i] * f.semantic_energy_32x32[i],
            "matte must equal saliency*semantic at cell {i}"
        );
    }
    // A real image has structure: the feature maps are not degenerate.
    let max_sal = f.saliency_32x32.iter().copied().fold(0.0, f32::max);
    let max_sem = f.semantic_energy_32x32.iter().copied().fold(0.0, f32::max);
    assert!(max_sal > 0.01, "saliency all ~zero");
    assert_eq!(max_sem, 1.0, "semantic energy should be max-normalized");
}

/// Global allocator counting allocations for the zero-alloc gate.
mod alloc_count {
    use core::sync::atomic::{AtomicUsize, Ordering};
    use std::alloc::{GlobalAlloc, Layout, System};

    pub static ALLOCS: AtomicUsize = AtomicUsize::new(0);

    pub struct Counting;
    unsafe impl GlobalAlloc for Counting {
        unsafe fn alloc(&self, l: Layout) -> *mut u8 {
            ALLOCS.fetch_add(1, Ordering::SeqCst);
            unsafe { System.alloc(l) }
        }
        unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
            unsafe { System.dealloc(p, l) }
        }
    }
}

#[global_allocator]
static COUNTING: alloc_count::Counting = alloc_count::Counting;

#[test]
fn w24_warm_forward_allocates_nothing() {
    use core::sync::atomic::Ordering;
    let _serial = SERIAL.lock().unwrap();
    let model = W24MicroModel::new().expect("model load");
    let input = load_input(TEST_IMAGES[0]);
    let mut ws = W24Workspace::new();
    let mut out = vec![0.0f32; 128 * 128];

    // Warm pass: token dispatch caches settle.
    model.forward_nhwc(&input, &mut ws, &mut out);

    let before = alloc_count::ALLOCS.load(Ordering::SeqCst);
    model.forward_nhwc(&input, &mut ws, &mut out);
    let allocs = alloc_count::ALLOCS.load(Ordering::SeqCst) - before;
    assert_eq!(allocs, 0, "warm w24 forward allocated {allocs} times");
}
