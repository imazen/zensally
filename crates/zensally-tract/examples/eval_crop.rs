/// Visual evaluation of smart crop heuristic.
///
/// Runs face detection (UltraFace) and saliency (MicroSalNet), then draws
/// annotated crop rectangles on each image for both Minimal and Maximal modes
/// across multiple aspect ratios. Generates an HTML viewer.
///
/// Usage:
///   cargo run --example eval_crop --features "ultraface,microsalnet" --release
fn main() {
    #[cfg(not(all(feature = "ultraface", feature = "microsalnet")))]
    {
        eprintln!("This example requires both 'ultraface' and 'microsalnet' features.");
        eprintln!(
            "Run: cargo run --example eval_crop --features \"ultraface,microsalnet\" --release"
        );
    }

    #[cfg(all(feature = "ultraface", feature = "microsalnet"))]
    run();
}

#[cfg(all(feature = "ultraface", feature = "microsalnet"))]
fn run() {
    use std::fmt::Write as FmtWrite;
    use std::path::{Path, PathBuf};
    use std::time::Instant;

    use zenlayout::smart_crop::{
        compute_crop, AspectRatio, CropConfig, CropMode, LANDSCAPE_16_9, PORTRAIT_3_4,
        PORTRAIT_9_16, SQUARE, FocusRect, HeatMap,
    };
    use zenlayout::Rect;
    use zensally::{FaceDetector, ImageRef, PixelFormat, SaliencyDetector};
    use zensally_tract::{MicroSalNet, UltraFaceDetector};

    let test_data = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("test_data");

    let wider = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("data/wider_face/WIDER_val/images");

    let output_dir = Path::new("/mnt/v/output/zensally/crop_eval");
    std::fs::create_dir_all(output_dir).expect("create output dir");

    // Collect test images
    let mut images: Vec<(String, PathBuf)> = Vec::new();

    let portrait = test_data.join("portrait.jpg");
    if portrait.exists() {
        images.push(("portrait".into(), portrait));
    }

    for name in &[
        "group",
        "person_dark",
        "dog_beach",
        "lighthouse",
        "food_plate",
    ] {
        let p = test_data.join("saliency").join(format!("{name}.jpg"));
        if p.exists() {
            images.push(((*name).into(), p));
        }
    }

    // WIDER FACE images — diverse scenarios
    let wider_picks: &[(&str, &str)] = &[
        ("parade", "0--Parade/0_Parade_marchingband_1_20.jpg"),
        (
            "interview1",
            "13--Interview/13_Interview_Interview_2_People_Visible_13_107.jpg",
        ),
        (
            "interview2",
            "13--Interview/13_Interview_Interview_2_People_Visible_13_155.jpg",
        ),
        (
            "family1",
            "20--Family_Group/20_Family_Group_Family_Group_20_100.jpg",
        ),
        (
            "family2",
            "20--Family_Group/20_Family_Group_Family_Group_20_1003.jpg",
        ),
        ("festival", "21--Festival/21_Festival_Festival_21_100.jpg"),
        (
            "football",
            "36--Football/36_Football_americanfootball_ball_36_111.jpg",
        ),
        (
            "dresses",
            "51--Dresses/51_Dresses_wearingdress_51_1012.jpg",
        ),
    ];
    for (name, relpath) in wider_picks {
        let p = wider.join(relpath);
        if p.exists() {
            images.push(((*name).into(), p));
        }
    }

    // A couple image (first in directory)
    let couple_dir = wider.join("19--Couple");
    if couple_dir.is_dir() {
        if let Some(entry) = std::fs::read_dir(&couple_dir)
            .ok()
            .and_then(|mut rd| rd.next())
            .and_then(|e| e.ok())
        {
            images.push(("couple".into(), entry.path()));
        }
    }

    if images.is_empty() {
        eprintln!("No test images found");
        return;
    }

    println!("=== Smart Crop Visual Evaluation ===\n");
    println!("Found {} test images", images.len());
    println!("Output: {}\n", output_dir.display());

    let t0 = Instant::now();
    let mut face_det = UltraFaceDetector::new().expect("UltraFace init");
    println!(
        "UltraFace load: {:.0}ms",
        t0.elapsed().as_secs_f64() * 1000.0
    );

    let t0 = Instant::now();
    let mut sal_det = MicroSalNet::new().expect("MicroSalNet init");
    println!(
        "MicroSalNet load: {:.0}ms\n",
        t0.elapsed().as_secs_f64() * 1000.0
    );

    let configs: &[(&str, AspectRatio, CropMode)] = &[
        ("9x16_min", PORTRAIT_9_16, CropMode::Minimal),
        ("9x16_max", PORTRAIT_9_16, CropMode::Maximal),
        ("3x4_min", PORTRAIT_3_4, CropMode::Minimal),
        ("3x4_max", PORTRAIT_3_4, CropMode::Maximal),
        ("1x1_min", SQUARE, CropMode::Minimal),
        ("1x1_max", SQUARE, CropMode::Maximal),
        ("16x9_min", LANDSCAPE_16_9, CropMode::Minimal),
        ("16x9_max", LANDSCAPE_16_9, CropMode::Maximal),
    ];

    let colors: &[(&str, [u8; 3])] = &[
        ("9x16_min", [0, 255, 255]),
        ("9x16_max", [255, 0, 255]),
        ("3x4_min", [255, 255, 0]),
        ("3x4_max", [255, 128, 0]),
        ("1x1_min", [128, 255, 128]),
        ("1x1_max", [255, 128, 128]),
        ("16x9_min", [128, 128, 255]),
        ("16x9_max", [255, 255, 255]),
    ];

    struct ImageResult {
        name: String,
        width: u32,
        height: u32,
        face_count: usize,
        face_ms: f64,
        sal_ms: f64,
        crops: Vec<(String, Rect)>,
    }

    let mut results: Vec<ImageResult> = Vec::new();

    for (name, path) in &images {
        let img = image::open(path).expect("open image");
        let rgb = img.to_rgb8();
        let (w, h) = rgb.dimensions();
        let pixels = rgb.as_raw();
        let image_ref = ImageRef::new(pixels, w, h, PixelFormat::Rgb).unwrap();

        let t0 = Instant::now();
        let faces = face_det.detect(&image_ref);
        let face_ms = t0.elapsed().as_secs_f64() * 1000.0;

        let t0 = Instant::now();
        let sal = sal_det.saliency_map(&image_ref);
        let sal_ms = t0.elapsed().as_secs_f64() * 1000.0;

        println!(
            "{:20} {:4}x{:<4}  faces={} ({:.0}ms)  saliency ({:.0}ms)",
            name,
            w,
            h,
            faces.len(),
            face_ms,
            sal_ms
        );

        // Annotated overview
        let mut annotated = rgb.clone();
        for face in &faces {
            let fx1 = (face.x1 / 100.0 * w as f32) as u32;
            let fy1 = (face.y1 / 100.0 * h as f32) as u32;
            let fx2 = (face.x2 / 100.0 * w as f32) as u32;
            let fy2 = (face.y2 / 100.0 * h as f32) as u32;
            draw_rect(&mut annotated, fx1, fy1, fx2, fy2, [0, 255, 0], 2);
        }

        let focus: Vec<FocusRect> = faces.iter().map(|f| FocusRect {
            x1: f.x1, y1: f.y1, x2: f.x2, y2: f.y2, weight: f.confidence,
        }).collect();
        let heatmap = HeatMap { data: sal.data.clone(), width: sal.width, height: sal.height };

        let mut crops: Vec<(String, Rect)> = Vec::new();
        for (cfg_name, ratio, mode) in configs {
            let config = CropConfig {
                target_aspect: *ratio,
                mode: *mode,
                ..CropConfig::default()
            };
            if let Some(crop) = compute_crop(w, h, &focus, Some(&heatmap), &config) {
                crops.push((cfg_name.to_string(), crop));
            }
        }

        for (cfg_name, crop) in &crops {
            let color = colors
                .iter()
                .find(|(n, _)| *n == cfg_name.as_str())
                .map(|(_, c)| *c)
                .unwrap_or([255, 255, 255]);
            draw_rect(
                &mut annotated,
                crop.x,
                crop.y,
                crop.x + crop.width,
                crop.y + crop.height,
                color,
                3,
            );
        }
        annotated
            .save(output_dir.join(format!("{name}_annotated.png")))
            .expect("save annotated");

        // Individual crops
        for (cfg_name, crop) in &crops {
            let cropped =
                image::imageops::crop_imm(&rgb, crop.x, crop.y, crop.width, crop.height).to_image();
            cropped
                .save(output_dir.join(format!("{name}_{cfg_name}.jpg")))
                .expect("save crop");
        }

        // Montage
        let montage_h = 400u32;
        let mut panels: Vec<image::RgbImage> = Vec::new();
        for (_cfg_name, ratio, mode) in configs {
            let config = CropConfig {
                target_aspect: *ratio,
                mode: *mode,
                ..CropConfig::default()
            };
            if let Some(crop) = compute_crop(w, h, &focus, Some(&heatmap), &config) {
                let cropped =
                    image::imageops::crop_imm(&rgb, crop.x, crop.y, crop.width, crop.height).to_image();
                let panel_w =
                    (montage_h as f64 * cropped.width() as f64 / cropped.height() as f64) as u32;
                let resized = image::imageops::resize(
                    &cropped,
                    panel_w.max(1),
                    montage_h,
                    image::imageops::FilterType::Lanczos3,
                );
                panels.push(resized);
            }
        }
        if !panels.is_empty() {
            let row1_w: u32 = panels.iter().take(4).map(|p| p.width() + 4).sum::<u32>();
            let row2_w: u32 = panels.iter().skip(4).map(|p| p.width() + 4).sum::<u32>();
            let total_w = row1_w.max(row2_w);
            let total_h = montage_h * 2 + 24;
            let label_h = 12;
            let mut montage =
                image::RgbImage::from_pixel(total_w, total_h, image::Rgb([32, 32, 32]));
            let mut x_off = 0u32;
            for (i, panel) in panels.iter().enumerate() {
                let y_off = if i < 4 {
                    if i == 0 {
                        x_off = 0;
                    }
                    label_h
                } else {
                    if i == 4 {
                        x_off = 0;
                    }
                    montage_h + label_h * 2
                };
                image::imageops::overlay(&mut montage, panel, x_off as i64, y_off as i64);
                x_off += panel.width() + 4;
            }
            montage
                .save(output_dir.join(format!("{name}_montage.png")))
                .expect("save montage");
        }

        for (cfg_name, crop) in &crops {
            println!(
                "  {:12} -> {}x{} at ({},{})",
                cfg_name, crop.width, crop.height, crop.x, crop.y
            );
        }
        println!();

        results.push(ImageResult {
            name: name.clone(),
            width: w,
            height: h,
            face_count: faces.len(),
            face_ms,
            sal_ms,
            crops,
        });
    }

    // Generate HTML viewer
    let html = generate_html("Smart Crop Evaluation", &results);
    let html_path = output_dir.join("index.html");
    std::fs::write(&html_path, &html).expect("write HTML");

    println!("Done. Open {}", html_path.display());

    fn generate_html(title: &str, results: &[ImageResult]) -> String {
        let total = results.len();
        let with_faces = results.iter().filter(|r| r.face_count > 0).count();
        let total_faces: usize = results.iter().map(|r| r.face_count).sum();
        let avg_face_ms =
            results.iter().map(|r| r.face_ms).sum::<f64>() / total.max(1) as f64;
        let avg_sal_ms =
            results.iter().map(|r| r.sal_ms).sum::<f64>() / total.max(1) as f64;

        let mut html = String::new();
        writeln!(html, "<!DOCTYPE html>").unwrap();
        writeln!(html, "<html lang=\"en\"><head><meta charset=\"utf-8\">").unwrap();
        writeln!(html, "<title>{title}</title>").unwrap();
        writeln!(html, "<style>").unwrap();
        writeln!(html, "{}", CSS).unwrap();
        writeln!(html, "</style></head><body>").unwrap();

        // Header
        writeln!(html, "<header>").unwrap();
        writeln!(html, "<h1>{title}</h1>").unwrap();
        writeln!(html, "<div class=\"stats\">").unwrap();
        writeln!(html, "<span>{total} images</span>").unwrap();
        writeln!(html, "<span>{with_faces} with faces ({total_faces} total)</span>").unwrap();
        writeln!(html, "<span>face det {avg_face_ms:.0}ms avg</span>").unwrap();
        writeln!(html, "<span>saliency {avg_sal_ms:.0}ms avg</span>").unwrap();
        writeln!(html, "</div>").unwrap();
        writeln!(html, "<div class=\"legend\">").unwrap();
        writeln!(html, "<span style=\"color:#0f0\">&#9632; faces</span>").unwrap();
        writeln!(html, "<span style=\"color:#0ff\">&#9632; 9:16 min</span>").unwrap();
        writeln!(html, "<span style=\"color:#f0f\">&#9632; 9:16 max</span>").unwrap();
        writeln!(html, "<span style=\"color:#ff0\">&#9632; 3:4 min</span>").unwrap();
        writeln!(html, "<span style=\"color:#f80\">&#9632; 3:4 max</span>").unwrap();
        writeln!(html, "<span style=\"color:#8f8\">&#9632; 1:1 min</span>").unwrap();
        writeln!(html, "<span style=\"color:#f88\">&#9632; 1:1 max</span>").unwrap();
        writeln!(html, "<span style=\"color:#88f\">&#9632; 16:9 min</span>").unwrap();
        writeln!(html, "<span style=\"color:#fff\">&#9632; 16:9 max</span>").unwrap();
        writeln!(html, "</div>").unwrap();
        writeln!(html, "</header>").unwrap();

        // Image cards
        for r in results {
            let note = if r.face_count == 0 {
                "saliency-only".to_string()
            } else {
                format!("{} face{}", r.face_count, if r.face_count != 1 { "s" } else { "" })
            };

            writeln!(html, "<section class=\"card\">").unwrap();
            writeln!(html, "<h2>{} <span class=\"dim\">{}x{} &mdash; {}</span></h2>",
                r.name, r.width, r.height, note).unwrap();

            // Top row: annotated + montage
            writeln!(html, "<div class=\"top-row\">").unwrap();
            writeln!(html, "<a href=\"{0}_annotated.png\" target=\"_blank\"><img src=\"{0}_annotated.png\" class=\"annotated\"></a>", r.name).unwrap();
            writeln!(html, "<a href=\"{0}_montage.png\" target=\"_blank\"><img src=\"{0}_montage.png\" class=\"montage\"></a>", r.name).unwrap();
            writeln!(html, "</div>").unwrap();

            // Crop grid
            writeln!(html, "<div class=\"crop-grid\">").unwrap();
            let cfg_labels = [
                ("9x16_min", "9:16 min"),
                ("9x16_max", "9:16 max"),
                ("3x4_min", "3:4 min"),
                ("3x4_max", "3:4 max"),
                ("1x1_min", "1:1 min"),
                ("1x1_max", "1:1 max"),
                ("16x9_min", "16:9 min"),
                ("16x9_max", "16:9 max"),
            ];
            for (cfg_name, label) in cfg_labels {
                if let Some((_, crop)) = r.crops.iter().find(|(n, _)| n == cfg_name) {
                    let file = format!("{}_{}.jpg", r.name, cfg_name);
                    writeln!(html, "<div class=\"crop-item\">").unwrap();
                    writeln!(html, "<a href=\"{file}\" target=\"_blank\"><img src=\"{file}\"></a>").unwrap();
                    writeln!(html, "<div class=\"crop-label\">{label}<br>{}x{}</div>", crop.width, crop.height).unwrap();
                    writeln!(html, "</div>").unwrap();
                }
            }
            writeln!(html, "</div>").unwrap();
            writeln!(html, "</section>").unwrap();
        }

        writeln!(html, "</body></html>").unwrap();
        html
    }
}

#[cfg(all(feature = "ultraface", feature = "microsalnet"))]
const CSS: &str = r#"
* { margin: 0; padding: 0; box-sizing: border-box; }
body { background: #1a1a1a; color: #ccc; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif; padding: 20px; }
header { margin-bottom: 30px; }
h1 { color: #fff; margin-bottom: 8px; }
.stats { display: flex; gap: 20px; font-size: 14px; color: #999; margin-bottom: 6px; }
.legend { display: flex; gap: 14px; font-size: 13px; flex-wrap: wrap; }
.card { background: #222; border-radius: 8px; padding: 16px; margin-bottom: 24px; }
.card h2 { color: #fff; font-size: 18px; margin-bottom: 12px; }
.dim { color: #888; font-weight: normal; font-size: 14px; }
.top-row { display: flex; gap: 12px; margin-bottom: 12px; flex-wrap: wrap; align-items: flex-start; }
.top-row .annotated { max-height: 400px; width: auto; border-radius: 4px; }
.top-row .montage { max-height: 400px; width: auto; border-radius: 4px; }
.crop-grid { display: flex; gap: 8px; flex-wrap: wrap; align-items: flex-end; }
.crop-item { text-align: center; }
.crop-item img { max-height: 220px; width: auto; border-radius: 4px; display: block; }
.crop-label { font-size: 11px; color: #999; margin-top: 4px; }
a { text-decoration: none; }
"#;

#[cfg(all(feature = "ultraface", feature = "microsalnet"))]
fn draw_rect(
    img: &mut image::RgbImage,
    x1: u32,
    y1: u32,
    x2: u32,
    y2: u32,
    color: [u8; 3],
    thickness: u32,
) {
    let (w, h) = img.dimensions();
    let x1 = x1.min(w.saturating_sub(1));
    let y1 = y1.min(h.saturating_sub(1));
    let x2 = x2.min(w.saturating_sub(1));
    let y2 = y2.min(h.saturating_sub(1));
    let px = image::Rgb(color);

    for t in 0..thickness {
        let top = y1.saturating_add(t).min(y2);
        let bot = y2.saturating_sub(t).max(y1);
        for x in x1..=x2 {
            img.put_pixel(x, top, px);
            img.put_pixel(x, bot, px);
        }
        let left = x1.saturating_add(t).min(x2);
        let right = x2.saturating_sub(t).max(x1);
        for y in y1..=y2 {
            img.put_pixel(left, y, px);
            img.put_pixel(right, y, px);
        }
    }
}
