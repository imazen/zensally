/// Holdout evaluation: 50 images from WIDER FACE categories not used during development.
///
/// For each image: annotated overlay, all crop variants, and a montage.
/// Generates an HTML viewer and markdown report.
///
/// Usage:
///   cargo run --example eval_crop_holdout --features "ultraface,microsalnet" --release
fn main() {
    #[cfg(not(all(feature = "ultraface", feature = "microsalnet")))]
    {
        eprintln!("Requires both 'ultraface' and 'microsalnet' features.");
        eprintln!(
            "Run: cargo run --example eval_crop_holdout --features \"ultraface,microsalnet\" --release"
        );
    }

    #[cfg(all(feature = "ultraface", feature = "microsalnet"))]
    run();
}

#[cfg(all(feature = "ultraface", feature = "microsalnet"))]
fn run() {
    use std::fmt::Write as FmtWrite;
    use std::path::Path;
    use std::time::Instant;

    use zenlayout::smart_crop::{
        compute_crop, AspectRatio, CropConfig, CropMode, LANDSCAPE_16_9, PORTRAIT_3_4,
        PORTRAIT_9_16, SQUARE, FocusRect, HeatMap,
    };
    use zenlayout::Rect;
    use zensally::{FaceDetector, ImageRef, PixelFormat, SaliencyDetector};
    use zensally_tract::{MicroSalNet, UltraFaceDetector};

    let wider = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("data/wider_face/WIDER_val/images");

    let output_dir = Path::new("/mnt/v/output/zensally/crop_holdout");
    std::fs::create_dir_all(output_dir).expect("create output dir");

    // 50 holdout images from categories NOT used during development.
    // Dev categories excluded: Parade, Interview, Couple, Family_Group, Festival, Football, Dresses
    let holdout: &[(&str, &str)] = &[
        ("march1", "10--People_Marching/10_People_Marching_People_Marching_10_People_Marching_People_Marching_10_368.jpg"),
        ("march2", "10--People_Marching/10_People_Marching_People_Marching_2_668.jpg"),
        ("meeting1", "11--Meeting/11_Meeting_Meeting_11_Meeting_Meeting_11_406.jpg"),
        ("meeting2", "11--Meeting/11_Meeting_Meeting_11_Meeting_Meeting_11_468.jpg"),
        ("group1", "12--Group/12_Group_Group_12_Group_Group_12_610.jpg"),
        ("stock1", "15--Stock_Market/15_Stock_Market_Stock_Market_15_286.jpg"),
        ("award1", "16--Award_Ceremony/16_Award_Ceremony_Awards_Ceremony_16_124.jpg"),
        ("ceremony1", "17--Ceremony/17_Ceremony_Ceremony_17_271.jpg"),
        ("concert1", "18--Concerts/18_Concerts_Concerts_18_27.jpg"),
        ("demo1", "2--Demonstration/2_Demonstration_Political_Rally_2_641.jpg"),
        ("picnic1", "22--Picnic/22_Picnic_Picnic_22_308.jpg"),
        ("firing1", "24--Soldier_Firing/24_Soldier_Firing_Soldier_Firing_24_405.jpg"),
        ("firing2", "24--Soldier_Firing/24_Soldier_Firing_Soldier_Firing_24_812.jpg"),
        ("patrol1", "25--Soldier_Patrol/25_Soldier_Patrol_Soldier_Patrol_25_614.jpg"),
        ("patrol2", "25--Soldier_Patrol/25_Soldier_Patrol_Soldier_Patrol_25_761.jpg"),
        ("drill1", "26--Soldier_Drilling/26_Soldier_Drilling_Soldiers_Drilling_26_1022.jpg"),
        ("spa1", "27--Spa/27_Spa_Spa_27_225.jpg"),
        ("students1", "29--Students_Schoolkids/29_Students_Schoolkids_Students_Schoolkids_29_250.jpg"),
        ("riot1", "3--Riot/3_Riot_Riot_3_480.jpg"),
        ("surgeon1", "30--Surgeons/30_Surgeons_Surgeons_30_746.jpg"),
        ("waiter1", "31--Waiter_Waitress/31_Waiter_Waitress_Waiter_Waitress_31_358.jpg"),
        ("worker1", "32--Worker_Laborer/32_Worker_Laborer_Worker_Laborer_32_494.jpg"),
        ("running1", "33--Running/33_Running_Running_33_332.jpg"),
        ("running2", "33--Running/33_Running_Running_33_891.jpg"),
        ("baseball1", "34--Baseball/34_Baseball_Baseball_34_895.jpg"),
        ("soccer1", "37--Soccer/37_Soccer_soccer_ball_37_269.jpg"),
        ("tennis1", "38--Tennis/38_Tennis_Tennis_38_497.jpg"),
        ("dancing1", "4--Dancing/4_Dancing_Dancing_4_1028.jpg"),
        ("dancing2", "4--Dancing/4_Dancing_Dancing_4_1036.jpg"),
        ("swimming1", "41--Swimming/41_Swimming_Swimmer_41_275.jpg"),
        ("aerobics1", "44--Aerobics/44_Aerobics_Aerobics_44_707.jpg"),
        ("matador1", "47--Matador_Bullfighter/47_Matador_Bullfighter_Matador_Bullfighter_47_617.jpg"),
        ("matador2", "47--Matador_Bullfighter/47_Matador_Bullfighter_matadorbullfighting_47_511.jpg"),
        ("greeting1", "49--Greeting/49_Greeting_peoplegreeting_49_387.jpg"),
        ("party1", "50--Celebration_Or_Party/50_Celebration_Or_Party_houseparty_50_173.jpg"),
        ("photo1", "52--Photographers/52_Photographers_taketouristphotos_52_141.jpg"),
        ("photo2", "52--Photographers/52_Photographers_taketouristphotos_52_266.jpg"),
        ("raid1", "53--Raid/53_Raid_policeraid_53_770.jpg"),
        ("rescue1", "54--Rescue/54_Rescue_rescuepeople_54_738.jpg"),
        ("coach1", "55--Sports_Coach_Trainer/55_Sports_Coach_Trainer_sportcoaching_55_859.jpg"),
        ("angler1", "57--Angler/57_Angler_peoplefishing_57_104.jpg"),
        ("angler2", "57--Angler/57_Angler_peoplefishing_57_254.jpg"),
        ("hockey1", "58--Hockey/58_Hockey_icehockey_puck_58_467.jpg"),
        ("driving1", "59--people--driving--car/59_peopledrivingcar_peopledrivingcar_59_117.jpg"),
        ("driving2", "59--people--driving--car/59_peopledrivingcar_peopledrivingcar_59_532.jpg"),
        ("funeral1", "6--Funeral/6_Funeral_Funeral_6_315.jpg"),
        ("funeral2", "6--Funeral/6_Funeral_Funeral_6_861.jpg"),
        ("cheering1", "7--Cheering/7_Cheering_Cheering_7_631.jpg"),
        ("election1", "8--Election_Campain/8_Election_Campain_Election_Campaign_8_236.jpg"),
        ("press1", "9--Press_Conference/9_Press_Conference_Press_Conference_9_278.jpg"),
    ];

    let mut images: Vec<(&str, std::path::PathBuf)> = Vec::new();
    for (name, relpath) in holdout {
        let p = wider.join(relpath);
        if p.exists() {
            images.push((name, p));
        }
    }

    println!(
        "=== Smart Crop Holdout Evaluation ({} images) ===\n",
        images.len()
    );
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
    let mut report = String::new();
    writeln!(report, "# Smart Crop Holdout Evaluation").unwrap();
    writeln!(report).unwrap();
    writeln!(
        report,
        "50 images from WIDER FACE val (categories not used during development)"
    )
    .unwrap();
    writeln!(report).unwrap();
    writeln!(
        report,
        "| # | Name | Size | Faces | Face ms | Sal ms | 9:16 min | 9:16 max | Notes |"
    )
    .unwrap();
    writeln!(
        report,
        "|---|------|------|-------|---------|--------|----------|----------|-------|"
    )
    .unwrap();

    let mut crop_failures = 0usize;

    for (idx, (name, path)) in images.iter().enumerate() {
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

        let focus: Vec<FocusRect> = faces.iter().map(|f| FocusRect {
            x1: f.x1, y1: f.y1, x2: f.x2, y2: f.y2, weight: f.confidence,
        }).collect();
        let heatmap = HeatMap { data: sal.data.clone(), width: sal.width, height: sal.height };

        // Compute all crops
        let mut crops: Vec<(String, Rect)> = Vec::new();
        for (cfg_name, ratio, mode) in configs {
            let config = CropConfig {
                target_aspect: *ratio,
                mode: *mode,
                ..CropConfig::default()
            };
            if let Some(crop) = compute_crop(w, h, &focus, Some(&heatmap), &config) {
                crops.push((cfg_name.to_string(), crop));
            } else {
                crop_failures += 1;
            }
        }

        // Annotated image
        let mut annotated = rgb.clone();
        for face in &faces {
            let fx1 = (face.x1 / 100.0 * w as f32) as u32;
            let fy1 = (face.y1 / 100.0 * h as f32) as u32;
            let fx2 = (face.x2 / 100.0 * w as f32) as u32;
            let fy2 = (face.y2 / 100.0 * h as f32) as u32;
            draw_rect(&mut annotated, fx1, fy1, fx2, fy2, [0, 255, 0], 2);
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
        let montage_h = 300u32;
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
            let total_h = montage_h * 2 + 8;
            let mut montage =
                image::RgbImage::from_pixel(total_w, total_h, image::Rgb([32, 32, 32]));
            let mut x_off = 0u32;
            for (i, panel) in panels.iter().enumerate() {
                let y_off = if i < 4 {
                    if i == 0 {
                        x_off = 0;
                    }
                    0
                } else {
                    if i == 4 {
                        x_off = 0;
                    }
                    montage_h + 8
                };
                image::imageops::overlay(&mut montage, panel, x_off as i64, y_off as i64);
                x_off += panel.width() + 4;
            }
            montage
                .save(output_dir.join(format!("{name}_montage.png")))
                .expect("save montage");
        }

        // Report row
        let min_9x16 = crops.iter().find(|(n, _)| n == "9x16_min").map(|(_, c)| c);
        let max_9x16 = crops.iter().find(|(n, _)| n == "9x16_max").map(|(_, c)| c);
        let min_str = min_9x16
            .map(|c| format!("{}x{} @({},{})", c.width, c.height, c.x, c.y))
            .unwrap_or_else(|| "NONE".into());
        let max_str = max_9x16
            .map(|c| format!("{}x{} @({},{})", c.width, c.height, c.x, c.y))
            .unwrap_or_else(|| "NONE".into());
        let notes = if faces.is_empty() {
            "saliency"
        } else if faces.len() == 1 {
            "1 face"
        } else {
            "multi-face"
        };
        writeln!(
            report,
            "| {} | {} | {}x{} | {} | {:.0} | {:.0} | {} | {} | {} |",
            idx + 1,
            name,
            w,
            h,
            faces.len(),
            face_ms,
            sal_ms,
            min_str,
            max_str,
            notes
        )
        .unwrap();

        println!(
            "{:3}. {:15} {:4}x{:<4} faces={:<3} ({:.0}ms + {:.0}ms)",
            idx + 1,
            name,
            w,
            h,
            faces.len(),
            face_ms,
            sal_ms
        );

        results.push(ImageResult {
            name: name.to_string(),
            width: w,
            height: h,
            face_count: faces.len(),
            face_ms,
            sal_ms,
            crops,
        });
    }

    // Summary for report
    let n = results.len() as f64;
    let with_faces = results.iter().filter(|r| r.face_count > 0).count();
    let total_faces: usize = results.iter().map(|r| r.face_count).sum();
    let avg_face_ms = results.iter().map(|r| r.face_ms).sum::<f64>() / n;
    let avg_sal_ms = results.iter().map(|r| r.sal_ms).sum::<f64>() / n;

    writeln!(report).unwrap();
    writeln!(report, "## Summary").unwrap();
    writeln!(report).unwrap();
    writeln!(report, "- **Images**: {}", results.len()).unwrap();
    writeln!(
        report,
        "- **Images with faces**: {} ({:.0}%)",
        with_faces,
        with_faces as f64 / n * 100.0
    )
    .unwrap();
    writeln!(report, "- **Total faces detected**: {}", total_faces).unwrap();
    writeln!(report, "- **Avg faces/image**: {:.1}", total_faces as f64 / n).unwrap();
    writeln!(report, "- **Avg face detection**: {:.1}ms", avg_face_ms).unwrap();
    writeln!(report, "- **Avg saliency**: {:.1}ms", avg_sal_ms).unwrap();
    writeln!(report, "- **Crop failures**: {}", crop_failures).unwrap();

    std::fs::write(output_dir.join("REPORT.md"), &report).expect("write report");

    // Generate HTML viewer
    let html = generate_html("Smart Crop Holdout — 50 images", &results);
    let html_path = output_dir.join("index.html");
    std::fs::write(&html_path, &html).expect("write HTML");

    println!("\n=== Summary ===");
    println!("Images: {}", results.len());
    println!(
        "With faces: {} ({:.0}%)",
        with_faces,
        with_faces as f64 / n * 100.0
    );
    println!("Total faces: {}", total_faces);
    println!(
        "Avg face det: {:.1}ms, saliency: {:.1}ms",
        avg_face_ms, avg_sal_ms
    );
    println!("Crop failures: {}", crop_failures);
    println!("\nHTML: {}", html_path.display());
    println!("Output: {}", output_dir.display());

    fn generate_html(title: &str, results: &[ImageResult]) -> String {
        let total = results.len();
        let with_faces = results.iter().filter(|r| r.face_count > 0).count();
        let total_faces: usize = results.iter().map(|r| r.face_count).sum();
        let avg_face_ms = results.iter().map(|r| r.face_ms).sum::<f64>() / total.max(1) as f64;
        let avg_sal_ms = results.iter().map(|r| r.sal_ms).sum::<f64>() / total.max(1) as f64;

        let mut html = String::new();
        writeln!(html, "<!DOCTYPE html>").unwrap();
        writeln!(html, "<html lang=\"en\"><head><meta charset=\"utf-8\">").unwrap();
        writeln!(html, "<title>{title}</title>").unwrap();
        writeln!(html, "<style>").unwrap();
        writeln!(html, "{}", CSS).unwrap();
        writeln!(html, "</style></head><body>").unwrap();

        writeln!(html, "<header>").unwrap();
        writeln!(html, "<h1>{title}</h1>").unwrap();
        writeln!(html, "<div class=\"stats\">").unwrap();
        writeln!(html, "<span>{total} images</span>").unwrap();
        writeln!(
            html,
            "<span>{with_faces} with faces ({total_faces} total)</span>"
        )
        .unwrap();
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

        for r in results {
            let note = if r.face_count == 0 {
                "saliency-only".to_string()
            } else {
                format!(
                    "{} face{}",
                    r.face_count,
                    if r.face_count != 1 { "s" } else { "" }
                )
            };

            writeln!(html, "<section class=\"card\">").unwrap();
            writeln!(
                html,
                "<h2>{} <span class=\"dim\">{}x{} &mdash; {}</span></h2>",
                r.name, r.width, r.height, note
            )
            .unwrap();

            writeln!(html, "<div class=\"top-row\">").unwrap();
            writeln!(
                html,
                "<a href=\"{0}_annotated.png\" target=\"_blank\"><img src=\"{0}_annotated.png\" class=\"annotated\"></a>",
                r.name
            ).unwrap();
            writeln!(
                html,
                "<a href=\"{0}_montage.png\" target=\"_blank\"><img src=\"{0}_montage.png\" class=\"montage\"></a>",
                r.name
            ).unwrap();
            writeln!(html, "</div>").unwrap();

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
                    writeln!(
                        html,
                        "<a href=\"{file}\" target=\"_blank\"><img src=\"{file}\"></a>"
                    )
                    .unwrap();
                    writeln!(
                        html,
                        "<div class=\"crop-label\">{label}<br>{}x{}</div>",
                        crop.width, crop.height
                    )
                    .unwrap();
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
