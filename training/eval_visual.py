"""Evaluate MicroSalNet + UltraFace on DUTS-TE.

Shows:
  - Red box: primary salient region (connected component with highest total saliency)
  - Green box: secondary salient region (2nd highest)
  - Cyan box: detected faces (UltraFace)

Usage:
    python3 training/eval_visual.py [--limit 200] [--threshold 0.4]
"""

import argparse
import json
import os
import numpy as np
import onnxruntime as ort
from pathlib import Path
from PIL import Image, ImageDraw
from scipy import ndimage

_REPO_ROOT = Path(__file__).resolve().parent.parent
SALIENCY_ONNX = Path(os.environ.get(
    "SALIENCY_ONNX_PATH",
    str(_REPO_ROOT / "training" / "checkpoints" / "microsalnet_w16_s256_tract.onnx"),
))
FACE_ONNX = Path("/tmp/ultraface-rfb-320.onnx")
FACE_GZ = _REPO_ROOT / "crates" / "zensally-tract" / "models" / "ultraface-rfb-320.onnx.gz"
DUTS_TE_IMG = Path(os.environ.get(
    "DUTS_TE_IMAGE_DIR",
    str(_REPO_ROOT / "data" / "DUTS-TE" / "DUTS-TE-Image"),
))
DUTS_TE_MASK = Path(os.environ.get(
    "DUTS_TE_MASK_DIR",
    str(_REPO_ROOT / "data" / "DUTS-TE" / "DUTS-TE-Mask"),
))
OUTPUT_DIR = Path(os.environ.get(
    "ZENSALLY_OUTPUT_DIR",
    "/mnt/v/output/zensally",
)) / "microsalnet_holdout"

SAL_INPUT = 256
SAL_OUTPUT = 128
FACE_W, FACE_H = 320, 240


def ensure_face_model():
    """Decompress UltraFace ONNX if needed."""
    if FACE_ONNX.exists():
        return
    import gzip
    with gzip.open(str(FACE_GZ), "rb") as f:
        data = f.read()
    FACE_ONNX.write_bytes(data)
    print(f"Decompressed UltraFace: {len(data) // 1024} KB")


def preprocess_saliency(img: Image.Image) -> np.ndarray:
    """256x256 stretch, [0,1], NCHW."""
    resized = img.convert("RGB").resize((SAL_INPUT, SAL_INPUT), Image.BILINEAR)
    arr = np.array(resized, dtype=np.float32) / 255.0
    return arr.transpose(2, 0, 1)[np.newaxis]


def preprocess_face(img: Image.Image):
    """320x240 letterbox, (pixel-127)/128, NCHW. Returns (input, pad_left, pad_top, ratio)."""
    w, h = img.size
    ratio = min(FACE_W / w, FACE_H / h)
    rw = int(round(w * ratio))
    rh = int(round(h * ratio))
    pad_left = (FACE_W - rw) // 2
    pad_top = (FACE_H - rh) // 2

    resized = img.convert("RGB").resize((rw, rh), Image.BILINEAR)
    arr = np.array(resized, dtype=np.float32)

    # Place into padded canvas
    canvas = np.full((FACE_H, FACE_W, 3), 127.0, dtype=np.float32)
    canvas[pad_top:pad_top + rh, pad_left:pad_left + rw] = arr

    # Normalize and NCHW
    inp = ((canvas - 127.0) / 128.0).transpose(2, 0, 1)[np.newaxis]
    return inp, pad_left, pad_top, ratio


def detect_faces(sess, img: Image.Image, score_thresh=0.7, nms_thresh=0.3):
    """Run UltraFace, return list of (x1, y1, x2, y2, confidence) in original pixel coords."""
    w, h = img.size
    inp, pad_left, pad_top, ratio = preprocess_face(img)
    input_name = sess.get_inputs()[0].name
    scores, boxes = sess.run(None, {input_name: inp})

    # scores: [1, 4420, 2], boxes: [1, 4420, 4]
    scores = scores[0]  # [4420, 2]
    boxes = boxes[0]    # [4420, 4]

    detections = []
    for i in range(len(scores)):
        conf = scores[i, 1]
        if conf < score_thresh:
            continue
        xmin = boxes[i, 0] * FACE_W
        ymin = boxes[i, 1] * FACE_H
        xmax = boxes[i, 2] * FACE_W
        ymax = boxes[i, 3] * FACE_H

        # Convert from padded space to original
        x1 = (xmin - pad_left) / ratio
        y1 = (ymin - pad_top) / ratio
        x2 = (xmax - pad_left) / ratio
        y2 = (ymax - pad_top) / ratio

        detections.append((x1, y1, x2, y2, float(conf)))

    # NMS
    detections.sort(key=lambda d: -d[4])
    keep = []
    for d in detections:
        overlap = False
        for k in keep:
            ix1 = max(d[0], k[0])
            iy1 = max(d[1], k[1])
            ix2 = min(d[2], k[2])
            iy2 = min(d[3], k[3])
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            area_d = (d[2] - d[0]) * (d[3] - d[1])
            area_k = (k[2] - k[0]) * (k[3] - k[1])
            union = area_d + area_k - inter
            if union > 0 and inter / union >= nms_thresh:
                overlap = True
                break
        if not overlap:
            keep.append(d)

    return keep


def find_salient_blobs(saliency: np.ndarray, threshold_frac: float, orig_w: int, orig_h: int, max_blobs=3):
    """Find connected components in thresholded saliency map.

    Returns list of (x1, y1, x2, y2, total_saliency) sorted by total saliency descending.
    Coordinates are in original image pixels.
    """
    sal_max = saliency.max()
    if sal_max < 0.05:
        return []

    thresh = sal_max * threshold_frac
    mask = saliency >= thresh

    labeled, n_labels = ndimage.label(mask)
    if n_labels == 0:
        return []

    scale_x = orig_w / saliency.shape[1]
    scale_y = orig_h / saliency.shape[0]

    blobs = []
    for label_id in range(1, n_labels + 1):
        component = labeled == label_id
        total_sal = float(saliency[component].sum())

        rows = np.any(component, axis=1)
        cols = np.any(component, axis=0)
        y1, y2 = np.where(rows)[0][[0, -1]]
        x1, x2 = np.where(cols)[0][[0, -1]]

        blobs.append((
            int(x1 * scale_x),
            int(y1 * scale_y),
            int((x2 + 1) * scale_x),
            int((y2 + 1) * scale_y),
            total_sal,
        ))

    blobs.sort(key=lambda b: -b[4])
    return blobs[:max_blobs]


def draw_rect(draw, bbox, color, line_width):
    """Draw a thick rectangle."""
    x1, y1, x2, y2 = bbox[:4]
    for offset in range(line_width):
        draw.rectangle(
            [x1 - offset, y1 - offset, x2 + offset, y2 + offset],
            outline=color,
        )


def generate_html(image_data: list, output_dir: Path):
    items_json = json.dumps(image_data)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>MicroSalNet + UltraFace — DUTS-TE</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ background: #1a1a1a; color: #e0e0e0; font-family: system-ui, -apple-system, sans-serif; }}
.header {{ padding: 12px 24px; background: #222; border-bottom: 1px solid #333; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px; }}
.header h1 {{ font-size: 16px; font-weight: 500; }}
.counter {{ font-size: 14px; color: #888; font-variant-numeric: tabular-nums; }}
.viewer {{ display: flex; justify-content: center; align-items: center; min-height: calc(100vh - 130px); padding: 16px; }}
.viewer img {{ max-width: 100%; max-height: calc(100vh - 170px); object-fit: contain; }}
.info {{ padding: 8px 24px; background: #222; border-top: 1px solid #333; display: flex; justify-content: space-between; align-items: center; font-size: 13px; flex-wrap: wrap; gap: 4px; }}
.info .name {{ color: #ccc; }}
.info .stats {{ color: #888; }}
.nav {{ display: flex; gap: 8px; align-items: center; }}
.nav button {{ background: #333; color: #ccc; border: 1px solid #444; padding: 6px 16px; cursor: pointer; border-radius: 4px; font-size: 13px; }}
.nav button:hover {{ background: #444; }}
.nav button:disabled {{ opacity: 0.3; cursor: default; }}
.legend {{ font-size: 12px; padding: 4px 24px; color: #777; }}
.legend span {{ margin-right: 16px; }}
.legend .red {{ color: #ff4444; }}
.legend .green {{ color: #44ff44; }}
.legend .cyan {{ color: #44ffff; }}
</style>
</head>
<body>
<div class="header">
    <h1>MicroSalNet + UltraFace — DUTS-TE</h1>
    <div class="nav">
        <button id="prev" onclick="go(-1)">&larr; Prev</button>
        <span class="counter" id="counter">1 / 1</span>
        <button id="next" onclick="go(1)">Next &rarr;</button>
    </div>
</div>
<div class="legend">
    <span class="red">&#9632; Primary salient region</span>
    <span class="green">&#9632; Secondary salient region</span>
    <span class="cyan">&#9632; Detected faces</span>
</div>
<div class="viewer">
    <img id="img" src="" alt="">
</div>
<div class="info">
    <span class="name" id="name"></span>
    <span class="stats" id="stats"></span>
</div>
<script>
const items = {items_json};
let idx = 0;

function show(i) {{
    idx = Math.max(0, Math.min(items.length - 1, i));
    const item = items[idx];
    document.getElementById('img').src = item.file;
    document.getElementById('counter').textContent = (idx + 1) + ' / ' + items.length;
    document.getElementById('name').textContent = item.name;
    let s = item.size;
    if (item.mae > 0) s += '  |  MAE: ' + item.mae.toFixed(4);
    s += '  |  Blobs: ' + item.n_blobs + '  |  Faces: ' + item.n_faces;
    document.getElementById('stats').textContent = s;
    document.getElementById('prev').disabled = idx === 0;
    document.getElementById('next').disabled = idx === items.length - 1;
}}

function go(delta) {{ show(idx + delta); }}

document.addEventListener('keydown', e => {{
    if (e.key === 'ArrowLeft') go(-1);
    else if (e.key === 'ArrowRight') go(1);
    else if (e.key === 'Home') show(0);
    else if (e.key === 'End') show(items.length - 1);
}});

show(0);
</script>
</body>
</html>"""

    html_path = output_dir / "index.html"
    html_path.write_text(html_content)
    return html_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--threshold", type=float, default=0.4,
                        help="Saliency threshold as fraction of max (for blob detection)")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    img_dir = OUTPUT_DIR / "images"
    img_dir.mkdir(exist_ok=True)

    ensure_face_model()

    # Load models
    sal_sess = ort.InferenceSession(str(SALIENCY_ONNX), providers=["CPUExecutionProvider"])
    sal_input = sal_sess.get_inputs()[0].name
    print(f"Saliency model: {SALIENCY_ONNX.name}")

    face_sess = ort.InferenceSession(str(FACE_ONNX), providers=["CPUExecutionProvider"])
    print(f"Face model: ultraface-rfb-320")

    image_files = sorted(DUTS_TE_IMG.glob("*.jpg"))[:args.limit]
    print(f"Processing {len(image_files)} images (threshold={args.threshold})")

    image_data = []
    total_mae = 0.0
    count = 0

    for i, img_path in enumerate(image_files):
        stem = img_path.stem
        img = Image.open(img_path).convert("RGB")
        orig_w, orig_h = img.size

        # Run saliency
        sal_inp = preprocess_saliency(img)
        result = sal_sess.run(None, {sal_input: sal_inp})[0]
        saliency = np.clip(result[0, 0], 0, 1)

        # Find salient blobs
        blobs = find_salient_blobs(saliency, args.threshold, orig_w, orig_h)

        # Run face detection
        faces = detect_faces(face_sess, img)

        # Compute MAE
        mae = 0.0
        gt_path = DUTS_TE_MASK / f"{stem}.png"
        if gt_path.exists():
            gt = Image.open(gt_path).convert("L").resize(
                (SAL_OUTPUT, SAL_OUTPUT), Image.BILINEAR
            )
            gt_arr = np.array(gt, dtype=np.float32) / 255.0
            mae = float(np.abs(saliency - gt_arr).mean())
            total_mae += mae
            count += 1

        # Draw annotations
        annotated = img.copy()
        draw = ImageDraw.Draw(annotated)
        lw = max(2, min(orig_w, orig_h) // 150)

        # Red: primary salient region
        if len(blobs) >= 1:
            draw_rect(draw, blobs[0], "red", lw)

        # Green: secondary salient region
        if len(blobs) >= 2:
            draw_rect(draw, blobs[1], "#00ff00", lw)

        # Cyan: faces
        for face in faces:
            draw_rect(draw, face, "cyan", lw)

        out_name = f"{stem}.jpg"
        annotated.save(img_dir / out_name, quality=85)

        image_data.append({
            "file": f"images/{out_name}",
            "name": stem,
            "mae": mae,
            "size": f"{orig_w}x{orig_h}",
            "n_blobs": len(blobs),
            "n_faces": len(faces),
        })

        if (i + 1) % 50 == 0 or i + 1 == len(image_files):
            avg_mae = total_mae / count if count > 0 else 0
            print(f"  [{i+1}/{len(image_files)}] MAE={avg_mae:.4f}  faces_avg={sum(d['n_faces'] for d in image_data)/len(image_data):.1f}")

    # Sort by MAE descending
    image_data_sorted = sorted(image_data, key=lambda x: -x["mae"])

    html_path = generate_html(image_data_sorted, OUTPUT_DIR)
    print(f"\nDone. {len(image_data)} images.")
    if count > 0:
        print(f"Average MAE: {total_mae / count:.4f}")
    print(f"HTML viewer: {html_path}")


if __name__ == "__main__":
    main()
