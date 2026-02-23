"""Generate U2-Netp teacher predictions on DUTS-TR for knowledge distillation.

Uses PyTorch + CUDA for GPU-accelerated inference of the ONNX model.
"""

import os
import numpy as np
import torch
import onnxruntime as ort
from PIL import Image
from pathlib import Path
import time

ONNX_PATH = "/tmp/u2netp.onnx"
DUTS_TR_IMG = Path("/home/lilith/work/zenfaces/data/DUTS-TR/DUTS-TR-Image")
OUTPUT_DIR = Path("/home/lilith/work/zenfaces/data/DUTS-TR/DUTS-TR-Teacher")

# ImageNet normalization (matching u2netp.rs preprocessing)
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)

INPUT_SIZE = 320
BATCH_SIZE = 16  # batch for GPU efficiency


def preprocess(img: Image.Image) -> np.ndarray:
    """Resize to 320x320, normalize, NCHW."""
    img = img.convert("RGB").resize((INPUT_SIZE, INPUT_SIZE), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)  # [3, H, W]
    arr = (arr - MEAN[0]) / STD[0]
    return arr


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Try CUDA provider first, fall back to CPU
    providers = []
    try:
        available = ort.get_available_providers()
        if "CUDAExecutionProvider" in available:
            providers.append("CUDAExecutionProvider")
    except Exception:
        pass
    providers.append("CPUExecutionProvider")

    sess = ort.InferenceSession(ONNX_PATH, providers=providers)
    active_provider = sess.get_providers()[0]
    input_name = sess.get_inputs()[0].name
    print(f"Provider: {active_provider}")

    image_files = sorted(DUTS_TR_IMG.glob("*.jpg"))
    print(f"Images: {len(image_files)}")

    # Filter out already-done images
    todo = []
    for p in image_files:
        out_path = OUTPUT_DIR / f"{p.stem}.npy"
        if not out_path.exists():
            todo.append(p)
    print(f"Remaining: {len(todo)}")

    if not todo:
        print("All teacher predictions already generated.")
        return

    t0 = time.time()
    for i in range(0, len(todo), BATCH_SIZE):
        batch_paths = todo[i : i + BATCH_SIZE]
        batch_inputs = []

        for img_path in batch_paths:
            img = Image.open(img_path)
            batch_inputs.append(preprocess(img))

        batch = np.stack(batch_inputs, axis=0)  # [B, 3, 320, 320]

        # ORT doesn't support variable batch on all models — run one at a time
        for j, inp in enumerate(batch_inputs):
            inp_4d = inp[np.newaxis]  # [1, 3, 320, 320]
            result = sess.run(None, {input_name: inp_4d})[0]  # [1, 1, 320, 320]
            saliency = result[0, 0]  # [320, 320]

            # Min-max normalize (matching u2netp.rs postprocessing)
            mn, mx = saliency.min(), saliency.max()
            if mx - mn > 1e-6:
                saliency = (saliency - mn) / (mx - mn)
            else:
                saliency = np.zeros_like(saliency)

            out_path = OUTPUT_DIR / f"{batch_paths[j].stem}.npy"
            np.save(out_path, saliency.astype(np.float16))

        done = min(i + BATCH_SIZE, len(todo))
        if done % 500 < BATCH_SIZE or done == len(todo):
            elapsed = time.time() - t0
            rate = done / elapsed
            eta = (len(todo) - done) / rate if rate > 0 else 0
            print(f"  [{done}/{len(todo)}] {rate:.1f} img/s  ETA {eta:.0f}s")

    elapsed = time.time() - t0
    print(f"Done: {len(todo)} images in {elapsed:.1f}s ({len(todo) / elapsed:.1f} img/s)")


if __name__ == "__main__":
    main()
