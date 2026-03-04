"""Export trained MicroSalNet to ONNX, patch for tract compatibility.

Usage:
    python3 training/export_onnx.py [--width 16] [--input-size 256] [--checkpoint best]
"""

import argparse
import os
import numpy as np
import torch
import onnx
from onnx import helper, numpy_helper
import onnx.shape_inference
from pathlib import Path

from model import MicroSalNet


def export(args):
    tag = f"microsalnet_w{args.width}_s{args.input_size}"
    ckpt_dir = Path(os.environ.get(
        "ZENFACES_CKPT_DIR",
        str(Path(__file__).resolve().parent / "checkpoints"),
    ))
    ckpt_path = ckpt_dir / f"{tag}_{args.checkpoint}.pth"

    if not ckpt_path.exists():
        print(f"Checkpoint not found: {ckpt_path}")
        return

    print(f"Loading {ckpt_path}")
    model = MicroSalNet(width=args.width)
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=True))
    model.eval()

    params = sum(p.numel() for p in model.parameters())
    print(f"Model: {params:,} params")

    # Verify output shape
    output_size = args.input_size // 2
    dummy = torch.randn(1, 3, args.input_size, args.input_size)
    with torch.no_grad():
        test_out = model(dummy)
    print(f"Output shape: {test_out.shape} (expected [1, 1, {output_size}, {output_size}])")

    # Export to ONNX
    onnx_path = ckpt_dir / f"{tag}.onnx"

    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        input_names=["input"],
        output_names=["output"],
        opset_version=17,
        dynamic_axes=None,  # fixed batch size = 1
    )
    print(f"Exported: {onnx_path} ({onnx_path.stat().st_size / 1024:.0f} KB)")

    # Simplify with onnxsim
    try:
        from onnxsim import simplify
        m = onnx.load(str(onnx_path))
        m_sim, ok = simplify(m)
        if ok:
            onnx.save(m_sim, str(onnx_path))
            print(f"Simplified: {onnx_path.stat().st_size / 1024:.0f} KB")
        else:
            print("Warning: onnxsim simplification failed, using unsimplified model")
    except ImportError:
        print("onnxsim not available, skipping simplification")

    # Patch Resize ops for tract compatibility:
    # 1. Convert sizes-based Resize to scales-based
    # 2. Fix coordinate_transformation_mode if needed
    m = onnx.load(str(onnx_path))
    m = onnx.shape_inference.infer_shapes(m)

    shape_map = {}
    for vi in list(m.graph.value_info) + list(m.graph.input) + list(m.graph.output):
        dims = [d.dim_value for d in vi.type.tensor_type.shape.dim]
        if all(d > 0 for d in dims):
            shape_map[vi.name] = dims

    patched = 0
    for node in m.graph.node:
        if node.op_type != "Resize":
            continue

        # Fix coordinate_transformation_mode
        for attr in node.attribute:
            if attr.name == "coordinate_transformation_mode":
                if attr.s == b"pytorch_half_pixel":
                    attr.s = b"half_pixel"
                    patched += 1

        # Convert sizes-based to scales-based
        inp_name = node.input[0]
        inp_shape = shape_map.get(inp_name)
        out_name = node.output[0]
        out_shape = shape_map.get(out_name)

        if inp_shape and out_shape and len(node.input) > 3 and node.input[3]:
            scales = [float(o) / float(i) for i, o in zip(inp_shape, out_shape)]
            scales_name = f"{node.name}_scales"
            scales_tensor = numpy_helper.from_array(
                np.array(scales, dtype=np.float32), name=scales_name
            )
            m.graph.initializer.append(scales_tensor)
            node.input[2] = scales_name
            node.input[3] = ""
            patched += 1

    print(f"Patched {patched} ops for tract compatibility")

    # Save patched model
    patched_path = ckpt_dir / f"{tag}_tract.onnx"
    onnx.save(m, str(patched_path))
    print(f"Tract-compatible: {patched_path} ({patched_path.stat().st_size / 1024:.0f} KB)")

    # Verify with ORT
    import onnxruntime as ort
    sess_orig = ort.InferenceSession(str(onnx_path))
    sess_patched = ort.InferenceSession(str(patched_path))

    test_input = np.random.randn(1, 3, args.input_size, args.input_size).astype(np.float32)
    out_orig = sess_orig.run(None, {"input": test_input})[0]
    out_patched = sess_patched.run(None, {"input": test_input})[0]
    max_diff = np.abs(out_orig - out_patched).max()
    print(f"Patch verification: max diff = {max_diff:.8f}")

    # Gzip compress for embedding
    import gzip
    gz_path = ckpt_dir / f"{tag}_tract.onnx.gz"
    with open(patched_path, "rb") as f_in:
        data = f_in.read()
    with gzip.open(gz_path, "wb", compresslevel=9) as f_out:
        f_out.write(data)
    print(f"Compressed: {gz_path} ({gz_path.stat().st_size / 1024:.0f} KB)")

    # Count ops
    from collections import Counter
    ops = Counter(n.op_type for n in m.graph.node)
    print(f"\nONNX ops ({len(m.graph.node)} total):")
    for op, cnt in ops.most_common():
        print(f"  {op}: {cnt}")

    # Test in tract CLI
    print(f"\nTo test in tract:")
    print(f"  tract {patched_path} -i 1,3,{args.input_size},{args.input_size},f32 -O dump --profile --cost -R")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--input-size", type=int, default=256)
    parser.add_argument("--checkpoint", type=str, default="best")
    args = parser.parse_args()
    export(args)
