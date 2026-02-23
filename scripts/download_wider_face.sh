#!/usr/bin/env bash
# Download WIDER FACE validation dataset for accuracy benchmarking.
# Images: ~363MB, Annotations: ~5MB
# License: Non-commercial research only. Do NOT commit to the repo.
set -euo pipefail

DATA_DIR="${1:-data/wider_face}"
mkdir -p "$DATA_DIR"

# Download annotations
if [ ! -f "$DATA_DIR/wider_face_split.zip" ]; then
    echo "Downloading annotations..."
    wget -q --show-progress \
        "http://shuoyang1213.me/WIDERFACE/support/bbx_annotation/wider_face_split.zip" \
        -O "$DATA_DIR/wider_face_split.zip"
fi

if [ ! -f "$DATA_DIR/wider_face_val_bbx_gt.txt" ]; then
    echo "Extracting annotations..."
    unzip -o "$DATA_DIR/wider_face_split.zip" -d "$DATA_DIR"
    # The zip contains wider_face_split/ subdirectory
    mv "$DATA_DIR/wider_face_split/"* "$DATA_DIR/" 2>/dev/null || true
    rmdir "$DATA_DIR/wider_face_split" 2>/dev/null || true
fi

# Download validation images
if [ ! -d "$DATA_DIR/WIDER_val" ]; then
    if [ ! -f "$DATA_DIR/WIDER_val.zip" ]; then
        echo "Downloading validation images (~363MB)..."
        # Try huggingface-cli first, then gdown, then direct
        if command -v huggingface-cli &>/dev/null; then
            huggingface-cli download CUHK-CSE/wider_face \
                --repo-type dataset \
                --include "data/WIDER_val.zip" \
                --local-dir "$DATA_DIR/hf_tmp"
            mv "$DATA_DIR/hf_tmp/data/WIDER_val.zip" "$DATA_DIR/WIDER_val.zip"
            rm -rf "$DATA_DIR/hf_tmp"
        elif command -v gdown &>/dev/null; then
            gdown 1GUCogbp16PMGa39thoMMeWxp7Rp5oM8Q -O "$DATA_DIR/WIDER_val.zip"
        else
            echo "ERROR: Need either 'huggingface-cli' or 'gdown' to download images."
            echo "  pip install huggingface-hub   # or"
            echo "  pip install gdown"
            exit 1
        fi
    fi

    echo "Extracting validation images..."
    unzip -o "$DATA_DIR/WIDER_val.zip" -d "$DATA_DIR"
fi

echo "WIDER FACE validation set ready in $DATA_DIR/"
echo "  Annotations: $DATA_DIR/wider_face_val_bbx_gt.txt"
echo "  Images:      $DATA_DIR/WIDER_val/images/"
ls "$DATA_DIR/WIDER_val/images/" | wc -l | xargs -I{} echo "  Event dirs:  {}"
