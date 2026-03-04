"""Train MicroSalNet with knowledge distillation from U2-Netp teacher.

Loss = alpha * BCE(pred, gt) + (1-alpha) * MSE(pred, teacher)

Usage:
    python3 training/train.py [--width 16] [--epochs 80] [--batch 32] [--lr 1e-3]
"""

import argparse
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image

from model import MicroSalNet


class DUTSDistillDataset(Dataset):
    """DUTS-TR with ground truth masks + teacher predictions."""

    def __init__(self, image_dir, mask_dir, teacher_dir, input_size=256, output_size=None):
        self.input_size = input_size
        self.output_size = output_size or input_size
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.teacher_dir = Path(teacher_dir)

        # Find all images with both mask and teacher
        self.samples = []
        for img_path in sorted(self.image_dir.glob("*.jpg")):
            stem = img_path.stem
            mask_path = self.mask_dir / f"{stem}.png"
            teacher_path = self.teacher_dir / f"{stem}.npy"
            if mask_path.exists() and teacher_path.exists():
                self.samples.append((img_path, mask_path, teacher_path))

        print(f"Dataset: {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path, teacher_path = self.samples[idx]

        # Load and preprocess image
        img = Image.open(img_path).convert("RGB")
        img = img.resize((self.input_size, self.input_size), Image.BILINEAR)
        img = np.array(img, dtype=np.float32) / 255.0  # [H, W, 3] in [0, 1]
        img = img.transpose(2, 0, 1)  # [3, H, W]

        # Load ground truth mask (at output_size resolution)
        mask = Image.open(mask_path).convert("L")
        mask = mask.resize((self.output_size, self.output_size), Image.BILINEAR)
        mask = np.array(mask, dtype=np.float32) / 255.0  # [H, W] in [0, 1]

        # Load teacher prediction (320x320 float16, resize to output_size)
        teacher = np.load(teacher_path).astype(np.float32)  # [320, 320]
        if teacher.shape[0] != self.output_size:
            teacher_img = Image.fromarray((teacher * 255).astype(np.uint8))
            teacher_img = teacher_img.resize(
                (self.output_size, self.output_size), Image.BILINEAR
            )
            teacher = np.array(teacher_img, dtype=np.float32) / 255.0

        # Random horizontal flip (data augmentation)
        if np.random.random() > 0.5:
            img = img[:, :, ::-1].copy()
            mask = mask[:, ::-1].copy()
            teacher = teacher[:, ::-1].copy()

        return (
            torch.from_numpy(img),
            torch.from_numpy(mask).unsqueeze(0),  # [1, H, W]
            torch.from_numpy(teacher).unsqueeze(0),  # [1, H, W]
        )


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    _repo_root = Path(__file__).resolve().parent.parent
    data_root = Path(os.environ.get(
        "DUTS_TR_DIR",
        str(_repo_root / "data" / "DUTS-TR"),
    ))
    output_dir = Path(os.environ.get(
        "ZENFACES_CKPT_DIR",
        str(_repo_root / "training" / "checkpoints"),
    ))
    output_dir.mkdir(parents=True, exist_ok=True)

    output_size = args.input_size // 2  # model outputs at half resolution

    # Dataset
    dataset = DUTSDistillDataset(
        image_dir=data_root / "DUTS-TR-Image",
        mask_dir=data_root / "DUTS-TR-Mask",
        teacher_dir=data_root / "DUTS-TR-Teacher",
        input_size=args.input_size,
        output_size=output_size,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    # Validation set: use last 500 images for validation
    val_dataset = DUTSDistillDataset(
        image_dir=data_root / "DUTS-TR-Image",
        mask_dir=data_root / "DUTS-TR-Mask",
        teacher_dir=data_root / "DUTS-TR-Teacher",
        input_size=args.input_size,
        output_size=output_size,
    )
    # Split: use first 10053 for train, last 500 for val
    train_indices = list(range(len(dataset) - 500))
    val_indices = list(range(len(dataset) - 500, len(dataset)))

    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset_sub = torch.utils.data.Subset(val_dataset, val_indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset_sub,
        batch_size=args.batch,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    # Model
    model = MicroSalNet(width=args.width).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"Model: MicroSalNet(width={args.width}), {params:,} params")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Loss
    bce_loss = nn.BCELoss()
    mse_loss = nn.MSELoss()
    alpha = args.alpha  # weight for GT loss vs teacher loss

    best_val_mae = float("inf")
    tag = f"microsalnet_w{args.width}_s{args.input_size}"

    print(f"\nTraining: {args.epochs} epochs, batch={args.batch}, lr={args.lr}")
    print(f"Loss: {alpha:.1f}*BCE(gt) + {1-alpha:.1f}*MSE(teacher)\n")

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        train_mae = 0.0
        n_batches = 0

        t0 = time.time()
        for imgs, masks, teachers in train_loader:
            imgs = imgs.to(device)
            masks = masks.to(device)
            teachers = teachers.to(device)

            pred = model(imgs)  # [B, 1, H, W]

            loss_gt = bce_loss(pred, masks)
            loss_teacher = mse_loss(pred, teachers)
            loss = alpha * loss_gt + (1 - alpha) * loss_teacher

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                mae = (pred - masks).abs().mean().item()
                train_loss += loss.item()
                train_mae += mae
                n_batches += 1

        scheduler.step()
        train_loss /= n_batches
        train_mae /= n_batches
        epoch_time = time.time() - t0

        # Validation
        model.eval()
        val_mae = 0.0
        val_n = 0
        with torch.no_grad():
            for imgs, masks, _ in val_loader:
                imgs = imgs.to(device)
                masks = masks.to(device)
                pred = model(imgs)
                val_mae += (pred - masks).abs().mean().item() * imgs.size(0)
                val_n += imgs.size(0)
        val_mae /= val_n

        lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch+1:3d}/{args.epochs}  "
            f"loss={train_loss:.4f}  MAE={train_mae:.4f}  "
            f"val_MAE={val_mae:.4f}  lr={lr:.6f}  {epoch_time:.1f}s"
        )

        # Save best
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save(model.state_dict(), output_dir / f"{tag}_best.pth")
            print(f"  -> New best val MAE: {val_mae:.4f}")

        # Save periodic checkpoints
        if (epoch + 1) % 20 == 0:
            torch.save(model.state_dict(), output_dir / f"{tag}_epoch{epoch+1}.pth")

    # Save final
    torch.save(model.state_dict(), output_dir / f"{tag}_final.pth")
    print(f"\nBest val MAE: {best_val_mae:.4f}")
    print(f"Checkpoints saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--input-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Weight for GT loss (1-alpha for teacher)")
    args = parser.parse_args()
    train(args)
