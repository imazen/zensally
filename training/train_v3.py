"""Train MicroSalNet v3d — 16×16 bottleneck with dilated blocks.

Usage:
    python3 training/train_v3.py --width 24 --epochs 120
    python3 training/train_v3.py --width 20 --epochs 120
"""

import argparse
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
from PIL import Image

# Import v3d model and v2's dataset/loss
from model_v3 import DilatedBlock, InvertedResidual, SEBlock, count_params
from train_v2 import DUTSDataset, structure_loss


class MicroSalNetV3d(nn.Module):
    """16×16 bottleneck with dilated blocks."""

    def __init__(self, width=24):
        super().__init__()
        w = width
        self.stem = nn.Sequential(
            nn.Conv2d(3, w, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(w), nn.ReLU(inplace=True),
        )
        self.enc1 = nn.Sequential(
            InvertedResidual(w, w, stride=2, expand_ratio=2),
            InvertedResidual(w, w, stride=1, expand_ratio=2),
        )
        self.enc2 = nn.Sequential(
            InvertedResidual(w, w * 2, stride=2, expand_ratio=2),
            InvertedResidual(w * 2, w * 2, stride=1, expand_ratio=2),
        )
        self.enc3 = nn.Sequential(
            InvertedResidual(w * 2, w * 3, stride=2, expand_ratio=2, use_se=True),
            InvertedResidual(w * 3, w * 3, stride=1, expand_ratio=2, use_se=True),
        )

        self.bottleneck = nn.Sequential(
            DilatedBlock(w * 3, dilation=1, expand_ratio=2),
            DilatedBlock(w * 3, dilation=2, expand_ratio=2),
            DilatedBlock(w * 3, dilation=4, expand_ratio=2),
            DilatedBlock(w * 3, dilation=8, expand_ratio=2),
        )

        self.up3 = nn.ConvTranspose2d(w * 3, w * 2, 2, stride=2, bias=False)
        self.dec3 = nn.Sequential(nn.BatchNorm2d(w * 2), nn.ReLU(inplace=True))

        self.up2 = nn.ConvTranspose2d(w * 2 + w * 2, w, 2, stride=2, bias=False)
        self.dec2 = nn.Sequential(nn.BatchNorm2d(w), nn.ReLU(inplace=True))

        self.up1 = nn.ConvTranspose2d(w + w, w, 2, stride=2, bias=False)
        self.dec1 = nn.Sequential(nn.BatchNorm2d(w), nn.ReLU(inplace=True))

        self.head = nn.Conv2d(w + w, 1, 1)

    def forward(self, x):
        s0 = self.stem(x)
        s1 = self.enc1(s0)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)
        b = self.bottleneck(s3)
        d3 = self.dec3(self.up3(b))
        d2 = self.dec2(self.up2(torch.cat([d3, s2], dim=1)))
        d1 = self.dec1(self.up1(torch.cat([d2, s1], dim=1)))
        return torch.sigmoid(self.head(torch.cat([d1, s0], dim=1)))


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    repo_root = Path(__file__).resolve().parent.parent
    data_root = Path(os.environ.get(
        "DUTS_TR_DIR", str(repo_root / "data" / "DUTS-TR"),
    ))
    output_dir = Path(os.environ.get(
        "ZENFACES_CKPT_DIR", str(repo_root / "training" / "checkpoints"),
    ))
    output_dir.mkdir(parents=True, exist_ok=True)

    output_size = args.input_size // 2
    teacher_dir = None if args.no_teacher else (data_root / "DUTS-TR-Teacher")

    dataset = DUTSDataset(
        image_dir=data_root / "DUTS-TR-Image",
        mask_dir=data_root / "DUTS-TR-Mask",
        teacher_dir=teacher_dir,
        input_size=args.input_size,
        output_size=output_size,
        augment=True,
    )

    n_val = 500
    train_indices = list(range(len(dataset) - n_val))
    val_indices = list(range(len(dataset) - n_val, len(dataset)))

    train_loader = DataLoader(
        torch.utils.data.Subset(dataset, train_indices),
        batch_size=args.batch, shuffle=True, num_workers=4,
        pin_memory=True, drop_last=True,
    )

    val_dataset = DUTSDataset(
        image_dir=data_root / "DUTS-TR-Image",
        mask_dir=data_root / "DUTS-TR-Mask",
        teacher_dir=teacher_dir,
        input_size=args.input_size,
        output_size=output_size,
        augment=False,
    )
    val_loader = DataLoader(
        torch.utils.data.Subset(val_dataset, val_indices),
        batch_size=args.batch, shuffle=False, num_workers=2,
        pin_memory=True,
    )

    model = MicroSalNetV3d(width=args.width).to(device)
    params = count_params(model)
    print(f"Model: MicroSalNetV3d(width={args.width}), {params:,} params")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    mse_loss = nn.MSELoss()
    best_val_mae = float("inf")
    tag = f"microsalnet_v3d_w{args.width}_s{args.input_size}"

    print(f"\nTraining: {args.epochs} epochs, batch={args.batch}, lr={args.lr}")
    print(f"Loss: structure_loss + {1 - args.alpha:.1f}*MSE(teacher)\n")

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

            pred = model(imgs)

            loss = structure_loss(pred, masks)
            if not args.no_teacher:
                loss = args.alpha * loss + (1 - args.alpha) * mse_loss(pred, teachers)

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
            f"Epoch {epoch + 1:3d}/{args.epochs}  "
            f"loss={train_loss:.4f}  MAE={train_mae:.4f}  "
            f"val_MAE={val_mae:.4f}  lr={lr:.6f}  {epoch_time:.1f}s"
        )

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save(model.state_dict(), output_dir / f"{tag}_best.pth")
            print(f"  -> New best val MAE: {val_mae:.4f}")

        if (epoch + 1) % 20 == 0:
            torch.save(model.state_dict(), output_dir / f"{tag}_epoch{epoch + 1}.pth")

    torch.save(model.state_dict(), output_dir / f"{tag}_final.pth")
    print(f"\nBest val MAE: {best_val_mae:.4f}")
    print(f"Checkpoints saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=int, default=24)
    parser.add_argument("--input-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--no-teacher", action="store_true")
    args = parser.parse_args()
    train(args)
