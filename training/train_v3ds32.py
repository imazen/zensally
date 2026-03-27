"""Train v3ds with 32×32 bottleneck for 50ms latency budget.

Usage:
    python3 training/train_v3ds32.py --width 40 --n-dilated 4 --epochs 150
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

from model_v3ds import DilatedBlock, InvertedResidual, SEBlock, count_params
from train_v3ds import DUTSDatasetV3, structure_loss


class MicroSalNetV3ds32(nn.Module):
    """Deep supervision + 32×32 bottleneck."""

    def __init__(self, width=32, n_dilated=6):
        super().__init__()
        w = width
        bw = w * 2

        self.stem = nn.Sequential(
            nn.Conv2d(3, w, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(w), nn.ReLU(inplace=True))
        self.enc1 = nn.Sequential(
            InvertedResidual(w, w, stride=2, expand_ratio=2),
            InvertedResidual(w, w, stride=1, expand_ratio=2))
        self.enc2 = nn.Sequential(
            InvertedResidual(w, bw, stride=2, expand_ratio=2, use_se=True),
            InvertedResidual(bw, bw, stride=1, expand_ratio=2, use_se=True))

        dilations = [1, 2, 4, 8, 1, 2, 4, 8][:n_dilated]
        self.bottleneck = nn.Sequential(
            *[DilatedBlock(bw, d, expand_ratio=2) for d in dilations])

        self.up2 = nn.ConvTranspose2d(bw, w, 2, stride=2, bias=False)
        self.dec2 = nn.Sequential(nn.BatchNorm2d(w), nn.ReLU(inplace=True))
        self.up1 = nn.ConvTranspose2d(w + w, w, 2, stride=2, bias=False)
        self.dec1 = nn.Sequential(nn.BatchNorm2d(w), nn.ReLU(inplace=True))
        self.head = nn.Conv2d(w + w, 1, 1)

        self.aux2 = nn.Conv2d(w, 1, 1)
        self.aux1 = nn.Conv2d(w, 1, 1)

    def forward(self, x):
        s0 = self.stem(x)
        s1 = self.enc1(s0)
        s2 = self.enc2(s1)
        b = self.bottleneck(s2)
        d2 = self.dec2(self.up2(b))
        d1 = self.dec1(self.up1(torch.cat([d2, s1], dim=1)))
        final = torch.sigmoid(self.head(torch.cat([d1, s0], dim=1)))
        if self.training:
            a2 = torch.sigmoid(self.aux2(d2))
            a1 = torch.sigmoid(self.aux1(d1))
            return final, [a2, a1]
        return final

    def export_mode(self):
        del self.aux2, self.aux1
        self.eval()
        return self


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    repo_root = Path(__file__).resolve().parent.parent
    data_root = Path(os.environ.get(
        "DUTS_TR_DIR", str(repo_root / "data" / "DUTS-TR")))
    output_dir = Path(os.environ.get(
        "ZENFACES_CKPT_DIR", str(repo_root / "training" / "checkpoints")))
    output_dir.mkdir(parents=True, exist_ok=True)

    output_size = args.input_size // 2
    teacher_dir = None if args.no_teacher else (data_root / "DUTS-TR-Teacher")

    dataset = DUTSDatasetV3(
        image_dir=data_root / "DUTS-TR-Image",
        mask_dir=data_root / "DUTS-TR-Mask",
        teacher_dir=teacher_dir,
        input_size=args.input_size, output_size=output_size, augment=True)

    n_val = 500
    train_loader = DataLoader(
        torch.utils.data.Subset(dataset, list(range(len(dataset) - n_val))),
        batch_size=args.batch, shuffle=True, num_workers=4,
        pin_memory=True, drop_last=True)

    val_dataset = DUTSDatasetV3(
        image_dir=data_root / "DUTS-TR-Image",
        mask_dir=data_root / "DUTS-TR-Mask",
        teacher_dir=teacher_dir,
        input_size=args.input_size, output_size=output_size, augment=False)
    val_loader = DataLoader(
        torch.utils.data.Subset(val_dataset, list(range(len(dataset) - n_val, len(dataset)))),
        batch_size=args.batch, shuffle=False, num_workers=2, pin_memory=True)

    model = MicroSalNetV3ds32(width=args.width, n_dilated=args.n_dilated).to(device)
    params = count_params(model)
    print(f"Model: V3ds32(w={args.width}, d={args.n_dilated}), {params:,} params")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    mse_loss = nn.MSELoss()
    best_val_mae = float("inf")
    tag = f"microsalnet_v3ds32_w{args.width}_d{args.n_dilated}_s{args.input_size}"

    print(f"\nTraining: {args.epochs} epochs, 32×32 bottleneck")

    for epoch in range(args.epochs):
        model.train()
        train_loss = train_mae = 0.0
        n_batches = 0
        t0 = time.time()
        current_aux = args.aux_weight * min(1.0, (epoch + 1) / 20.0)

        for imgs, masks, teachers in train_loader:
            imgs, masks, teachers = imgs.to(device), masks.to(device), teachers.to(device)
            final, auxes = model(imgs)
            loss = structure_loss(final, masks)
            loss_aux = torch.tensor(0.0, device=device)
            for aux_pred in auxes:
                ah, aw = aux_pred.shape[2], aux_pred.shape[3]
                gt_r = F.interpolate(masks, size=(ah, aw), mode='bilinear', align_corners=False) if ah != masks.shape[2] else masks
                loss_aux = loss_aux + structure_loss(aux_pred, gt_r)
            loss = loss + current_aux * loss_aux
            if not args.no_teacher:
                loss = args.alpha * loss + (1 - args.alpha) * mse_loss(final, teachers)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            with torch.no_grad():
                train_loss += loss.item(); train_mae += (final - masks).abs().mean().item(); n_batches += 1

        scheduler.step()
        model.eval()
        val_mae = val_n = 0.0
        with torch.no_grad():
            for imgs, masks, _ in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                val_mae += (model(imgs) - masks).abs().mean().item() * imgs.size(0)
                val_n += imgs.size(0)
        val_mae /= val_n
        print(f"Epoch {epoch+1:3d}/{args.epochs}  loss={train_loss/n_batches:.4f}  MAE={train_mae/n_batches:.4f}  val_MAE={val_mae:.4f}  {time.time()-t0:.1f}s")
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save(model.state_dict(), output_dir / f"{tag}_best.pth")
            print(f"  -> New best val MAE: {val_mae:.4f}")
        if (epoch + 1) % 20 == 0:
            torch.save(model.state_dict(), output_dir / f"{tag}_epoch{epoch+1}.pth")

    torch.save(model.state_dict(), output_dir / f"{tag}_final.pth")
    print(f"\nBest val MAE: {best_val_mae:.4f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--width", type=int, default=32)
    p.add_argument("--n-dilated", type=int, default=6)
    p.add_argument("--input-size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--alpha", type=float, default=0.7)
    p.add_argument("--aux-weight", type=float, default=0.4)
    p.add_argument("--no-teacher", action="store_true")
    train(p.parse_args())
