"""Train MicroSalNet v2 — improved loss for better precision.

Changes from v1:
- BCE + IoU loss (instead of pure BCE + teacher MSE)
- Higher alpha on ground truth (less teacher dependence)
- Optional: skip teacher entirely (--no-teacher)
- More augmentation: random scale, color jitter

Usage:
    python3 training/train_v2.py --width 24 --epochs 120 --batch 32
    python3 training/train_v2.py --width 24 --epochs 120 --no-teacher
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


def iou_loss(pred, target, smooth=1.0):
    """Differentiable IoU loss — penalizes false positives and false negatives."""
    pred_flat = pred.view(pred.size(0), -1)
    target_flat = target.view(target.size(0), -1)
    intersection = (pred_flat * target_flat).sum(dim=1)
    union = pred_flat.sum(dim=1) + target_flat.sum(dim=1) - intersection
    iou = (intersection + smooth) / (union + smooth)
    return 1.0 - iou.mean()


def structure_loss(pred, mask):
    """Weighted BCE + weighted IoU — standard SOD training loss."""
    weit = 1 + 5 * torch.abs(
        F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask
    )
    wbce = F.binary_cross_entropy(pred, mask, reduction='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)

    return (wbce + wiou).mean()


class DUTSDataset(Dataset):
    """DUTS-TR with optional teacher predictions."""

    def __init__(self, image_dir, mask_dir, teacher_dir=None,
                 input_size=256, output_size=None, augment=True):
        self.input_size = input_size
        self.output_size = output_size or input_size
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.teacher_dir = Path(teacher_dir) if teacher_dir else None
        self.augment = augment

        self.samples = []
        for img_path in sorted(self.image_dir.glob("*.jpg")):
            stem = img_path.stem
            mask_path = self.mask_dir / f"{stem}.png"
            if not mask_path.exists():
                continue
            teacher_path = None
            if self.teacher_dir:
                tp = self.teacher_dir / f"{stem}.npy"
                if tp.exists():
                    teacher_path = tp
            self.samples.append((img_path, mask_path, teacher_path))

        print(f"Dataset: {len(self.samples)} samples"
              f" (teacher: {sum(1 for s in self.samples if s[2] is not None)})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path, teacher_path = self.samples[idx]

        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        # Random scale augmentation
        if self.augment and np.random.random() > 0.5:
            scale = np.random.uniform(0.75, 1.25)
            new_size = max(int(self.input_size * scale), 64)
            img = img.resize((new_size, new_size), Image.BILINEAR)
            mask = mask.resize((new_size, new_size), Image.BILINEAR)
            # Center crop/pad to input_size
            img = self._center_crop_pad(img, self.input_size)
            mask = self._center_crop_pad(mask, self.input_size)

        img = img.resize((self.input_size, self.input_size), Image.BILINEAR)
        mask = mask.resize((self.output_size, self.output_size), Image.BILINEAR)

        img = np.array(img, dtype=np.float32) / 255.0
        mask = np.array(mask, dtype=np.float32) / 255.0

        # Color jitter
        if self.augment and np.random.random() > 0.5:
            # Brightness
            img = img * np.random.uniform(0.8, 1.2)
            img = np.clip(img, 0, 1)

        img = img.transpose(2, 0, 1)  # [3, H, W]

        # Random horizontal flip
        if self.augment and np.random.random() > 0.5:
            img = img[:, :, ::-1].copy()
            mask = mask[:, ::-1].copy()

        # Teacher
        teacher = np.zeros_like(mask)
        if teacher_path is not None:
            teacher = np.load(teacher_path).astype(np.float32)
            if teacher.shape[0] != self.output_size:
                teacher_img = Image.fromarray((teacher * 255).astype(np.uint8))
                teacher_img = teacher_img.resize(
                    (self.output_size, self.output_size), Image.BILINEAR
                )
                teacher = np.array(teacher_img, dtype=np.float32) / 255.0
            if self.augment and img.shape[2] != mask.shape[1]:
                pass  # skip teacher if augmented differently
            elif self.augment and np.random.random() > 0.5:
                teacher = teacher[:, ::-1].copy()

        return (
            torch.from_numpy(img),
            torch.from_numpy(mask).unsqueeze(0),
            torch.from_numpy(teacher).unsqueeze(0),
        )

    @staticmethod
    def _center_crop_pad(img, target_size):
        w, h = img.size
        if w >= target_size and h >= target_size:
            left = (w - target_size) // 2
            top = (h - target_size) // 2
            return img.crop((left, top, left + target_size, top + target_size))
        else:
            result = Image.new(img.mode, (target_size, target_size), 0)
            left = (target_size - w) // 2
            top = (target_size - h) // 2
            result.paste(img, (left, top))
            return result


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

    # Split: last 500 for validation
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

    model = MicroSalNet(width=args.width).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"Model: MicroSalNet(width={args.width}), {params:,} params")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    mse_loss = nn.MSELoss()
    best_val_mae = float("inf")
    tag = f"microsalnet_w{args.width}_s{args.input_size}_v2"

    loss_desc = "structure_loss(wBCE+wIoU)"
    if not args.no_teacher:
        loss_desc += f" + {1-args.alpha:.1f}*MSE(teacher)"

    print(f"\nTraining: {args.epochs} epochs, batch={args.batch}, lr={args.lr}")
    print(f"Loss: {loss_desc}\n")

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

            # Structure loss: weighted BCE + weighted IoU (boundary-aware)
            loss = structure_loss(pred, masks)

            # Optional teacher distillation
            if not args.no_teacher:
                loss_teacher = mse_loss(pred, teachers)
                loss = args.alpha * loss + (1 - args.alpha) * loss_teacher

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

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save(model.state_dict(), output_dir / f"{tag}_best.pth")
            print(f"  -> New best val MAE: {val_mae:.4f}")

        if (epoch + 1) % 20 == 0:
            torch.save(model.state_dict(), output_dir / f"{tag}_epoch{epoch+1}.pth")

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
    parser.add_argument("--alpha", type=float, default=0.7,
                        help="Weight for structure loss (1-alpha for teacher)")
    parser.add_argument("--no-teacher", action="store_true",
                        help="Train without teacher distillation (GT only)")
    args = parser.parse_args()
    train(args)
