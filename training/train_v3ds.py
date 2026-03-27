"""Train MicroSalNet v3ds — deep supervision + stronger augmentation.

Loss = structure_loss(final, gt) + 0.4 * sum(structure_loss(aux_i, gt_i))
     + (1-alpha) * MSE(final, teacher)

Usage:
    python3 training/train_v3ds.py --width 24 --epochs 150
    python3 training/train_v3ds.py --width 28 --epochs 150
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
from PIL import Image, ImageFilter

from model_v3ds import MicroSalNetV3ds, count_params


def structure_loss(pred, mask):
    """Weighted BCE + weighted IoU — boundary-aware."""
    weit = 1 + 5 * torch.abs(
        F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask
    )
    wbce = F.binary_cross_entropy(pred, mask, reduction='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)

    return (wbce + wiou).mean()


class DUTSDatasetV3(Dataset):
    """DUTS-TR with strong augmentation for v3 training."""

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

        print(f"Dataset: {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path, teacher_path = self.samples[idx]

        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.augment:
            # Random horizontal flip
            if np.random.random() > 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

            # Random scale (0.75–1.25)
            if np.random.random() > 0.5:
                scale = np.random.uniform(0.75, 1.25)
                new_w = max(int(img.width * scale), 64)
                new_h = max(int(img.height * scale), 64)
                img = img.resize((new_w, new_h), Image.BILINEAR)
                mask = mask.resize((new_w, new_h), Image.BILINEAR)

            # Random rotation (±10°)
            if np.random.random() > 0.7:
                angle = np.random.uniform(-10, 10)
                img = img.rotate(angle, resample=Image.BILINEAR, fillcolor=(0, 0, 0))
                mask = mask.rotate(angle, resample=Image.BILINEAR, fillcolor=0)

            # Color jitter
            if np.random.random() > 0.5:
                # Brightness
                factor = np.random.uniform(0.7, 1.3)
                img = Image.fromarray(
                    np.clip(np.array(img, dtype=np.float32) * factor, 0, 255).astype(np.uint8)
                )

            # Random Gaussian blur
            if np.random.random() > 0.8:
                img = img.filter(ImageFilter.GaussianBlur(radius=np.random.uniform(0.5, 1.5)))

        img = img.resize((self.input_size, self.input_size), Image.BILINEAR)
        mask = mask.resize((self.output_size, self.output_size), Image.BILINEAR)

        img = np.array(img, dtype=np.float32) / 255.0
        mask = np.array(mask, dtype=np.float32) / 255.0
        img = img.transpose(2, 0, 1)

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

        return (
            torch.from_numpy(img),
            torch.from_numpy(mask).unsqueeze(0),
            torch.from_numpy(teacher).unsqueeze(0),
        )


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

    dataset = DUTSDatasetV3(
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

    val_dataset = DUTSDatasetV3(
        image_dir=data_root / "DUTS-TR-Image",
        mask_dir=data_root / "DUTS-TR-Mask",
        teacher_dir=teacher_dir,
        input_size=args.input_size,
        output_size=output_size,
        augment=False,
    )
    val_loader = DataLoader(
        torch.utils.data.Subset(val_dataset, val_indices),
        batch_size=args.batch, shuffle=False, num_workers=2, pin_memory=True,
    )

    model = MicroSalNetV3ds(width=args.width).to(device)
    params = count_params(model)
    print(f"Model: MicroSalNetV3ds(width={args.width}), {params:,} params")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    mse_loss = nn.MSELoss()

    best_val_mae = float("inf")
    tag = f"microsalnet_v3ds_w{args.width}_s{args.input_size}"

    # Aux loss weight ramps up over first 20 epochs
    aux_weight = args.aux_weight

    print(f"\nTraining: {args.epochs} epochs, batch={args.batch}, lr={args.lr}")
    print(f"Deep supervision: aux_weight={aux_weight}")
    print(f"Teacher: alpha={args.alpha}\n")

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        train_mae = 0.0
        n_batches = 0
        t0 = time.time()

        # Ramp aux weight over first 20 epochs
        current_aux = aux_weight * min(1.0, (epoch + 1) / 20.0)

        for imgs, masks, teachers in train_loader:
            imgs = imgs.to(device)
            masks = masks.to(device)
            teachers = teachers.to(device)

            final, auxes = model(imgs)

            # Main loss on final output
            loss_main = structure_loss(final, masks)

            # Deep supervision: aux losses at each decoder scale
            loss_aux = torch.tensor(0.0, device=device)
            for aux_pred in auxes:
                # Resize GT to match aux prediction size
                aux_h, aux_w = aux_pred.shape[2], aux_pred.shape[3]
                if aux_h != masks.shape[2] or aux_w != masks.shape[3]:
                    gt_resized = F.interpolate(masks, size=(aux_h, aux_w),
                                               mode='bilinear', align_corners=False)
                else:
                    gt_resized = masks
                loss_aux = loss_aux + structure_loss(aux_pred, gt_resized)

            loss = loss_main + current_aux * loss_aux

            # Teacher distillation on final output
            if not args.no_teacher:
                loss = args.alpha * loss + (1 - args.alpha) * mse_loss(final, teachers)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                mae = (final - masks).abs().mean().item()
                train_loss += loss.item()
                train_mae += mae
                n_batches += 1

        scheduler.step()
        train_loss /= n_batches
        train_mae /= n_batches
        epoch_time = time.time() - t0

        # Validation (eval mode — only final output)
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
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--aux-weight", type=float, default=0.4)
    parser.add_argument("--no-teacher", action="store_true")
    args = parser.parse_args()
    train(args)
