"""Lightweight saliency model: MobileNetV3-Small encoder + minimal decoder.

Target: ~100-200K params, 256x256 input/output, all standard ONNX ops.
Architecture inspired by MediaPipe Selfie Segmentation (106K params, 31ms in tract).

Encoder: MobileNetV3-Small backbone (torchvision, pretrained) with reduced width.
Decoder: ConvTranspose2d upsampling + 1x1 conv to merge multi-scale features.

NOTE: Uses ConvTranspose2d instead of F.interpolate for upsampling because
tract's Resize op is pathologically slow (~60ms for 5 bilinear resizes).
ConvTranspose2d maps to efficient Conv ops in tract.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class InvertedResidual(nn.Module):
    """MobileNetV2-style inverted residual with optional SE and activation choice."""

    def __init__(self, inp, oup, stride, expand_ratio, use_se=False, activation="relu"):
        super().__init__()
        self.use_residual = stride == 1 and inp == oup
        hidden = int(inp * expand_ratio)

        act = nn.Hardswish if activation == "hardswish" else nn.ReLU

        layers = []
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(inp, hidden, 1, bias=False),
                nn.BatchNorm2d(hidden),
                act(inplace=True),
            ])

        layers.extend([
            # Depthwise
            nn.Conv2d(hidden, hidden, 3, stride=stride, padding=1, groups=hidden, bias=False),
            nn.BatchNorm2d(hidden),
            act(inplace=True),
        ])

        if use_se:
            layers.append(SEBlock(hidden))

        layers.extend([
            # Pointwise
            nn.Conv2d(hidden, oup, 1, bias=False),
            nn.BatchNorm2d(oup),
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_residual:
            return x + self.conv(x)
        return self.conv(x)


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block."""

    def __init__(self, channels, reduction=4):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.fc1 = nn.Conv2d(channels, mid, 1)
        self.fc2 = nn.Conv2d(mid, channels, 1)

    def forward(self, x):
        w = F.adaptive_avg_pool2d(x, 1)
        w = F.relu(self.fc1(w), inplace=True)
        w = torch.sigmoid(self.fc2(w))
        return x * w


class MicroSalNet(nn.Module):
    """Tiny saliency network (~200K params).

    Encoder stages at 256, 128, 64, 32, 16 resolution.
    Decoder upsamples from 8 back to 128 with skip connections.
    Output is 128x128 (half input resolution) — sufficient for smart cropping.
    """

    def __init__(self, width=16):
        super().__init__()
        w = width  # base width

        # Encoder
        # Stage 0: 256x256 -> 128x128
        self.stem = nn.Sequential(
            nn.Conv2d(3, w, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(w),
            nn.ReLU(inplace=True),
        )

        # Stage 1: 128x128 -> 64x64
        self.enc1 = nn.Sequential(
            InvertedResidual(w, w, stride=2, expand_ratio=1),
            InvertedResidual(w, w, stride=1, expand_ratio=1),
        )

        # Stage 2: 64x64 -> 32x32
        self.enc2 = nn.Sequential(
            InvertedResidual(w, w * 2, stride=2, expand_ratio=2),
            InvertedResidual(w * 2, w * 2, stride=1, expand_ratio=2),
        )

        # Stage 3: 32x32 -> 16x16
        self.enc3 = nn.Sequential(
            InvertedResidual(w * 2, w * 4, stride=2, expand_ratio=2, use_se=True),
            InvertedResidual(w * 4, w * 4, stride=1, expand_ratio=2, use_se=True),
        )

        # Stage 4: 16x16 (bottleneck)
        self.enc4 = nn.Sequential(
            InvertedResidual(w * 4, w * 8, stride=2, expand_ratio=2, use_se=True,
                             activation="hardswish"),
            InvertedResidual(w * 8, w * 8, stride=1, expand_ratio=2, use_se=True,
                             activation="hardswish"),
        )

        # Decoder (ConvTranspose2d upsample + merge skip connections)
        # 8x8 -> 16x16
        self.up4 = nn.ConvTranspose2d(w * 8, w * 4, kernel_size=2, stride=2, bias=False)
        self.dec4 = nn.Sequential(
            nn.BatchNorm2d(w * 4),
            nn.ReLU(inplace=True),
        )
        # 16x16 -> 32x32
        self.up3 = nn.ConvTranspose2d(w * 4 + w * 4, w * 2, kernel_size=2, stride=2, bias=False)
        self.dec3 = nn.Sequential(
            nn.BatchNorm2d(w * 2),
            nn.ReLU(inplace=True),
        )
        # 32x32 -> 64x64
        self.up2 = nn.ConvTranspose2d(w * 2 + w * 2, w, kernel_size=2, stride=2, bias=False)
        self.dec2 = nn.Sequential(
            nn.BatchNorm2d(w),
            nn.ReLU(inplace=True),
        )
        # 64x64 -> 128x128
        self.up1 = nn.ConvTranspose2d(w + w, w, kernel_size=2, stride=2, bias=False)
        self.dec1 = nn.Sequential(
            nn.BatchNorm2d(w),
            nn.ReLU(inplace=True),
        )

        # Output head at 128x128 (skip last upsample to 256 — saves 12ms in tract)
        # Concat d1 (w) + s0 (w) = 2w channels, then 1x1 conv to saliency
        self.head = nn.Conv2d(w * 2, 1, 1)

    def forward(self, x):
        # Encoder
        s0 = self.stem(x)      # [B, w, 128, 128]
        s1 = self.enc1(s0)     # [B, w, 64, 64]
        s2 = self.enc2(s1)     # [B, 2w, 32, 32]
        s3 = self.enc3(s2)     # [B, 4w, 16, 16]
        s4 = self.enc4(s3)     # [B, 8w, 8, 8]

        # Decoder (ConvTranspose2d for 2x upsample, no Resize ops)
        d4 = self.dec4(self.up4(s4))                 # [B, 4w, 16, 16]
        d3 = self.dec3(self.up3(torch.cat([d4, s3], dim=1)))  # [B, 2w, 32, 32]
        d2 = self.dec2(self.up2(torch.cat([d3, s2], dim=1)))  # [B, w, 64, 64]
        d1 = self.dec1(self.up1(torch.cat([d2, s1], dim=1)))  # [B, w, 128, 128]

        # Output at 128x128 (concat with stem skip, head to saliency)
        out = torch.sigmoid(self.head(torch.cat([d1, s0], dim=1)))  # [B, 1, 128, 128]
        return out


def count_params(model):
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    model = MicroSalNet(width=16)
    print(f"Parameters: {count_params(model):,}")

    x = torch.randn(1, 3, 256, 256)
    y = model(x)
    print(f"Input: {x.shape} -> Output: {y.shape}")
    print(f"Output range: [{y.min():.4f}, {y.max():.4f}]")

    # Try different widths
    for w in [8, 12, 16, 20, 24]:
        m = MicroSalNet(width=w)
        print(f"  width={w:2d}: {count_params(m):>8,} params")
