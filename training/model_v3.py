"""MicroSalNet v3 — hybrid: v2's encoder structure + dilated bottleneck at 32×32.

Key changes from v2:
- Encoder stops at 32×32 (removes enc3 at 16×16 and enc4 at 8×8)
- Adds dilated MSBlocks at 32×32 for receptive field
- Fewer, wider bottleneck with expand ratio 4 for capacity
- Only 2 decoder upsamples (32→64→128) instead of 4 (8→16→32→64→128)
- Skip connections from stem (128) and enc1 (64)

All standard ops: Conv2d, ConvTranspose2d, ReLU, BN, Concat — no Resize.
"""

import torch
import torch.nn as nn


class InvertedResidual(nn.Module):
    """MobileNetV2-style inverted residual block."""

    def __init__(self, inp, oup, stride, expand_ratio, use_se=False):
        super().__init__()
        self.use_residual = stride == 1 and inp == oup
        hidden = int(inp * expand_ratio)

        layers = []
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(inp, hidden, 1, bias=False),
                nn.BatchNorm2d(hidden),
                nn.ReLU(inplace=True),
            ])

        layers.extend([
            nn.Conv2d(hidden, hidden, 3, stride=stride, padding=1,
                      groups=hidden, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
        ])

        if use_se:
            layers.append(SEBlock(hidden))

        layers.extend([
            nn.Conv2d(hidden, oup, 1, bias=False),
            nn.BatchNorm2d(oup),
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_residual:
            return x + self.conv(x)
        return self.conv(x)


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.fc1 = nn.Conv2d(channels, mid, 1)
        self.fc2 = nn.Conv2d(mid, channels, 1)

    def forward(self, x):
        w = x.mean(dim=(2, 3), keepdim=True)
        w = torch.relu(self.fc1(w))
        w = torch.sigmoid(self.fc2(w))
        return x * w


class DilatedBlock(nn.Module):
    """Depthwise dilated conv + pointwise, with residual.

    Single dilation rate per block — simpler and more parameter-efficient
    than the full MSBlock. Stack multiple with different dilations.
    """

    def __init__(self, channels, dilation, expand_ratio=2):
        super().__init__()
        hidden = channels * expand_ratio
        self.conv = nn.Sequential(
            # Expand
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            # Dilated depthwise
            nn.Conv2d(hidden, hidden, 3, padding=dilation, dilation=dilation,
                      groups=hidden, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            # Project back
            nn.Conv2d(hidden, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x):
        return x + self.conv(x)


class MicroSalNetV3(nn.Module):
    """Saliency network with dilated bottleneck at 32×32.

    Architecture:
      256→128 (stem) → 64 (enc1) → 32 (enc2)
      → dilated blocks at 32×32 (d=1,2,4,8,1,2,4,8)
      → 64 (up2+skip) → 128 (up1+skip) → head
    """

    def __init__(self, width=24):
        super().__init__()
        w = width

        # Encoder: 256→128→64→32
        self.stem = nn.Sequential(
            nn.Conv2d(3, w, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(w),
            nn.ReLU(inplace=True),
        )

        self.enc1 = nn.Sequential(
            InvertedResidual(w, w, stride=2, expand_ratio=2),
            InvertedResidual(w, w, stride=1, expand_ratio=2),
        )

        self.enc2 = nn.Sequential(
            InvertedResidual(w, w * 2, stride=2, expand_ratio=2),
            InvertedResidual(w * 2, w * 2, stride=1, expand_ratio=2),
        )

        # Bottleneck: stacked dilated blocks at 32×32
        # Each block: expand→dilated_depthwise→project, with residual
        # Dilations 1,2,4,8 repeated = effective RF covers entire 32×32 map
        self.bottleneck = nn.Sequential(
            DilatedBlock(w * 2, dilation=1, expand_ratio=2),
            DilatedBlock(w * 2, dilation=2, expand_ratio=2),
            DilatedBlock(w * 2, dilation=4, expand_ratio=2),
            DilatedBlock(w * 2, dilation=8, expand_ratio=2),
            DilatedBlock(w * 2, dilation=1, expand_ratio=2),
            DilatedBlock(w * 2, dilation=2, expand_ratio=2),
            DilatedBlock(w * 2, dilation=4, expand_ratio=2),
            DilatedBlock(w * 2, dilation=8, expand_ratio=2),
        )

        # Decoder: 32→64→128
        self.up2 = nn.ConvTranspose2d(w * 2, w, kernel_size=2, stride=2, bias=False)
        self.dec2 = nn.Sequential(
            nn.BatchNorm2d(w),
            nn.ReLU(inplace=True),
        )

        self.up1 = nn.ConvTranspose2d(w + w, w, kernel_size=2, stride=2, bias=False)
        self.dec1 = nn.Sequential(
            nn.BatchNorm2d(w),
            nn.ReLU(inplace=True),
        )

        # Head: concat dec1 + stem skip → saliency
        self.head = nn.Conv2d(w + w, 1, 1)

    def forward(self, x):
        s0 = self.stem(x)          # [w, 128, 128]
        s1 = self.enc1(s0)         # [w, 64, 64]
        s2 = self.enc2(s1)         # [2w, 32, 32]

        b = self.bottleneck(s2)    # [2w, 32, 32]

        d2 = self.dec2(self.up2(b))                           # [w, 64, 64]
        d1 = self.dec1(self.up1(torch.cat([d2, s1], dim=1)))  # [w, 128, 128]

        out = torch.sigmoid(self.head(torch.cat([d1, s0], dim=1)))
        return out


def count_params(model):
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    for w in [16, 20, 24, 28, 32, 40]:
        m = MicroSalNetV3(width=w)
        x = torch.randn(1, 3, 256, 256)
        y = m(x)
        print(f"  width={w:2d}: {count_params(m):>8,} params  "
              f"output={y.shape}  ONNX~{count_params(m)*4/1024:.0f}KB")
