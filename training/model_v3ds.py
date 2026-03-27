"""MicroSalNet v3ds — v3d + deep supervision + stronger augmentation.

Changes from v3d:
- Auxiliary saliency heads at each decoder stage (deep supervision)
- During training: loss = sum of weighted losses at all scales
- During inference: only the final head is used (aux heads removed for ONNX)
- Optional: wider bottleneck channels via bottleneck_mult
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, use_se=False):
        super().__init__()
        self.use_residual = stride == 1 and inp == oup
        hidden = int(inp * expand_ratio)
        layers = []
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(inp, hidden, 1, bias=False),
                nn.BatchNorm2d(hidden), nn.ReLU(inplace=True),
            ])
        layers.extend([
            nn.Conv2d(hidden, hidden, 3, stride=stride, padding=1,
                      groups=hidden, bias=False),
            nn.BatchNorm2d(hidden), nn.ReLU(inplace=True),
        ])
        if use_se:
            layers.append(SEBlock(hidden))
        layers.extend([
            nn.Conv2d(hidden, oup, 1, bias=False),
            nn.BatchNorm2d(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.conv(x) if self.use_residual else self.conv(x)


class SEBlock(nn.Module):
    def __init__(self, ch, r=4):
        super().__init__()
        mid = max(ch // r, 4)
        self.fc1 = nn.Conv2d(ch, mid, 1)
        self.fc2 = nn.Conv2d(mid, ch, 1)

    def forward(self, x):
        w = x.mean(dim=(2, 3), keepdim=True)
        return x * torch.sigmoid(self.fc2(torch.relu(self.fc1(w))))


class DilatedBlock(nn.Module):
    def __init__(self, ch, dilation, expand_ratio=2):
        super().__init__()
        hid = ch * expand_ratio
        self.conv = nn.Sequential(
            nn.Conv2d(ch, hid, 1, bias=False),
            nn.BatchNorm2d(hid), nn.ReLU(inplace=True),
            nn.Conv2d(hid, hid, 3, padding=dilation, dilation=dilation,
                      groups=hid, bias=False),
            nn.BatchNorm2d(hid), nn.ReLU(inplace=True),
            nn.Conv2d(hid, ch, 1, bias=False),
            nn.BatchNorm2d(ch),
        )

    def forward(self, x):
        return x + self.conv(x)


class MicroSalNetV3ds(nn.Module):
    """v3d with deep supervision.

    During training, returns (final_pred, [aux3, aux2, aux1]) where each
    aux is a saliency map at the corresponding decoder resolution.
    During eval/export, returns only final_pred.
    """

    def __init__(self, width=24, bottleneck_mult=3):
        super().__init__()
        w = width
        bw = w * bottleneck_mult  # bottleneck channels

        # Encoder
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
            InvertedResidual(w * 2, bw, stride=2, expand_ratio=2, use_se=True),
            InvertedResidual(bw, bw, stride=1, expand_ratio=2, use_se=True),
        )

        # Dilated bottleneck at 16×16
        self.bottleneck = nn.Sequential(
            DilatedBlock(bw, dilation=1, expand_ratio=2),
            DilatedBlock(bw, dilation=2, expand_ratio=2),
            DilatedBlock(bw, dilation=4, expand_ratio=2),
            DilatedBlock(bw, dilation=8, expand_ratio=2),
        )

        # Decoder
        self.up3 = nn.ConvTranspose2d(bw, w * 2, 2, stride=2, bias=False)
        self.dec3 = nn.Sequential(nn.BatchNorm2d(w * 2), nn.ReLU(inplace=True))

        self.up2 = nn.ConvTranspose2d(w * 2 + w * 2, w, 2, stride=2, bias=False)
        self.dec2 = nn.Sequential(nn.BatchNorm2d(w), nn.ReLU(inplace=True))

        self.up1 = nn.ConvTranspose2d(w + w, w, 2, stride=2, bias=False)
        self.dec1 = nn.Sequential(nn.BatchNorm2d(w), nn.ReLU(inplace=True))

        # Final head
        self.head = nn.Conv2d(w + w, 1, 1)

        # Auxiliary heads for deep supervision (training only)
        self.aux3 = nn.Conv2d(w * 2, 1, 1)  # at 32×32
        self.aux2 = nn.Conv2d(w, 1, 1)      # at 64×64
        self.aux1 = nn.Conv2d(w, 1, 1)      # at 128×128 (before stem skip)

    def forward(self, x):
        s0 = self.stem(x)
        s1 = self.enc1(s0)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)
        b = self.bottleneck(s3)

        d3 = self.dec3(self.up3(b))
        d2 = self.dec2(self.up2(torch.cat([d3, s2], dim=1)))
        d1 = self.dec1(self.up1(torch.cat([d2, s1], dim=1)))

        final = torch.sigmoid(self.head(torch.cat([d1, s0], dim=1)))

        if self.training:
            a3 = torch.sigmoid(self.aux3(d3))   # [1, 1, 32, 32]
            a2 = torch.sigmoid(self.aux2(d2))   # [1, 1, 64, 64]
            a1 = torch.sigmoid(self.aux1(d1))   # [1, 1, 128, 128]
            return final, [a3, a2, a1]

        return final

    def export_mode(self):
        """Remove auxiliary heads for ONNX export (saves ~1KB)."""
        del self.aux3, self.aux2, self.aux1
        self.eval()
        return self


def count_params(model):
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    for w in [20, 24, 28, 32]:
        m = MicroSalNetV3ds(width=w)
        x = torch.randn(1, 3, 256, 256)
        m.train()
        final, auxes = m(x)
        m.eval()
        y = m(x)
        p = count_params(m)
        print(f"  w={w:2d}: {p:>8,} params  "
              f"final={final.shape} aux=[{','.join(str(list(a.shape)) for a in auxes)}]  "
              f"eval={y.shape}")
