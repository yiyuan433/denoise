"""
Lightweight baseline models for 2D denoising.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DnCNN2D(nn.Module):
    """DnCNN-style residual denoiser (2D)."""

    def __init__(self, in_channels=1, num_features=64, depth=17, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        layers = [
            nn.Conv2d(in_channels, num_features, kernel_size, padding=padding),
            nn.ReLU(inplace=True),
        ]

        for _ in range(depth - 2):
            layers += [
                nn.Conv2d(num_features, num_features, kernel_size, padding=padding),
                nn.BatchNorm2d(num_features),
                nn.ReLU(inplace=True),
            ]

        layers.append(nn.Conv2d(num_features, in_channels, kernel_size, padding=padding))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        noise = self.net(x)
        return x - noise


def _conv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
    )


class UNet2D(nn.Module):
    """Compact UNet for 2D denoising."""

    def __init__(self, in_channels=1, base_channels=32, out_channels=1):
        super().__init__()
        self.enc1 = _conv_block(in_channels, base_channels)
        self.enc2 = _conv_block(base_channels, base_channels * 2)
        self.enc3 = _conv_block(base_channels * 2, base_channels * 4)
        self.enc4 = _conv_block(base_channels * 4, base_channels * 8)

        self.pool = nn.MaxPool2d(2)

        self.bottleneck = _conv_block(base_channels * 8, base_channels * 16)

        self.up4 = nn.ConvTranspose2d(base_channels * 16, base_channels * 8, kernel_size=2, stride=2)
        self.dec4 = _conv_block(base_channels * 16, base_channels * 8)

        self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.dec3 = _conv_block(base_channels * 8, base_channels * 4)

        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.dec2 = _conv_block(base_channels * 4, base_channels * 2)

        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.dec1 = _conv_block(base_channels * 2, base_channels)

        self.out_conv = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def forward(self, x):
        original_h, original_w = x.shape[-2:]

        # Pad to a multiple of 16 so four encoder/decoder stages keep spatial sizes aligned.
        target_h = ((original_h + 15) // 16) * 16
        target_w = ((original_w + 15) // 16) * 16
        pad_h = target_h - original_h
        pad_w = target_w - original_w
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")

        def _align_to(tensor, reference):
            if tensor.shape[-2:] != reference.shape[-2:]:
                tensor = F.interpolate(tensor, size=reference.shape[-2:], mode="bilinear", align_corners=False)
            return tensor

        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b = self.bottleneck(self.pool(e4))

        d4 = _align_to(self.up4(b), e4)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))

        d3 = _align_to(self.up3(d4), e3)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = _align_to(self.up2(d3), e2)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = _align_to(self.up1(d2), e1)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        out = self.out_conv(d1)

        if pad_h or pad_w:
            out = out[..., :original_h, :original_w]

        return out


def create_baseline_model(name, **kwargs):
    """Factory for baseline models."""
    name = name.lower()
    if name == "dncnn":
        return DnCNN2D(**kwargs)
    if name == "unet":
        return UNet2D(**kwargs)
    raise ValueError(f"Unknown baseline model: {name}")
