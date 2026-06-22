"""
USL DIP 模型的 PyTorch 复现
=========================
原始论文: Unsupervised Deep Learning for Random Noise Attenuation of Seismic Data
原始框架: Keras/TensorFlow
本文件: PyTorch 等价实现，保持与原始架构一致

架构:
- 3层编码-解码器 (1D Conv autoencoder)
- 双路径卷积 (kernel_size=3 和 kernel_size=6) 每层  (大核 = 小核 + 3, 与原始一致)
- ECA 通道注意力 skip connection (两层Conv1D, relu+sigmoid, 与原始一致)
- 自监督 DIP 训练 (input = noisy, target = noisy)

原始参数: D1=128, D2=32, D3=8, kernel_size=3/6 (大核=小核+3)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ECABlock1D(nn.Module):
    """
    Efficient Channel Attention (1D) —— 与原始 USL eca_block 完全对应

    原始 Keras 参数: b=1, gama=8, in_channel=1 (硬编码)
    → kernel_size = int(abs((log(1,2) + 1) / 8)) = 0 → 补为 1
    原始结构: GAP → Conv1D(relu) → Conv1D(sigmoid) → 通道加权
    """

    def __init__(self, channels: int, b: int = 1, gama: int = 8):
        super().__init__()
        # ★ 与原始一致: 使用 in_channel=1 (硬编码) 计算 kernel_size
        in_channel = 1
        kernel_size = int(abs((math.log(in_channel + 1e-12, 2) + b) / gama))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        # in_channel=1 → log(1,2)=0 → k=int(1/8)=0 → 补1 → kernel_size=1

        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        # ★ 原始有两层 Conv1D, 中间用 relu, 最后用 sigmoid
        self.conv1 = nn.Conv1d(
            1, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False
        )
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(
            1, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        x: (B, C, L)
        return: (B, C, L) 加权后
        """
        # (B, C, L) -> GAP -> (B, C, 1)
        y = self.avg_pool(x)
        # (B, C, 1) -> transpose -> (B, 1, C)
        y = y.transpose(1, 2)
        # ★ 两层 Conv1D: conv(relu) → conv(sigmoid)  与原始 eca_block 一致
        y = self.relu(self.conv1(y))
        y = self.sigmoid(self.conv2(y))
        # (B, 1, C) -> transpose -> (B, C, 1)
        y = y.transpose(1, 2)
        return x * y


class DualPathConv1D(nn.Module):
    """
    双路径 Conv1D 块 —— 与原始 USL mff_block 完全对应

    原始 Keras mff_block:
      path a: Conv1D(filters, k, strides) → BN → PReLU
      path b: Conv1D(filters, k+3, strides+2) → BN → PReLU
      concat → Conv1D(1, 1, tanh) → Add(residual)

    PyTorch 适配说明:
      - 两条路径均保留 BN + PReLU (与原始一致)
      - 大核默认 = 小核 + 3 (原始 k=3 → 大核=6)
      - 投影层输出 1 通道: Conv1d(2*filters, 1, 1) + Tanh (与原始 Conv1D(1,1,tanh) 一致)
      - 保留残差连接 Add (与原始一致, 输入/输出均为 1 通道)
      - strides 统一用 1 (原始 path B strides+2 会导致空间维度不一致,
        此处保持 stride=1 以确保 concat 可行)
    """

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_small: int = 3, kernel_large: int = None):
        super().__init__()
        # ★ 原始: kernel_large = kernel_small + 3 (如 3→6)
        if kernel_large is None:
            kernel_large = kernel_small + 3

        # Path A: small kernel + BN + PReLU
        self.conv_a = nn.Conv1d(in_channels, out_channels, kernel_small, padding='same')
        self.bn_a = nn.BatchNorm1d(out_channels)
        self.act_a = nn.PReLU(out_channels)

        # Path B: large kernel + BN + PReLU  ★ 原始两条路径都有 BN
        self.conv_b = nn.Conv1d(in_channels, out_channels, kernel_large, padding='same')
        self.bn_b = nn.BatchNorm1d(out_channels)
        self.act_b = nn.PReLU(out_channels)

        # Projection: 2*filters -> 1 channel  ★ 与原始 Conv1D(1, 1, tanh) 完全一致
        self.proj = nn.Conv1d(out_channels * 2, 1, 1)
        self.act_proj = nn.Tanh()

        # ★ 残差连接: 投影到 1 通道 (与原始 Add 一致)
        if in_channels != 1:
            self.skip_proj = nn.Conv1d(in_channels, 1, 1)
        else:
            self.skip_proj = nn.Identity()

    def forward(self, x):
        identity = self.skip_proj(x)  # → (B, 1, L)

        a = self.act_a(self.bn_a(self.conv_a(x)))   # (B, filters, L)
        b = self.act_b(self.bn_b(self.conv_b(x)))   # (B, filters, L)
        out = torch.cat([a, b], dim=1)               # (B, 2*filters, L)
        out = self.act_proj(self.proj(out))           # (B, 1, L)  ★ Tanh

        out = out + identity  # ★ 残差连接 (B, 1, L) + (B, 1, L)
        return out


class USLAutoencoder(nn.Module):
    """
    USL DIP Autoencoder (PyTorch)
    =============================
    与原始 Keras 版本的 USL_DIP_FORGE_EQ36_Train.ipynb 架构一一对应:
      - 3 层编码器, 每层: DualPathConv1D(→1通道) + ECA skip
      - 3 层解码器, 每层: DualPathConv1D(→1通道) + 加法 skip (与原始 Add 一致)
      - 全程 1 通道 (与原始 mff_block 输出 Conv1D(1,1,tanh) 一致)

    输入/输出: (B, 1, H, W) 2D patch
      内部将 patch 展平为 1D 序列处理 (与 USL 原始 yc_patch 一致)

    Parameters
    ----------
    patch_h, patch_w : int
        Patch 的高和宽, 默认 24×24 (与 USL 原始一致)
    D1, D2, D3 : int
        3 层通道数, 默认 128/32/8 (与 USL 原始一致)
    kernel_small, kernel_large : int
        双路径卷积核大小, 默认 3/6 (kernel_large = kernel_small + 3, 与原始一致)
    """

    def __init__(
        self,
        patch_h: int = 24,
        patch_w: int = 24,
        D1: int = 128,
        D2: int = 32,
        D3: int = 8,
        kernel_small: int = 3,
        kernel_large: int = None,
    ):
        super().__init__()
        self.patch_h = patch_h
        self.patch_w = patch_w
        self.seq_len = patch_h * patch_w  # 576

        # kernel_large 默认 = kernel_small + 3 (与原始 mff_block 一致)
        if kernel_large is None:
            kernel_large = kernel_small + 3  # 3 → 6

        # ---- Encoder (每层输出 1 通道, 与原始 mff_block 一致) ----
        self.enc1 = DualPathConv1D(1, D1, kernel_small, kernel_large)
        self.eca1 = ECABlock1D(1)

        self.enc2 = DualPathConv1D(1, D2, kernel_small, kernel_large)
        self.eca2 = ECABlock1D(1)

        self.enc3 = DualPathConv1D(1, D3, kernel_small, kernel_large)
        self.eca3 = ECABlock1D(1)

        # ---- Decoder (每层输入/输出 1 通道, skip 用加法与原始 Add 一致) ----
        self.dec3 = DualPathConv1D(1, D3, kernel_small, kernel_large)
        self.dec2 = DualPathConv1D(1, D2, kernel_small, kernel_large)
        self.dec1 = DualPathConv1D(1, D1, kernel_small, kernel_large)

    def forward(self, x):
        """
        x : (B, 1, H, W)
        return : (output, aux_dict)
            output : (B, 1, H, W)
            aux_dict : dict with optional diagnostics
        """
        B, C, H, W = x.shape
        # Flatten to 1D: (B, 1, H*W) = (B, 1, L)
        x_flat = x.view(B, 1, -1)

        # Encoder (全程 1 通道, 与原始一致)
        e1 = self.enc1(x_flat)   # (B, 1, L)
        skip1 = self.eca1(e1)    # (B, 1, L)

        e2 = self.enc2(e1)       # (B, 1, L)
        skip2 = self.eca2(e2)    # (B, 1, L)

        e3 = self.enc3(e2)       # (B, 1, L)
        skip3 = self.eca3(e3)    # (B, 1, L)

        # Decoder (skip 用加法, 与原始 Add 一致)
        d3 = self.dec3(e3)       # (B, 1, L)
        d3 = d3 + skip3          # (B, 1, L)

        d2 = self.dec2(d3)       # (B, 1, L)
        d2 = d2 + skip2          # (B, 1, L)

        d1 = self.dec1(d2)       # (B, 1, L)
        d1 = d1 + skip1          # (B, 1, L)

        # Reshape back to 2D (已经是 1 通道, 无需额外投影)
        out = d1.view(B, 1, H, W)

        # 中间特征: 编码器瓶颈层
        features_2d = e3.view(B, 1, H, W)

        return out, {"model_type": "usl", "features": features_2d}


def create_usl_model(config: dict) -> nn.Module:
    """
    根据配置创建 USL 模型

    Parameters
    ----------
    config : dict
        模型配置, 可选键:
        - patch_size: (H, W) 默认 (24, 24)
        - D1, D2, D3: 通道数, 默认 128/32/8
        - kernel_small, kernel_large: 默认 3/6 (kernel_large = kernel_small + 3)
    """
    patch_size = config.get("patch_size", (24, 24))
    model = USLAutoencoder(
        patch_h=patch_size[0],
        patch_w=patch_size[1],
        D1=config.get("D1", 128),
        D2=config.get("D2", 32),
        D3=config.get("D3", 8),
        kernel_small=config.get("kernel_small", 3),
        kernel_large=config.get("kernel_large", None),  # None → 自动 kernel_small+3
    )
    return model


if __name__ == "__main__":
    print("=" * 60)
    print("USL DIP Model (PyTorch) — Testing")
    print("=" * 60)

    model = create_usl_model({"patch_size": (24, 24)})
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    x = torch.randn(4, 1, 24, 24)
    out, aux = model(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {out.shape}")
    print(f"Aux:    {aux}")
    assert out.shape == x.shape, "Shape mismatch!"
    print("✓ Shape test passed")
