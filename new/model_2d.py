"""
完全2D处理的降噪模型 - 整合所有9个创新点 + FDA + NAAG
将所有创新点用2D操作实现
新增：
- FDA: Frequency Disentangled Attention (频率解耦注意力)
- NAAG: Noise-Aware Adaptive Gating (噪声感知自适应门控)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Tuple


class FrequencyDisentangledAttention(nn.Module):
    """
    频率解耦注意力模块 (FDA) - 原创
    在频域显式分离信号和噪声的频率成分
    """
    
    def __init__(self, in_channels=1, d_model=64, num_freq_bands=3):
        super().__init__()
        self.in_channels = in_channels
        self.d_model = d_model
        self.num_freq_bands = num_freq_bands  # 信号主导、混合、噪声主导
        
        # 频率特征提取器
        self.freq_encoder = nn.Sequential(
            nn.Conv2d(in_channels, d_model, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(d_model, d_model, kernel_size=1)
        )
        
        # 可学习的频率mask生成网络
        self.freq_mask_net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, num_freq_bands),
            nn.Softmax(dim=-1)
        )
        
        # 每个频带的处理网络
        self.band_processors = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, d_model, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(d_model, in_channels, kernel_size=3, padding=1)
            ) for _ in range(num_freq_bands)
        ])
        
        # 频域自注意力
        self.freq_attention = nn.MultiheadAttention(d_model, num_heads=4, batch_first=True)
        
        # 输出融合
        self.output_fusion = nn.Conv2d(in_channels * num_freq_bands, in_channels, kernel_size=1)
    
    def forward(self, x):
        """
        x: (B, C, H, W)
        返回: 解耦后的特征, 频率mask权重
        """
        B, C, H, W = x.shape
        
        # === 步骤1: FFT到频域 ===
        x_float32 = x.float()
        x_fft = torch.fft.rfft2(x_float32, dim=(-2, -1))
        x_mag = torch.abs(x_fft)  # 幅度谱 (B, C, H, W_freq)
        x_phase = torch.angle(x_fft)  # 相位谱
        
        freq_h, freq_w = x_mag.shape[-2:]
        
        # === 步骤2: 频域特征编码 ===
        # 将幅度谱作为输入
        freq_feat = self.freq_encoder(x_mag)  # (B, d_model, freq_h, freq_w)
        
        # === 步骤3: 生成频率band的mask权重 ===
        band_weights = self.freq_mask_net(freq_feat)  # (B, num_freq_bands)
        
        # === 步骤4: 将频谱分成多个频带 ===
        # 低频 -> 高频划分
        band_h = freq_h // self.num_freq_bands
        freq_bands_complex = []
        
        for i in range(self.num_freq_bands):
            start_h = i * band_h
            end_h = (i + 1) * band_h if i < self.num_freq_bands - 1 else freq_h
            
            # 创建band mask
            band_mask = torch.zeros_like(x_mag)
            band_mask[..., start_h:end_h, :] = 1.0
            
            # 应用mask到复数频谱
            band_fft = x_fft * band_mask
            
            # IFFT回到空域
            band_spatial = torch.fft.irfft2(band_fft, s=(H, W), dim=(-2, -1))
            band_spatial = band_spatial.to(x.dtype)
            
            freq_bands_complex.append(band_spatial)
        
        # === 步骤5: 每个频带单独处理 ===
        processed_bands = []
        for i, band_signal in enumerate(freq_bands_complex):
            # 应用频带特定的处理网络
            processed = self.band_processors[i](band_signal)
            
            # 加权
            weight = band_weights[:, i:i+1, None, None]  # (B, 1, 1, 1)
            processed_bands.append(processed * weight)
        
        # === 步骤6: 融合所有频带 ===
        concatenated = torch.cat(processed_bands, dim=1)  # (B, C*num_bands, H, W)
        output = self.output_fusion(concatenated)  # (B, C, H, W)
        
        # 残差连接
        output = output + x
        
        return output, band_weights


class NoiseAwareAdaptiveGating(nn.Module):
    """
    噪声感知自适应门控网络 (NAAG) - 原创
    动态估计每个patch的局部噪声水平，并自适应调整特征提取的强度
    """
    
    def __init__(self, in_channels=1, d_model=64):
        super().__init__()
        self.in_channels = in_channels
        self.d_model = d_model
        
        # 轻量级噪声水平估计器
        self.noise_estimator = nn.Sequential(
            # 使用多尺度卷积捕获噪声特征
            nn.Conv2d(in_channels, d_model // 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(d_model // 4, d_model // 4, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(d_model // 4, d_model // 4, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(d_model // 4, d_model // 8),
            nn.ReLU(),
            nn.Linear(d_model // 8, 1),
            nn.Sigmoid()  # 输出噪声水平 [0, 1]
        )
        
        # 基于噪声水平的门控特征提取器
        self.gated_encoder = nn.ModuleList([
            # 弱去噪分支（保留细节）
            nn.Sequential(
                nn.Conv2d(in_channels, d_model, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(d_model, d_model, kernel_size=3, padding=1)
            ),
            # 中等去噪分支
            nn.Sequential(
                nn.Conv2d(in_channels, d_model, kernel_size=5, padding=2),
                nn.ReLU(),
                nn.Conv2d(d_model, d_model, kernel_size=5, padding=2)
            ),
            # 强去噪分支（强抑制）
            nn.Sequential(
                nn.Conv2d(in_channels, d_model, kernel_size=7, padding=3),
                nn.ReLU(),
                nn.Conv2d(d_model, d_model, kernel_size=7, padding=3)
            )
        ])
        
        # 门控权重生成网络
        self.gate_net = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 3),
            nn.Softmax(dim=-1)  # 3个分支的权重
        )
        
        # 输出投影
        self.output_proj = nn.Conv2d(d_model, in_channels, kernel_size=1)
    
    def forward(self, x):
        """
        x: (B, C, H, W)
        返回: 自适应去噪后的特征, 噪声水平, 门控权重
        """
        B, C, H, W = x.shape
        
        # === 步骤1: 估计噪声水平 ===
        noise_level = self.noise_estimator(x)  # (B, 1)
        
        # === 步骤2: 生成门控权重 ===
        gate_weights = self.gate_net(noise_level)  # (B, 3)
        
        # === 步骤3: 多分支特征提取 ===
        branch_features = []
        for branch in self.gated_encoder:
            feat = branch(x)  # (B, d_model, H, W)
            branch_features.append(feat)
        
        # === 步骤4: 自适应加权融合 ===
        # 根据噪声水平动态选择分支
        fused_features = torch.zeros_like(branch_features[0])
        for i, feat in enumerate(branch_features):
            weight = gate_weights[:, i:i+1, None, None]  # (B, 1, 1, 1)
            fused_features = fused_features + feat * weight
        
        # === 步骤5: 输出投影 ===
        output = self.output_proj(fused_features)  # (B, C, H, W)
        
        # 残差连接（基于噪声水平自适应调整）
        residual_weight = 1.0 - noise_level.unsqueeze(-1).unsqueeze(-1)  # 低噪声保留更多原始信号
        output = output + x * residual_weight
        
        return output, noise_level, gate_weights


class LearnableWaveletDecomposition2D(nn.Module):
    """可学习的2D多尺度小波分解 - 创新点1"""
    
    def __init__(self, level=3, init_channels=16):
        super().__init__()
        self.level = level
        
        # 2D低通和高通滤波器
        self.low_pass_filters = nn.ModuleList()
        self.high_pass_filters = nn.ModuleList()
        
        for i in range(level):
            kernel_size = 2 ** (i + 2)  # 4, 8, 16
            # 低通滤波器
            self.low_pass_filters.append(
                nn.Conv2d(1, 1, kernel_size=kernel_size, stride=2,
                         padding=kernel_size//2, bias=False)
            )
            # 高通滤波器
            self.high_pass_filters.append(
                nn.Conv2d(1, 1, kernel_size=kernel_size, stride=2,
                         padding=kernel_size//2, bias=False)
            )
        
        self._init_wavelet_filters()
    
    def _init_wavelet_filters(self):
        """初始化为2D小波基"""
        for low_filter, high_filter in zip(self.low_pass_filters, self.high_pass_filters):
            kernel_size = low_filter.weight.shape[-1]
            
            with torch.no_grad():
                # 低通：平滑
                low_filter.weight.fill_(1.0 / kernel_size)
                
                # 高通：边缘检测
                high_weight = torch.zeros_like(high_filter.weight)
                half = kernel_size // 2
                high_weight[..., :half, :] = 1.0 / half
                high_weight[..., half:, :] = -1.0 / half
                high_filter.weight.copy_(high_weight)
    
    def forward(self, x):
        """
        2D多级分解
        x: (B, 1, H, W)
        返回: List of (approx, detail) tuples
        """
        coeffs = []
        current = x
        
        for i in range(self.level):
            approx = self.low_pass_filters[i](current)
            detail = self.high_pass_filters[i](current)
            coeffs.append((approx, detail))
            current = approx
        
        return coeffs


class MultiHeadSelfAttention2D(nn.Module):
    """2D多头自注意力机制"""
    
    def __init__(self, d_model, num_heads=8, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        """
        x: (B, H*W, D)
        """
        B, N, D = x.shape
        
        qkv = self.qkv_proj(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -65000)
        
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        out = attn @ v
        out = out.transpose(1, 2).reshape(B, N, D)
        out = self.out_proj(out)
        
        return out, attn


class DynamicSparseAttention2D(nn.Module):
    """2D动态稀疏注意力 - 创新点6"""
    
    def __init__(self, d_model, num_heads=8, dropout=0.1, sparsity_ratio=0.3,
                 local_window=(5, 5)):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.sparsity_ratio = sparsity_ratio
        self.local_window = local_window
        self.scale = self.head_dim ** -0.5
        
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
        # 重要性评分网络
        self.importance_scorer = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, 1)
        )
    
    def forward(self, x, mask=None):
        """
        x: (B, H*W, D)
        """
        B, N, D = x.shape
        
        # 计算重要性分数
        importance = self.importance_scorer(x).squeeze(-1)  # (B, N)
        
        # 选择top-k重要位置
        k = max(1, int(N * self.sparsity_ratio))
        _, top_indices = torch.topk(importance, k, dim=-1)
        
        # 创建稀疏mask
        sparse_mask = torch.zeros(B, N, device=x.device, dtype=torch.bool)
        sparse_mask.scatter_(1, top_indices, True)
        
        # 标准注意力
        qkv = self.qkv_proj(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        # 应用稀疏mask
        sparse_mask_4d = sparse_mask.unsqueeze(1).unsqueeze(2)
        attn = attn.masked_fill(~sparse_mask_4d, -65000)
        
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        out = attn @ v
        out = out.transpose(1, 2).reshape(B, N, D)
        out = self.out_proj(out)
        
        return out, attn, importance


class FrequencyAwareGating2D(nn.Module):
    """2D频率感知门控 - 创新点7"""
    
    def __init__(self, num_bands=4, d_model=64):
        super().__init__()
        self.num_bands = num_bands
        
        # 频带门控网络
        self.band_gates = nn.Sequential(
            nn.Linear(num_bands, num_bands * 2),
            nn.ReLU(),
            nn.Linear(num_bands * 2, num_bands),
            nn.Sigmoid()
        )
        
        # 频带处理器
        self.band_processors = nn.ModuleList([
            nn.Conv2d(1, d_model, kernel_size=3, padding=1)
            for _ in range(num_bands)
        ])
        
        self.fusion = nn.Conv2d(d_model * num_bands, d_model, kernel_size=1)
    
    def forward(self, x):
        """
        x: (B, 1, H, W)
        """
        B, C, H, W = x.shape
        
        # 2D FFT (使用float32避免cuFFT的half precision限制)
        original_dtype = x.dtype
        x_float32 = x.float()
        x_fft = torch.fft.rfft2(x_float32, dim=(-2, -1))
        x_mag = torch.abs(x_fft)
        
        # 将频谱分成多个频带
        freq_h = x_mag.shape[-2]
        freq_w = x_mag.shape[-1]
        band_h = freq_h // self.num_bands
        
        band_energies = []
        for i in range(self.num_bands):
            start_h = i * band_h
            end_h = (i + 1) * band_h if i < self.num_bands - 1 else freq_h
            band_energy = torch.mean(x_mag[..., start_h:end_h, :], dim=(-2, -1))  # (B, C)
            band_energies.append(band_energy)
        
        band_energies = torch.stack(band_energies, dim=-1)  # (B, C, num_bands)
        band_energies = band_energies.squeeze(1)  # (B, num_bands) - 移除channel维度
        
        # 计算门控权重
        gate_weights = self.band_gates(band_energies)  # (B, num_bands)
        
        # 处理每个频带
        band_features = []
        for i in range(self.num_bands):
            feat = self.band_processors[i](x)  # (B, d_model, H, W)
            weight = gate_weights[:, i:i+1].unsqueeze(-1).unsqueeze(-1)  # (B, 1, 1, 1)
            band_features.append(feat * weight)  # 广播到 (B, d_model, H, W)
        
        # 融合
        fused = torch.cat(band_features, dim=1)  # (B, d_model*num_bands, H, W)
        output = self.fusion(fused)  # (B, d_model, H, W)
        
        return output, gate_weights


class NoiseEstimator2D(nn.Module):
    """2D噪声估计器 - 创新点8"""
    
    def __init__(self, num_scales=3, d_model=64):
        super().__init__()
        self.num_scales = num_scales
        
        # 多尺度噪声统计估计
        self.scale_estimators = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(1, d_model, kernel_size=2**(i+2), 
                         stride=2**(i+1), padding=2**(i+1)),
                nn.ReLU(),
                nn.Conv2d(d_model, d_model, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(d_model, 2)  # mean和std
            )
            for i in range(num_scales)
        ])
        
        # 全局噪声水平估计
        self.global_estimator = nn.Sequential(
            nn.Conv2d(1, d_model, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(d_model, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """
        x: (B, 1, H, W)
        返回: 噪声估计, 噪声特征, 全局噪声水平
        """
        # 多尺度噪声统计
        noise_stats = []
        for estimator in self.scale_estimators:
            stats = estimator(x)  # (B, 2)
            noise_stats.append(stats)
        
        noise_stats = torch.stack(noise_stats, dim=1)  # (B, num_scales, 2)
        
        # 全局噪声水平
        global_noise = self.global_estimator(x)  # (B, 1)
        
        # 构造噪声估计图
        noise_mean = torch.mean(noise_stats[..., 0], dim=1, keepdim=True)
        noise_estimate = x * 0 + noise_mean.unsqueeze(-1).unsqueeze(-1)
        
        return noise_estimate, noise_stats, global_noise


class ResidualDenseBlock2D(nn.Module):
    """2D残差密集块 - 创新点9"""
    
    def __init__(self, in_channels, growth_rate=16, num_layers=4, dropout=0.1):
        super().__init__()
        self.num_layers = num_layers
        
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(
                nn.Sequential(
                    nn.Conv2d(in_channels + i * growth_rate, growth_rate,
                             kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.Dropout2d(dropout)
                )
            )
        
        # 特征融合
        self.fusion = nn.Conv2d(in_channels + num_layers * growth_rate,
                               in_channels, kernel_size=1)
    
    def forward(self, x):
        """
        x: (B, C, H, W)
        """
        features = [x]
        
        for layer in self.layers:
            out = layer(torch.cat(features, dim=1))
            features.append(out)
        
        # 融合所有特征
        fused = self.fusion(torch.cat(features, dim=1))
        
        # 残差连接
        return x + fused


class CrossScaleAttention2D(nn.Module):
    """2D跨尺度注意力 - 创新点2"""
    
    def __init__(self, d_model, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
    
    def forward(self, x_query, x_key_value):
        """
        x_query: (B, N1, D)
        x_key_value: (B, N2, D)
        """
        B, N1, D = x_query.shape
        N2 = x_key_value.shape[1]
        
        q = self.q_proj(x_query).reshape(B, N1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x_key_value).reshape(B, N2, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x_key_value).reshape(B, N2, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        
        out = attn @ v
        out = out.transpose(1, 2).reshape(B, N1, D)
        out = self.out_proj(out)
        
        return out


class AdaptiveWaveletFusion2D(nn.Module):
    """2D自适应小波融合 - 创新点3"""
    
    def __init__(self, num_scales):
        super().__init__()
        self.scale_weights = nn.Parameter(torch.ones(num_scales))
        self.softmax = nn.Softmax(dim=0)
    
    def forward(self, scale_features):
        """
        scale_features: List of (B, H*W, D) tensors
        """
        weights = self.softmax(self.scale_weights)
        fused = sum(w * feat for w, feat in zip(weights, scale_features))
        return fused, weights


class TransformerBlock2D(nn.Module):
    """2D Transformer块"""
    
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1,
                 use_sparse_attention=False, sparsity_ratio=0.3):
        super().__init__()
        self.use_sparse_attention = use_sparse_attention
        
        if use_sparse_attention:
            self.attention = DynamicSparseAttention2D(d_model, num_heads, dropout,
                                                     sparsity_ratio)
        else:
            self.attention = MultiHeadSelfAttention2D(d_model, num_heads, dropout)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        x: (B, H*W, D)
        """
        # Self-attention
        if self.use_sparse_attention:
            attn_out, attn_weights, importance = self.attention(self.norm1(x))
        else:
            attn_out, attn_weights = self.attention(self.norm1(x))
            importance = None
        
        x = x + self.dropout(attn_out)
        
        # Feed-forward
        ff_out = self.feed_forward(self.norm2(x))
        x = x + ff_out
        
        return x, attn_weights


class WaveletScaleProcessor2D(nn.Module):
    """2D小波尺度处理器"""
    
    def __init__(self, d_model, num_heads, num_layers, d_ff, dropout=0.1,
                 use_sparse_attention=False, sparsity_ratio=0.3):
        super().__init__()
        self.d_model = d_model
        
        # 输入投影
        self.input_proj = nn.Conv2d(1, d_model, kernel_size=3, padding=1)
        
        # Transformer层
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock2D(d_model, num_heads, d_ff, dropout,
                             use_sparse_attention, sparsity_ratio)
            for _ in range(num_layers)
        ])
        
        # 输出投影
        self.output_proj = nn.Conv2d(d_model, 1, kernel_size=3, padding=1)
    
    def forward(self, x):
        """
        x: (B, 1, H, W)
        """
        B, C, H, W = x.shape
        
        # 输入投影
        x_proj = self.input_proj(x)  # (B, d_model, H, W)
        
        # 转换为序列格式
        x_seq = x_proj.flatten(2).transpose(1, 2)  # (B, H*W, d_model)
        
        # Transformer处理
        attn_weights_list = []
        for block in self.transformer_blocks:
            x_seq, attn_weights = block(x_seq)
            attn_weights_list.append(attn_weights)
        
        # 转回2D
        x_proj = x_seq.transpose(1, 2).reshape(B, self.d_model, H, W)
        
        # 输出投影
        output = self.output_proj(x_proj)  # (B, 1, H, W)
        
        return output, x_proj, attn_weights_list


class WaveletTransformerDenoiser2D(nn.Module):
    """完全2D处理的降噪模型 - 整合所有9个创新点 + FDA + NAAG"""
    
    def __init__(
        self,
        wavelet_level=3,
        d_model=64,
        num_heads=8,
        num_layers=4,
        d_ff=256,
        dropout=0.2,
        use_cross_scale=True,
        use_sparse_attention=True,
        sparsity_ratio=0.3,
        use_fag=True,
        fag_num_bands=4,
        use_noise_estimator=True,
        noise_num_scales=3,
        use_rdc=True,
        rdc_growth_rate=16,
        rdc_num_blocks=3,
        use_fda=True,
        fda_num_bands=3,
        use_naag=True
    ):
        super().__init__()
        
        self.wavelet_level = wavelet_level
        self.use_cross_scale = use_cross_scale
        self.use_fag = use_fag
        self.use_noise_estimator = use_noise_estimator
        self.use_rdc = use_rdc
        self.use_fda = use_fda
        self.use_naag = use_naag
        
        # === 新增：FDA和NAAG模块（优先级最高） ===
        if use_naag:
            self.naag = NoiseAwareAdaptiveGating(in_channels=1, d_model=d_model)
        
        if use_fda:
            self.fda = FrequencyDisentangledAttention(in_channels=1, d_model=d_model, 
                                                      num_freq_bands=fda_num_bands)
        
        # 创新点1: 可学习小波分解
        self.wavelet_decomp = LearnableWaveletDecomposition2D(level=wavelet_level)
        
        # 每个尺度的处理器
        num_scales = wavelet_level * 2
        self.scale_processors = nn.ModuleList([
            WaveletScaleProcessor2D(d_model, num_heads, num_layers, d_ff, dropout,
                                   use_sparse_attention, sparsity_ratio)
            for _ in range(num_scales)
        ])
        
        # 创新点2: 跨尺度注意力
        if use_cross_scale:
            self.cross_scale_attn = nn.ModuleList([
                CrossScaleAttention2D(d_model, num_heads=4)
                for _ in range(num_scales - 1)
            ])
        
        # 创新点3: 自适应融合
        self.adaptive_fusion = AdaptiveWaveletFusion2D(num_scales)
        
        # 融合后处理
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(d_model, 1, kernel_size=3, padding=1)
        )
        
        # 创新点4: 双路径 - 空域路径
        self.spatial_path = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=5, padding=2)
        )
        
        # 双路径融合
        self.dual_path_fusion = nn.Conv2d(2, 1, kernel_size=1)
        
        # 创新点7: 频率感知门控
        if use_fag:
            self.fag = FrequencyAwareGating2D(num_bands=fag_num_bands, d_model=d_model)
            self.fag_fusion = nn.Conv2d(d_model + 1, 1, kernel_size=1)
        
        # 创新点8: 噪声估计器
        if use_noise_estimator:
            self.noise_estimator = NoiseEstimator2D(num_scales=noise_num_scales,
                                                    d_model=d_model)
        
        # 创新点9: 残差密集连接
        if use_rdc:
            self.rdc_blocks = nn.ModuleList([
                ResidualDenseBlock2D(d_model, growth_rate=rdc_growth_rate, dropout=dropout)
                for _ in range(rdc_num_blocks)
            ])
    
    def forward(self, x):
        """
        x: (B, 1, H, W)
        返回: 降噪后的信号
        """
        B, C, H, W = x.shape
        
        # === 新增：Step 0 - NAAG噪声感知预处理 ===
        naag_output = None
        noise_level = None
        naag_gate_weights = None
        if self.use_naag:
            naag_output, noise_level, naag_gate_weights = self.naag(x)
            # 使用NAAG的输出作为后续处理的输入
            x_processed = naag_output
        else:
            x_processed = x
        
        # === 新增：Step 0.5 - FDA频率解耦 ===
        fda_output = None
        freq_band_weights = None
        if self.use_fda:
            fda_output, freq_band_weights = self.fda(x_processed)
            x_processed = fda_output
        
        # 创新点8: 噪声估计
        if self.use_noise_estimator:
            noise_est, noise_stats, global_noise = self.noise_estimator(x_processed)
        
        # 创新点1: 小波分解（使用经过NAAG+FDA预处理的数据）
        coeffs = self.wavelet_decomp(x_processed)
        
        # 处理每个尺度
        processed_coeffs = []
        scale_features = []
        all_attn_weights = []
        
        idx = 0
        for approx, detail in coeffs:
            # 处理近似系数
            proc_approx, feat_approx, attn_approx = self.scale_processors[idx](approx)
            processed_coeffs.append((proc_approx, None))
            scale_features.append(feat_approx)
            all_attn_weights.append(attn_approx)
            idx += 1
            
            # 处理细节系数
            proc_detail, feat_detail, attn_detail = self.scale_processors[idx](detail)
            processed_coeffs[-1] = (proc_approx, proc_detail)
            scale_features.append(feat_detail)
            all_attn_weights.append(attn_detail)
            idx += 1
        
        # 创新点2: 跨尺度注意力
        if self.use_cross_scale and len(scale_features) > 1:
            enhanced_features = [scale_features[0]]
            for i in range(1, len(scale_features)):
                curr_feat = scale_features[i]
                base_feat = scale_features[0]
                
                # 转换为序列格式
                B_f, C_f, H_f, W_f = curr_feat.shape
                curr_seq = curr_feat.flatten(2).transpose(1, 2)  # (B, H*W, d_model)
                
                B_b, C_b, H_b, W_b = base_feat.shape
                base_seq = base_feat.flatten(2).transpose(1, 2)
                
                # 如果大小不同，插值
                if H_f * W_f != H_b * W_b:
                    base_feat_resized = F.interpolate(base_feat, size=(H_f, W_f),
                                                     mode='bilinear', align_corners=False)
                    base_seq = base_feat_resized.flatten(2).transpose(1, 2)
                
                # 跨尺度注意力
                cross_attn_out = self.cross_scale_attn[i-1](curr_seq, base_seq)
                enhanced = curr_seq + cross_attn_out
                
                # 转回2D
                enhanced_2d = enhanced.transpose(1, 2).reshape(B_f, C_f, H_f, W_f)
                enhanced_features.append(enhanced_2d)
            
            scale_features = enhanced_features
        
        # 创新点9: 残差密集连接
        if self.use_rdc:
            for i in range(len(scale_features)):
                for rdc_block in self.rdc_blocks:
                    scale_features[i] = rdc_block(scale_features[i])
        
        # 创新点3: 自适应融合
        # 将所有特征调整到同一大小
        target_H, target_W = scale_features[0].shape[-2:]
        aligned_features = []
        for feat in scale_features:
            if feat.shape[-2:] != (target_H, target_W):
                feat = F.interpolate(feat, size=(target_H, target_W),
                                   mode='bilinear', align_corners=False)
            # 转为序列
            feat_seq = feat.flatten(2).transpose(1, 2)  # (B, H*W, d_model)
            aligned_features.append(feat_seq)
        
        fused_features, fusion_weights = self.adaptive_fusion(aligned_features)
        
        # 转回2D
        fused_2d = fused_features.transpose(1, 2).reshape(B, -1, target_H, target_W)
        
        # 调整到原始大小
        if (target_H, target_W) != (H, W):
            fused_2d = F.interpolate(fused_2d, size=(H, W),
                                    mode='bilinear', align_corners=False)
        
        # 频域路径输出
        wavelet_output = self.fusion_conv(fused_2d)
        
        # 创新点4: 空域路径
        spatial_output = self.spatial_path(x)
        
        # 双路径融合
        combined = torch.cat([wavelet_output, spatial_output], dim=1)
        output = self.dual_path_fusion(combined)
        
        # 创新点7: 频率感知门控
        fag_gate_weights = None
        if self.use_fag:
            fag_output, fag_gate_weights = self.fag(output)
            combined_fag = torch.cat([output, fag_output], dim=1)
            output = self.fag_fusion(combined_fag)
        
        # 创建输出字典（包含所有中间结果）
        aux_outputs = {
            'attn_weights': all_attn_weights,
            'naag_noise_level': noise_level,
            'naag_gate_weights': naag_gate_weights,
            'fda_freq_bands': freq_band_weights,
            'fag_gate_weights': fag_gate_weights,
            'naag_output': naag_output,
            'fda_output': fda_output,
            'features': fused_2d,  # 多尺度融合后的中间特征 (B, d_model, H, W)
        }
        
        return output, aux_outputs


class BlindSpotNetwork2D(nn.Module):
    """2D Blind-spot网络 - 创新点5"""
    
    def __init__(self, base_model, mask_ratio=0.05):
        super().__init__()
        self.model = base_model
        self.mask_ratio = mask_ratio
    
    def forward(self, x):
        """
        x: (B, 1, H, W)
        """
        if self.training and self.mask_ratio > 0:
            B, C, H, W = x.shape
            mask = torch.ones_like(x)
            
            # 随机mask
            num_masked = int(H * W * self.mask_ratio)
            for b in range(B):
                masked_indices = torch.randperm(H * W)[:num_masked]
                mask_2d = mask[b, 0].flatten()
                mask_2d[masked_indices] = 0
                mask[b, 0] = mask_2d.reshape(H, W)
            
            x_masked = x * mask
            output, attn_weights = self.model(x_masked)
            return output, attn_weights, mask
        else:
            output, attn_weights = self.model(x)
            return output, attn_weights, None


def create_model_2d(config):
    """创建2D模型"""
    model_type = config.get('model_type', 'advanced')

    if model_type == 'usl':
        # 延迟导入，避免不必要依赖
        from model_usl import create_usl_model
        return create_usl_model(config)

    # 兼容旧配置键名
    wavelet_level = config.get('wavelet_level', config.get('wavelet_levels', 3))
    use_fag = config.get('use_fag', config.get('use_frequency_gating', True))
    fag_num_bands = config.get('fag_num_bands', config.get('num_freq_bands', 4))
    noise_num_scales = config.get('noise_num_scales', config.get('noise_scales', 3))

    base_model = WaveletTransformerDenoiser2D(
        wavelet_level=wavelet_level,
        d_model=config.get('d_model', 64),
        num_heads=config.get('num_heads', 8),
        num_layers=config.get('num_layers', 4),
        d_ff=config.get('d_ff', 256),
        dropout=config.get('dropout', 0.2),
        use_cross_scale=config.get('use_cross_scale', True),
        use_sparse_attention=config.get('use_sparse_attention', True),
        sparsity_ratio=config.get('sparsity_ratio', 0.3),
        use_fag=use_fag,
        fag_num_bands=fag_num_bands,
        use_noise_estimator=config.get('use_noise_estimator', True),
        noise_num_scales=noise_num_scales,
        use_rdc=config.get('use_rdc', True),
        rdc_growth_rate=config.get('rdc_growth_rate', 16),
        rdc_num_blocks=config.get('rdc_num_blocks', 3),
        use_fda=config.get('use_fda', True),
        fda_num_bands=config.get('fda_num_bands', 3),
        use_naag=config.get('use_naag', True)
    )

    # 创新点5: Blind-spot策略
    model = BlindSpotNetwork2D(base_model, mask_ratio=config.get('mask_ratio', 0.05))

    return model


if __name__ == "__main__":
    print("=" * 70)
    print("Testing 2D Wavelet-Transformer Denoiser with ALL innovations + FDA + NAAG")
    print("=" * 70)
    
    from config_2d import MODEL_2D_CONFIG
    
    # 启用新模块
    MODEL_2D_CONFIG['use_fda'] = True
    MODEL_2D_CONFIG['fda_num_bands'] = 3
    MODEL_2D_CONFIG['use_naag'] = True
    
    model = create_model_2d(MODEL_2D_CONFIG)
    model.eval()
    
    # 测试
    x = torch.randn(2, 1, 24, 24)
    print(f"\nInput shape: {x.shape}")
    
    with torch.no_grad():
        output, aux_outputs = model(x)
    
    print(f"Output shape: {output.shape}")
    print(f"\n=== Auxiliary Outputs ===")
    print(f"Number of attention weight sets: {len(aux_outputs['attn_weights'])}")
    if aux_outputs['naag_noise_level'] is not None:
        print(f"NAAG noise level: {aux_outputs['naag_noise_level'].shape}")
        print(f"  - Mean noise level: {aux_outputs['naag_noise_level'].mean().item():.4f}")
    if aux_outputs['naag_gate_weights'] is not None:
        print(f"NAAG gate weights: {aux_outputs['naag_gate_weights'].shape}")
        print(f"  - Weights: {aux_outputs['naag_gate_weights'][0].cpu().numpy()}")
    if aux_outputs['fda_freq_bands'] is not None:
        print(f"FDA frequency band weights: {aux_outputs['fda_freq_bands'].shape}")
        print(f"  - Weights: {aux_outputs['fda_freq_bands'][0].cpu().numpy()}")
    
    # 参数统计
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n{'='*70}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"{'='*70}")
    print("\nInnovations implemented (All in 2D):")
    print("  1. ✓ Multi-scale learnable wavelet decomposition (2D)")
    print("  2. ✓ Cross-scale attention mechanism (2D)")
    print("  3. ✓ Adaptive wavelet feature fusion (2D)")
    print("  4. ✓ Dual-path (frequency + spatial) processing (2D)")
    print("  5. ✓ Blind-spot unsupervised training strategy (2D)")
    print("  6. ✓ Dynamic Sparse Attention (2D)")
    print("  7. ✓ Frequency-Aware Gating (2D FFT)")
    print("  8. ✓ Noise Estimator Module (2D)")
    print("  9. ✓ Residual Dense Connection (2D)")
    print(" 10. ✓ FDA - Frequency Disentangled Attention (NEW!)")
    print(" 11. ✓ NAAG - Noise-Aware Adaptive Gating (NEW!)")
    print("=" * 70)
