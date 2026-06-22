"""
完全2D处理的工具函数
基于USL DIP的数据加载策略，完全采用2D处理
新增：FDA + NAAG可视化工具
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy import signal
from scipy.ndimage import gaussian_filter
import os

try:
    from skimage.metrics import structural_similarity as _skimage_ssim
    HAS_SKIMAGE_SSIM = True
except Exception:
    HAS_SKIMAGE_SSIM = False


def cseis():
    """创建地震数据可视化的色标 (seismic colormap)"""
    colors = [(0, 0, 1), (1, 1, 1), (1, 0, 0)]  # Blue -> White -> Red
    n_bins = 256
    cmap_name = 'seismic'
    return LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)


def yc_patch(data, w1, w2, z1, z2):
    """
    2D Patch提取（参考USL DIP策略）
    
    Args:
        data: 2D numpy array (n1, n2)
        w1, w2: patch大小 (高度, 宽度)
        z1, z2: stride步长
    
    Returns:
        patches: (num_patches, w1, w2)
    """
    n1, n2 = data.shape
    
    # 计算patch数量
    num_patches_h = (n1 - w1) // z1 + 1
    num_patches_w = (n2 - w2) // z2 + 1
    num_patches = num_patches_h * num_patches_w
    
    # 提取patches
    patches = np.zeros((num_patches, w1, w2), dtype=data.dtype)
    
    idx = 0
    for i in range(0, n1 - w1 + 1, z1):
        for j in range(0, n2 - w2 + 1, z2):
            patches[idx] = data[i:i+w1, j:j+w2]
            idx += 1
    
    print(f"Extracted {num_patches} patches of size ({w1}, {w2})")
    print(f"Grid: {num_patches_h} × {num_patches_w}")
    
    return patches


def yc_patch_inv(patches, n1, n2, w1, w2, z1, z2):
    """
    2D Patch逆变换（加权平均重建）
    
    Args:
        patches: (num_patches, w1, w2)
        n1, n2: 原始数据大小
        w1, w2: patch大小
        z1, z2: stride步长
    
    Returns:
        reconstructed: (n1, n2)
    """
    # 初始化重建矩阵和权重矩阵
    reconstructed = np.zeros((n1, n2), dtype=patches.dtype)
    weight = np.zeros((n1, n2), dtype=np.float32)
    
    # 计算patch数量
    num_patches_h = (n1 - w1) // z1 + 1
    num_patches_w = (n2 - w2) // z2 + 1
    
    idx = 0
    for i in range(0, n1 - w1 + 1, z1):
        for j in range(0, n2 - w2 + 1, z2):
            reconstructed[i:i+w1, j:j+w2] += patches[idx]
            weight[i:i+w1, j:j+w2] += 1.0
            idx += 1
    
    # 加权平均
    weight[weight == 0] = 1.0  # 避免除零
    reconstructed = reconstructed / weight
    
    return reconstructed


class DASDataset2D(Dataset):
    """2D Patch DAS数据集 - 基于USL DIP策略"""
    
    def __init__(self, data_path, patch_size=(24, 24), stride=(6, 6), 
                 augment=True, normalize=True):
        """
        Args:
            data_path: 数据文件路径 (.npy)
            patch_size: patch大小 (H, W)
            stride: 滑动步长 (stride_h, stride_w)
            augment: 是否数据增强
            normalize: 是否归一化
        """
        # 加载数据
        self.data = np.load(data_path)
        if len(self.data.shape) != 2:
            raise ValueError(f"Expected 2D data, got shape {self.data.shape}")
        
        self.patch_size = patch_size
        self.stride = stride
        self.augment = augment
        self.normalize = normalize
        
        # 数据统计
        self.data_mean = np.mean(self.data)
        self.data_std = np.std(self.data)
        self.data_min = np.min(self.data)
        self.data_max = np.max(self.data)
        
        # 使用百分位数作为clip范围（更稳健）
        self.clip_min = np.percentile(self.data, 1)
        self.clip_max = np.percentile(self.data, 99)
        
        print(f"=" * 60)
        print(f"2D DAS Dataset Initialized")
        print(f"Data shape: {self.data.shape}")
        print(f"Value range: [{self.data_min:.2f}, {self.data_max:.2f}]")
        print(f"Mean: {self.data_mean:.4f}, Std: {self.data_std:.4f}")
        print(f"Clip range (1%-99%): [{self.clip_min:.2f}, {self.clip_max:.2f}]")
        
        # 提取所有patches
        self.patches = yc_patch(self.data, patch_size[0], patch_size[1], 
                               stride[0], stride[1])
        
        overlap_h = (patch_size[0] - stride[0]) / patch_size[0] * 100
        overlap_w = (patch_size[1] - stride[1]) / patch_size[1] * 100
        print(f"Patch size: {patch_size}, Stride: {stride}")
        print(f"Overlap: H={overlap_h:.1f}%, W={overlap_w:.1f}%")
        print(f"Total patches: {len(self.patches)}")
        print(f"=" * 60)
        
    def __len__(self):
        return len(self.patches)
    
    def __getitem__(self, idx):
        """获取一个2D patch"""
        patch = self.patches[idx].copy().astype(np.float32)
        
        # 数据增强
        if self.augment:
            # 随机翻转
            if np.random.rand() > 0.5:
                patch = np.flip(patch, axis=0).copy()
            if np.random.rand() > 0.5:
                patch = np.flip(patch, axis=1).copy()
            
            # 随机转置（90度旋转）
            if np.random.rand() > 0.5:
                patch = patch.T.copy()
            
            # 随机旋转（180度）
            if np.random.rand() > 0.5:
                patch = np.rot90(patch, k=2).copy()
            
            # 轻微随机缩放
            if np.random.rand() > 0.5:
                scale = np.random.uniform(0.95, 1.05)
                patch = patch * scale
        
        # 归一化
        if self.normalize:
            # Clip到合理范围
            patch = np.clip(patch, self.clip_min, self.clip_max)
            # Z-score归一化
            if self.data_std > 1e-6:
                patch = (patch - self.data_mean) / self.data_std
        
        # 转换为tensor (1, H, W)
        patch = torch.from_numpy(patch).unsqueeze(0)
        
        return patch


def add_noise_2d(clean_data, noise_level=0.1, noise_type='gaussian'):
    """
    添加2D噪声
    
    Args:
        clean_data: (B, 1, H, W)
        noise_level: 噪声水平
        noise_type: 噪声类型
    
    Returns:
        noisy_data: (B, 1, H, W)
    """
    if noise_type == 'gaussian':
        noise = torch.randn_like(clean_data) * noise_level
    elif noise_type == 'uniform':
        noise = (torch.rand_like(clean_data) - 0.5) * 2 * noise_level
    elif noise_type == 'speckle':
        noise = clean_data * torch.randn_like(clean_data) * noise_level
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")
    
    return clean_data + noise


def _compute_ssim_2d(pred, target):
    """Compute SSIM for 2D arrays with a safe fallback."""
    pred = pred.astype(np.float64)
    target = target.astype(np.float64)

    data_range = target.max() - target.min()
    if data_range <= 1e-12:
        return 1.0

    if HAS_SKIMAGE_SSIM:
        return float(_skimage_ssim(target, pred, data_range=data_range))

    # Fallback SSIM (Gaussian window)
    k1, k2 = 0.01, 0.03
    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2
    sigma = 1.5

    mu_x = gaussian_filter(pred, sigma)
    mu_y = gaussian_filter(target, sigma)
    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x2 = gaussian_filter(pred * pred, sigma) - mu_x2
    sigma_y2 = gaussian_filter(target * target, sigma) - mu_y2
    sigma_xy = gaussian_filter(pred * target, sigma) - mu_xy

    numerator = (2.0 * mu_xy + c1) * (2.0 * sigma_xy + c2)
    denominator = (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2)
    ssim_map = numerator / (denominator + 1e-12)
    return float(np.mean(ssim_map))


def calculate_metrics_2d(pred, target):
    """
    计算2D评估指标
    
    Args:
        pred: (B, 1, H, W) or (H, W)
        target: (B, 1, H, W) or (H, W)
    
    Returns:
        metrics: dict
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()
    
    # 确保是numpy数组
    pred = np.array(pred)
    target = np.array(target)
    
    # MSE
    mse = np.mean((pred - target) ** 2)
    
    # PSNR
    if mse > 1e-10:
        max_val = np.max(np.abs(target))
        psnr = 20 * np.log10(max_val / np.sqrt(mse)) if max_val > 0 else 0
    else:
        psnr = 100
    
    # SNR
    signal_power = np.mean(target ** 2)
    noise_power = mse
    if noise_power > 1e-10:
        snr = 10 * np.log10(signal_power / noise_power)
    else:
        snr = 100
    
    # MAE
    mae = np.mean(np.abs(pred - target))
    
    # Correlation
    pred_flat = pred.flatten()
    target_flat = target.flatten()
    if pred_flat.size < 2:
        correlation = 0.0
    else:
        correlation = float(np.corrcoef(pred_flat, target_flat)[0, 1])
        if not np.isfinite(correlation):
            correlation = 0.0

    # SSIM
    try:
        ssim = _compute_ssim_2d(pred, target)
    except Exception:
        ssim = 0.0
    
    return {
        'mse': mse,
        'psnr': psnr,
        'snr': snr,
        'mae': mae,
        'correlation': correlation,
        'ssim': ssim
    }


def calculate_no_ref_metrics(raw, denoised, weights=None, eps=1e-8):
    """
    无参考指标（真实数据场景）

    Args:
        raw: 原始含噪数据 (H, W)
        denoised: 去噪结果 (H, W)
        weights: dict, e.g. {'residual': 0.4, 'corr': 0.4, 'smooth': 0.2}
        eps: 数值稳定项

    Returns:
        dict: residual_energy_ratio, signal_corr_with_raw, smoothness_gain, no_ref_score
    """
    raw = np.array(raw)
    denoised = np.array(denoised)

    residual = raw - denoised
    residual_energy_ratio = float(np.sum(residual ** 2) / (np.sum(raw ** 2) + eps))

    raw_flat = raw.flatten()
    den_flat = denoised.flatten()
    if raw_flat.size < 2:
        corr = 0.0
    else:
        corr = float(np.corrcoef(raw_flat, den_flat)[0, 1])
        if np.isnan(corr):
            corr = 0.0

    # Smoothness gain: larger means denoised is smoother
    grad_raw = np.sum(np.abs(np.diff(raw, axis=0))) + np.sum(np.abs(np.diff(raw, axis=1)))
    grad_den = np.sum(np.abs(np.diff(denoised, axis=0))) + np.sum(np.abs(np.diff(denoised, axis=1)))
    smoothness_gain = float(grad_raw / (grad_den + eps))

    if weights is None:
        weights = {'residual': 0.4, 'corr': 0.4, 'smooth': 0.2}

    # Clip smoothness gain to avoid dominating the score
    smooth_clip = min(max(smoothness_gain, 0.0), 2.0)
    no_ref_score = (
        weights.get('residual', 0.4) * (1.0 - residual_energy_ratio)
        + weights.get('corr', 0.4) * corr
        + weights.get('smooth', 0.2) * smooth_clip
    )

    return {
        'residual_energy_ratio': residual_energy_ratio,
        'signal_corr_with_raw': corr,
        'smoothness_gain': smoothness_gain,
        'no_ref_score': float(no_ref_score),
    }


def get_dataloader_2d(data_path, batch_size=128, patch_size=(24, 24), 
                     stride=(6, 6), augment=True, shuffle=True, num_workers=0):
    """
    创建2D DataLoader
    
    Args:
        data_path: 数据路径
        batch_size: batch大小
        patch_size: patch大小
        stride: 步长
        augment: 是否增强
        shuffle: 是否打乱
        num_workers: 工作进程数
    
    Returns:
        dataloader: DataLoader
    """
    dataset = DASDataset2D(data_path, patch_size=patch_size, stride=stride, 
                          augment=augment)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    return dataloader


def plot_2d_comparison(original, denoised, noise=None, save_path=None, 
                      vmin=-30, vmax=30, figsize=(15, 5)):
    """
    可视化2D降噪结果（参考USL DIP风格）
    
    Args:
        original: (H, W) 原始数据
        denoised: (H, W) 去噪数据
        noise: (H, W) 噪声（可选）
        save_path: 保存路径
        vmin, vmax: 色标范围
        figsize: 图形大小
    """
    if isinstance(original, torch.Tensor):
        original = original.squeeze().detach().cpu().numpy()
    if isinstance(denoised, torch.Tensor):
        denoised = denoised.squeeze().detach().cpu().numpy()
    if noise is not None and isinstance(noise, torch.Tensor):
        noise = noise.squeeze().detach().cpu().numpy()
    
    # 创建图形
    n_plots = 3 if noise is not None else 2
    fig, axes = plt.subplots(1, n_plots, figsize=figsize)
    
    # 时间轴
    n1, n2 = original.shape
    time_interval = 0.001
    time = np.arange(0, n1) * time_interval
    
    # 绘图参数
    extent = (1, n2, time[-1], 0)
    
    # 原始数据
    im1 = axes[0].imshow(original, cmap=cseis(), vmin=vmin, vmax=vmax, 
                        aspect='auto', extent=extent)
    axes[0].set_xlabel("Trace", fontsize=12)
    axes[0].set_ylabel("Time (s)", fontsize=12)
    axes[0].set_title('Raw Data', fontsize=14, fontweight='bold')
    
    # 去噪数据
    im2 = axes[1].imshow(denoised, cmap=cseis(), vmin=vmin, vmax=vmax, 
                        aspect='auto', extent=extent)
    axes[1].set_xlabel("Trace", fontsize=12)
    axes[1].set_yticks([])
    axes[1].set_title('Denoised Data', fontsize=14, fontweight='bold')
    
    # 噪声
    if noise is not None:
        im3 = axes[2].imshow(noise, cmap=cseis(), vmin=vmin, vmax=vmax, 
                            aspect='auto', extent=extent)
        axes[2].set_xlabel("Trace", fontsize=12)
        axes[2].set_yticks([])
        axes[2].set_title('Removed Noise', fontsize=14, fontweight='bold')
    
    # 调整布局
    plt.tight_layout()
    
    # 添加colorbar
    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.905, 0.2, 0.015, 0.6])
    cb = plt.colorbar(im2, cax=cbar_ax)
    cb.ax.tick_params(labelsize=10)
    
    # 保存
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    
    return fig


def plot_ipynb_style(data, denoised, save_path=None, data_name='data',
                     vmin=-30, vmax=30, figsize=(10, 5)):
    """
    IPYNB风格的三栏可视化（完全复刻USL DIP效果）
    
    Args:
        data: (H, W) 原始含噪数据
        denoised: (H, W) 降噪后数据
        save_path: 保存路径
        data_name: 数据名称
        vmin, vmax: 色标范围
        figsize: 图形大小
    """
    # 转换为numpy
    if isinstance(data, torch.Tensor):
        data = data.squeeze().detach().cpu().numpy()
    if isinstance(denoised, torch.Tensor):
        denoised = denoised.squeeze().detach().cpu().numpy()
    
    # 数据维度
    num_samples = data.shape[0]
    num_samples_x = data.shape[1]
    time_interval = 0.001
    time = np.arange(0, num_samples) * time_interval
    
    # 标注位置
    mm = -0.03
    nn = 1.05
    
    # 创建图形
    fig = plt.figure(figsize=figsize)
    
    # 子图1: 原始数据 (Raw Data)
    ax1 = fig.add_subplot(1, 3, 1)
    im1 = ax1.imshow(
        data.T, cmap=cseis(), vmin=vmin, vmax=vmax, 
        aspect='auto', extent=(1, num_samples_x, time[-1], 0)
    )
    ax1.set_xlabel("Trace", fontsize=12)
    ax1.set_ylabel("Time (s)", fontsize=12)
    ax1.set_title('Raw Data', fontsize=14)
    ax1.annotate('(a)', xy=(mm, nn), xycoords='axes fraction', 
                fontsize=14, fontweight='bold', va='top')
    
    # 子图2: 降噪数据 (Denoised Data)
    ax2 = fig.add_subplot(1, 3, 2)
    im2 = ax2.imshow(
        denoised.T, cmap=cseis(), vmin=vmin, vmax=vmax, 
        aspect='auto', extent=(1, num_samples_x, time[-1], 0)
    )
    ax2.set_xlabel("Trace", fontsize=12)
    ax2.set_yticks([])  # 移除y轴刻度
    ax2.set_title('Denoised Data', fontsize=14)
    ax2.annotate('(b)', xy=(mm, nn), xycoords='axes fraction', 
                fontsize=14, fontweight='bold', va='top')
    
    # 子图3: 去除的噪声 (Removed Noise)
    ax3 = fig.add_subplot(1, 3, 3)
    im3 = ax3.imshow(
        (data - denoised).T, cmap=cseis(), vmin=vmin, vmax=vmax, 
        aspect='auto', extent=(1, num_samples_x, time[-1], 0)
    )
    ax3.set_xlabel("Trace", fontsize=12)
    ax3.set_yticks([])  # 移除y轴刻度
    ax3.set_title('Removed Noise', fontsize=14)
    ax3.annotate('(c)', xy=(mm, nn), xycoords='axes fraction', 
                fontsize=14, fontweight='bold', va='top')
    
    # 调整布局并添加colorbar
    plt.tight_layout()
    fig.subplots_adjust(right=0.9)  # 为colorbar留出空间
    
    # 添加colorbar（与IPYNB完全一致）
    cbar_ax = fig.add_axes([0.905, 0.2, 0.015, 0.6])  # [left, bottom, width, height]
    cb = plt.colorbar(im3, cax=cbar_ax)
    cb.ax.tick_params(labelsize=8)
    cbar_ticks = np.linspace(vmin, vmax, 5)  # 5个刻度
    cb.set_ticks(cbar_ticks)
    cb.set_ticklabels([f'{tick:.0f}' for tick in cbar_ticks])
    
    # 保存
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"✓ IPYNB-style figure saved to {save_path}")
    
    return fig


class EarlyStopping:
    """早停策略"""
    
    def __init__(self, patience=10, min_delta=0, verbose=True):
        """
        Args:
            patience: 容忍的epoch数
            min_delta: 最小改善量
            verbose: 是否打印信息
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

        # 返回是否触发早停，便于训练循环直接判断并中断
        return self.early_stop


def save_checkpoint(model, optimizer, epoch, loss, metrics, path):
    """保存checkpoint"""
    # 确保目录存在
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'metrics': metrics
    }, path)
    print(f"Checkpoint saved to {path}")


def load_checkpoint(model, optimizer, path):
    """加载checkpoint"""
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint['epoch'], checkpoint['loss'], checkpoint.get('metrics', {})


def plot_naag_analysis(noise_levels, gate_weights, save_path=None):
    """
    可视化NAAG模块的噪声估计和门控权重分析
    
    Args:
        noise_levels: (N,) 噪声水平数组
        gate_weights: (N, 3) 门控权重数组（弱/中/强去噪分支）
        save_path: 保存路径
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # 1. 噪声水平分布直方图
    ax1 = axes[0, 0]
    ax1.hist(noise_levels, bins=50, alpha=0.7, color='blue', edgecolor='black')
    ax1.set_xlabel('Estimated Noise Level', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('NAAG: Noise Level Distribution', fontsize=14, fontweight='bold')
    ax1.axvline(np.mean(noise_levels), color='red', linestyle='--', 
                label=f'Mean: {np.mean(noise_levels):.4f}')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # 2. 门控权重分布（箱线图）
    ax2 = axes[0, 1]
    branch_names = ['Weak\nDenoising', 'Medium\nDenoising', 'Strong\nDenoising']
    ax2.boxplot([gate_weights[:, 0], gate_weights[:, 1], gate_weights[:, 2]],
                labels=branch_names)
    ax2.set_ylabel('Gate Weight', fontsize=12)
    ax2.set_title('NAAG: Gate Weight Distribution', fontsize=14, fontweight='bold')
    ax2.grid(alpha=0.3)
    
    # 3. 噪声水平与门控权重的关系（散点图）
    ax3 = axes[1, 0]
    for i, (name, color) in enumerate(zip(branch_names, ['green', 'orange', 'red'])):
        ax3.scatter(noise_levels, gate_weights[:, i], alpha=0.5, 
                   s=10, color=color, label=name.replace('\n', ' '))
    ax3.set_xlabel('Estimated Noise Level', fontsize=12)
    ax3.set_ylabel('Gate Weight', fontsize=12)
    ax3.set_title('NAAG: Noise Level vs Gate Weights', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(alpha=0.3)
    
    # 4. 平均门控权重条形图
    ax4 = axes[1, 1]
    mean_weights = np.mean(gate_weights, axis=0)
    bars = ax4.bar(branch_names, mean_weights, color=['green', 'orange', 'red'], 
                   alpha=0.7, edgecolor='black')
    ax4.set_ylabel('Average Weight', fontsize=12)
    ax4.set_title('NAAG: Average Branch Selection', fontsize=14, fontweight='bold')
    ax4.set_ylim([0, 1])
    # 添加数值标签
    for bar, weight in zip(bars, mean_weights):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{weight:.3f}', ha='center', va='bottom', fontsize=10)
    ax4.grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"✓ NAAG analysis saved to {save_path}")
    
    return fig


def plot_fda_analysis(freq_band_weights, save_path=None):
    """
    可视化FDA模块的频率band权重分析
    
    Args:
        freq_band_weights: (N, num_bands) 频率band权重数组
        save_path: 保存路径
    """
    num_bands = freq_band_weights.shape[1]
    band_names = [f'Band {i+1}\n(Freq {i})' for i in range(num_bands)]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # 1. 频率band权重分布（箱线图）
    ax1 = axes[0]
    ax1.boxplot([freq_band_weights[:, i] for i in range(num_bands)],
                labels=band_names)
    ax1.set_ylabel('Band Weight', fontsize=12)
    ax1.set_title('FDA: Frequency Band Weight Distribution', 
                 fontsize=14, fontweight='bold')
    ax1.grid(alpha=0.3)
    
    # 2. 平均频率band权重
    ax2 = axes[1]
    mean_weights = np.mean(freq_band_weights, axis=0)
    colors = plt.cm.viridis(np.linspace(0, 1, num_bands))
    bars = ax2.bar(band_names, mean_weights, color=colors, 
                   alpha=0.7, edgecolor='black')
    ax2.set_ylabel('Average Weight', fontsize=12)
    ax2.set_title('FDA: Average Frequency Band Selection', 
                 fontsize=14, fontweight='bold')
    ax2.set_ylim([0, max(mean_weights) * 1.2])
    # 添加数值标签
    for bar, weight in zip(bars, mean_weights):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{weight:.3f}', ha='center', va='bottom', fontsize=10)
    ax2.grid(alpha=0.3, axis='y')
    
    # 3. 频率band权重热图（样本 × band）
    ax3 = axes[2]
    # 只显示前100个样本（避免太密）
    display_samples = min(100, freq_band_weights.shape[0])
    im = ax3.imshow(freq_band_weights[:display_samples].T, 
                    aspect='auto', cmap='viridis', interpolation='nearest')
    ax3.set_xlabel('Sample Index', fontsize=12)
    ax3.set_ylabel('Frequency Band', fontsize=12)
    ax3.set_title(f'FDA: Band Weights Heatmap\n(First {display_samples} samples)', 
                 fontsize=14, fontweight='bold')
    ax3.set_yticks(range(num_bands))
    ax3.set_yticklabels([f'Band {i+1}' for i in range(num_bands)])
    plt.colorbar(im, ax=ax3, label='Weight')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"✓ FDA analysis saved to {save_path}")
    
    return fig


def visualize_fda_naag_combined(noise_levels, gate_weights, freq_band_weights, 
                                save_path=None):
    """
    综合可视化FDA和NAAG的协同工作
    
    Args:
        noise_levels: (N,) NAAG估计的噪声水平
        gate_weights: (N, 3) NAAG门控权重
        freq_band_weights: (N, num_bands) FDA频率权重
        save_path: 保存路径
    """
    # 确保所有数组维度正确
    noise_levels = np.squeeze(noise_levels)  # 移除所有size=1的维度
    gate_weights = np.squeeze(gate_weights)
    if gate_weights.ndim == 1:  # 如果只有一个样本
        gate_weights = gate_weights.reshape(1, -1)
    freq_band_weights = np.squeeze(freq_band_weights)
    if freq_band_weights.ndim == 1:  # 如果只有一个样本
        freq_band_weights = freq_band_weights.reshape(1, -1)
    
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 顶部：NAAG分析
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(noise_levels, bins=50, alpha=0.7, color='blue', edgecolor='black')
    ax1.set_xlabel('Noise Level', fontsize=11)
    ax1.set_ylabel('Count', fontsize=11)
    ax1.set_title('NAAG: Noise Distribution', fontsize=12, fontweight='bold')
    ax1.axvline(np.mean(noise_levels), color='red', linestyle='--', linewidth=2)
    
    ax2 = fig.add_subplot(gs[0, 1])
    branch_names = ['Weak', 'Medium', 'Strong']
    mean_gate = np.mean(gate_weights, axis=0)
    ax2.bar(branch_names, mean_gate, color=['green', 'orange', 'red'], alpha=0.7)
    ax2.set_ylabel('Avg Weight', fontsize=11)
    ax2.set_title('NAAG: Branch Selection', fontsize=12, fontweight='bold')
    ax2.set_ylim([0, 1])
    
    ax3 = fig.add_subplot(gs[0, 2])
    for i, (name, color) in enumerate(zip(branch_names, ['green', 'orange', 'red'])):
        ax3.scatter(noise_levels, gate_weights[:, i], alpha=0.3, 
                   s=5, color=color, label=name)
    ax3.set_xlabel('Noise Level', fontsize=11)
    ax3.set_ylabel('Gate Weight', fontsize=11)
    ax3.set_title('NAAG: Noise vs Gates', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=9)
    
    # 中部：FDA分析
    num_bands = freq_band_weights.shape[1]
    
    ax4 = fig.add_subplot(gs[1, 0])
    mean_freq = np.mean(freq_band_weights, axis=0)
    colors = plt.cm.viridis(np.linspace(0, 1, num_bands))
    ax4.bar(range(num_bands), mean_freq, color=colors, alpha=0.7)
    ax4.set_xlabel('Frequency Band', fontsize=11)
    ax4.set_ylabel('Avg Weight', fontsize=11)
    ax4.set_title('FDA: Band Selection', fontsize=12, fontweight='bold')
    ax4.set_xticks(range(num_bands))
    ax4.set_xticklabels([f'B{i+1}' for i in range(num_bands)])
    
    ax5 = fig.add_subplot(gs[1, 1:])
    display_samples = min(100, freq_band_weights.shape[0])
    im = ax5.imshow(freq_band_weights[:display_samples].T, 
                    aspect='auto', cmap='viridis', interpolation='nearest')
    ax5.set_xlabel('Sample Index', fontsize=11)
    ax5.set_ylabel('Freq Band', fontsize=11)
    ax5.set_title(f'FDA: Band Weight Heatmap ({display_samples} samples)', 
                 fontsize=12, fontweight='bold')
    ax5.set_yticks(range(num_bands))
    ax5.set_yticklabels([f'B{i+1}' for i in range(num_bands)])
    plt.colorbar(im, ax=ax5)
    
    # 底部：FDA + NAAG联合分析
    ax6 = fig.add_subplot(gs[2, :])
    # 将样本按噪声水平排序
    sorted_idx = np.argsort(noise_levels)
    sorted_noise = noise_levels[sorted_idx]
    sorted_freq = freq_band_weights[sorted_idx]
    
    # 绘制噪声水平曲线
    ax6_twin = ax6.twinx()
    ax6.plot(sorted_noise, color='red', linewidth=2, alpha=0.7, label='NAAG Noise Level')
    ax6.set_xlabel('Sample (sorted by noise level)', fontsize=11)
    ax6.set_ylabel('Noise Level (NAAG)', fontsize=11, color='red')
    ax6.tick_params(axis='y', labelcolor='red')
    ax6.set_title('FDA + NAAG: Joint Analysis (Sorted by Noise)', 
                 fontsize=12, fontweight='bold')
    
    # 绘制频率band权重
    for i in range(num_bands):
        ax6_twin.plot(sorted_freq[:, i], alpha=0.6, label=f'FDA Band {i+1}')
    ax6_twin.set_ylabel('Frequency Band Weight (FDA)', fontsize=11)
    ax6_twin.legend(loc='upper right', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"✓ FDA+NAAG combined analysis saved to {save_path}")
    
    return fig
