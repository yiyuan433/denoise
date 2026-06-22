"""
传统去噪方法集合 + 深度学习与传统方法联合去噪
支持的方法：
  1. 高斯滤波 (Gaussian Filter)
  2. 中值滤波 (Median Filter)
  3. 小波软阈值去噪 (Wavelet Soft-Threshold)
  4. FK滤波 (FK Filter, 频率-波数域滤波)
  5. 带通滤波 (Bandpass Filter)
  6. SVD秩截断 (SVD Rank-Reduction)
  7. 总变差去噪 (Total Variation, TV)
  8. Wiener滤波 (Wiener Filter)
"""

import numpy as np
from scipy import signal
from scipy.ndimage import gaussian_filter, median_filter, uniform_filter
from scipy.signal import butter, sosfilt

try:
    import pywt
    HAS_PYWT = True
except ImportError:
    HAS_PYWT = False
    print("⚠ pywt未安装，小波去噪不可用。运行: pip install PyWavelets")

try:
    from skimage.restoration import denoise_tv_chambolle
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False
    print("⚠ skimage未安装，TV去噪使用内置实现。运行: pip install scikit-image")


# ==============================================================================
# 传统去噪方法
# ==============================================================================

def gaussian_denoise(data: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """
    高斯滤波去噪
    Args:
        data: 2D numpy array (H, W)
        sigma: 高斯核标准差，越大模糊程度越高
    Returns:
        denoised: (H, W)
    """
    return gaussian_filter(data.astype(np.float64), sigma=sigma).astype(np.float32)


def median_denoise(data: np.ndarray, size: int = 3) -> np.ndarray:
    """
    中值滤波去噪（对脉冲噪声效果好）
    Args:
        data: 2D numpy array (H, W)
        size: 滤波窗口大小
    Returns:
        denoised: (H, W)
    """
    return median_filter(data.astype(np.float64), size=size).astype(np.float32)


def wavelet_denoise(data: np.ndarray, wavelet: str = 'db4', level: int = 3,
                    threshold_mode: str = 'soft', sigma: float = None) -> np.ndarray:
    """
    小波阈值去噪（软/硬阈值）
    Args:
        data: 2D numpy array (H, W)
        wavelet: 小波基，常用 'db4', 'sym8', 'coif4', 'bior6.8'
        level: 分解层数
        threshold_mode: 'soft' 或 'hard'
        sigma: 噪声标准差估计，None则自动估计
    Returns:
        denoised: (H, W)
    """
    if not HAS_PYWT:
        print("  [wavelet_denoise] pywt不可用，返回原始数据")
        return data.copy()

    data64 = data.astype(np.float64)

    # 2D小波分解
    coeffs = pywt.wavedec2(data64, wavelet=wavelet, level=level)

    # 自动估计噪声标准差（使用最细尺度高频系数的MAD估计）
    if sigma is None:
        detail_coeffs = coeffs[-1]  # (cH, cV, cD) at finest level
        all_detail = np.concatenate([c.ravel() for c in detail_coeffs])
        sigma = np.median(np.abs(all_detail)) / 0.6745

    # 计算阈值（BayesShrink/VisuShrink: lambda = sigma * sqrt(2 * log(N)))
    n_pixels = data.size
    threshold = sigma * np.sqrt(2 * np.log(n_pixels))

    # 应用阈值（跳过近似系数 coeffs[0]）
    new_coeffs = [coeffs[0]]
    for detail_level in coeffs[1:]:
        new_detail = tuple(
            pywt.threshold(c, threshold, mode=threshold_mode)
            for c in detail_level
        )
        new_coeffs.append(new_detail)

    # 2D小波重建
    denoised = pywt.waverec2(new_coeffs, wavelet=wavelet)

    # 裁剪到原始尺寸（小波填充可能导致尺寸略大）
    h, w = data.shape
    denoised = denoised[:h, :w]

    return denoised.astype(np.float32)


def fk_filter_denoise(data: np.ndarray, v_cut: float = 0.6,
                      taper_width: float = 0.1) -> np.ndarray:
    """
    FK滤波去噪（频率-波数域低通滤波，保留低波数有效信号）
    Args:
        data: 2D numpy array (时间 × 道数, H × W)
        v_cut: 截止视速度归一化值 (0~1)，越小保留的倾斜程度越低
        taper_width: 软化过渡带宽度（归一化）
    Returns:
        denoised: (H, W)
    """
    data64 = data.astype(np.float64)
    h, w = data.shape

    # 2D FFT（时间轴+空间轴）
    FK = np.fft.fft2(data64)
    FK_shift = np.fft.fftshift(FK)

    # 构建视速度锥形掩膜
    # 频率轴归一化索引 [-0.5, 0.5]
    f_idx = np.fft.fftshift(np.fft.fftfreq(h))   # (H,)
    k_idx = np.fft.fftshift(np.fft.fftfreq(w))   # (W,)
    K, F = np.meshgrid(k_idx, f_idx)

    # 视速度掩膜：|k| <= v_cut * |f| （锥形区域）
    # 避免除零：仅在|f|>0时判断
    with np.errstate(divide='ignore', invalid='ignore'):
        cone = np.where(np.abs(F) > 1e-12, np.abs(K) / (np.abs(F) + 1e-12), 0.0)

    # 硬截止 + 余弦平滑过渡带
    mask = np.ones_like(cone)
    v_low = v_cut - taper_width / 2
    v_high = v_cut + taper_width / 2
    in_taper = (cone > v_low) & (cone < v_high)
    mask[cone >= v_high] = 0.0
    mask[in_taper] = 0.5 * (1 + np.cos(np.pi * (cone[in_taper] - v_low) / taper_width))

    # 应用掩膜并逆变换
    FK_filt = np.fft.ifftshift(FK_shift * mask)
    denoised = np.real(np.fft.ifft2(FK_filt))

    return denoised.astype(np.float32)


def bandpass_denoise(data: np.ndarray, lowcut: float = 0.05,
                     highcut: float = 0.45, order: int = 4) -> np.ndarray:
    """
    Butterworth带通滤波（沿时间轴，即行方向）
    Args:
        data: 2D numpy array (H, W)，H为时间轴
        lowcut: 归一化低截止频率 (0~0.5)
        highcut: 归一化高截止频率 (0~0.5)
        order: 滤波器阶数
    Returns:
        denoised: (H, W)
    """
    sos = butter(order, [lowcut, highcut], btype='bandpass',
                 output='sos', fs=1.0)  # fs=1 => 归一化频率

    data64 = data.astype(np.float64)
    denoised = sosfilt(sos, data64, axis=0)   # 沿时间轴（行方向）滤波

    return denoised.astype(np.float32)


def svd_denoise(data: np.ndarray, keep_ratio: float = 0.1) -> np.ndarray:
    """
    SVD秩截断去噪（保留最大奇异值对应的成分）
    Args:
        data: 2D numpy array (H, W)
        keep_ratio: 保留奇异值的比例 (0~1)；也可以传入整数表示绝对秩数
    Returns:
        denoised: (H, W)
    """
    data64 = data.astype(np.float64)
    U, s, Vt = np.linalg.svd(data64, full_matrices=False)

    if isinstance(keep_ratio, float):
        rank = max(1, int(len(s) * keep_ratio))
    else:
        rank = int(keep_ratio)

    # 截断
    U_r = U[:, :rank]
    s_r = s[:rank]
    Vt_r = Vt[:rank, :]

    denoised = (U_r * s_r) @ Vt_r

    return denoised.astype(np.float32)


def tv_denoise(data: np.ndarray, weight: float = 0.1) -> np.ndarray:
    """
    总变差 (TV) 去噪
    优先使用 skimage 实现，否则使用简化的各向同性TV梯度下降
    Args:
        data: 2D numpy array (H, W)
        weight: 正则化权重，越大越平滑
    Returns:
        denoised: (H, W)
    """
    if HAS_SKIMAGE:
        # skimage期望值域在[0,1]或[-1,1]，做归一化
        vmin, vmax = data.min(), data.max()
        rng = vmax - vmin + 1e-12
        norm = (data - vmin) / rng
        dn = denoise_tv_chambolle(norm, weight=weight)
        return (dn * rng + vmin).astype(np.float32)
    else:
        # 简化各向同性TV (Chambolle投影法, 20次迭代)
        return _tv_chambolle_numpy(data, weight=weight, n_iter=20)


def _tv_chambolle_numpy(data: np.ndarray, weight: float = 0.1,
                        n_iter: int = 20) -> np.ndarray:
    """内置TV Chambolle去噪（不依赖skimage）"""
    u = data.astype(np.float64)
    px = np.zeros_like(u)
    py = np.zeros_like(u)
    tau = 0.25 / weight

    for _ in range(n_iter):
        # 计算梯度散度
        div_p = np.zeros_like(u)
        div_p[:-1, :] += px[:-1, :]
        div_p[1:, :] -= px[:-1, :]
        div_p[:, :-1] += py[:, :-1]
        div_p[:, 1:] -= py[:, :-1]

        u = data - weight * div_p

        # 更新对偶变量
        grad_ux = np.diff(u, axis=0, append=u[-1:, :])
        grad_uy = np.diff(u, axis=1, append=u[:, -1:])

        px_new = px + tau * grad_ux
        py_new = py + tau * grad_uy

        norm = np.sqrt(px_new**2 + py_new**2)
        norm = np.maximum(norm, 1.0)

        px = px_new / norm
        py = py_new / norm

    return u.astype(np.float32)


def wiener_denoise(data: np.ndarray, mysize: int = 3,
                   noise_var: float = None) -> np.ndarray:
    """
    Wiener滤波（自适应局部均值方差）
    Args:
        data: 2D numpy array (H, W)
        mysize: 局部窗口大小
        noise_var: 噪声方差，None则自动估计
    Returns:
        denoised: (H, W)
    """
    from scipy.signal import wiener
    return wiener(data.astype(np.float64), mysize=mysize,
                  noise=noise_var).astype(np.float32)


# ==============================================================================
# 联合去噪策略
# ==============================================================================

TRAD_METHODS = {
    'gaussian':  {'fn': gaussian_denoise,  'kwargs': {'sigma': 1.0},           'name': 'Gaussian'},
    'median':    {'fn': median_denoise,     'kwargs': {'size': 3},              'name': 'Median'},
    'wavelet':   {'fn': wavelet_denoise,    'kwargs': {'wavelet': 'db4', 'level': 3}, 'name': 'Wavelet'},
    'fk':        {'fn': fk_filter_denoise, 'kwargs': {'v_cut': 0.6},           'name': 'FK Filter'},
    'bandpass':  {'fn': bandpass_denoise,   'kwargs': {'lowcut': 0.05, 'highcut': 0.45}, 'name': 'Bandpass'},
    'svd':       {'fn': svd_denoise,        'kwargs': {'keep_ratio': 0.15},     'name': 'SVD'},
    'tv':        {'fn': tv_denoise,         'kwargs': {'weight': 0.05},         'name': 'TV'},
    'wiener':    {'fn': wiener_denoise,     'kwargs': {'mysize': 3},            'name': 'Wiener'},
}


def apply_traditional(data: np.ndarray, method: str, **override_kwargs) -> np.ndarray:
    """
    应用指定传统去噪方法
    Args:
        data: (H, W)
        method: TRAD_METHODS中的键名
        override_kwargs: 覆盖默认参数
    Returns:
        denoised: (H, W)
    """
    if method not in TRAD_METHODS:
        raise ValueError(f"未知方法: {method}，可选: {list(TRAD_METHODS.keys())}")
    entry = TRAD_METHODS[method]
    kwargs = {**entry['kwargs'], **override_kwargs}
    return entry['fn'](data, **kwargs)


def dl_then_traditional(dl_denoised: np.ndarray, method: str, **kwargs) -> np.ndarray:
    """
    深度学习后处理：DL → 传统方法（去除DL残差噪声）
    """
    return apply_traditional(dl_denoised, method, **kwargs)


def traditional_then_dl(raw: np.ndarray, method: str,
                        dl_model_fn, **trad_kwargs) -> np.ndarray:
    """
    传统方法预处理：传统 → DL（先粗降噪再精细还原）
    Args:
        raw: 原始含噪数据 (H, W)
        method: 传统方法名
        dl_model_fn: 调用签名 fn(data: np.ndarray) -> np.ndarray
        trad_kwargs: 传统方法额外参数
    """
    pre = apply_traditional(raw, method, **trad_kwargs)
    return dl_model_fn(pre)


def run_all_hybrid(raw: np.ndarray, dl_denoised: np.ndarray,
                   methods: list = None) -> dict:
    """
    对DL去噪结果批量应用所有传统后处理方法
    Args:
        raw: 原始含噪数据 (H, W)
        dl_denoised: DL去噪结果 (H, W)
        methods: 要使用的方法名列表，None则使用全部
    Returns:
        results: {method_key: denoised_array}
    """
    if methods is None:
        methods = list(TRAD_METHODS.keys())

    results = {'raw': raw, 'dl': dl_denoised}
    for m in methods:
        try:
            results[f'dl+{m}'] = dl_then_traditional(dl_denoised, m)
            print(f"  ✓ DL+{TRAD_METHODS[m]['name']} 完成")
        except Exception as e:
            print(f"  ✗ DL+{TRAD_METHODS[m]['name']} 失败: {e}")
    return results


def run_trad_only(raw: np.ndarray, methods: list = None) -> dict:
    """
    Apply each traditional denoising method directly to raw noisy data
    (no deep learning involved).

    Args:
        raw: Noisy input data (H, W)
        methods: List of method keys to apply. None = all TRAD_METHODS.
    Returns:
        results: {'raw': raw, 'gaussian': array, 'wavelet': array, ...}
    """
    if methods is None:
        methods = list(TRAD_METHODS.keys())
    results = {'raw': raw}
    for m in methods:
        try:
            results[m] = apply_traditional(raw, m)
            print(f"  ✓ {TRAD_METHODS[m]['name']} done")
        except Exception as e:
            print(f"  ✗ {TRAD_METHODS[m]['name']} failed: {e}")
    return results


# ==============================================================================
# 多方法组合策略
# ==============================================================================

# 预定义的多传统方法组合（每个 combo 是按顺序串行应用的方法链）
# 键名使用 '+' 分隔方法名，与 run_all_hybrid 的单方法键保持一致风格
COMBO_METHODS = {
    'wavelet+tv':        [('wavelet',  {}), ('tv',       {})],
    'fk+bandpass':       [('fk',       {}), ('bandpass', {})],
    'svd+wavelet':       [('svd',      {}), ('wavelet',  {})],
    'wiener+tv':         [('wiener',   {}), ('tv',       {})],
    'bandpass+median':   [('bandpass', {}), ('median',   {})],
    'svd+tv':            [('svd',      {}), ('tv',       {})],
    'svd+bandpass':      [('svd',      {}), ('bandpass', {})],
    'fk+wavelet+tv':     [('fk',       {}), ('wavelet',  {}), ('tv',      {})],
    'svd+wavelet+tv':    [('svd',      {}), ('wavelet',  {}), ('tv',      {})],
    'bandpass+wavelet+tv': [('bandpass', {}), ('wavelet', {}), ('tv',     {})],
}


def combo_display_name(combo_key: str) -> str:
    """
    将组合键 (如 'wavelet+tv') 转换为可读显示名称 (如 'Wavelet → TV')。
    也可处理单方法键（如 'wavelet' → 'Wavelet'）。
    """
    parts = combo_key.split('+')
    names = [TRAD_METHODS[p]['name'] if p in TRAD_METHODS else p.capitalize()
             for p in parts]
    return ' → '.join(names)


def run_combo_hybrid(raw: np.ndarray, dl_denoised: np.ndarray,
                     combos: dict = None) -> dict:
    """
    深度学习后，串行应用多个传统方法组合进行后处理。

    Args:
        raw: 原始含噪数据 (H, W)
        dl_denoised: DL 去噪结果 (H, W)
        combos: dict {combo_key: [(method1, kwargs1), (method2, kwargs2), ...]}
                None 则使用 COMBO_METHODS 中的全部默认组合

    Returns:
        results: {'raw': raw, 'dl': dl_denoised,
                  'dl+wavelet+tv': array, 'dl+fk+bandpass': array, ...}
    """
    if combos is None:
        combos = COMBO_METHODS

    results = {'raw': raw, 'dl': dl_denoised}
    for combo_key, steps in combos.items():
        arr = dl_denoised.copy()
        try:
            for method, kwargs in steps:
                arr = apply_traditional(arr, method, **kwargs)
            results[f'dl+{combo_key}'] = arr
            step_names = ' → '.join(
                TRAD_METHODS[m]['name'] for m, _ in steps)
            print(f"  ✓ DL + {step_names} 完成")
        except Exception as e:
            step_names = ' + '.join(m for m, _ in steps)
            print(f"  ✗ DL + {step_names} 失败: {e}")
    return results


# ==============================================================================
# 可视化
# ==============================================================================

def plot_hybrid_comparison(results: dict, reference: np.ndarray = None,
                           vmin: float = -30, vmax: float = 30,
                           save_path: str = None, data_name: str = ''):
    """
    综合对比图：原始 / DL / DL+各传统方法
    Args:
        results: run_all_hybrid的返回值 dict {key: (H,W) array}
        reference: 干净参考（可为None）
        vmin, vmax: 色标范围
        save_path: 保存路径
        data_name: 数据名称（用于标题）
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from utils_2d import cseis, calculate_metrics_2d

    keys = list(results.keys())
    n = len(keys)

    ncols = 4
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 4))
    if nrows == 1:
        axes = axes.reshape(1, -1)

    cmap = cseis()
    raw = results.get('raw', list(results.values())[0])
    h, w = raw.shape
    t = np.arange(h) * 0.001
    extent = (1, w, t[-1], 0)

    label_map = {'raw': 'Raw (Noisy)'}
    if reference is not None:
        label_map['reference'] = 'Clean Reference'
    for m in TRAD_METHODS:
        label_map[m] = TRAD_METHODS[m]['name']
        label_map[f'dl+{m}'] = f"DL + {TRAD_METHODS[m]['name']}"
    label_map['dl'] = 'DL Only'

    for idx, key in enumerate(keys):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]
        arr = results[key]

        im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax,
                       aspect='auto', extent=extent)
        title = label_map.get(key, key)

        # 计算指标（如有参考）
        if reference is not None and key not in ('raw',):
            met = calculate_metrics_2d(arr, reference)
            title += f"\nSNR={met['snr']:.1f}dB  PSNR={met['psnr']:.1f}dB"
        elif key == 'raw' and reference is not None:
            met = calculate_metrics_2d(arr, reference)
            title += f"\nSNR={met['snr']:.1f}dB  PSNR={met['psnr']:.1f}dB"

        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.set_xlabel('Trace', fontsize=9)
        if col == 0:
            ax.set_ylabel('Time (s)', fontsize=9)
        else:
            ax.set_yticks([])

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # 隐藏多余子图
    for idx in range(len(keys), nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    suptitle = f"深度学习 + 传统方法联合去噪对比"
    if data_name:
        suptitle += f" — {data_name}"
    fig.suptitle(suptitle, fontsize=14, fontweight='bold', y=1.01)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=180, bbox_inches='tight')
        print(f"✓ 联合对比图保存至: {save_path}")

    return fig


def plot_metrics_bar(results: dict, reference: np.ndarray,
                     save_path: str = None, data_name: str = ''):
    """
    指标对比柱状图（SNR 和 PSNR）
    Args:
        results: dict {key: (H,W) array}，必须包含 'raw' 和 'dl'
        reference: 干净参考 (H,W)
        save_path: 保存路径
        data_name: 数据名称
    """
    import matplotlib.pyplot as plt
    from utils_2d import calculate_metrics_2d

    keys = [k for k in results if k != 'raw']
    snrs = []
    psnrs = []
    labels = []

    label_map = {'dl': 'DL Only'}
    for m in TRAD_METHODS:
        label_map[f'dl+{m}'] = f"DL+{TRAD_METHODS[m]['name']}"

    for key in keys:
        met = calculate_metrics_2d(results[key], reference)
        snrs.append(met['snr'])
        psnrs.append(met['psnr'])
        labels.append(label_map.get(key, key))

    # 也加上raw的数据作为基准
    raw_met = calculate_metrics_2d(results['raw'], reference)

    fig, axes = plt.subplots(1, 2, figsize=(max(12, len(keys) * 1.5), 5))

    x = np.arange(len(labels))
    width = 0.6

    # SNR 柱状图
    ax1 = axes[0]
    colors = ['#2196F3' if k == 'dl' else '#4CAF50' for k in keys]
    bars1 = ax1.bar(x, snrs, width, color=colors, edgecolor='black', alpha=0.8)
    ax1.axhline(raw_met['snr'], color='red', linestyle='--', linewidth=1.5,
                label=f"Raw SNR = {raw_met['snr']:.1f} dB")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=30, ha='right', fontsize=9)
    ax1.set_ylabel('SNR (dB)', fontsize=12)
    ax1.set_title('SNR 对比', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars1, snrs):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                 f'{val:.1f}', ha='center', va='bottom', fontsize=8)

    # PSNR 柱状图
    ax2 = axes[1]
    bars2 = ax2.bar(x, psnrs, width, color=colors, edgecolor='black', alpha=0.8)
    ax2.axhline(raw_met['psnr'], color='red', linestyle='--', linewidth=1.5,
                label=f"Raw PSNR = {raw_met['psnr']:.1f} dB")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=30, ha='right', fontsize=9)
    ax2.set_ylabel('PSNR (dB)', fontsize=12)
    ax2.set_title('PSNR 对比', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars2, psnrs):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                 f'{val:.1f}', ha='center', va='bottom', fontsize=8)

    suptitle = "深度学习 + 传统方法 — SNR / PSNR 指标对比"
    if data_name:
        suptitle += f" ({data_name})"
    fig.suptitle(suptitle, fontsize=13, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=180, bbox_inches='tight')
        print(f"✓ 指标对比图保存至: {save_path}")

    return fig


def plot_noise_residuals(results: dict, raw: np.ndarray,
                         vmin: float = -30, vmax: float = 30,
                         save_path: str = None, data_name: str = ''):
    """
    去除噪声残差图（含噪 - 去噪 = 被去除的噪声）
    Args:
        results: dict {key: (H,W) array}，必须包含 'raw' 和 'dl'
        raw: 原始含噪数据
        save_path: 保存路径
    """
    import matplotlib.pyplot as plt
    from utils_2d import cseis

    # 只展示方法结果（排除 raw）
    keys = [k for k in results if k != 'raw']
    n = len(keys)
    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 4))
    if nrows == 1:
        axes = axes.reshape(1, -1)

    cmap = cseis()
    h, w = raw.shape
    t = np.arange(h) * 0.001
    extent = (1, w, t[-1], 0)

    label_map = {'dl': 'DL Only'}
    for m in TRAD_METHODS:
        label_map[m] = TRAD_METHODS[m]['name']
        label_map[f'dl+{m}'] = f"DL + {TRAD_METHODS[m]['name']}"

    for idx, key in enumerate(keys):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]
        residual = raw - results[key]   # 被去除的噪声

        im = ax.imshow(residual, cmap=cmap, vmin=vmin, vmax=vmax,
                       aspect='auto', extent=extent)
        ax.set_title(f"Removed by {label_map.get(key, key)}", fontsize=10, fontweight='bold')
        ax.set_xlabel('Trace', fontsize=9)
        if col == 0:
            ax.set_ylabel('Time (s)', fontsize=9)
        else:
            ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for idx in range(len(keys), nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    suptitle = "各方法去除的噪声成分"
    if data_name:
        suptitle += f" — {data_name}"
    fig.suptitle(suptitle, fontsize=14, fontweight='bold', y=1.01)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=180, bbox_inches='tight')
        print(f"✓ 噪声残差图保存至: {save_path}")

    return fig


def plot_single_trace_comparison(results: dict, trace_idx: int = None,
                                 raw: np.ndarray = None,
                                 reference: np.ndarray = None,
                                 save_path: str = None, data_name: str = ''):
    """
    单道波形对比图（时域），展示各方法在单个道上的效果
    """
    import matplotlib.pyplot as plt

    arr0 = raw if raw is not None else list(results.values())[0]
    h, w = arr0.shape
    if trace_idx is None:
        trace_idx = w // 2  # 取中间道

    t = np.arange(h) * 0.001

    label_map = {'raw': 'Raw (Noisy)', 'dl': 'DL Only'}
    for m in TRAD_METHODS:
        label_map[m] = TRAD_METHODS[m]['name']
        label_map[f'dl+{m}'] = f"DL+{TRAD_METHODS[m]['name']}"
    if reference is not None:
        label_map['reference'] = 'Clean Reference'

    fig, ax = plt.subplots(figsize=(14, 5))

    # 原始噪声道
    if raw is not None:
        ax.plot(t, raw[:, trace_idx], color='gray', linewidth=0.8,
                alpha=0.6, label='Raw (Noisy)')

    # 参考道
    if reference is not None:
        ax.plot(t, reference[:, trace_idx], color='black', linewidth=1.5,
                alpha=0.9, label='Clean Reference', linestyle='--')

    # 各去噪方法
    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))
    for (key, arr), color in zip(results.items(), colors):
        if key == 'raw':
            continue
        label = label_map.get(key, key)
        ax.plot(t, arr[:, trace_idx], linewidth=1.0,
                alpha=0.85, label=label, color=color)

    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Amplitude', fontsize=12)
    title = f"单道波形对比 (道 #{trace_idx + 1})"
    if data_name:
        title += f" — {data_name}"
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend(fontsize=9, ncol=2)
    ax.grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=180, bbox_inches='tight')
        print(f"✓ 单道对比图保存至: {save_path}")

    return fig


def plot_trad_comparison(results: dict, reference: np.ndarray = None,
                        vmin: float = -30, vmax: float = 30,
                        save_path: str = None, data_name: str = ''):
    """
    Seismic image comparison of pure traditional denoising methods.

    Args:
        results: run_trad_only() output dict {key: (H,W) array}
        reference: Clean reference array (optional)
        vmin, vmax: Colorbar range
        save_path: Output file path
        data_name: Dataset name for figure title
    """
    import matplotlib.pyplot as plt
    from utils_2d import cseis, calculate_metrics_2d

    keys = list(results.keys())
    n = len(keys)
    ncols = 4
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 4.2))
    if nrows == 1:
        axes = axes.reshape(1, -1)

    cmap = cseis()
    raw = results.get('raw', list(results.values())[0])
    h, w = raw.shape
    t = np.arange(h) * 0.001
    extent = (1, w, t[-1], 0)

    def _label(key):
        if key == 'raw':
            return 'Raw (Noisy)'
        return TRAD_METHODS[key]['name'] if key in TRAD_METHODS else key

    for idx, key in enumerate(keys):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]
        arr = results[key]

        im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax,
                       aspect='auto', extent=extent)
        title = _label(key)
        if reference is not None:
            met = calculate_metrics_2d(arr, reference)
            title += f"\nSNR={met['snr']:.1f} dB  PSNR={met['psnr']:.1f} dB"

        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.set_xlabel('Trace', fontsize=9)
        if col == 0:
            ax.set_ylabel('Time (s)', fontsize=9)
        else:
            ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for idx in range(len(keys), nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    suptitle = 'Pure Traditional Denoising Methods Comparison'
    if data_name:
        suptitle += f' — {data_name}'
    fig.suptitle(suptitle, fontsize=14, fontweight='bold', y=1.01)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=180, bbox_inches='tight')
        print(f'✓ Traditional methods comparison saved to: {save_path}')

    plt.close(fig)
    return fig


def plot_trad_metrics_bar(results: dict, reference: np.ndarray,
                         save_path: str = None, data_name: str = ''):
    """
    SNR/PSNR bar chart for pure traditional denoising methods, ranked by SNR.

    Args:
        results: run_trad_only() output (must include 'raw')
        reference: Clean reference (H, W)
        save_path: Output file path
        data_name: Dataset name
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from utils_2d import calculate_metrics_2d

    keys = [k for k in results if k != 'raw']
    snrs, psnrs, labels = [], [], []
    for key in keys:
        met = calculate_metrics_2d(results[key], reference)
        snrs.append(met['snr'])
        psnrs.append(met['psnr'])
        labels.append(TRAD_METHODS[key]['name'] if key in TRAD_METHODS else key)

    raw_met = calculate_metrics_2d(results['raw'], reference)
    best_snr_idx = int(np.argmax(snrs))
    best_psnr_idx = int(np.argmax(psnrs))

    # Print ranking table
    print(f"\n{'='*60}")
    print(f"  Traditional Methods SNR Ranking  (Raw SNR = {raw_met['snr']:.2f} dB)")
    print(f"  {'Rank':<6} {'Method':<20} {'SNR':>8} {'PSNR':>8}")
    print(f"  {'-'*44}")
    sorted_items = sorted(zip(keys, snrs, psnrs), key=lambda x: x[1], reverse=True)
    for rank, (key, snr, psnr) in enumerate(sorted_items, 1):
        tag = ' ★' if rank == 1 else ''
        name = TRAD_METHODS[key]['name'] if key in TRAD_METHODS else key
        print(f"  {rank:<6} {name:<20} {snr:>7.2f}  {psnr:>7.2f}{tag}")
    print(f"{'='*60}\n")

    fig, axes = plt.subplots(1, 2, figsize=(max(14, len(keys) * 1.8), 5))
    x = np.arange(len(labels))
    width = 0.65

    for ax, vals, metric_key, ylabel, title, best_idx in zip(
            axes,
            [snrs, psnrs],
            ['snr', 'psnr'],
            ['SNR (dB)', 'PSNR (dB)'],
            ['SNR Comparison', 'PSNR Comparison'],
            [best_snr_idx, best_psnr_idx]):

        baseline = raw_met[metric_key]
        bar_colors = ['#FFC107' if i == best_idx else '#4CAF50'
                      for i in range(len(keys))]
        bars = ax.bar(x, vals, width, color=bar_colors,
                      edgecolor='black', alpha=0.88)
        ax.axhline(baseline, color='red', linestyle='--', linewidth=1.5,
                   label=f'Raw = {baseline:.1f} dB')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=9)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(axis='y', alpha=0.3)

        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.05,
                    f'{val:.1f}', ha='center', va='bottom', fontsize=8)

        ax.text(x[best_idx], vals[best_idx] + 0.3, '★ Best',
                ha='center', va='bottom', fontsize=9,
                color='#E65100', fontweight='bold')

    legend_handles = [
        Patch(facecolor='#FFC107', edgecolor='black', label='Best Method'),
        Patch(facecolor='#4CAF50', edgecolor='black', label='Other Methods'),
    ]
    fig.legend(handles=legend_handles, loc='lower center',
               ncol=2, fontsize=9, bbox_to_anchor=(0.5, -0.02))

    suptitle = 'Pure Traditional Methods — SNR / PSNR Ranking'
    if data_name:
        suptitle += f' ({data_name})'
    fig.suptitle(suptitle, fontsize=13, fontweight='bold')

    plt.tight_layout(rect=[0, 0.04, 1, 1])

    if save_path:
        plt.savefig(save_path, dpi=180, bbox_inches='tight')
        print(f'✓ Traditional methods metrics chart saved to: {save_path}')

    plt.close(fig)
    return fig


def plot_combo_comparison(results: dict, reference: np.ndarray = None,
                          vmin: float = -30, vmax: float = 30,
                          save_path: str = None, data_name: str = ''):
    """
    多传统方法组合对比图（地震剖面风格）。
    与 plot_hybrid_comparison 类似，但展示 DL + 多方法组合结果。

    Args:
        results: run_combo_hybrid 的返回值 dict {key: (H,W) array}
        reference: 干净参考（可为 None）
        vmin, vmax: 色标范围
        save_path: 保存路径
        data_name: 数据名称（用于标题）
    """
    import matplotlib.pyplot as plt
    from utils_2d import cseis, calculate_metrics_2d

    keys = list(results.keys())
    n = len(keys)
    ncols = 4
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 4.2))
    if nrows == 1:
        axes = axes.reshape(1, -1)

    cmap = cseis()
    raw = results.get('raw', list(results.values())[0])
    h, w = raw.shape
    t = np.arange(h) * 0.001
    extent = (1, w, t[-1], 0)

    def _label(key):
        if key == 'raw':
            return 'Raw (Noisy)'
        if key == 'dl':
            return 'DL Only'
        suffix = key[3:]   # strip leading 'dl+'
        return 'DL + ' + combo_display_name(suffix)

    for idx, key in enumerate(keys):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]
        arr = results[key]

        im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax,
                       aspect='auto', extent=extent)
        title = _label(key)
        if reference is not None:
            met = calculate_metrics_2d(arr, reference)
            title += f"\nSNR={met['snr']:.1f} dB  PSNR={met['psnr']:.1f} dB"

        ax.set_title(title, fontsize=9, fontweight='bold')
        ax.set_xlabel('Trace', fontsize=8)
        if col == 0:
            ax.set_ylabel('Time (s)', fontsize=8)
        else:
            ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for idx in range(len(keys), nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    suptitle = "DL + 多传统方法组合去噪对比"
    if data_name:
        suptitle += f" — {data_name}"
    fig.suptitle(suptitle, fontsize=14, fontweight='bold', y=1.01)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=180, bbox_inches='tight')
        print(f"✓ 组合方法对比图保存至: {save_path}")

    plt.close(fig)
    return fig


def plot_combo_metrics_bar(results: dict, reference: np.ndarray,
                           save_path: str = None, data_name: str = ''):
    """
    多传统方法组合 SNR/PSNR 柱状图对比，并打印排名。
    同时高亮最优 combo。

    Args:
        results: run_combo_hybrid 的返回值（包含 'raw' 和 'dl'）
        reference: 干净参考 (H,W)
        save_path: 保存路径
        data_name: 数据名称
    """
    import matplotlib.pyplot as plt
    from utils_2d import calculate_metrics_2d

    def _label(key):
        if key == 'dl':
            return 'DL Only'
        suffix = key[3:]
        return 'DL+\n' + combo_display_name(suffix).replace(' → ', '\n→ ')

    keys = [k for k in results if k != 'raw']
    snrs, psnrs, labels = [], [], []
    for key in keys:
        met = calculate_metrics_2d(results[key], reference)
        snrs.append(met['snr'])
        psnrs.append(met['psnr'])
        labels.append(_label(key))

    raw_met = calculate_metrics_2d(results['raw'], reference)

    # 确定最优 combo 索引（按 SNR）
    best_snr_idx = int(np.argmax(snrs))
    best_psnr_idx = int(np.argmax(psnrs))

    # 打印排名
    print(f"\n{'='*65}")
    print(f"  组合方法 SNR 排名 (参考基准 Raw SNR = {raw_met['snr']:.2f} dB)")
    print(f"  {'排名':<5} {'方法':<35} {'SNR':>8} {'PSNR':>8}")
    print(f"  {'-'*58}")
    sorted_items = sorted(zip(keys, snrs, psnrs), key=lambda x: x[1], reverse=True)
    for rank, (key, snr, psnr) in enumerate(sorted_items, 1):
        tag = ' ★' if rank == 1 else ''
        name = 'DL Only' if key == 'dl' else 'DL + ' + combo_display_name(key[3:])
        print(f"  {rank:<5} {name:<35} {snr:>7.2f}  {psnr:>7.2f}{tag}")
    print(f"{'='*65}\n")

    fig, axes = plt.subplots(1, 2, figsize=(max(16, len(keys) * 2.0), 6))
    x = np.arange(len(labels))
    width = 0.65

    for ax, vals, metric_key, ylabel, title, best_idx in zip(
            axes,
            [snrs, psnrs],
            ['snr', 'psnr'],
            ['SNR (dB)', 'PSNR (dB)'],
            ['SNR 对比', 'PSNR 对比'],
            [best_snr_idx, best_psnr_idx]):

        baseline = raw_met[metric_key]
        # 颜色：DL Only = 蓝，最佳组合 = 金，其余 = 橙
        bar_colors = []
        for i, k in enumerate(keys):
            if k == 'dl':
                bar_colors.append('#2196F3')
            elif i == best_idx:
                bar_colors.append('#FFC107')
            else:
                bar_colors.append('#FF7043')

        bars = ax.bar(x, vals, width, color=bar_colors, edgecolor='black', alpha=0.88)
        ax.axhline(baseline, color='red', linestyle='--', linewidth=1.5,
                   label=f"Raw = {baseline:.1f} dB")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=7)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(axis='y', alpha=0.3)

        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.05,
                    f'{val:.1f}', ha='center', va='bottom', fontsize=7)

        # 标注最优
        ax.text(x[best_idx], vals[best_idx] + 0.3, '★ Best',
                ha='center', va='bottom', fontsize=8,
                color='#E65100', fontweight='bold')

    # 图例说明颜色
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor='#2196F3', edgecolor='black', label='DL Only'),
        Patch(facecolor='#FFC107', edgecolor='black', label='Best Combo'),
        Patch(facecolor='#FF7043', edgecolor='black', label='Other Combos'),
    ]
    fig.legend(handles=legend_handles, loc='lower center',
               ncol=3, fontsize=9, bbox_to_anchor=(0.5, -0.02))

    suptitle = "DL + 多传统方法组合 — SNR / PSNR 指标排名"
    if data_name:
        suptitle += f" ({data_name})"
    fig.suptitle(suptitle, fontsize=13, fontweight='bold')

    plt.tight_layout(rect=[0, 0.04, 1, 1])

    if save_path:
        plt.savefig(save_path, dpi=180, bbox_inches='tight')
        print(f"✓ 组合方法指标图保存至: {save_path}")

    plt.close(fig)
    return fig


if __name__ == '__main__':
    # 简单自测
    import numpy as np
    rng = np.random.default_rng(42)
    clean = rng.standard_normal((300, 128)).astype(np.float32) * 20
    noisy = clean + rng.standard_normal((300, 128)).astype(np.float32) * 5

    print("测试各传统去噪方法 ...")
    for key, entry in TRAD_METHODS.items():
        try:
            out = apply_traditional(noisy, key)
            from utils_2d import calculate_metrics_2d
            met = calculate_metrics_2d(out, clean)
            print(f"  {entry['name']:15s}  SNR={met['snr']:.2f}  PSNR={met['psnr']:.2f}")
        except Exception as e:
            print(f"  {entry['name']:15s}  ERROR: {e}")
