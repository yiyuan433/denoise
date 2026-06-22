"""
针对极低SNR数据（SNR < 5 dB）的优化配置
专门为EQ-36数据（SNR=3.45dB）设计
"""

import os
import numpy as np

# ============ 路径配置 ============
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(BASE_DIR), 'data')
CHECKPOINT_DIR = os.path.join(BASE_DIR, 'checkpoints_low_snr')
RESULTS_DIR = os.path.join(BASE_DIR, 'results_low_snr')
FIGURES_DIR = os.path.join(BASE_DIR, 'figures_low_snr')

# 创建必要目录
for dir_path in [CHECKPOINT_DIR, RESULTS_DIR, FIGURES_DIR]:
    os.makedirs(dir_path, exist_ok=True)

PATHS = {
    'data': DATA_DIR,
    'checkpoints': CHECKPOINT_DIR,
    'results': RESULTS_DIR,
    'figures': FIGURES_DIR
}


# ============ 数据配置（针对低SNR优化）============
DATA_CONFIG = {
    'datasets': {
        'eq-36': os.path.join(DATA_DIR, 'eq-36.npy'),
        'eq-68': os.path.join(DATA_DIR, 'eq-68.npy'),
        'slice_german': os.path.join(DATA_DIR, 'slice_german.npy'),
        'slice_german_1': os.path.join(DATA_DIR, 'slice_german_1.npy'),
    },
    # 大Patch尺寸 - 捕获更多上下文信息
    'patch_size': (48, 48),      # 24→48（显著增加）
    'stride': (12, 12),          # 75%重叠率保持不变
    'batch_size': 32,            # 降低batch size以稳定训练
    
    # 数据预处理
    'clip_percentile': 99.5,     # 裁剪极端异常值
    'normalize_method': 'robust',  # 使用鲁棒归一化（对异常值不敏感）
}


# ============ 模型配置（大容量模型）============
MODEL_LOW_SNR_CONFIG = {
    # 基础参数 - 大幅增加模型容量
    'd_model': 256,              # 64→256（4倍增加）
    'dropout': 0.1,              # 0.2→0.1（降低，避免欠拟合）
    
    # 1) 多尺度可学习小波分解（增强版）
    'wavelet_levels': 4,         # 3→4（更多尺度）
    'use_learnable_wavelet': True,
    'wavelet_channels': [64, 128, 256, 256],  # 每层通道数
    
    # 2) 跨尺度注意力机制（增强版）
    'use_cross_scale': True,
    'num_cross_scale_layers': 4,  # 2→4（更深）
    'num_attention_heads': 8,     # 增加注意力头
    
    # 3) 自适应小波特征融合（增强版）
    'use_adaptive_fusion': True,
    'fusion_method': 'learnable_weighted',  # 可学习权重
    
    # 4) 频域-空域双路径处理（增强版）
    'use_dual_path': True,
    'spatial_path_channels': 128,  # 32→128
    'frequency_path_channels': 128,
    
    # 5) Blind-spot无监督训练（优化版）
    'use_blind_spot': True,
    'mask_ratio': 0.1,           # 0.05→0.1（更激进）
    'mask_type': 'random',       # random/checkerboard
    
    # 6) 动态稀疏注意力（优化版）
    'use_sparse_attention': True,
    'sparsity_ratio': 0.2,       # 0.3→0.2（保留更多信息）
    'attention_window': (7, 7),  # (5,5)→(7,7)（更大感受野）
    'adaptive_sparsity': True,   # 自适应稀疏度
    
    # 7) 频率感知门控（增强版）
    'use_frequency_gating': True,
    'num_freq_bands': 8,         # 4→8（更精细）
    'freq_gate_method': 'adaptive',  # 自适应频率门控
    
    # 8) 噪声估计器模块（增强版）
    'use_noise_estimator': True,
    'noise_scales': 5,           # 3→5（更多尺度）
    'noise_estimator_channels': 128,
    'explicit_noise_modeling': True,  # 显式噪声建模
    
    # 9) 残差密集连接（增强版）
    'use_residual_dense': True,
    'rdc_growth_rate': 32,       # 增长率
    'rdc_num_blocks': 6,         # 增加块数
    'rdc_compression': 0.5,      # 压缩率
}


# ============ 训练配置（长时间强化训练）============
TRAIN_LOW_SNR_CONFIG = {
    # 训练基本参数
    'epochs': 300,               # 50→300（大幅增加）
    'batch_size': 32,            # 128→32（降低）
    'learning_rate': 5e-5,       # 初始学习率（保守）
    'weight_decay': 1e-5,
    
    # 优化器配置
    'optimizer': 'adamw',
    'betas': (0.9, 0.999),
    'eps': 1e-8,
    
    # 学习率调度（余弦退火 + 重启）
    'scheduler': {
        'type': 'cosine_warmup_restarts',
        'warmup_epochs': 20,      # 预热期
        'T_0': 50,                # 第一个周期长度
        'T_mult': 2,              # 周期倍增因子
        'eta_min': 1e-7,          # 最小学习率
        'restart_decay': 0.8,     # 重启衰减
    },
    
    # 损失函数权重（针对极低SNR）
    'loss_weights': {
        'mse': 0.2,              # 降低MSE（对异常值敏感）
        'l1': 0.5,               # 增加L1（鲁棒）
        'spectral': 0.15,        # 频谱损失
        'perceptual': 0.1,       # 感知损失
        'ssim': 0.05,            # 结构相似性
    },
    
    # 数据增强（激进）
    'augmentation': {
        'horizontal_flip': True,
        'vertical_flip': True,
        'rotation': [0, 90, 180, 270],
        'transpose': True,
        'noise_augment': True,
        'noise_range': (0.05, 0.15),  # 额外噪声强度
        'cutout': True,               # 随机遮挡
        'cutout_ratio': 0.1,
    },
    
    # 梯度管理
    'gradient_clip': 1.0,         # 梯度裁剪
    'gradient_accumulation': 2,   # 梯度累积（等效batch_size=64）
    
    # Early stopping（宽松）
    'patience': 50,               # 50个epoch不改进才停止
    'min_delta': 1e-6,
    'monitor': 'val_loss',
    
    # 检查点保存
    'save_every': 10,             # 每10个epoch保存
    'save_best_only': False,      # 保存所有检查点（便于分析）
    
    # 验证配置
    'val_split': 0.15,            # 15%用于验证
    'val_every': 1,               # 每个epoch验证
    
    # 混合精度训练（加速）
    'use_amp': True,              # 自动混合精度
    'amp_level': 'O1',
}


# ============ 数据预处理配置 ============
PREPROCESSING_CONFIG = {
    # 异常值处理
    'clip_method': 'percentile',
    'clip_lower': 0.5,            # 下限百分位
    'clip_upper': 99.5,           # 上限百分位
    
    # 归一化方法
    'normalization': 'robust',    # robust/standard/minmax
    'robust_params': {
        'with_centering': True,
        'with_scaling': True,
        'quantile_range': (5, 95),
    },
    
    # 去趋势
    'detrend': True,
    'detrend_method': 'linear',
    
    # 滤波预处理
    'prefilter': False,           # 谨慎使用，可能损失信号
    'filter_type': 'bandpass',
    'filter_params': {
        'low': 1,
        'high': 100,
        'order': 4,
    },
}


# ============ 评估配置 ============
EVAL_CONFIG = {
    'metrics': [
        'mse', 'rmse', 'psnr', 'ssim',
        'snr_improvement',        # SNR提升
        'noise_reduction_ratio',  # 噪声降低比例
    ],
    
    # 可视化
    'plot_every': 10,
    'plot_samples': 5,
    'save_predictions': True,
}


# ============ 实验配置 ============
EXPERIMENT_CONFIG = {
    'name': 'low_snr_eq36_v1',
    'description': 'Optimized for SNR=3.45dB, patch=48x48, d_model=256',
    'tags': ['low-snr', 'eq-36', 'heavy-noise'],
    
    'seed': 42,
    'deterministic': True,
    
    'log_every': 10,
    'verbose': True,
}


def get_low_snr_config():
    """获取完整的低SNR配置"""
    return {
        'paths': PATHS,
        'data': DATA_CONFIG,
        'model': MODEL_LOW_SNR_CONFIG,
        'train': TRAIN_LOW_SNR_CONFIG,
        'preprocess': PREPROCESSING_CONFIG,
        'eval': EVAL_CONFIG,
        'experiment': EXPERIMENT_CONFIG,
    }


def print_config_summary(config=None):
    """打印配置摘要"""
    if config is None:
        config = get_low_snr_config()
    
    print("\n" + "="*70)
    print("极低SNR数据优化配置摘要")
    print("="*70)
    
    print("\n【数据配置】")
    print(f"  Patch尺寸: {config['data']['patch_size']}")
    print(f"  Stride: {config['data']['stride']}")
    print(f"  Batch size: {config['data']['batch_size']}")
    
    print("\n【模型配置】")
    print(f"  特征维度: {config['model']['d_model']}")
    print(f"  Dropout: {config['model']['dropout']}")
    print(f"  小波层数: {config['model']['wavelet_levels']}")
    print(f"  跨尺度层数: {config['model']['num_cross_scale_layers']}")
    print(f"  频率频带数: {config['model']['num_freq_bands']}")
    
    print("\n【训练配置】")
    print(f"  训练轮数: {config['train']['epochs']}")
    print(f"  学习率: {config['train']['learning_rate']}")
    print(f"  调度器: {config['train']['scheduler']['type']}")
    print(f"  Early stopping patience: {config['train']['patience']}")
    
    print("\n【损失权重】")
    for k, v in config['train']['loss_weights'].items():
        print(f"  {k}: {v}")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    # 测试配置
    config = get_low_snr_config()
    print_config_summary(config)
