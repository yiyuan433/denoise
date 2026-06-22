"""
完全2D处理的DAS降噪模型配置
保留所有9个创新点，但全部采用2D实现
"""

import os
import numpy as np

# ============ 路径配置 ============
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(BASE_DIR), 'data')
CHECKPOINT_DIR = os.path.join(BASE_DIR, 'checkpoints')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
FIGURES_DIR = os.path.join(BASE_DIR, 'figures')

# 创建必要目录
for dir_path in [CHECKPOINT_DIR, RESULTS_DIR, FIGURES_DIR]:
    os.makedirs(dir_path, exist_ok=True)

PATHS = {
    'data': DATA_DIR,
    'checkpoints': CHECKPOINT_DIR,
    'results': RESULTS_DIR,
    'figures': FIGURES_DIR
}


# ============ 数据配置 ============
DATA_CONFIG = {
    'datasets': {
        'eq-36': os.path.join(DATA_DIR, 'eq-36.npy'),
        'eq-68': os.path.join(DATA_DIR, 'eq-68.npy'),
        'slice_german': os.path.join(DATA_DIR, 'slice_german.npy'),
        'slice_german_1': os.path.join(DATA_DIR, 'slice_german_1.npy'),
    },
    # 2D Patch参数（参考USL DIP）
    'patch_size': (24, 24),  # 高度 × 宽度
    'stride': (6, 6),        # 75%重叠率
    'batch_size': 128,       # 小patch允许大batch
}


# ============ 模型配置（Wavelet-Transformer + FDA + NAAG）============
MODEL_2D_CONFIG = {
    # 模型类型: 'advanced' = 本文提出的2D Wavelet-Transformer
    #          'usl' = USL DIP 原始架构（仅供baseline）
    'model_type': 'advanced',

    # 基础参数
    'd_model': 64,
    'dropout': 0.2,
    'num_heads': 8,
    'num_layers': 4,
    'd_ff': 256,

    # 1) 多尺度可学习小波分解
    'wavelet_level': 3,

    # 2) 跨尺度注意力
    'use_cross_scale': True,

    # 3) 自适应小波融合
    'use_adaptive_fusion': True,

    # 4) 频域-空域双路径
    'use_dual_path': True,

    # 5) Blind-spot (可关)
    'mask_ratio': 0.05,

    # 6) 动态稀疏注意力
    'use_sparse_attention': True,
    'sparsity_ratio': 0.3,

    # 7) 频率感知门控 (FAG)
    'use_fag': True,
    'fag_num_bands': 4,

    # 8) 噪声估计器
    'use_noise_estimator': True,
    'noise_num_scales': 3,

    # 9) 残差密集连接
    'use_rdc': True,
    'rdc_growth_rate': 16,
    'rdc_num_blocks': 3,

    # 新增模块：FDA + NAAG
    'use_fda': True,
    'fda_num_bands': 3,
    'use_naag': True,
}


# ============ 训练配置 ============
TRAIN_CONFIG = {
    # 训练范式: 'supervised' = 合成噪声监督, 'blind_spot' = 自监督mask
    'training_paradigm': 'supervised',

    # 训练参数
    'epochs': 80,
    'batch_size': 128,
    'learning_rate': 5e-5,
    'weight_decay': 1e-5,
    'grad_clip': 1.0,

    # 合成噪声强度
    'noise_levels': [0.05, 0.1, 0.15],

    # 损失权重
    'loss_weights': {
        'mse': 0.4,
        'l1': 0.3,
        'spectral': 0.2,
        'perceptual': 0.1,
    },

    # 数据增强
    'augmentation': True,

    # 验证与调度
    'val_split': 0.1,
    'early_stopping_patience': 10,
    'lr_scheduler': 'cosine',
    'warmup_epochs': 5,

    # 训练辅助
    'use_amp': True,
    'save_every': 10,
}


# ============ 测试配置 ============
TEST_CONFIG = {
    'num_visualizations': 5,
    'save_predictions': True,
    'calculate_metrics': True,
    
    # 重建权重模式
    'reconstruction_mode': 'uniform',  # 'uniform', 'gaussian', 'distance'
    
    # 可视化参数
    'vmin': -20,
    'vmax': 20,
    'cmap': 'seismic',
}


# ============ 优化配置 ============
OPTIMIZE_CONFIG = {
    'n_trials': 50,
    'timeout': 3600 * 8,  # 8小时
    
    # 搜索空间
    'search_space': {
        'learning_rate': (1e-5, 1e-3),
        'batch_size': [64, 128, 256],
        'd_model': [32, 64, 128],
        'dropout': (0.1, 0.3),
        'wavelet_level': [2, 3, 4],
        'rdc_growth_rate': [8, 16, 32],
    },
    
    # 优化目标
    'metric': 'snr_improvement',  # 'snr_improvement', 'psnr', 'ssim'
    'direction': 'maximize',
}


def get_config(mode='train'):
    """
    获取配置
    
    Args:
        mode: 'train', 'test', 'optimize'
    
    Returns:
        config: 配置字典
    """
    config = {
        'paths': PATHS,
        'data': DATA_CONFIG,
        'model': MODEL_2D_CONFIG,
    }
    
    if mode == 'train':
        config['train'] = TRAIN_CONFIG
    elif mode == 'test':
        config['test'] = TEST_CONFIG
    elif mode == 'optimize':
        config['optimize'] = OPTIMIZE_CONFIG
        config['train'] = TRAIN_CONFIG
    
    return config


def print_config(config):
    """打印配置"""
    print("\n" + "="*60)
    print("配置信息 - 2D Wavelet-Transformer (Advanced)")
    print("="*60)
    
    if 'model' in config:
        print("\n【模型配置】")
        model_cfg = config['model']
        model_type = model_cfg.get('model_type', 'advanced')
        print(f"  模型类型: {model_type}")
        if model_type == 'usl':
            print("  ★ USL DIP 原始架构 (baseline)")
        else:
            print(f"  特征维度: {model_cfg['d_model']}")
            print(f"  Dropout: {model_cfg['dropout']}")
            print(f"  Wavelet levels: {model_cfg['wavelet_level']}")
            print(f"  FDA bands: {model_cfg['fda_num_bands']} | NAAG: {model_cfg['use_naag']}")
    
    if 'data' in config:
        print(f"\n【数据配置】")
        data_cfg = config['data']
        print(f"  Patch大小: {data_cfg['patch_size']}")
        print(f"  Stride: {data_cfg['stride']}")
        overlap_h = (1 - data_cfg['stride'][0] / data_cfg['patch_size'][0]) * 100
        overlap_w = (1 - data_cfg['stride'][1] / data_cfg['patch_size'][1]) * 100
        print(f"  重叠率: {overlap_h:.0f}% × {overlap_w:.0f}%")
        print(f"  Batch size: {data_cfg['batch_size']}")
    
    if 'train' in config:
        print(f"\n【训练配置】")
        train_cfg = config['train']
        print(f"  Epochs: {train_cfg['epochs']}")
        print(f"  学习率: {train_cfg['learning_rate']}")
        print(f"  噪声水平: {train_cfg['noise_levels']}")
        print(f"  损失权重: {train_cfg['loss_weights']}")
    
    print("="*60 + "\n")


if __name__ == '__main__':
    # 测试配置
    config = get_config('train')
    print_config(config)
    
    print("\n可用数据集:")
    for name, path in config['data']['datasets'].items():
        exists = "✓" if os.path.exists(path) else "✗"
        print(f"  {exists} {name}: {path}")
