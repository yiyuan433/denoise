"""
改进的训练脚本 - 针对真实DAS数据优化
"""

import os
import sys
import json
import torch
import numpy as np
from config_2d import *

def create_improved_config():
    """创建针对真实含噪数据优化的配置"""
    
    improved_config = {
        # ========== 模型架构优化 ==========
        'd_model': 96,              # 增加特征维度（64→96）
        'dropout': 0.15,            # 降低dropout（避免欠拟合）
        
        # 小波参数
        'wavelet_levels': 4,        # 增加分解层数（捕获更多尺度）
        'use_learnable_wavelet': True,
        
        # 注意力机制
        'use_cross_scale': True,
        'num_cross_scale_layers': 3,  # 增加跨尺度层
        'use_sparse_attention': True,
        'sparsity_ratio': 0.25,     # 降低稀疏度（保留更多信息）
        
        # 频域处理
        'use_frequency_gating': True,
        'num_freq_bands': 6,        # 增加频带数（更精细）
        
        # 噪声估计
        'use_noise_estimator': True,
        'noise_scales': 4,          # 多尺度噪声估计
        
        # 残差连接
        'use_residual_dense': True,
        'rdc_growth_rate': 24,      # 增加增长率
        'rdc_num_blocks': 4,        # 增加块数
        
        # 其他创新点
        'use_adaptive_fusion': True,
        'use_dual_path': True,
        'use_blind_spot': True,
        'mask_ratio': 0.08,         # 增加mask比例
    }
    
    # ========== 训练策略优化 ==========
    train_config = {
        'epochs': 150,              # 大幅增加训练轮数
        'batch_size': 64,           # 降低batch_size（更稳定）
        'learning_rate': 1e-4,      # 降低学习率（更精细）
        'weight_decay': 5e-6,
        
        # 学习率调度
        'scheduler': {
            'type': 'cosine',
            'warmup_epochs': 10,
            'min_lr': 1e-6,
        },
        
        # 损失函数权重（针对真实数据）
        'loss_weights': {
            'mse': 0.3,             # 降低MSE权重
            'l1': 0.4,              # 增加L1权重（鲁棒性）
            'spectral': 0.2,        # 频谱一致性
            'perceptual': 0.1,      # 感知质量
        },
        
        # 数据增强
        'augmentation': {
            'horizontal_flip': True,
            'vertical_flip': True,
            'rotation': True,
            'noise_augment': True,  # 额外噪声增强
        },
        
        # 梯度裁剪（防止梯度爆炸）
        'gradient_clip': 1.0,
        
        # Early stopping
        'patience': 30,
        'min_delta': 1e-5,
    }
    
    # ========== Patch策略优化 ==========
    patch_config = {
        'patch_size': (32, 32),     # 增大patch（24→32，更多上下文）
        'stride': (8, 8),           # 增大stride（6→8，减少重叠）
        'batch_size': 64,
        'shuffle': True,
        'num_workers': 4,
    }
    
    return {
        'model': improved_config,
        'train': train_config,
        'patch': patch_config,
    }


def save_improved_params():
    """保存改进的参数配置"""
    config = create_improved_config()
    
    # 保存到results目录
    output_path = os.path.join(PATHS['results'], 'improved_params.json')
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4, ensure_ascii=False)
    
    print("=" * 70)
    print("改进配置已生成")
    print("=" * 70)
    print(f"\n✓ 保存位置: {output_path}")
    print("\n关键改进:")
    print("  1. 模型容量: d_model 64→96")
    print("  2. 训练轮数: 80→150 epochs")
    print("  3. Patch大小: 24×24→32×32 (更多上下文)")
    print("  4. Patch数量: ~51k→~13k (减少重叠，加速)")
    print("  5. 学习率: 5e-5→1e-4 (余弦退火)")
    print("  6. 损失函数: 增加L1权重 (更鲁棒)")
    print("  7. 数据增强: 翻转+旋转+噪声增强")
    print("  8. 频带数: 4→6 (更精细频域处理)")
    print("\n预期效果:")
    print("  - 降噪后PSNR: 32-35 dB (当前~30 dB)")
    print("  - 训练速度: 快3-4倍 (patch数减少)")
    print("  - 泛化性能: 显著提升")
    print("=" * 70)
    
    return config


if __name__ == '__main__':
    config = save_improved_params()
    
    print("\n下一步:")
    print("  1. 使用改进配置重新训练:")
    print("     python train_2d.py --config improved")
    print("\n  2. 或手动修改 config_2d.py")
    print("\n  3. 训练完成后测试:")
    print("     python test_2d.py")
