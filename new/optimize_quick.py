"""
快速超参数优化 - 简化版
只优化最关键的参数，快速找到合理配置
"""

import os
import sys
import json
from glob import glob
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model_2d import create_model_2d
from utils_2d import get_dataloader_2d, add_noise_2d, calculate_metrics_2d
from config_2d import DATA_CONFIG, PATHS

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    print("❌ Optuna未安装。请运行: pip install optuna")
    sys.exit(1)


def objective(trial, train_loader, val_loader, device):
    """优化目标函数 - 只优化关键参数"""
    
    print(f"\n{'='*60}")
    print(f"Trial {trial.number}")
    print(f"{'='*60}")
    
    # 只优化最关键的参数
    config = {
        # 关键参数（优化）
        'd_model': trial.suggest_categorical('d_model', [32, 48, 64]),
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 5e-4, log=True),
        'dropout': trial.suggest_float('dropout', 0.1, 0.3),
        
        # 次要参数（小范围优化）
        'num_layers': trial.suggest_int('num_layers', 2, 4),
        'batch_size': trial.suggest_categorical('batch_size', [64, 128]),
        
        # 固定参数（使用经验值）
        'wavelet_level': 3,
        'num_heads': 8,
        'd_ff': 256,
        'noise_level': 0.1,
        'gradient_clip': 1.0,
        'weight_decay': 1e-5,
        'mask_ratio': 0.05,
        
        # 创新点开关（先全部启用）
        'use_cross_scale': True,
        'use_sparse_attention': True,
        'sparsity_ratio': 0.3,
        'use_fag': True,
        'fag_num_bands': 4,
        'use_noise_estimator': True,
        'noise_num_scales': 3,
        'use_rdc': True,
        'rdc_growth_rate': 16,
        'rdc_num_blocks': 3,
    }
    
    print(f"  d_model: {config['d_model']}")
    print(f"  learning_rate: {config['learning_rate']:.6f}")
    print(f"  dropout: {config['dropout']:.2f}")
    print(f"  num_layers: {config['num_layers']}")
    print(f"  batch_size: {config['batch_size']}")
    
    try:
        # 创建模型
        model = create_model_2d(config).to(device)
        
        # 统计参数
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {total_params:,}")
        
        # 训练配置
        criterion = nn.MSELoss()
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        
        # 快速训练（只训练10个epochs）
        max_epochs = 10
        best_val_loss = float('inf')
        
        for epoch in range(max_epochs):
            # 训练
            model.train()
            train_loss = 0
            train_batches = 0
            
            # 只用部分数据加速
            max_train_batches = 50
            for batch_idx, batch in enumerate(train_loader):
                if batch_idx >= max_train_batches:
                    break
                    
                batch = batch.to(device)
                noisy = add_noise_2d(batch, noise_level=config['noise_level'])
                
                optimizer.zero_grad()
                output, _ = model(noisy)
                loss = criterion(output, batch)
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), 
                                              config['gradient_clip'])
                optimizer.step()
                
                train_loss += loss.item()
                train_batches += 1
            
            train_loss /= train_batches
            
            # 验证
            model.eval()
            val_loss = 0
            val_batches = 0
            
            # 只用部分数据加速
            max_val_batches = 20
            with torch.no_grad():
                for batch_idx, batch in enumerate(val_loader):
                    if batch_idx >= max_val_batches:
                        break
                        
                    batch = batch.to(device)
                    noisy = add_noise_2d(batch, noise_level=config['noise_level'])
                    
                    output, _ = model(noisy)
                    loss = criterion(output, batch)
                    val_loss += loss.item()
                    val_batches += 1
            
            val_loss /= val_batches
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
            
            # 报告中间结果（用于pruning）
            trial.report(val_loss, epoch)
            
            # 检查是否应该剪枝
            if trial.should_prune():
                print(f"  ✂ Pruned at epoch {epoch+1}")
                raise optuna.TrialPruned()
            
            if epoch % 3 == 0:
                print(f"  Epoch {epoch+1}/{max_epochs}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")
        
        print(f"  ✓ Best val_loss: {best_val_loss:.6f}")
        
        # 清理内存
        del model
        torch.cuda.empty_cache()
        
        return best_val_loss
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return float('inf')


if __name__ == "__main__":
    print("\n" + "="*70)
    print("快速超参数优化 - 简化版")
    print("只优化关键参数：d_model, learning_rate, dropout, num_layers")
    print("="*70)
    
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # 加载数据
    print("\n加载数据...")
    available_data = sorted(glob(os.path.join(PATHS['data'], '*.npy')))
    train_data_path = available_data[0] if available_data else None
    val_data_path = available_data[1] if len(available_data) > 1 else train_data_path
    
    if not train_data_path or not os.path.exists(train_data_path):
        print(f"❌ 数据不存在: {train_data_path}")
        sys.exit(1)
    
    # 小batch加速
    train_loader = get_dataloader_2d(
        train_data_path,
        batch_size=128,
        patch_size=DATA_CONFIG['patch_size'],
        stride=DATA_CONFIG['stride'],
        augment=True,
        shuffle=True
    )
    
    val_loader = get_dataloader_2d(
        val_data_path if os.path.exists(val_data_path) else train_data_path,
        batch_size=128,
        patch_size=DATA_CONFIG['patch_size'],
        stride=DATA_CONFIG['stride'],
        augment=False,
        shuffle=False
    )
    
    print(f"训练数据: {len(train_loader.dataset)} patches")
    print(f"验证数据: {len(val_loader.dataset)} patches")
    
    # 创建study
    print("\n开始优化...")
    print("  - Trials: 20 (快速模式)")
    print("  - 每个trial训练: 10 epochs")
    print("  - 使用MedianPruner加速")
    
    study = optuna.create_study(
        direction='minimize',
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=3
        ),
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    
    # 运行优化
    study.optimize(
        lambda trial: objective(trial, train_loader, val_loader, device),
        n_trials=20,
        timeout=3600,  # 最多1小时
        show_progress_bar=True
    )
    
    # 结果
    print("\n" + "="*70)
    print("优化完成！")
    print("="*70)
    
    print(f"\nBest trial:")
    print(f"  Value (loss): {study.best_trial.value:.6f}")
    print(f"\nBest hyperparameters:")
    for key, value in study.best_trial.params.items():
        print(f"  {key}: {value}")
    
    # 保存结果
    best_params = {
        'best_loss': float(study.best_trial.value),
        'best_params': study.best_trial.params,
        'n_trials': len(study.trials),
        'optimization_time': sum(t.duration.total_seconds() 
                                for t in study.trials if t.duration is not None)
    }
    
    results_path = os.path.join(PATHS['results'], 'best_params_quick.json')
    with open(results_path, 'w') as f:
        json.dump(best_params, f, indent=4)
    
    print(f"\n✓ 结果已保存到: {results_path}")
    
    # 优化历史
    history_path = os.path.join(PATHS['results'], 'optimization_history_quick.json')
    history = {
        'trials': [
            {
                'number': t.number,
                'value': t.value,
                'params': t.params,
                'state': str(t.state)
            }
            for t in study.trials
        ]
    }
    
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=4)
    
    print(f"✓ 优化历史已保存到: {history_path}")
    
    print("\n" + "="*70)
    print("下一步：使用最佳参数训练完整模型")
    print("  python train_2d.py")
    print("="*70)
