"""
针对极低SNR数据的训练脚本
专门优化用于SNR < 5 dB的情况（如EQ-36: SNR=3.45dB）
新增：FDA + NAAG模块支持
"""

import os
import sys
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
from tqdm import tqdm
import warnings
from torch.cuda.amp import autocast, GradScaler
warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model_2d import create_model_2d
from utils_2d import (
    DASDataset2D, calculate_metrics_2d,
    plot_2d_comparison, EarlyStopping,
    save_checkpoint, load_checkpoint
)
from config_low_snr import get_low_snr_config, print_config_summary


class RobustPreprocessor:
    """鲁棒数据预处理器"""
    
    def __init__(self, config):
        self.config = config
        self.fitted = False
        self.stats = {}
    
    def fit(self, data):
        """拟合预处理参数"""
        # 计算鲁棒统计量
        if self.config['normalization'] == 'robust':
            q_low, q_high = self.config['robust_params']['quantile_range']
            self.stats['q_low'] = np.percentile(data, q_low)
            self.stats['q_high'] = np.percentile(data, q_high)
            self.stats['median'] = np.median(data)
            self.stats['iqr'] = self.stats['q_high'] - self.stats['q_low']
        
        # 异常值裁剪阈值
        self.stats['clip_low'] = np.percentile(data, self.config['clip_lower'])
        self.stats['clip_high'] = np.percentile(data, self.config['clip_upper'])
        
        self.fitted = True
        return self
    
    def transform(self, data):
        """应用预处理"""
        if not self.fitted:
            raise RuntimeError("Preprocessor must be fitted before transform")
        
        # 1. 裁剪异常值
        data = np.clip(data, self.stats['clip_low'], self.stats['clip_high'])
        
        # 2. 去趋势（可选）
        if self.config.get('detrend', False):
            for i in range(data.shape[1]):
                trend = np.polyfit(np.arange(len(data)), data[:, i], 1)
                data[:, i] -= np.polyval(trend, np.arange(len(data)))
        
        # 3. 归一化
        if self.config['normalization'] == 'robust':
            if self.config['robust_params']['with_centering']:
                data = data - self.stats['median']
            if self.config['robust_params']['with_scaling']:
                data = data / (self.stats['iqr'] + 1e-8)
        
        return data
    
    def inverse_transform(self, data):
        """逆变换"""
        if self.config['normalization'] == 'robust':
            if self.config['robust_params']['with_scaling']:
                data = data * (self.stats['iqr'] + 1e-8)
            if self.config['robust_params']['with_centering']:
                data = data + self.stats['median']
        
        return data


class EnhancedLoss2D(nn.Module):
    """增强的2D组合损失（针对低SNR + FDA/NAAG）"""
    
    def __init__(self, weights):
        super().__init__()
        self.weights = weights
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()
    
    def forward(self, pred, target, aux_outputs=None):
        """
        计算总损失
        aux_outputs: 辅助输出（包含NAAG和FDA的中间结果）
        """
        losses = {}
        
        # L1损失（主要，鲁棒）
        if 'l1' in self.weights:
            losses['l1'] = self.l1(pred, target)
        
        # MSE损失（辅助）
        if 'mse' in self.weights:
            losses['mse'] = self.mse(pred, target)
        
        # 频谱损失
        if 'spectral' in self.weights:
            losses['spectral'] = self.spectral_loss_2d(pred, target)
        
        # SSIM损失
        if 'ssim' in self.weights:
            losses['ssim'] = 1.0 - self.ssim_loss_2d(pred, target)
        
        # 感知损失（简化版）
        if 'perceptual' in self.weights:
            losses['perceptual'] = self.perceptual_loss_2d(pred, target)
        
        # === 新增：NAAG正则化损失 ===
        if aux_outputs is not None and aux_outputs.get('naag_noise_level') is not None:
            # 鼓励噪声估计准确
            if 'naag_reg' in self.weights:
                noise_level = aux_outputs['naag_noise_level']
                # 噪声水平应该在合理范围内 [0, 1]
                naag_reg = torch.mean((noise_level - 0.5).pow(2))  # 鼓励适中的噪声估计
                losses['naag_reg'] = naag_reg
        
        # === 新增：FDA频域一致性损失 ===
        if aux_outputs is not None and aux_outputs.get('fda_freq_bands') is not None:
            if 'fda_consistency' in self.weights:
                freq_bands = aux_outputs['fda_freq_bands']
                # 鼓励频带权重的平滑分布
                fda_consistency = torch.var(freq_bands, dim=-1).mean()
                losses['fda_consistency'] = fda_consistency
        
        # 加权求和
        total_loss = sum(self.weights.get(k, 0) * v for k, v in losses.items())
        
        return total_loss, losses
    
    def spectral_loss_2d(self, pred, target):
        """2D频谱损失"""
        # 使用float32避免cuFFT的half precision限制
        pred_float32 = pred.float()
        target_float32 = target.float()
        
        pred_fft = torch.fft.rfft2(pred_float32, dim=(-2, -1))
        target_fft = torch.fft.rfft2(target_float32, dim=(-2, -1))
        
        pred_mag = torch.abs(pred_fft)
        target_mag = torch.abs(target_fft)
        
        return self.l1(pred_mag, target_mag)
    
    def ssim_loss_2d(self, pred, target, window_size=11):
        """简化的SSIM损失"""
        C1, C2 = 0.01**2, 0.03**2
        
        mu_pred = torch.nn.functional.avg_pool2d(pred, window_size, stride=1, padding=window_size//2)
        mu_target = torch.nn.functional.avg_pool2d(target, window_size, stride=1, padding=window_size//2)
        
        mu_pred_sq = mu_pred ** 2
        mu_target_sq = mu_target ** 2
        mu_pred_target = mu_pred * mu_target
        
        sigma_pred = torch.nn.functional.avg_pool2d(pred * pred, window_size, stride=1, padding=window_size//2) - mu_pred_sq
        sigma_target = torch.nn.functional.avg_pool2d(target * target, window_size, stride=1, padding=window_size//2) - mu_target_sq
        sigma_pred_target = torch.nn.functional.avg_pool2d(pred * target, window_size, stride=1, padding=window_size//2) - mu_pred_target
        
        ssim_map = ((2 * mu_pred_target + C1) * (2 * sigma_pred_target + C2)) / \
                   ((mu_pred_sq + mu_target_sq + C1) * (sigma_pred + sigma_target + C2))
        
        return ssim_map.mean()
    
    def perceptual_loss_2d(self, pred, target):
        """简化的感知损失（使用梯度）"""
        # 计算Sobel梯度
        pred_grad_x = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        pred_grad_y = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        
        target_grad_x = target[:, :, :, 1:] - target[:, :, :, :-1]
        target_grad_y = target[:, :, 1:, :] - target[:, :, :-1, :]
        
        loss_x = self.l1(pred_grad_x, target_grad_x)
        loss_y = self.l1(pred_grad_y, target_grad_y)
        
        return loss_x + loss_y


class CosineWarmupRestartsScheduler:
    """余弦退火 + 预热 + 重启调度器"""
    
    def __init__(self, optimizer, config):
        self.optimizer = optimizer
        self.warmup_epochs = config['warmup_epochs']
        self.T_0 = config['T_0']
        self.T_mult = config['T_mult']
        self.eta_min = config['eta_min']
        self.restart_decay = config['restart_decay']
        self.base_lr = optimizer.param_groups[0]['lr']
        self.current_epoch = 0
        self.T_cur = 0
        self.T_i = self.T_0
        self.restart_count = 0
    
    def step(self):
        """更新学习率"""
        if self.current_epoch < self.warmup_epochs:
            # 预热阶段：线性增长
            lr = self.base_lr * (self.current_epoch + 1) / self.warmup_epochs
        else:
            # 余弦退火
            epoch_after_warmup = self.current_epoch - self.warmup_epochs
            
            if self.T_cur >= self.T_i:
                # 重启
                self.T_cur = 0
                self.T_i = int(self.T_i * self.T_mult)
                self.restart_count += 1
                self.base_lr = self.base_lr * self.restart_decay
            
            progress = self.T_cur / self.T_i
            lr = self.eta_min + (self.base_lr - self.eta_min) * (1 + np.cos(np.pi * progress)) / 2
            self.T_cur += 1
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        self.current_epoch += 1
        return lr


def train_epoch(model, dataloader, criterion, optimizer, device, config, scaler=None):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    loss_components = {}
    
    pbar = tqdm(dataloader, desc='Training', leave=False)
    accumulation_steps = config['train']['gradient_accumulation']
    
    for batch_idx, (noisy, clean) in enumerate(pbar):
        noisy = noisy.to(device)
        clean = clean.to(device)
        
        # 混合精度训练
        if scaler is not None:
            with autocast():
                pred, aux_outputs = model(noisy)
                loss, losses_dict = criterion(pred, clean, aux_outputs)
                loss = loss / accumulation_steps
            
            scaler.scale(loss).backward()
            
            if (batch_idx + 1) % accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['train']['gradient_clip'])
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:
            pred, aux_outputs = model(noisy)
            loss, losses_dict = criterion(pred, clean, aux_outputs)
            loss = loss / accumulation_steps
            loss.backward()
            
            if (batch_idx + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['train']['gradient_clip'])
                optimizer.step()
                optimizer.zero_grad()
        
        total_loss += loss.item() * accumulation_steps
        
        # 累积损失分量
        for k, v in losses_dict.items():
            if k not in loss_components:
                loss_components[k] = 0
            loss_components[k] += v.item()
        
        pbar.set_postfix({'loss': loss.item() * accumulation_steps})
    
    avg_loss = total_loss / len(dataloader)
    avg_components = {k: v / len(dataloader) for k, v in loss_components.items()}
    
    return avg_loss, avg_components


def validate(model, dataloader, criterion, device):
    """验证"""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for noisy, clean in tqdm(dataloader, desc='Validating', leave=False):
            noisy = noisy.to(device)
            clean = clean.to(device)
            
            pred, aux_outputs = model(noisy)
            loss, _ = criterion(pred, clean, aux_outputs)
            
            total_loss += loss.item()
    
    return total_loss / len(dataloader)


def main():
    """主训练流程"""
    # 加载配置
    config = get_low_snr_config()
    print_config_summary(config)
    
    # 设置随机种子
    torch.manual_seed(config['experiment']['seed'])
    np.random.seed(config['experiment']['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config['experiment']['seed'])
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")
    
    # 加载数据
    print("\n加载数据...")
    data_path = config['data']['datasets']['eq-36']
    data = np.load(data_path)
    print(f"原始数据形状: {data.shape}")
    
    # 数据预处理
    print("应用鲁棒预处理...")
    preprocessor = RobustPreprocessor(config['preprocess'])
    preprocessor.fit(data)
    data_processed = preprocessor.transform(data)
    print(f"预处理后: mean={np.mean(data_processed):.4f}, std={np.std(data_processed):.4f}")
    
    # 创建数据集
    from data_augmentation import create_patches_2d, AugmentedDataset
    patches = create_patches_2d(
        data_processed,
        patch_size=config['data']['patch_size'],
        stride=config['data']['stride']
    )
    print(f"生成patch数量: {len(patches)}")
    
    # 分割训练/验证集
    val_size = int(len(patches) * config['train']['val_split'])
    train_size = len(patches) - val_size
    train_indices = list(range(train_size))
    val_indices = list(range(train_size, len(patches)))
    
    # 使用索引创建子集
    train_patches = patches[train_indices]
    val_patches = patches[val_indices]
    
    # 创建增强数据集
    train_dataset = AugmentedDataset(
        train_patches,
        config['train']['augmentation'],
        add_noise=True
    )
    val_dataset = AugmentedDataset(
        val_patches,
        config['train']['augmentation'],
        add_noise=True
    )
    
    # 创建DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    print(f"训练batch数: {len(train_loader)}, 验证batch数: {len(val_loader)}")
    
    # 创建模型
    print("\n创建模型...")
    model = create_model_2d(config['model'])
    model = model.to(device)
    
    # 统计参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")
    
    # 创建优化器
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['train']['learning_rate'],
        weight_decay=config['train']['weight_decay'],
        betas=config['train']['betas']
    )
    
    # 创建学习率调度器
    scheduler = CosineWarmupRestartsScheduler(optimizer, config['train']['scheduler'])
    
    # 创建损失函数
    criterion = EnhancedLoss2D(config['train']['loss_weights'])
    
    # 混合精度训练
    scaler = GradScaler() if config['train']['use_amp'] else None
    
    # Early stopping
    early_stopping = EarlyStopping(
        patience=config['train']['patience'],
        min_delta=config['train']['min_delta']
    )
    
    # 训练历史
    history = {
        'train_loss': [],
        'val_loss': [],
        'learning_rate': [],
        'loss_components': [],
    }
    
    # 训练循环
    print(f"\n开始训练 {config['train']['epochs']} 个epoch...")
    best_val_loss = float('inf')
    start_time = time.time()
    
    for epoch in range(config['train']['epochs']):
        epoch_start = time.time()
        
        # 训练
        train_loss, loss_components = train_epoch(
            model, train_loader, criterion, optimizer, device, config, scaler
        )
        
        # 验证
        val_loss = validate(model, val_loader, criterion, device)
        
        # 更新学习率
        current_lr = scheduler.step()
        
        # 记录历史
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['learning_rate'].append(current_lr)
        history['loss_components'].append(loss_components)
        
        epoch_time = time.time() - epoch_start
        
        # 打印进度
        print(f"\nEpoch [{epoch+1}/{config['train']['epochs']}] ({epoch_time:.1f}s)")
        print(f"  Train Loss: {train_loss:.6f}")
        print(f"  Val Loss: {val_loss:.6f}")
        print(f"  LR: {current_lr:.2e}")
        print(f"  Components: {', '.join(f'{k}={v:.4f}' for k, v in loss_components.items())}")
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                model, optimizer, epoch, val_loss,
                loss_components,
                os.path.join(config['paths']['checkpoints'], 'best_model.pth')
            )
            print(f"  ✓ 保存最佳模型 (val_loss={val_loss:.6f})")
        
        # 定期保存
        if (epoch + 1) % config['train']['save_every'] == 0:
            save_checkpoint(
                model, optimizer, epoch, val_loss,
                loss_components,
                os.path.join(config['paths']['checkpoints'], f'checkpoint_epoch_{epoch+1}.pth')
            )
        
        # Early stopping检查
        if early_stopping(val_loss):
            print(f"\n早停触发！在epoch {epoch+1}")
            break
    
    total_time = time.time() - start_time
    print(f"\n训练完成！总耗时: {total_time/3600:.2f}小时")
    print(f"最佳验证损失: {best_val_loss:.6f}")
    
    # 保存训练历史
    history_path = os.path.join(config['paths']['results'], 'training_history.json')
    with open(history_path, 'w') as f:
        # 转换numpy类型为Python类型
        history_json = {k: [float(x) if isinstance(x, (np.floating, float)) else x for x in v] 
                       for k, v in history.items() if k != 'loss_components'}
        json.dump(history_json, f, indent=2)
    print(f"训练历史已保存: {history_path}")


if __name__ == "__main__":
    main()
