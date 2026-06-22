"""
针对无干净真值数据的自监督训练脚本
- 使用 blind-spot 自监督方式训练
- 仅用无参考指标评估
"""

import argparse
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from config_2d import get_config
from model_2d import create_model_2d
from baseline_models import create_baseline_model
from utils_2d import (
    DASDataset2D,
    calculate_no_ref_metrics,
    EarlyStopping,
    save_checkpoint,
)


class BlindSpotLoss(nn.Module):
    """
    Blind-Spot 自监督损失
    在掩盖的区域上计算损失，使模型通过自己预测自己来学习去噪
    """
    def __init__(self, mask_ratio=0.05):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.mse = nn.MSELoss()

    def forward(self, pred, noisy, mask=None):
        """
        Args:
            pred: 模型预测输出 (B, 1, H, W)
            noisy: 输入的噪声数据 (B, 1, H, W)
            mask: 预生成的mask (B, 1, H, W) [0=masked, 1=unmasked]
        
        Returns:
            loss: 标量损失
        """
        B, C, H, W = pred.shape
        
        # 生成或使用提供的mask
        if mask is None:
            # 随机生成mask：mask_ratio比例的像素被掩盖
            mask = torch.rand(B, 1, H, W, device=pred.device) > self.mask_ratio
            mask = mask.float()  # (B, 1, H, W)
        
        # 在掩盖区域计算损失
        # mask=1 表示该像素被保留（不掩盖）
        # mask=0 表示该像素被掩盖
        masked_pred = pred * (1 - mask)  # 只保留掩盖区域的预测
        masked_target = noisy * (1 - mask)  # 掩盖区域的目标也是输入本身
        
        # 损失只在掩盖区域计算
        num_masked = torch.sum(1 - mask)
        if num_masked > 0:
            loss = self.mse(masked_pred, masked_target)
        else:
            loss = torch.tensor(0.0, device=pred.device, requires_grad=True)
        
        return loss


def _build_self_supervised_model(model_name, cfg):
    """构建支持自监督的模型"""
    if model_name == "ours":
        model_cfg = cfg["model"].copy()
        # 确保启用 mask_ratio
        if model_cfg.get("mask_ratio", 0) == 0:
            model_cfg["mask_ratio"] = 0.05
        return create_model_2d(model_cfg)
    
    if model_name == "dncnn":
        # DnCNN 基线无原生自监督支持，需要包装
        return create_baseline_model("dncnn")
    
    if model_name == "unet":
        # U-Net 基线无原生自监督支持，需要包装
        return create_baseline_model("unet")
    
    raise ValueError(f"Unknown model: {model_name}")


def _forward_model(model, noisy, mask=None):
    """前向传播"""
    out = model(noisy)
    if isinstance(out, (tuple, list)):
        pred = out[0]
        aux = out[1] if len(out) > 1 and isinstance(out[1], dict) else {}
        if len(out) > 2 and out[2] is not None:
            aux = dict(aux)
            aux["mask"] = out[2]
    else:
        pred = out
        aux = {}
    return pred, aux


def train_one_epoch_self_supervised(model, loader, criterion, optimizer, device, mask_ratio=0.05):
    """使用 blind-spot 自监督方式训练一个 epoch"""
    model.train()
    total_loss = 0.0

    for batch in tqdm(loader, desc="Train (self-supervised)", leave=False):
        noisy = batch.to(device)  # 直接使用噪声数据作为输入

        # 生成随机 mask
        B, C, H, W = noisy.shape
        mask = torch.rand(B, 1, H, W, device=device) > mask_ratio
        mask = mask.float()

        optimizer.zero_grad()
        pred, aux = _forward_model(model, noisy)
        
        # 使用 blind-spot 损失
        loss = criterion(pred, noisy, mask)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / max(len(loader), 1)
    return avg_loss


def validate_self_supervised(model, loader, criterion, device, mask_ratio=0.05):
    """验证阶段（自监督）"""
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Val (self-supervised)", leave=False):
            noisy = batch.to(device)

            # 生成随机 mask
            B, C, H, W = noisy.shape
            mask = torch.rand(B, 1, H, W, device=device) > mask_ratio
            mask = mask.float()

            pred, aux = _forward_model(model, noisy)
            loss = criterion(pred, noisy, mask)
            total_loss += loss.item()

    return total_loss / max(len(loader), 1)


def train_model_self_supervised(cfg, model_name, train_loader, val_loader, device, tag=None):
    """自监督模型训练（无需干净真值）"""
    train_cfg = cfg["train"]
    mask_ratio = train_cfg.get("mask_ratio", 0.05)

    model = _build_self_supervised_model(model_name, cfg).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {model_name} | Params: {total_params:,}")
    print(f"Training paradigm: SELF-SUPERVISED (Blind-Spot)")
    print(f"Mask ratio: {mask_ratio}")

    optimizer = optim.AdamW(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
    )

    criterion = BlindSpotLoss(mask_ratio=mask_ratio)

    early_stopping = EarlyStopping(
        patience=train_cfg.get("early_stopping_patience", 10),
        min_delta=1e-6,
        verbose=True,
    )

    history = {"train_loss": [], "val_loss": []}
    best_val = float("inf")
    start_time = time.time()

    ckpt_dir = cfg["paths"]["checkpoints"]
    os.makedirs(ckpt_dir, exist_ok=True)
    tag_part = f"_{tag}" if tag else ""
    ckpt_name = f"best_{model_name}{tag_part}_ss.pth"  # ss = self-supervised
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)

    for epoch in range(train_cfg["epochs"]):
        train_loss = train_one_epoch_self_supervised(
            model, train_loader, criterion, optimizer, device, mask_ratio
        )
        val_loss = validate_self_supervised(model, val_loader, criterion, device, mask_ratio)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        print(
            f"Epoch {epoch + 1}/{train_cfg['epochs']} | "
            f"train={train_loss:.6f} val={val_loss:.6f}"
        )

        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(model, optimizer, epoch, val_loss, {}, ckpt_path)

        if early_stopping(val_loss):
            print("Early stopping triggered.")
            break

    total_time = time.time() - start_time
    print(f"Training done in {total_time / 60:.1f} min. Best val={best_val:.6f}")

    return ckpt_path, history


def main():
    parser = argparse.ArgumentParser(description="Train 2D denoising models (self-supervised)")
    parser.add_argument("--model", choices=["ours", "dncnn", "unet"], default="ours")
    parser.add_argument("--dataset", choices=["eq-36", "eq-68", "slice_german", "slice_german_1"], default="eq-36")
    parser.add_argument("--val-dataset", choices=["eq-36", "eq-68", "slice_german", "slice_german_1"], default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--mask-ratio", type=float, default=0.05)
    args = parser.parse_args()

    cfg = get_config("train")
    train_cfg = cfg["train"]

    if args.epochs is not None:
        train_cfg["epochs"] = args.epochs
    if args.batch_size is not None:
        train_cfg["batch_size"] = args.batch_size
    if args.mask_ratio is not None:
        train_cfg["mask_ratio"] = args.mask_ratio

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    data_cfg = cfg["data"]
    train_path = data_cfg["datasets"][args.dataset]
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Training data not found: {train_path}")

    dataset = DASDataset2D(
        train_path,
        patch_size=data_cfg["patch_size"],
        stride=data_cfg["stride"],
        augment=train_cfg.get("augmentation", True),
    )

    if args.val_dataset:
        val_path = data_cfg["datasets"][args.val_dataset]
        if not os.path.exists(val_path):
            raise FileNotFoundError(f"Validation data not found: {val_path}")
        val_dataset = DASDataset2D(
            val_path,
            patch_size=data_cfg["patch_size"],
            stride=data_cfg["stride"],
            augment=False,
        )
    else:
        val_size = int(len(dataset) * train_cfg.get("val_split", 0.1))
        train_size = len(dataset) - val_size
        dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    ckpt_path, history = train_model_self_supervised(
        cfg, args.model, train_loader, val_loader, device, tag=args.dataset
    )

    # 保存训练历史
    history_path = ckpt_path.replace(".pth", "_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"Checkpoint: {ckpt_path}")
    print(f"History: {history_path}")


if __name__ == "__main__":
    main()
