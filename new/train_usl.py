"""
USL DIP 自监督训练脚本
====================
完全复现原始 USL DIP 训练范式:
1. 2D patch 提取 (yc_patch)
2. 基于峰度 (kurtosis) 的 patch 筛选
3. 自监督训练: input = noisy → target = noisy (DIP 核心思想)
4. Adam + MSE loss + EarlyStopping

原始参数 (USL_DIP_FORGE_EQ36_Train.ipynb):
  - patch_size: 24×24, stride: 6×6
  - drop_rate: 0.2 (kurtosis 筛选)
  - batch_size: 1024, epochs: 50
  - lr: 0.01, EarlyStopping patience: 5
  - loss: MSE

用法:
  python train_usl.py --dataset eq-36 --epochs 50
  python train_usl.py --dataset eq-36 --epochs 50 --tag A
"""

import os
import sys
import json
import time
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from scipy.stats import kurtosis
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model_usl import create_usl_model
from utils_2d import yc_patch, EarlyStopping, save_checkpoint


# ============================================================
#  数据处理: 与 USL 原始完全一致
# ============================================================

def remove_patches_kurtosis(patches: np.ndarray, drop_rate: float = 0.2) -> np.ndarray:
    """
    基于峰度筛选 patch —— 与原始 USL remove_columns_kurtosis 一致

    高峰度 patch 包含更多信号成分, 优先保留;
    低峰度 patch 以 drop_rate 比例丢弃 (多为纯噪声 patch)

    Parameters
    ----------
    patches : (N, L)  展平的 patch 数组
    drop_rate : float  丢弃的低峰度 patch 比例, 默认 0.2

    Returns
    -------
    selected : (M, L)  筛选后的 patch 数组
    """
    kurt_values = kurtosis(patches, axis=1)
    threshold = np.percentile(kurt_values, drop_rate * 100)
    selected = patches[kurt_values > threshold]
    return selected


class USLPatchDataset(Dataset):
    """
    USL 自监督 Dataset
    - input = noisy patch
    - target = 同一个 noisy patch (DIP 自监督)
    """

    def __init__(self, patches: np.ndarray, patch_h: int, patch_w: int):
        """
        Parameters
        ----------
        patches : (N, L) 展平 patch
        patch_h, patch_w : patch 空间尺寸
        """
        self.patches = patches.astype(np.float32)
        self.patch_h = patch_h
        self.patch_w = patch_w

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        patch_flat = self.patches[idx]  # (L,)
        # 重塑为 (1, H, W) 以匹配模型输入
        patch_2d = patch_flat.reshape(1, self.patch_h, self.patch_w)
        tensor = torch.from_numpy(patch_2d)
        # 自监督: input = target
        return tensor, tensor


# ============================================================
#  训练逻辑
# ============================================================

def train_usl(
    dataset_name: str = "eq-36",
    tag: str = "A",
    epochs: int = 50,
    batch_size: int = 1024,
    lr: float = 0.01,
    drop_rate: float = 0.2,
    patch_size: tuple = (24, 24),
    stride: tuple = (6, 6),
    patience: int = 5,
    val_split: float = 0.1,
    transpose_data: bool = True,
    model_config: dict = None,
    seed: int = 42,
):
    """
    USL DIP 自监督训练 —— 与原始 USL 训练流程完全一致

    Parameters
    ----------
    dataset_name : str   数据集名, 如 'eq-36' 或 'eq-68'
    tag : str            模型标签, 默认 'A' (用于融合时标识)
    epochs : int         训练轮数, 默认 50
    batch_size : int     批大小, 默认 1024 (与 USL 一致)
    lr : float           学习率, 默认 0.01 (与 USL 一致)
    drop_rate : float    峰度筛选丢弃率, 默认 0.2
    patch_size : tuple   (H, W) patch 尺寸
    stride : tuple       (sH, sW) 步长
    patience : int       EarlyStopping patience, 默认 5
    val_split : float    验证集比例, 默认 0.1
    transpose_data : bool  是否转置数据 (USL 原始对 eq-36 做了 data.T)
    model_config : dict  模型配置 (可选, 默认使用 USL 原始参数)
    seed : int           随机种子
    """
    # 固定随机种子
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"USL DIP Training — Model {tag}")
    print(f"{'='*60}")
    print(f"Device: {device}")

    # ---- 路径 ----
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(os.path.dirname(base_dir), "data")
    ckpt_dir = os.path.join(base_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    data_path = os.path.join(data_dir, f"{dataset_name}.npy")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data not found: {data_path}")

    # ---- 加载数据 ----
    data = np.load(data_path)
    print(f"Original data shape: {data.shape}")

    if transpose_data:
        data = data.T
        print(f"After transpose: {data.shape}")

    # ---- 2D Patching (与 USL yc_patch 完全一致) ----
    w1, w2 = patch_size
    z1, z2 = stride
    patches = yc_patch(data, w1, w2, z1, z2)
    # yc_patch 可能返回 (N, H, W)，需要展平为 (N, H*W) 以匹配原始 USL 格式
    if patches.ndim == 3:
        patches = patches.reshape(patches.shape[0], -1)
    print(f"Patches shape: {patches.shape}")

    # ---- 峰度筛选 (与 USL remove_columns_kurtosis 一致) ----
    patches_selected = remove_patches_kurtosis(patches, drop_rate)
    print(f"After kurtosis selection (drop_rate={drop_rate}): {patches_selected.shape}")

    # ---- 划分训练/验证 ----
    num_patches = len(patches_selected)
    indices = np.random.permutation(num_patches)
    val_size = int(num_patches * val_split)
    train_patches = patches_selected[indices[val_size:]]
    val_patches = patches_selected[indices[:val_size]]
    print(f"Train: {len(train_patches)}, Val: {len(val_patches)}")

    train_dataset = USLPatchDataset(train_patches, w1, w2)
    val_dataset = USLPatchDataset(val_patches, w1, w2)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=torch.cuda.is_available(),
    )

    # ---- 模型 ----
    if model_config is None:
        model_config = {"patch_size": patch_size}
    else:
        model_config.setdefault("patch_size", patch_size)

    model = create_usl_model(model_config).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # ---- 优化器 & 损失 (与 USL 一致: Adam + MSE) ----
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # ---- EarlyStopping ----
    early_stopping = EarlyStopping(patience=patience, min_delta=1e-6, verbose=True)

    # ---- 训练循环 ----
    best_val = float("inf")
    best_path = os.path.join(ckpt_dir, f"fusion_{tag}_best.pth")
    last_path = os.path.join(ckpt_dir, f"fusion_{tag}_last.pth")

    history = {"train_loss": [], "val_loss": []}

    for epoch in range(epochs):
        start = time.time()

        # --- Train ---
        model.train()
        train_loss_sum = 0.0
        for noisy, target in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            noisy = noisy.to(device)
            target = target.to(device)

            optimizer.zero_grad()
            pred, _ = model(noisy)  # 自监督: pred 应逼近 noisy
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item()

        train_loss = train_loss_sum / max(len(train_loader), 1)

        # --- Val ---
        model.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            for noisy, target in val_loader:
                noisy = noisy.to(device)
                target = target.to(device)
                pred, _ = model(noisy)
                loss = criterion(pred, target)
                val_loss_sum += loss.item()

        val_loss = val_loss_sum / max(len(val_loader), 1)

        elapsed = time.time() - start
        print(f"[{tag}] Epoch {epoch+1}/{epochs} | train={train_loss:.6f} | val={val_loss:.6f} | {elapsed:.1f}s")

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        # Save
        metrics = {"train_loss": train_loss, "val_loss": val_loss}
        save_checkpoint(model, optimizer, epoch, val_loss, metrics, last_path)

        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(model, optimizer, epoch, val_loss, metrics, best_path)
            print(f"  → Best model saved (val_loss={val_loss:.6f})")

        early_stopping(val_loss)
        if early_stopping.early_stop:
            print(f"[{tag}] EarlyStopping triggered at epoch {epoch+1}")
            break

    # ---- 保存训练摘要 ----
    results_dir = os.path.join(base_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    summary = {
        "model_type": "usl",
        "tag": tag,
        "dataset": dataset_name,
        "patch_size": list(patch_size),
        "stride": list(stride),
        "drop_rate": drop_rate,
        "batch_size": batch_size,
        "lr": lr,
        "epochs_trained": len(history["train_loss"]),
        "best_val_loss": best_val,
        "total_params": total_params,
        "checkpoint": best_path,
        "history": history,
    }
    summary_path = os.path.join(results_dir, f"usl_train_{tag}_{dataset_name}_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n✓ Training complete. Best checkpoint: {best_path}")
    print(f"✓ Summary saved: {summary_path}")
    return best_path


# ============================================================
#  CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="USL DIP Self-supervised Training")
    parser.add_argument("--dataset", type=str, default="eq-36", help="Dataset stem in data/ (e.g. eq-36, eq-68, slice_german, slice_german_1)")
    parser.add_argument("--tag", type=str, default="A", help="Model tag (for fusion)")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs (USL default: 50)")
    parser.add_argument("--batch-size", type=int, default=1024, help="Batch size (USL default: 1024)")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate (USL default: 0.01)")
    parser.add_argument("--drop-rate", type=float, default=0.2, help="Kurtosis drop rate (USL default: 0.2)")
    parser.add_argument("--patch-h", type=int, default=24, help="Patch height")
    parser.add_argument("--patch-w", type=int, default=24, help="Patch width")
    parser.add_argument("--stride-h", type=int, default=6, help="Stride height")
    parser.add_argument("--stride-w", type=int, default=6, help="Stride width")
    parser.add_argument("--patience", type=int, default=5, help="EarlyStopping patience")
    parser.add_argument("--no-transpose", action="store_true", help="Don't transpose data")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    train_usl(
        dataset_name=args.dataset,
        tag=args.tag,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        drop_rate=args.drop_rate,
        patch_size=(args.patch_h, args.patch_w),
        stride=(args.stride_h, args.stride_w),
        patience=args.patience,
        transpose_data=not args.no_transpose,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
