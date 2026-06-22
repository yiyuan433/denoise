"""
Supervised training for 2D denoising models (ours + baselines).
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
    add_noise_2d,
    calculate_metrics_2d,
    EarlyStopping,
    save_checkpoint,
)


class CompositeLoss2D(nn.Module):
    """Combination of MSE/L1/Spectral/Perceptual losses."""

    def __init__(self, weights):
        super().__init__()
        self.weights = weights
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()

    def forward(self, pred, target, aux_outputs=None):
        losses = {}

        mask = None
        if isinstance(aux_outputs, dict):
            mask = aux_outputs.get("mask")

        pred_pixel = pred
        target_pixel = target
        if mask is not None and mask.shape == pred.shape:
            masked_region = mask < 0.5
            if masked_region.any():
                pred_pixel = pred[masked_region]
                target_pixel = target[masked_region]

        if self.weights.get("mse", 0) > 0:
            losses["mse"] = self.mse(pred_pixel, target_pixel)
        if self.weights.get("l1", 0) > 0:
            losses["l1"] = self.l1(pred_pixel, target_pixel)
        if self.weights.get("spectral", 0) > 0:
            losses["spectral"] = self.spectral_loss(pred, target)
        if self.weights.get("perceptual", 0) > 0:
            losses["perceptual"] = self.perceptual_loss(pred, target)

        total = sum(self.weights.get(k, 0) * v for k, v in losses.items())
        return total, losses

    def spectral_loss(self, pred, target):
        pred_fft = torch.fft.rfft2(pred.float(), dim=(-2, -1))
        target_fft = torch.fft.rfft2(target.float(), dim=(-2, -1))
        return self.l1(torch.abs(pred_fft), torch.abs(target_fft))

    def perceptual_loss(self, pred, target):
        pred_grad_x = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        pred_grad_y = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        target_grad_x = target[:, :, :, 1:] - target[:, :, :, :-1]
        target_grad_y = target[:, :, 1:, :] - target[:, :, :-1, :]
        return self.l1(pred_grad_x, target_grad_x) + self.l1(pred_grad_y, target_grad_y)


def _build_model(model_name, cfg):
    if model_name == "ours":
        model_cfg = cfg["model"].copy()
        if cfg["train"].get("training_paradigm", "supervised") == "supervised":
            # Disable blind-spot mask when using supervised targets.
            model_cfg["mask_ratio"] = 0.0
        return create_model_2d(model_cfg)

    if model_name == "dncnn":
        return create_baseline_model("dncnn")

    if model_name == "unet":
        return create_baseline_model("unet")

    raise ValueError(f"Unknown model: {model_name}")


def _forward_model(model, noisy):
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


def train_one_epoch(model, loader, criterion, optimizer, device, noise_levels, training_paradigm):
    model.train()
    total_loss = 0.0
    comp = {}

    for batch in tqdm(loader, desc="Train", leave=False):
        clean = batch.to(device)

        if training_paradigm == "blind_spot":
            noisy = clean
        else:
            noise_level = random.choice(noise_levels)
            noisy = add_noise_2d(clean, noise_level=noise_level)

        optimizer.zero_grad()
        pred, aux = _forward_model(model, noisy)
        loss, losses_dict = criterion(pred, clean, aux)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        for k, v in losses_dict.items():
            comp[k] = comp.get(k, 0.0) + v.item()

    avg_loss = total_loss / max(len(loader), 1)
    avg_comp = {k: v / max(len(loader), 1) for k, v in comp.items()}
    return avg_loss, avg_comp


def validate(model, loader, criterion, device, noise_levels, training_paradigm):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Val", leave=False):
            clean = batch.to(device)
            if training_paradigm == "blind_spot":
                noisy = clean
            else:
                noise_level = random.choice(noise_levels)
                noisy = add_noise_2d(clean, noise_level=noise_level)
            pred, aux = _forward_model(model, noisy)
            loss, _ = criterion(pred, clean, aux)
            total_loss += loss.item()

    return total_loss / max(len(loader), 1)


def build_dataloaders(cfg, dataset_name, val_dataset_name=None):
    train_cfg = cfg["train"]
    data_cfg = cfg["data"]

    train_path = data_cfg["datasets"][dataset_name]
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Training data not found: {train_path}")

    dataset = DASDataset2D(
        train_path,
        patch_size=data_cfg["patch_size"],
        stride=data_cfg["stride"],
        augment=train_cfg.get("augmentation", True),
    )

    if val_dataset_name:
        val_path = data_cfg["datasets"][val_dataset_name]
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

    return train_loader, val_loader


def train_model(cfg, model_name, train_loader, val_loader, device, tag=None):
    train_cfg = cfg["train"]
    training_paradigm = train_cfg.get("training_paradigm", "supervised")

    model = _build_model(model_name, cfg).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {model_name} | Params: {total_params:,}")

    optimizer = optim.AdamW(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
    )

    criterion = CompositeLoss2D(train_cfg["loss_weights"])
    noise_levels = train_cfg.get("noise_levels", [0.1])

    early_stopping = EarlyStopping(
        patience=train_cfg.get("early_stopping_patience", 10),
        min_delta=1e-6,
        verbose=True,
    )

    history = {"train_loss": [], "val_loss": [], "components": []}
    best_val = float("inf")
    start_time = time.time()

    ckpt_dir = cfg["paths"]["checkpoints"]
    os.makedirs(ckpt_dir, exist_ok=True)
    tag_part = f"_{tag}" if tag else ""
    ckpt_name = f"best_{model_name}{tag_part}.pth"
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)

    for epoch in range(train_cfg["epochs"]):
        train_loss, comp = train_one_epoch(
            model, train_loader, criterion, optimizer, device, noise_levels, training_paradigm
        )
        val_loss = validate(model, val_loader, criterion, device, noise_levels, training_paradigm)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["components"].append(comp)

        print(
            f"Epoch {epoch + 1}/{train_cfg['epochs']} | "
            f"train={train_loss:.6f} val={val_loss:.6f}"
        )

        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(model, optimizer, epoch, val_loss, comp, ckpt_path)

        if early_stopping(val_loss):
            print("Early stopping triggered.")
            break

    total_time = time.time() - start_time
    print(f"Training done in {total_time / 60:.1f} min. Best val={best_val:.6f}")

    return ckpt_path, history


def main():
    parser = argparse.ArgumentParser(description="Train 2D denoising models")
    parser.add_argument("--model", choices=["ours", "dncnn", "unet"], default="ours")
    parser.add_argument("--dataset", choices=["eq-36", "eq-68", "slice_german", "slice_german_1"], default="eq-36")
    parser.add_argument("--val-dataset", choices=["eq-36", "eq-68", "slice_german", "slice_german_1"], default=None)
    parser.add_argument("--training-paradigm", choices=["auto", "supervised", "blind_spot"], default="auto")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    args = parser.parse_args()

    cfg = get_config("train")
    train_cfg = cfg["train"]

    if args.epochs is not None:
        train_cfg["epochs"] = args.epochs
    if args.batch_size is not None:
        train_cfg["batch_size"] = args.batch_size

    if args.training_paradigm == "auto":
        train_cfg["training_paradigm"] = "blind_spot" if args.model == "ours" else "supervised"
    else:
        train_cfg["training_paradigm"] = args.training_paradigm

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_loader, val_loader = build_dataloaders(cfg, args.dataset, args.val_dataset)

    ckpt_path, history = train_model(
        cfg, args.model, train_loader, val_loader, device, tag=args.dataset
    )

    # Save history
    results_dir = cfg["paths"]["results"]
    os.makedirs(results_dir, exist_ok=True)
    hist_path = os.path.join(results_dir, f"train_history_{args.model}_{args.dataset}.json")
    with open(hist_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"History saved: {hist_path}")
    print(f"Checkpoint saved: {ckpt_path}")


if __name__ == "__main__":
    main()
