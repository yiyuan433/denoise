"""
Paper sweeps for TV sensitivity and patch strategy studies.
"""

import argparse
import json
import os
import time
from glob import glob

import numpy as np
import torch

from config_2d import get_config
from test_2d import load_trained_model, denoise_2d_data
from traditional_denoise import tv_denoise
from utils_2d import add_noise_2d, calculate_metrics_2d, calculate_no_ref_metrics


def _resolve_path(path: str) -> str:
    if not path:
        return path
    if os.path.isabs(path):
        return path
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), path)


def _find_default_checkpoint(checkpoints_dir: str, datasets: list[str]) -> str | None:
    candidate = os.path.join(checkpoints_dir, "best_model_2d.pth")
    if os.path.exists(candidate):
        return candidate

    for ds in datasets:
        candidate = os.path.join(checkpoints_dir, f"best_ours_{ds}.pth")
        if os.path.exists(candidate):
            return candidate

    candidates = glob(os.path.join(checkpoints_dir, "best_ours_*.pth"))
    if candidates:
        candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        return candidates[0]

    return None


def _parse_crop(crop_str: str | None):
    if not crop_str:
        return None
    if "x" not in crop_str:
        raise ValueError("Crop must be like 512x512")
    h, w = crop_str.lower().split("x")
    return int(h), int(w)


def _parse_weights(weights_str: str) -> list[float]:
    return [float(x) for x in weights_str.split(",") if x.strip()]


def _parse_patch_specs(spec_str: str) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    specs = []
    for item in spec_str.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError("Patch spec must be like 24x24:6x6")
        patch, stride = item.split(":")
        ph, pw = [int(x) for x in patch.lower().split("x")]
        sh, sw = [int(x) for x in stride.lower().split("x")]
        specs.append(((ph, pw), (sh, sw)))
    return specs


def _load_data(data_path: str, crop=None):
    data = np.load(data_path)
    if crop:
        h, w = crop
        data = data[:h, :w]
    return data


def _add_synthetic_noise(data: np.ndarray, noise_level: float) -> np.ndarray:
    tensor = torch.from_numpy(data).unsqueeze(0).unsqueeze(0).float()
    noisy = add_noise_2d(tensor, noise_level=noise_level).squeeze().numpy()
    return noisy


def _save_json(path: str, payload: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def run_tv_sweep(cfg, args):
    paths = cfg["paths"]
    datasets = list(cfg["data"]["datasets"].keys())

    if args.checkpoint:
        ckpt_path = _resolve_path(args.checkpoint)
    else:
        ckpt_path = _find_default_checkpoint(paths["checkpoints"], datasets)
    if not ckpt_path or not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    data_path = cfg["data"]["datasets"][args.dataset]
    data = _load_data(data_path, crop=_parse_crop(args.crop))

    if args.real:
        noisy = data
        clean = None
    else:
        noisy = _add_synthetic_noise(data, args.noise_level)
        clean = data

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_trained_model(ckpt_path, device, use_best_params=True)

    ph, pw = cfg["data"]["patch_size"]
    sh, sw = cfg["data"]["stride"]

    start = time.time()
    dl_denoised, _stats = denoise_2d_data(
        model,
        noisy,
        patch_size=(ph, pw),
        stride=(sh, sw),
        device=device,
        batch_size=args.batch_size,
    )
    dl_time = time.time() - start

    results = []
    for weight in _parse_weights(args.weights):
        t0 = time.time()
        tv_out = tv_denoise(dl_denoised, weight=weight)
        t1 = time.time() - t0

        if clean is not None:
            metrics = calculate_metrics_2d(tv_out, clean)
        else:
            metrics = calculate_no_ref_metrics(noisy, tv_out)

        results.append({
            "weight": weight,
            "tv_time": t1,
            "metrics": metrics,
        })

    out_name = f"{args.dataset}_tv_sweep"
    suffix = "_real" if args.real else f"_noise{args.noise_level:g}"
    out_json = os.path.join(paths["results"], f"{out_name}{suffix}.json")
    _save_json(out_json, {
        "dataset": args.dataset,
        "mode": "real" if args.real else "synthetic",
        "noise_level": args.noise_level if not args.real else None,
        "checkpoint": ckpt_path,
        "dl_time": dl_time,
        "results": results,
    })

    if args.plot:
        import matplotlib.pyplot as plt

        weights = [r["weight"] for r in results]
        snrs = [r["metrics"].get("snr", 0.0) for r in results]
        psnrs = [r["metrics"].get("psnr", 0.0) for r in results]
        ssims = [r["metrics"].get("ssim", 0.0) for r in results]

        fig = plt.figure(figsize=(6.5, 4))
        plt.plot(weights, snrs, marker="o", label="SNR")
        plt.plot(weights, psnrs, marker="s", label="PSNR")
        if any(ssims):
            plt.plot(weights, ssims, marker="^", label="SSIM")
        plt.xlabel("TV weight (lambda)")
        plt.ylabel("Metric")
        title = f"TV sensitivity - {args.dataset}"
        if not args.real:
            title += f" (noise={args.noise_level:g})"
        plt.title(title)
        plt.grid(alpha=0.3)
        plt.legend()

        fig_path = os.path.join(paths["figures"], f"{out_name}{suffix}.png")
        plt.tight_layout()
        plt.savefig(fig_path, dpi=200, bbox_inches="tight")
        plt.close(fig)



def run_patch_sweep(cfg, args):
    paths = cfg["paths"]
    datasets = list(cfg["data"]["datasets"].keys())

    if args.checkpoint:
        ckpt_path = _resolve_path(args.checkpoint)
    else:
        ckpt_path = _find_default_checkpoint(paths["checkpoints"], datasets)
    if not ckpt_path or not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    data_path = cfg["data"]["datasets"][args.dataset]
    data = _load_data(data_path, crop=_parse_crop(args.crop))

    if args.real:
        noisy = data
        clean = None
    else:
        noisy = _add_synthetic_noise(data, args.noise_level)
        clean = data

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_trained_model(ckpt_path, device, use_best_params=True)

    results = []
    for (ph, pw), (sh, sw) in _parse_patch_specs(args.patch_specs):
        start = time.time()
        denoised, _stats = denoise_2d_data(
            model,
            noisy,
            patch_size=(ph, pw),
            stride=(sh, sw),
            device=device,
            batch_size=args.batch_size,
        )
        elapsed = time.time() - start

        if clean is not None:
            metrics = calculate_metrics_2d(denoised, clean)
        else:
            metrics = calculate_no_ref_metrics(noisy, denoised)

        overlap_h = 1.0 - (sh / float(ph))
        overlap_w = 1.0 - (sw / float(pw))

        results.append({
            "patch_size": [ph, pw],
            "stride": [sh, sw],
            "overlap": [overlap_h, overlap_w],
            "time": elapsed,
            "metrics": metrics,
        })

    out_name = f"{args.dataset}_patch_sweep"
    suffix = "_real" if args.real else f"_noise{args.noise_level:g}"
    out_json = os.path.join(paths["results"], f"{out_name}{suffix}.json")
    _save_json(out_json, {
        "dataset": args.dataset,
        "mode": "real" if args.real else "synthetic",
        "noise_level": args.noise_level if not args.real else None,
        "checkpoint": ckpt_path,
        "results": results,
    })

    if args.plot:
        import matplotlib.pyplot as plt

        labels = [f"{r['patch_size'][0]}x{r['patch_size'][1]}" for r in results]
        snrs = [r["metrics"].get("snr", 0.0) for r in results]
        psnrs = [r["metrics"].get("psnr", 0.0) for r in results]
        times = [r["time"] for r in results]

        fig, ax1 = plt.subplots(figsize=(7, 4))
        ax1.plot(labels, snrs, marker="o", label="SNR")
        ax1.plot(labels, psnrs, marker="s", label="PSNR")
        ax1.set_xlabel("Patch size")
        ax1.set_ylabel("dB")
        ax1.grid(alpha=0.3)

        ax2 = ax1.twinx()
        ax2.plot(labels, times, marker="^", color="tab:red", label="Time (s)")
        ax2.set_ylabel("Seconds")

        lines, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels1 + labels2, loc="best")

        title = f"Patch sweep - {args.dataset}"
        if not args.real:
            title += f" (noise={args.noise_level:g})"
        plt.title(title)

        fig_path = os.path.join(paths["figures"], f"{out_name}{suffix}.png")
        plt.tight_layout()
        plt.savefig(fig_path, dpi=200, bbox_inches="tight")
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Paper sweep utilities")
    parser.add_argument("--mode", choices=["tv", "patch"], required=True)
    parser.add_argument("--dataset", choices=["eq-36", "eq-68", "slice_german", "slice_german_1"], default="eq-36")
    parser.add_argument("--noise-level", type=float, default=0.1)
    parser.add_argument("--real", action="store_true")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--weights", type=str, default="0.01,0.03,0.05,0.1,0.2")
    parser.add_argument(
        "--patch-specs",
        type=str,
        default="24x24:6x6,32x32:8x8,48x48:12x12",
        help="Comma-separated patch specs like 24x24:6x6",
    )
    parser.add_argument("--crop", type=str, default=None, help="Optional crop like 512x512")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    cfg = get_config("test")

    if args.mode == "tv":
        run_tv_sweep(cfg, args)
    elif args.mode == "patch":
        run_patch_sweep(cfg, args)


if __name__ == "__main__":
    main()
