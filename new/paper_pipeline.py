"""
Paper experiment pipeline:
- run synthetic + real experiments
- collect tables and figure manifest for paper
"""

import argparse
import json
import os
import time
from glob import glob

import numpy as np
import torch

from config_2d import get_config
from test_2d import (
    load_trained_model,
    test_on_data,
    test_trad_only,
    test_hybrid_denoise,
    test_combo_denoise,
)
from utils_2d import calculate_no_ref_metrics
from traditional_denoise import TRAD_METHODS, combo_display_name

try:
    from baseline_models import create_baseline_model
    from model_usl import create_usl_model
    BASELINES_AVAILABLE = True
except Exception:
    BASELINES_AVAILABLE = False

_BASELINE_NAME_MAP = {
    "dncnn": "DnCNN",
    "unet": "U-Net",
    "usl": "USL-DIP",
}


def _resolve_path(path: str) -> str:
    if not path:
        return path
    if os.path.isabs(path):
        return path
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), path)


def _find_default_checkpoint(checkpoints_dir: str, datasets: list[str]) -> str | None:
    """Pick a reasonable default checkpoint if none is provided."""
    candidate = os.path.join(checkpoints_dir, "best_model_2d.pth")
    if os.path.exists(candidate):
        return candidate

    for ds in datasets:
        candidate = os.path.join(checkpoints_dir, f"best_ours_{ds}_ss.pth")
        if os.path.exists(candidate):
            return candidate

        candidate = os.path.join(checkpoints_dir, f"best_ours_{ds}.pth")
        if os.path.exists(candidate):
            return candidate

    candidates = glob(os.path.join(checkpoints_dir, "best_ours_*.pth"))
    if candidates:
        candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        return candidates[0]

    return None


def _find_dataset_checkpoint(checkpoints_dir: str, tag: str, dataset: str) -> str | None:
    candidates = [
        os.path.join(checkpoints_dir, f"best_{tag}_{dataset}_ss.pth"),
        os.path.join(checkpoints_dir, f"best_{tag}_{dataset}.pth"),
        os.path.join(checkpoints_dir, f"best_{tag}_ss.pth"),
        os.path.join(checkpoints_dir, f"best_{tag}.pth"),
    ]

    if tag == "ours":
        candidates.insert(0, os.path.join(checkpoints_dir, "best_model_2d.pth"))

    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return None


def _load_state_dict(model, checkpoint_path):
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)
    return model


def _load_baseline(tag, checkpoint_path, device):
    if not BASELINES_AVAILABLE:
        raise RuntimeError("Baseline modules not available")

    if tag == "usl":
        model = create_usl_model({"patch_size": (24, 24)})
    elif tag == "dncnn":
        model = create_baseline_model("dncnn")
    elif tag == "unet":
        model = create_baseline_model("unet")
    else:
        raise ValueError(f"Unknown baseline tag: {tag}")

    model = _load_state_dict(model, checkpoint_path)
    model = model.to(device)
    model.eval()
    return model


def _method_display(key: str) -> str:
    if key == "raw":
        return "Raw (Noisy)"
    if key == "dl":
        return "DL Only"
    if key.startswith("dl+"):
        rest = key.replace("dl+", "")
        if rest in TRAD_METHODS:
            return f"DL + {TRAD_METHODS[rest]['name']}"
        return f"DL + {combo_display_name(rest)}"
    if key in TRAD_METHODS:
        return TRAD_METHODS[key]["name"]
    return key


def _read_json(path):
    if not path or not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return None


def _collect_synthetic(dataset_name, noise_level, results_dir, baseline_tags=None):
    records = []

    # Ours (DL only)
    ours_path = os.path.join(results_dir, f"{dataset_name}_noise{noise_level}_test_results.json")
    ours = _read_json(ours_path)
    if ours and ours.get("metrics_noisy"):
        raw_met = ours["metrics_noisy"]
        records.append({"method_name": "Raw (Noisy)", **raw_met, "source": os.path.basename(ours_path)})
    if ours and ours.get("metrics_denoised"):
        dl_met = ours["metrics_denoised"]
        records.append({"method_name": "DL Only", **dl_met, "source": os.path.basename(ours_path)})

    # Baselines (tagged)
    if baseline_tags:
        for tag in baseline_tags:
            tag_path = os.path.join(results_dir, f"{dataset_name}_{tag}_noise{noise_level}_test_results.json")
            tag_json = _read_json(tag_path)
            if tag_json and tag_json.get("metrics_denoised"):
                records.append({
                    "method_name": _BASELINE_NAME_MAP.get(tag, tag.upper()),
                    **tag_json["metrics_denoised"],
                    "source": os.path.basename(tag_path),
                })

    # Traditional only
    trad_path = os.path.join(results_dir, f"{dataset_name}_trad_noise{noise_level}_metrics.json")
    trad = _read_json(trad_path)
    if isinstance(trad, dict):
        for key, met in trad.items():
            name = _method_display(key)
            records.append({"method_name": name, **met, "source": os.path.basename(trad_path)})

    # Hybrid (DL + traditional)
    hybrid_path = os.path.join(results_dir, f"{dataset_name}_hybrid_noise{noise_level}_metrics.json")
    hybrid = _read_json(hybrid_path)
    if isinstance(hybrid, dict):
        for key, met in hybrid.items():
            name = _method_display(key)
            records.append({"method_name": name, **met, "source": os.path.basename(hybrid_path)})

    # Combo (DL + multi-traditional)
    combo_path = os.path.join(results_dir, f"{dataset_name}_combo_noise{noise_level}_combo_metrics.json")
    combo = _read_json(combo_path)
    if isinstance(combo, dict):
        for key, met in combo.items():
            name = _method_display(key)
            records.append({"method_name": name, **met, "source": os.path.basename(combo_path)})

    # Deduplicate by method_name (keep best SNR if duplicates)
    merged = {}
    for rec in records:
        name = rec["method_name"]
        if name not in merged or rec.get("snr", -1) > merged[name].get("snr", -1):
            merged[name] = rec

    return list(merged.values())


def _collect_real(dataset_name, results_dir):
    records = []

    def _append_metrics(method_name, metrics, source):
        records.append({
            "method_name": method_name,
            "no_ref_score": metrics.get("no_ref_score"),
            "residual_energy_ratio": metrics.get("residual_energy_ratio"),
            "signal_corr_with_raw": metrics.get("signal_corr_with_raw"),
            "smoothness_gain": metrics.get("smoothness_gain"),
            "source": source,
        })

    # Ours
    ours_path = os.path.join(results_dir, f"{dataset_name}_real_test_results.json")
    ours = _read_json(ours_path)
    if ours and ours.get("no_ref_metrics"):
        _append_metrics("DL Only", ours["no_ref_metrics"], os.path.basename(ours_path))

    # Traditional
    trad_path = os.path.join(results_dir, f"{dataset_name}_trad_real_metrics.json")
    trad = _read_json(trad_path)
    if isinstance(trad, dict):
        for key, met in trad.items():
            name = _method_display(key)
            _append_metrics(name, met, os.path.basename(trad_path))

    # Hybrid
    hybrid_path = os.path.join(results_dir, f"{dataset_name}_hybrid_real_metrics.json")
    hybrid = _read_json(hybrid_path)
    if isinstance(hybrid, dict):
        for key, met in hybrid.items():
            name = _method_display(key)
            _append_metrics(name, met, os.path.basename(hybrid_path))

    # Combo
    combo_path = os.path.join(results_dir, f"{dataset_name}_combo_real_combo_metrics.json")
    combo = _read_json(combo_path)
    if isinstance(combo, dict):
        for key, met in combo.items():
            name = _method_display(key)
            _append_metrics(name, met, os.path.basename(combo_path))

    # Fallback: compute from arrays if metrics missing
    if not records:
        raw_path = os.path.join(os.path.dirname(results_dir), "data", f"{dataset_name}.npy")
        if os.path.exists(raw_path):
            raw = np.load(raw_path)
            for npy_path in glob(os.path.join(results_dir, f"{dataset_name}_*real*.npy")):
                arr = np.load(npy_path)
                met = calculate_no_ref_metrics(raw, arr)
                method_name = os.path.splitext(os.path.basename(npy_path))[0]
                _append_metrics(method_name, met, os.path.basename(npy_path))

    return records


def _write_table_md(path, title, columns, rows):
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"\n\n## {title}\n\n")
        f.write("| " + " | ".join(columns) + " |\n")
        f.write("| " + " | ".join(["---"] * len(columns)) + " |\n")
        for r in rows:
            f.write("| " + " | ".join(r) + " |\n")


def write_tables(output_path, datasets, noise_level, results_dir, baseline_tags=None):
    if os.path.exists(output_path):
        os.remove(output_path)

    for ds in datasets:
        records = _collect_synthetic(ds, noise_level, results_dir, baseline_tags=baseline_tags)
        records = sorted(records, key=lambda x: x.get("snr", -1), reverse=True)
        rows = []
        for r in records:
            ssim = r.get("ssim")
            ssim_str = f"{ssim:.4f}" if isinstance(ssim, (int, float)) else "-"
            rows.append([
                r.get("method_name", ""),
                f"{r.get('snr', 0):.4f}",
                f"{r.get('psnr', 0):.4f}",
                ssim_str,
                f"{r.get('correlation', 0):.4f}",
                f"{r.get('mse', 0):.4f}",
                f"{r.get('mae', 0):.4f}",
            ])
        _write_table_md(
            output_path,
            f"{ds} - Synthetic main comparison (noise={noise_level})",
            ["method_name", "snr", "psnr", "ssim", "correlation", "mse", "mae"],
            rows,
        )

    for ds in datasets:
        records = _collect_real(ds, results_dir)
        records = sorted(records, key=lambda x: x.get("no_ref_score", -1), reverse=True)
        rows = []
        for r in records:
            rows.append([
                r.get("method_name", ""),
                f"{r.get('no_ref_score', 0):.4f}",
                f"{r.get('residual_energy_ratio', 0):.4f}",
                f"{r.get('signal_corr_with_raw', 0):.4f}",
                f"{r.get('smoothness_gain', 0):.4f}",
                r.get("source", ""),
            ])
        _write_table_md(
            output_path,
            f"{ds} - Real data no-reference metrics",
            [
                "method_name",
                "no_ref_score",
                "residual_energy_ratio",
                "signal_corr_with_raw",
                "smoothness_gain",
                "source_file",
            ],
            rows,
        )


def write_figure_manifest(fig_dir, output_path):
    files = sorted([
        f for f in os.listdir(fig_dir)
        if os.path.isfile(os.path.join(fig_dir, f))
    ])

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# Figure Manifest\n\n")
        for name in files:
            f.write(f"- {name}\n")


def write_ablation_table(output_path, dataset, ablation_path, noise_level=None):
    data = _read_json(ablation_path)
    if not isinstance(data, list):
        return False

    rows = []
    for item in data:
        met = item.get("metrics", {}) or {}
        ssim = met.get("ssim")
        ssim_str = f"{ssim:.4f}" if isinstance(ssim, (int, float)) else "-"
        rows.append([
            item.get("ablation", ""),
            f"{met.get('snr', 0):.4f}",
            f"{met.get('psnr', 0):.4f}",
            ssim_str,
            f"{met.get('correlation', 0):.4f}",
            f"{met.get('mse', 0):.4f}",
            f"{met.get('mae', 0):.4f}",
            os.path.basename(item.get("checkpoint", "")),
        ])

    title_suffix = f" (noise={noise_level})" if noise_level is not None else ""
    _write_table_md(
        output_path,
        f"{dataset} - Ablation study{title_suffix}",
        ["ablation", "snr", "psnr", "ssim", "correlation", "mse", "mae", "checkpoint"],
        rows,
    )
    return True


def write_noise_robustness_plot(datasets, noise_levels, results_dir, figures_dir):
    import matplotlib.pyplot as plt

    os.makedirs(figures_dir, exist_ok=True)

    def _noise_tag(val: float) -> str:
        return f"{val:g}"

    for ds in datasets:
        raw_snrs = []
        dl_snrs = []
        x_vals = []

        for nl in noise_levels:
            tag = _noise_tag(nl)
            path = os.path.join(results_dir, f"{ds}_noise{tag}_test_results.json")
            data = _read_json(path)
            if not data or not data.get("metrics_noisy") or not data.get("metrics_denoised"):
                continue

            x_vals.append(float(nl))
            raw_snrs.append(data["metrics_noisy"].get("snr", 0.0))
            dl_snrs.append(data["metrics_denoised"].get("snr", 0.0))

        if len(x_vals) < 2:
            continue

        fig = plt.figure(figsize=(6, 4))
        plt.plot(x_vals, raw_snrs, marker="o", label="Raw (Noisy)")
        plt.plot(x_vals, dl_snrs, marker="s", label="DL Only")
        plt.xlabel("Synthetic noise level")
        plt.ylabel("SNR (dB)")
        plt.title(f"Noise Robustness - {ds}")
        plt.grid(alpha=0.3)
        plt.legend()

        out_path = os.path.join(figures_dir, f"{ds}_noise_robustness.png")
        plt.tight_layout()
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Run paper experiments")
    parser.add_argument("--mode", choices=["all", "synthetic", "real", "collect"], default="all")
    parser.add_argument("--datasets", type=str, default="auto")
    parser.add_argument("--noise-levels", type=str, default="0.05,0.1,0.15")
    parser.add_argument("--comparison-noise", type=float, default=0.1)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--dncnn-ckpt", type=str, default=None)
    parser.add_argument("--unet-ckpt", type=str, default=None)
    parser.add_argument("--usl-ckpt", type=str, default=None)
    parser.add_argument("--skip-trad", action="store_true")
    parser.add_argument("--skip-hybrid", action="store_true")
    parser.add_argument("--skip-combo", action="store_true")
    parser.add_argument("--collect-ablation", action="store_true")
    parser.add_argument("--plot-robustness", action="store_true")
    args = parser.parse_args()

    cfg = get_config("test")
    paths = cfg["paths"]
    data_cfg = cfg["data"]

    if args.datasets.strip().lower() == "auto":
        datasets = ["eq-36", "eq-68"]
    else:
        datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    noise_levels = [float(x) for x in args.noise_levels.split(",") if x.strip()]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load baselines if provided
    baselines = {}
    for tag, p in [("dncnn", args.dncnn_ckpt), ("unet", args.unet_ckpt), ("usl", args.usl_ckpt)]:
        if p:
            baselines[tag] = _load_baseline(tag, _resolve_path(p), device)

    patch_size = data_cfg.get("patch_size")
    stride = data_cfg.get("stride")

    def _load_models_for_dataset(dataset_name: str):
        if args.checkpoint:
            ours_ckpt = _resolve_path(args.checkpoint)
        else:
            ours_ckpt = _find_dataset_checkpoint(paths["checkpoints"], "ours", dataset_name)

        if not ours_ckpt or not os.path.exists(ours_ckpt):
            raise FileNotFoundError(f"Checkpoint not found for ours on {dataset_name}: {ours_ckpt}")

        ours_model = load_trained_model(ours_ckpt, device, use_best_params=True)

        baseline_models = {}
        if not baselines:
            for tag in ("dncnn", "unet", "usl"):
                ckpt = _find_dataset_checkpoint(paths["checkpoints"], tag, dataset_name)
                if ckpt:
                    baseline_models[tag] = _load_baseline(tag, ckpt, device)
        else:
            baseline_models.update(baselines)

        return ours_model, baseline_models

    if args.mode in ("all", "synthetic"):
        for ds in datasets:
            data_path = data_cfg["datasets"].get(ds)
            if not data_path or not os.path.exists(data_path):
                print(f"[skip] dataset not found: {ds}")
                continue

            model, dataset_baselines = _load_models_for_dataset(ds)

            for nl in noise_levels:
                test_on_data(
                    model,
                    data_path,
                    noise_level=nl,
                    add_synthetic_noise=True,
                    save_results=True,
                    patch_size=patch_size,
                    stride=stride,
                )

            # Baselines at comparison noise level
            for tag, bmodel in dataset_baselines.items():
                test_on_data(
                    bmodel,
                    data_path,
                    noise_level=args.comparison_noise,
                    add_synthetic_noise=True,
                    save_results=True,
                    patch_size=patch_size,
                    stride=stride,
                    tag=tag,
                )

            if not args.skip_trad:
                test_trad_only(
                    data_path,
                    noise_level=args.comparison_noise,
                    add_synthetic_noise=True,
                    save_results=True,
                )
            if not args.skip_hybrid:
                test_hybrid_denoise(
                    model,
                    data_path,
                    noise_level=args.comparison_noise,
                    add_synthetic_noise=True,
                    save_results=True,
                    patch_size=patch_size,
                    stride=stride,
                )
            if not args.skip_combo:
                test_combo_denoise(
                    model,
                    data_path,
                    noise_level=args.comparison_noise,
                    add_synthetic_noise=True,
                    save_results=True,
                    patch_size=patch_size,
                    stride=stride,
                )

    if args.mode in ("all", "real"):
        for ds in datasets:
            data_path = data_cfg["datasets"].get(ds)
            if not data_path or not os.path.exists(data_path):
                print(f"[skip] dataset not found: {ds}")
                continue

            model, dataset_baselines = _load_models_for_dataset(ds)

            test_on_data(
                model,
                data_path,
                add_synthetic_noise=False,
                save_results=True,
                patch_size=patch_size,
                stride=stride,
            )

            for tag, bmodel in dataset_baselines.items():
                test_on_data(
                    bmodel,
                    data_path,
                    add_synthetic_noise=False,
                    save_results=True,
                    patch_size=patch_size,
                    stride=stride,
                    tag=tag,
                )

            if not args.skip_trad:
                test_trad_only(
                    data_path,
                    add_synthetic_noise=False,
                    save_results=True,
                )
            if not args.skip_hybrid:
                test_hybrid_denoise(
                    model,
                    data_path,
                    add_synthetic_noise=False,
                    save_results=True,
                    patch_size=patch_size,
                    stride=stride,
                )
            if not args.skip_combo:
                test_combo_denoise(
                    model,
                    data_path,
                    add_synthetic_noise=False,
                    save_results=True,
                    patch_size=patch_size,
                    stride=stride,
                )

    # Collect tables + manifest
    if args.mode in ("all", "collect"):
        paper_dir = os.path.join(os.path.dirname(__file__), "paper_outputs")
        os.makedirs(paper_dir, exist_ok=True)
        table_path = os.path.join(paper_dir, "tables_for_paper.md")
        fig_manifest = os.path.join(paper_dir, "figure_manifest.md")
        run_summary = os.path.join(paper_dir, "run_summary.json")

        baseline_tags = list(baselines.keys()) if baselines else None
        write_tables(table_path, datasets, args.comparison_noise, paths["results"], baseline_tags)
        write_figure_manifest(paths["figures"], fig_manifest)

        if args.collect_ablation:
            for ds in datasets:
                ablation_path = os.path.join(paths["results"], f"ablation_{ds}.json")
                if os.path.exists(ablation_path):
                    write_ablation_table(table_path, ds, ablation_path, args.comparison_noise)

        summary = {
            "mode": args.mode,
            "datasets": datasets,
            "noise_levels": noise_levels,
            "comparison_noise_level": args.comparison_noise,
            "results_dir": paths["results"],
            "figures_dir": paths["figures"],
            "paper_outputs": paper_dir,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        with open(run_summary, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        print(f"Tables written: {table_path}")
        print(f"Figure manifest: {fig_manifest}")
        print(f"Run summary: {run_summary}")

    if args.plot_robustness and args.mode in ("all", "synthetic", "collect"):
        write_noise_robustness_plot(datasets, noise_levels, paths["results"], paths["figures"])


if __name__ == "__main__":
    main()
