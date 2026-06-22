"""
Ablation runner for the proposed model.
"""

import argparse
import json
import os

import torch

from config_2d import get_config
from train_2d import build_dataloaders, train_model
from test_2d import test_on_data


def _ablation_ckpt_path(cfg, dataset, tag):
    return os.path.join(cfg["paths"]["checkpoints"], f"best_ours_{dataset}_{tag}.pth")


def main():
    parser = argparse.ArgumentParser(description="Run ablation studies")
    parser.add_argument("--dataset", choices=["eq-36"], default="eq-36")
    parser.add_argument("--noise", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    args = parser.parse_args()

    cfg = get_config("train")
    if args.epochs is not None:
        cfg["train"]["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["train"]["batch_size"] = args.batch_size

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ablations = [
        ("full", {}),
        ("no_fda", {"use_fda": False}),
        ("no_naag", {"use_naag": False}),
        ("no_wavelet", {"use_cross_scale": False, "use_adaptive_fusion": False}),
        ("no_sparse_attn", {"use_sparse_attention": False}),
        ("no_rdc", {"use_rdc": False}),
        ("no_fag", {"use_fag": False}),
    ]

    train_loader, val_loader = build_dataloaders(cfg, args.dataset)

    results = []
    for tag, updates in ablations:
        print("\n" + "=" * 70)
        print(f"Ablation: {tag}")
        print("=" * 70)

        cfg_local = get_config("train")
        cfg_local["train"].update(cfg["train"])
        cfg_local["model"].update(cfg["model"])
        cfg_local["model"].update(updates)

        ckpt_path = _ablation_ckpt_path(cfg_local, args.dataset, tag)
        if os.path.exists(ckpt_path):
            print(f"Reusing existing checkpoint: {ckpt_path}")
        else:
            ckpt_path, _history = train_model(
                cfg_local,
                "ours",
                train_loader,
                val_loader,
                device,
                tag=f"{args.dataset}_{tag}",
            )

        # Evaluate on synthetic noise
        data_path = cfg_local["data"]["datasets"][args.dataset]
        model = cfg_local["model"]

        # Use the trained model instance by reloading the checkpoint
        # (ensures same weights used for evaluation)
        from model_2d import create_model_2d
        eval_model = create_model_2d(model).to(device)
        state = torch.load(ckpt_path, map_location=device)
        eval_model.load_state_dict(state.get("model_state_dict", state))
        eval_model.eval()

        _, metrics = test_on_data(
            eval_model,
            data_path,
            noise_level=args.noise,
            add_synthetic_noise=True,
            save_results=True,
            patch_size=cfg_local["data"]["patch_size"],
            stride=cfg_local["data"]["stride"],
            tag=f"abl_{tag}",
        )

        results.append({
            "ablation": tag,
            "checkpoint": ckpt_path,
            "metrics": metrics,
        })

    out_path = os.path.join(cfg["paths"]["results"], f"ablation_{args.dataset}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Ablation summary saved: {out_path}")


if __name__ == "__main__":
    main()
