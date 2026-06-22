"""
针对无干净真值数据的推理和评估脚本
- 仅使用无参考指标
- 支持所有对比方法
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config_2d import get_config
from model_2d import create_model_2d
from baseline_models import create_baseline_model
from utils_2d import yc_patch, yc_patch_inv, calculate_no_ref_metrics, plot_2d_comparison
from traditional_denoise import run_all_hybrid, run_trad_only, TRAD_METHODS, tv_denoise


def _resolve_path(path: str) -> str:
    if not path:
        return path
    if os.path.isabs(path):
        return path
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), path)


def _extract_prediction(output):
    if isinstance(output, (tuple, list)):
        return output[0]
    return output


def load_model_ss(ckpt_path, device):
    """加载自监督训练的模型"""
    cfg = get_config("test")
    model = create_model_2d(cfg["model"]).to(device)
    
    state = torch.load(ckpt_path, map_location=device)
    model_state = state.get("model_state_dict", state)
    model.load_state_dict(model_state)
    model.eval()
    
    return model


def load_baseline_model(model_name, ckpt_path, device):
    """加载基线模型（DnCNN / U-Net）"""
    model = create_baseline_model(model_name).to(device)

    state = torch.load(ckpt_path, map_location=device)
    model_state = state.get("model_state_dict", state)
    model.load_state_dict(model_state)
    model.eval()

    return model


def denoise_2d_data_patch_based(model, data_path, patch_size=(24, 24), stride=(6, 6)):
    """
    基于 patch 的 2D 去噪推理（无真值）
    
    Args:
        model: 训练好的去噪模型
        data_path: 数据文件路径
        patch_size: patch 大小
        stride: 步长
    
    Returns:
        denoised: (H, W) 去噪结果
    """
    device = next(model.parameters()).device
    
    # 加载数据
    data = np.load(data_path).astype(np.float32)
    n1, n2 = data.shape
    
    # 归一化
    data_mean = np.mean(data)
    data_std = np.std(data)
    data = (data - data_mean) / (data_std + 1e-6)
    
    # 提取 patches
    patches = yc_patch(data, patch_size[0], patch_size[1], stride[0], stride[1])
    
    # 去噪
    denoised_patches = []
    model.eval()
    with torch.no_grad():
        for patch in tqdm(patches, desc="Denoising patches", leave=False):
            patch_tensor = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).to(device)
            denoised_patch = _extract_prediction(model(patch_tensor)).squeeze().cpu().numpy()
            denoised_patches.append(denoised_patch)
    
    denoised_patches = np.array(denoised_patches)
    
    # 逆向转换
    denoised_norm = yc_patch_inv(
        denoised_patches, n1, n2, 
        patch_size[0], patch_size[1], 
        stride[0], stride[1]
    )
    
    # 反归一化
    denoised = denoised_norm * data_std + data_mean
    
    return denoised


def denoise_2d_data_patch_based_dl_tv(model, data_path, patch_size=(24, 24), stride=(6, 6), tv_weight=0.1):
    """先做深度模型推理，再对结果做 TV 后处理。"""
    dl_denoised = denoise_2d_data_patch_based(model, data_path, patch_size, stride)
    return tv_denoise(dl_denoised, weight=tv_weight)


def test_denoise_no_reference(model, data_path, dataset_name, model_name, 
                             results_dir, patch_size=(24, 24), stride=(6, 6)):
    """
    测试去噪模型（无参考指标）
    
    Args:
        model: 去噪模型
        data_path: 数据路径
        dataset_name: 数据集名称
        model_name: 模型名称
        results_dir: 结果保存目录
        patch_size: patch 大小
        stride: 步长
    
    Returns:
        metrics: 无参考指标字典
    """
    device = next(model.parameters()).device
    
    print(f"\n[{model_name}] Processing {dataset_name}...")
    
    # 加载原始数据
    raw = np.load(data_path).astype(np.float32)
    print(f"Raw data shape: {raw.shape}")
    
    # 去噪
    denoised = denoise_2d_data_patch_based(model, data_path, patch_size, stride)
    print(f"Denoised shape: {denoised.shape}")
    
    # 计算无参考指标
    metrics = calculate_no_ref_metrics(raw, denoised)
    
    print(f"No-ref metrics:")
    print(f"  residual_energy_ratio: {metrics['residual_energy_ratio']:.6f}")
    print(f"  signal_corr_with_raw: {metrics['signal_corr_with_raw']:.6f}")
    print(f"  smoothness_gain: {metrics['smoothness_gain']:.6f}")
    print(f"  no_ref_score: {metrics['no_ref_score']:.6f}")
    
    # 保存去噪结果
    os.makedirs(results_dir, exist_ok=True)
    denoised_path = os.path.join(results_dir, f"{dataset_name}_{model_name}_denoised.npy")
    np.save(denoised_path, denoised)
    print(f"Saved denoised result: {denoised_path}")
    
    # 保存指标
    metrics_path = os.path.join(results_dir, f"{dataset_name}_{model_name}_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics: {metrics_path}")
    
    # 生成可视化
    figures_dir = os.path.join(os.path.dirname(results_dir), "figures")
    os.makedirs(figures_dir, exist_ok=True)
    fig_path = os.path.join(figures_dir, f"{dataset_name}_{model_name}_comparison.png")
    plot_2d_comparison(raw, denoised, save_path=fig_path, figsize=(15, 5))
    plt.close()
    
    return metrics


def test_traditional_methods_no_reference(data_path, dataset_name, results_dir):
    """
    测试传统去噪方法（无参考指标）
    """
    print(f"\n[Traditional Methods] Processing {dataset_name}...")
    
    raw = np.load(data_path).astype(np.float32)
    results = {}
    
    # 运行所有传统方法
    for method_key, method_info in TRAD_METHODS.items():
        method_name = method_info["name"]
        method_fn = method_info["fn"]
        
        print(f"  Applying {method_name}...")
        try:
            denoised = method_fn(raw)
            metrics = calculate_no_ref_metrics(raw, denoised)
            results[method_key] = metrics
            
            print(f"    no_ref_score: {metrics['no_ref_score']:.6f}")
            
            # 保存去噪结果
            denoised_path = os.path.join(results_dir, f"{dataset_name}_trad_{method_key}_denoised.npy")
            np.save(denoised_path, denoised)
        except Exception as e:
            print(f"    Error: {e}")
            results[method_key] = None
    
    # 保存指标
    metrics_path = os.path.join(results_dir, f"{dataset_name}_trad_metrics.json")
    os.makedirs(results_dir, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved traditional methods metrics: {metrics_path}")
    
    return results


def compare_methods_no_reference(data_path, dataset_name, results_dir, 
                                 dl_model=None, baseline_models=None):
    """
    对所有方法进行综合对比（无参考指标）
    
    Returns:
        comparison_table: 对比表格
    """
    raw = np.load(data_path).astype(np.float32)
    comparison = []
    
    # 1. 原始数据（baseline）
    comparison.append({
        "method": "Raw (Noisy)",
        "residual_energy_ratio": 1.0,
        "signal_corr_with_raw": 1.0,
        "smoothness_gain": 1.0,
        "no_ref_score": 0.0,
    })
    
    # 2. 深度学习模型
    if dl_model is not None:
        print(f"\nTesting DL+TV model on {dataset_name}...")
        denoised = denoise_2d_data_patch_based_dl_tv(dl_model, data_path)
        metrics = calculate_no_ref_metrics(raw, denoised)
        comparison.append({
            "method": "DL+TV (Ours)",
            **metrics
        })
    
    # 3. 基线深度学习模型
    if baseline_models:
        for model_name, model in baseline_models.items():
            print(f"\nTesting {model_name} on {dataset_name}...")
            denoised = denoise_2d_data_patch_based(model, data_path)
            metrics = calculate_no_ref_metrics(raw, denoised)
            comparison.append({
                "method": f"{model_name.upper()} (Baseline)",
                **metrics
            })
    
    # 4. 传统方法
    print(f"\nTesting traditional methods on {dataset_name}...")
    for method_key, method_info in TRAD_METHODS.items():
        method_name = method_info["name"]
        method_fn = method_info["fn"]
        
        try:
            denoised = method_fn(raw)
            metrics = calculate_no_ref_metrics(raw, denoised)
            comparison.append({
                "method": method_name,
                **metrics
            })
        except Exception as e:
            print(f"  Error in {method_name}: {e}")
    
    # 排序：按 no_ref_score 降序
    comparison = sorted(comparison, key=lambda x: x.get("no_ref_score", 0), reverse=True)
    
    # 保存对比结果
    os.makedirs(results_dir, exist_ok=True)
    comparison_path = os.path.join(results_dir, f"{dataset_name}_comparison_no_ref.json")
    with open(comparison_path, "w") as f:
        json.dump(comparison, f, indent=2)
    
    # 打印对比表
    print(f"\n{'='*80}")
    print(f"Comparison Results for {dataset_name}")
    print(f"{'='*80}")
    print(f"{'Method':<30} {'no_ref_score':<15} {'residual':<15} {'smoothness':<15}")
    print(f"{'-'*80}")
    for item in comparison:
        method = item["method"]
        score = item.get("no_ref_score", 0)
        residual = item.get("residual_energy_ratio", 0)
        smooth = item.get("smoothness_gain", 0)
        print(f"{method:<30} {score:<15.6f} {residual:<15.6f} {smooth:<15.6f}")
    
    return comparison


def main():
    parser = argparse.ArgumentParser(description="Denoise and compare methods (no reference)")
    parser.add_argument("--model", choices=["ours", "dncnn", "unet"], default="ours")
    parser.add_argument("--dataset", choices=["eq-36", "eq-68", "slice_german", "slice_german_1"], default="eq-36")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--include-baselines", action="store_true")
    parser.add_argument("--include-traditional", action="store_true")
    parser.add_argument("--compare-all", action="store_true")
    args = parser.parse_args()

    cfg = get_config("test")
    paths = cfg["paths"]
    data_cfg = cfg["data"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 加载数据路径
    data_path = data_cfg["datasets"].get(args.dataset)
    if not data_path or not os.path.exists(data_path):
        raise FileNotFoundError(f"Data not found: {data_path}")

    # 加载模型
    if args.checkpoint:
        ckpt_path = _resolve_path(args.checkpoint)
    else:
        # 自动查找检查点
        model_prefix = "best_ours" if args.model == "ours" else f"best_{args.model}"
        candidates = [
            os.path.join(paths["checkpoints"], f"{model_prefix}_{args.dataset}_ss.pth"),
            os.path.join(paths["checkpoints"], f"{model_prefix}_ss.pth"),
        ]
        ckpt_path = next((c for c in candidates if os.path.exists(c)), None)
    
    if not ckpt_path or not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found for {args.model}")

    model = load_model_ss(ckpt_path, device)

    # 测试模型
    results_dir = paths["results"]
    metrics = test_denoise_no_reference(
        model, data_path, args.dataset, args.model,
        results_dir, patch_size=data_cfg.get("patch_size"),
        stride=data_cfg.get("stride")
    )

    # 可选：测试其他方法
    if args.include_traditional or args.compare_all:
        test_traditional_methods_no_reference(data_path, args.dataset, results_dir)

    if args.compare_all:
        baseline_models = {}
        if args.include_baselines:
            # 加载基线模型
            for baseline_model in ["dncnn", "unet"]:
                try:
                    baseline_ckpt = os.path.join(
                        paths["checkpoints"], 
                        f"best_{baseline_model}_{args.dataset}_ss.pth"
                    )
                    if os.path.exists(baseline_ckpt):
                        bmodel = load_baseline_model(baseline_model, baseline_ckpt, device)
                        baseline_models[baseline_model] = bmodel
                except:
                    pass
        
        # 综合对比
        comparison = compare_methods_no_reference(
            data_path, args.dataset, results_dir,
            dl_model=model,
            baseline_models=baseline_models if baseline_models else None
        )


if __name__ == "__main__":
    main()
