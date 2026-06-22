"""
完全2D处理的测试脚本
基于USL DIP的预测策略
新增：FDA + NAAG模块可视化和分析
"""

import os
import sys
import json
import time
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model_2d import create_model_2d
from utils_2d import (
    yc_patch, yc_patch_inv, add_noise_2d,
    calculate_metrics_2d, calculate_no_ref_metrics,
    plot_2d_comparison, plot_ipynb_style, cseis,
    plot_naag_analysis, plot_fda_analysis, visualize_fda_naag_combined
)
from config_2d import MODEL_2D_CONFIG, DATA_CONFIG, PATHS
from traditional_denoise import (
    run_all_hybrid, plot_hybrid_comparison, plot_metrics_bar,
    plot_noise_residuals, plot_single_trace_comparison, TRAD_METHODS,
    run_combo_hybrid, plot_combo_comparison, plot_combo_metrics_bar,
    COMBO_METHODS, combo_display_name,
    run_trad_only, plot_trad_comparison, plot_trad_metrics_bar,
)


def _resolve_path(path: str) -> str:
    """Resolve relative paths against this file's directory."""
    if not path:
        return path
    if os.path.isabs(path):
        return path
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), path)


def _select_runtime_config(args) -> tuple[dict, dict, dict, bool]:
    """Select (model_config, data_config, paths, use_best_params) for test run."""
    checkpoint_hint = (args.checkpoint or "").replace('\\', '/').lower()
    auto_low_snr = (
        (not args.low_snr)
        and checkpoint_hint
        and (
            'checkpoints_low_snr' in checkpoint_hint
            or checkpoint_hint.endswith('/best_model.pth')
            or checkpoint_hint.endswith('best_model.pth')
        )
    )

    if args.low_snr or auto_low_snr:
        from config_low_snr import get_low_snr_config

        cfg = get_low_snr_config()
        model_config = cfg['model']
        data_config = cfg['data']
        paths = cfg['paths']
        use_best_params = False
        if auto_low_snr and not args.low_snr:
            print("\n⚠ 检测到低SNR checkpoint 路径，自动切换到 low_snr 配置")
        return model_config, data_config, paths, use_best_params

    return MODEL_2D_CONFIG, DATA_CONFIG, PATHS, True


def load_best_params():
    """加载优化后的最佳超参数"""
    quick_params_path = os.path.join(PATHS['results'], 'best_params_quick.json')
    full_params_path = os.path.join(PATHS['results'], 'best_params.json')
    
    params_path = None
    if os.path.exists(quick_params_path):
        params_path = quick_params_path
        print(f"✓ 找到快速优化参数: {quick_params_path}")
    elif os.path.exists(full_params_path):
        params_path = full_params_path
        print(f"✓ 找到完整优化参数: {full_params_path}")
    else:
        print("⚠ 未找到优化参数文件，使用默认配置")
        return None
    
    try:
        with open(params_path, 'r') as f:
            data = json.load(f)
        
        best_params = data.get('best_params', {})
        best_loss = data.get('best_loss', None)
        
        if best_loss:
            print(f"  优化得到的最佳验证损失: {best_loss:.6f}")
        print(f"  加载参数: {len(best_params)} 个")
        
        return best_params
    except Exception as e:
        print(f"⚠ 加载参数失败: {e}")
        return None


def infer_config_from_state_dict(state_dict: dict) -> dict:
    """
    从 checkpoint 的 state_dict 自动反推模型超参数，
    避免 config_2d.py 与实际训练时的配置不一致导致 size mismatch。
    """
    inferred = {}

    # d_model: scale_processors.0.input_proj.weight -> (d_model, 1, 3, 3)
    key_d = 'model.scale_processors.0.input_proj.weight'
    if key_d in state_dict:
        inferred['d_model'] = state_dict[key_d].shape[0]

    # rdc_num_blocks: 统计 rdc_blocks.X.fusion.weight 的最大 X+1
    rdc_indices = [
        int(k.split('.')[2])
        for k in state_dict
        if k.startswith('model.rdc_blocks.') and k.endswith('.fusion.weight')
    ]
    if rdc_indices:
        inferred['rdc_num_blocks'] = max(rdc_indices) + 1

    # rdc_growth_rate: rdc_blocks.0.layers.0.0.weight -> (growth_rate, d_model, 3, 3)
    key_gr = 'model.rdc_blocks.0.layers.0.0.weight'
    if key_gr in state_dict:
        inferred['rdc_growth_rate'] = state_dict[key_gr].shape[0]

    # wavelet_levels: scale_processors 数量 / 2  (每个 level 有 2 个方向)
    sp_indices = set(
        int(k.split('.')[2])
        for k in state_dict
        if k.startswith('model.scale_processors.')
    )
    if sp_indices:
        # scale_processors 数量 = wavelet_levels * 2  (LL/LH/HL/HH 各尺度)
        # 根据实际模型结构：通常 num_scale_processors = wavelet_levels * num_subbands
        # 安全起见直接把数量存进去让 create_model_2d 按需使用
        inferred['_n_scale_processors'] = len(sp_indices)

    # num_freq_bands (fda): band_processors 数量
    fda_bp = set(
        int(k.split('.')[3])
        for k in state_dict
        if k.startswith('model.fda.band_processors.')
    )
    if fda_bp:
        inferred['num_freq_bands'] = len(fda_bp)

    # fag_num_bands: fag.band_processors 数量
    fag_bp = set(
        int(k.split('.')[3])
        for k in state_dict
        if k.startswith('model.fag.band_processors.')
    )
    if fag_bp:
        inferred['fag_num_bands'] = len(fag_bp)

    # noise_scales: noise_estimator.scale_estimators 数量
    ne_idx = set(
        int(k.split('.')[3])
        for k in state_dict
        if k.startswith('model.noise_estimator.scale_estimators.')
    )
    if ne_idx:
        inferred['noise_scales'] = len(ne_idx)

    return inferred


def load_trained_model(checkpoint_path, device, use_best_params=True):
    """
    加载训练好的模型。
    优先从 checkpoint state_dict 自动反推模型结构，
    再用 best_params（若存在）覆盖其余超参数。
    """
    print(f"\n{'='*70}")
    print("加载模型和参数")
    print(f"{'='*70}")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"模型文件不存在: {checkpoint_path}")

    # 先加载 checkpoint（只读 state_dict 用于反推结构）
    print(f"\n加载模型权重: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    sd = checkpoint.get('model_state_dict', checkpoint)

    # 1. 从 state_dict 自动反推结构参数
    inferred = infer_config_from_state_dict(sd)
    if inferred:
        print("\n🔍 从 checkpoint 自动检测到以下模型结构：")
        for k, v in inferred.items():
            if not k.startswith('_'):
                print(f"  {k}: {v}")

    # 2. 用反推结果覆盖 MODEL_2D_CONFIG（作为副本，不污染全局）
    import copy
    model_config = copy.deepcopy(MODEL_2D_CONFIG)
    for k, v in inferred.items():
        if not k.startswith('_') and k in model_config:
            model_config[k] = v

    # 3. 再用 best_params 文件覆盖（优先级最高）
    if use_best_params:
        best_params = load_best_params()
        if best_params:
            print("\n🎯 使用优化后的最佳超参数（覆盖自动检测值）")
            for key in ['d_model', 'num_layers', 'dropout', 'wavelet_level',
                        'num_heads', 'd_ff', 'sparsity_ratio', 'fag_num_bands',
                        'noise_num_scales', 'rdc_growth_rate', 'rdc_num_blocks']:
                if key in best_params:
                    model_config[key] = best_params[key]
                    print(f"  {key}: {best_params[key]}")
        else:
            print("\n⚠ 未找到 best_params 文件，仅使用自动检测配置")

    # 4. 用最终配置创建模型并加载权重
    model = create_model_2d(model_config)
    model.load_state_dict(sd)
    model = model.to(device)
    model.eval()
    
    print(f"\n✓ 模型加载成功")
    if 'epoch' in checkpoint:
        print(f"  训练轮数: {checkpoint['epoch']+1}")
    if 'loss' in checkpoint:
        try:
            print(f"  最佳损失: {checkpoint['loss']:.6f}")
        except Exception:
            pass
    
    if 'metrics' in checkpoint and checkpoint['metrics']:
        metrics = checkpoint['metrics']
        if 'psnr' in metrics:
            print(f"  训练PSNR: {metrics['psnr']:.2f} dB")
        if 'snr' in metrics:
            print(f"  训练SNR: {metrics['snr']:.2f} dB")
    
    # 统计参数
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  模型参数: {total_params:,}")
    print(f"{'='*70}\n")
    
    return model


def denoise_2d_data(model, data, patch_size, stride, device, batch_size=128):
    """
    对完整2D数据进行降噪
    使用patch策略（完全模仿USL DIP的IPYNB流程）
    
    Args:
        model: 训练好的模型
        data: 2D numpy array (H, W)
        patch_size: (h, w)
        stride: (sh, sw)
        device: torch device
        batch_size: batch大小（PyTorch需要分批，但逻辑与IPYNB一致）
    
    Returns:
        denoised: 降噪后的2D数据
        avg_noise_level: 平均噪声水平（来自NAAG）
        avg_freq_weights: 平均频率权重（来自FDA）
    """
    model.eval()
    
    h, w = data.shape
    ph, pw = patch_size
    sh, sw = stride
    
    # Step 1: 提取patches（模仿IPYNB: data_noisy = yc_patch(data, w1, w2, z1, z2)）
    print("Extracting patches...")
    data_noisy = yc_patch(data, ph, pw, sh, sw)  # 使用与IPYNB相同的变量名
    num_patches = len(data_noisy)
    print(f"Processing {num_patches} patches...")
    
    # Step 2: 预测（模仿IPYNB: out = model.predict(data_noisy)）
    # PyTorch需要分批处理，但逻辑与IPYNB一致
    denoised_patches = []
    all_noise_levels = []
    all_freq_weights = []
    all_gate_weights = []  # NAAG门控权重
    
    with torch.no_grad():
        for i in tqdm(range(0, num_patches, batch_size), desc='Denoising'):
            batch_patches = data_noisy[i:i+batch_size]
            
            # 转为tensor (B, 1, H, W)
            batch_tensor = torch.from_numpy(batch_patches).unsqueeze(1).float()
            batch_tensor = batch_tensor.to(device)
            
            # 降噪（兼容仅返回output的模型）
            model_out = model(batch_tensor)
            if isinstance(model_out, (tuple, list)):
                output = model_out[0]
                aux_outputs = model_out[1] if len(model_out) > 1 and isinstance(model_out[1], dict) else {}
            else:
                output = model_out
                aux_outputs = {}
            
            # 转回numpy
            output_np = output.squeeze(1).cpu().numpy()
            denoised_patches.append(output_np)
            
            # 收集NAAG和FDA的统计信息
            if aux_outputs.get('naag_noise_level') is not None:
                all_noise_levels.append(aux_outputs['naag_noise_level'].cpu().numpy())
            if aux_outputs.get('naag_gate_weights') is not None:
                all_gate_weights.append(aux_outputs['naag_gate_weights'].cpu().numpy())
            if aux_outputs.get('fda_freq_bands') is not None:
                all_freq_weights.append(aux_outputs['fda_freq_bands'].cpu().numpy())
    
    # 合并所有批次
    out = np.concatenate(denoised_patches, axis=0)
    
    # 统计NAAG和FDA信息
    avg_noise_level = None
    avg_freq_weights = None
    noise_levels_array = None
    gate_weights_array = None
    freq_weights_array = None
    
    if all_noise_levels:
        noise_levels_array = np.concatenate(all_noise_levels, axis=0)
        avg_noise_level = noise_levels_array.mean()
        print(f"  Average noise level (NAAG): {avg_noise_level:.4f}")
    
    if all_gate_weights:
        gate_weights_array = np.concatenate(all_gate_weights, axis=0)
        print(f"  Average gate weights (NAAG): {gate_weights_array.mean(axis=0)}")
    
    if all_freq_weights:
        freq_weights_array = np.concatenate(all_freq_weights, axis=0)
        avg_freq_weights = freq_weights_array.mean(axis=0)
        print(f"  Average frequency band weights (FDA): {avg_freq_weights}")
    
    # Step 4: 重建图像（模仿IPYNB: predicted = yc_patch_inv(out, n1, n2, w1, w2, z1, z2)）
    print("Reconstructing image...")
    predicted = yc_patch_inv(out, h, w, ph, pw, sh, sw)
    
    # 返回所有统计信息用于可视化
    stats = {
        'avg_noise_level': avg_noise_level,
        'avg_freq_weights': avg_freq_weights,
        'noise_levels': noise_levels_array,
        'gate_weights': gate_weights_array,
        'freq_weights': freq_weights_array
    }
    
    return predicted, stats


def test_on_data(model, data_path, noise_level=None, save_results=True,
                add_synthetic_noise=False, patch_size=None, stride=None,
                tag=None):
    """
    在单个数据上测试
    
    Args:
        model: 模型
        data_path: 数据路径
        noise_level: 噪声水平（如果add_synthetic_noise=True）
        save_results: 是否保存结果
        add_synthetic_noise: 是否添加合成噪声（False表示直接对真实数据降噪）
    """
    device = next(model.parameters()).device
    
    # 加载数据
    print(f"\nLoading test data from {data_path}...")
    data = np.load(data_path)
    print(f"Data shape: {data.shape}")
    
    # 数据统计
    data_mean = np.mean(data)
    data_std = np.std(data)
    print(f"Mean: {data_mean:.4f}, Std: {data_std:.4f}")
    
    if add_synthetic_noise and noise_level is not None:
        # 模式1：添加合成噪声测试
        print(f"\n[测试模式] 添加合成噪声 (level={noise_level})")
        data_tensor = torch.from_numpy(data).unsqueeze(0).unsqueeze(0).float()
        noisy_tensor = add_noise_2d(data_tensor, noise_level=noise_level)
        noisy = noisy_tensor.squeeze().numpy()
        
        # 计算噪声前的指标
        metrics_noisy = calculate_metrics_2d(noisy, data)
        print(f"Noisy PSNR: {metrics_noisy['psnr']:.2f} dB")
        print(f"Noisy SNR: {metrics_noisy['snr']:.2f} dB")
        clean_reference = data
    else:
        # 模式2：直接对真实含噪数据降噪（推荐）
        print(f"\n[真实模式] 直接对真实含噪数据降噪")
        noisy = data
        metrics_noisy = None
        clean_reference = None
        print("⚠ 无干净参考，将只输出降噪结果")
    
    # 降噪
    print("\nDenoising...")
    start_time = time.time()
    
    ph, pw = patch_size or DATA_CONFIG['patch_size']
    sh, sw = stride or DATA_CONFIG['stride']
    denoised, stats = denoise_2d_data(
        model, noisy,
        patch_size=(ph, pw),
        stride=(sh, sw),
        device=device,
        batch_size=128
    )
    
    denoise_time = time.time() - start_time
    print(f"Denoising completed in {denoise_time:.2f}s")
    
    # 提取统计信息
    avg_noise_level = stats['avg_noise_level']
    avg_freq_weights = stats['avg_freq_weights']
    noise_levels = stats['noise_levels']
    gate_weights = stats['gate_weights']
    freq_weights = stats['freq_weights']
    
    # 计算指标
    if clean_reference is not None:
        print("\nCalculating metrics...")
        metrics_denoised = calculate_metrics_2d(denoised, clean_reference)
        
        print("\nResults:")
        print(f"  PSNR improvement: {metrics_noisy['psnr']:.2f} → {metrics_denoised['psnr']:.2f} dB "
              f"(+{metrics_denoised['psnr'] - metrics_noisy['psnr']:.2f} dB)")
        print(f"  SNR improvement:  {metrics_noisy['snr']:.2f} → {metrics_denoised['snr']:.2f} dB "
              f"(+{metrics_denoised['snr'] - metrics_noisy['snr']:.2f} dB)")
        print(f"  Correlation: {metrics_denoised['correlation']:.4f}")
    else:
        print("\n✓ 降噪完成（真实数据模式，无定量指标）")
        print("  请查看可视化结果判断效果")
        metrics_denoised = {}
        no_ref_metrics = calculate_no_ref_metrics(noisy, denoised)
    
    # 保存结果
    if save_results:
        # 获取数据名称
        data_name = os.path.splitext(os.path.basename(data_path))[0]
        
        # 生成文件名后缀（区分不同测试模式和噪声水平）
        tag_part = f"_{tag}" if tag else ""
        if add_synthetic_noise and noise_level is not None:
            file_suffix = f"{tag_part}_noise{noise_level}"
        else:
            file_suffix = f"{tag_part}_real"
        
        # === 生成多种可视化图像 ===
        print("\n生成可视化图像...")
        
        # 1. IPYNB风格（三栏布局）
        fig_path_ipynb = os.path.join(PATHS['figures'], 
                                      f'{data_name}{file_suffix}_ipynb_style.png')
        plot_ipynb_style(
            data=noisy,
            denoised=denoised,
            save_path=fig_path_ipynb,
            data_name=data_name,
            vmin=-30,
            vmax=30
        )
        
        # 2. NAAG分析图（如果有NAAG统计信息）
        if noise_levels is not None and gate_weights is not None:
            fig_path_naag = os.path.join(PATHS['figures'], 
                                        f'{data_name}{file_suffix}_naag_analysis.png')
            plot_naag_analysis(noise_levels, gate_weights, save_path=fig_path_naag)
        
        # 3. FDA分析图（如果有FDA统计信息）
        if freq_weights is not None:
            fig_path_fda = os.path.join(PATHS['figures'], 
                                       f'{data_name}{file_suffix}_fda_analysis.png')
            plot_fda_analysis(freq_weights, save_path=fig_path_fda)
        
        # 4. FDA+NAAG联合分析图（如果两者都有）
        if noise_levels is not None and gate_weights is not None and freq_weights is not None:
            fig_path_combined = os.path.join(PATHS['figures'], 
                                            f'{data_name}{file_suffix}_fda_naag_combined.png')
            visualize_fda_naag_combined(noise_levels, gate_weights, freq_weights, 
                                       save_path=fig_path_combined)
        
        # 保存图像 - 标准风格（可选，包含原始干净数据对比）
        # fig_path = os.path.join(PATHS['figures'], 
        #                        f'{data_name}_denoised_2d.png')
        # plot_2d_comparison(
        #     original=noisy,
        #     denoised=denoised,
        #     noise=noisy - denoised,
        #     save_path=fig_path,
        #     vmin=-30,
        #     vmax=30
        # )
        
        # 保存数值结果
        results = {
            'data_name': data_name,
            'test_mode': 'synthetic_noise' if add_synthetic_noise else 'real_data',
            'noise_level': noise_level if add_synthetic_noise else 'N/A',
            'denoise_time': denoise_time,
            'naag_avg_noise_level': float(avg_noise_level) if avg_noise_level is not None else None,
            'fda_avg_freq_weights': avg_freq_weights.tolist() if avg_freq_weights is not None else None
        }
        
        if metrics_noisy is not None and metrics_denoised:
            results['metrics_noisy'] = metrics_noisy
            results['metrics_denoised'] = metrics_denoised
            results['improvement'] = {
                'psnr': float(metrics_denoised.get('psnr', 0) - metrics_noisy.get('psnr', 0)),
                'snr': float(metrics_denoised.get('snr', 0) - metrics_noisy.get('snr', 0))
            }
        elif not add_synthetic_noise:
            results['no_ref_metrics'] = no_ref_metrics
        
        results_path = os.path.join(PATHS['results'], 
                                   f'{data_name}{file_suffix}_test_results.json')
        
        with open(results_path, 'w') as f:
            # 转换numpy类型
            def convert_types(obj):
                if isinstance(obj, dict):
                    return {k: convert_types(v) for k, v in obj.items()}
                elif isinstance(obj, (np.integer, np.floating)):
                    return float(obj)
                return obj
            
            json.dump(convert_types(results), f, indent=4)
        
        print(f"\n✓ Results saved:")
        print(f"  📊 Figures:")
        print(f"     - IPYNB风格: {fig_path_ipynb}")
        if noise_levels is not None and gate_weights is not None:
            print(f"     - NAAG分析: {os.path.join(PATHS['figures'], f'{data_name}{file_suffix}_naag_analysis.png')}")
        if freq_weights is not None:
            print(f"     - FDA分析: {os.path.join(PATHS['figures'], f'{data_name}{file_suffix}_fda_analysis.png')}")
        if noise_levels is not None and gate_weights is not None and freq_weights is not None:
            print(f"     - FDA+NAAG联合: {os.path.join(PATHS['figures'], f'{data_name}{file_suffix}_fda_naag_combined.png')}")
        print(f"  📄 Metrics: {results_path}")
        
        # 保存去噪后的数据
        denoised_path = os.path.join(PATHS['results'], 
                                    f'{data_name}{file_suffix}_denoised.npy')
        np.save(denoised_path, denoised)
        print(f"  Denoised data: {denoised_path}")
    
    return denoised, metrics_denoised


def test_trad_only(data_path, noise_level=None, add_synthetic_noise=False,
                   save_results=True):
    """
    Pure traditional denoising test — no deep learning involved.
    Runs all 8 traditional methods directly on raw/noisy data and
    generates comparison plots and a metrics JSON.

    Generated figures:
      1. Seismic image comparison (all methods)
      2. SNR/PSNR bar chart (when clean reference is available)
      3. Removed noise residuals
      4. Single-trace waveform comparison

    Args:
        data_path: Path to .npy data file
        noise_level: Synthetic noise level (used when add_synthetic_noise=True)
        add_synthetic_noise: True → add Gaussian noise then denoise; False → use real noisy data
        save_results: Whether to save figures and JSON
    """
    print(f"\n{'='*70}")
    print("[Pure Traditional Denoising] Loading data...")
    data = np.load(data_path)
    print(f"  Data shape: {data.shape}")
    data_name = os.path.splitext(os.path.basename(data_path))[0]

    if add_synthetic_noise and noise_level is not None:
        data_tensor = torch.from_numpy(data).unsqueeze(0).unsqueeze(0).float()
        noisy_tensor = add_noise_2d(data_tensor, noise_level=noise_level)
        noisy = noisy_tensor.squeeze().numpy()
        reference = data
        file_suffix = f"_trad_noise{noise_level}"
        print(f"  Mode: synthetic noise (level={noise_level})")
    else:
        noisy = data
        reference = None
        file_suffix = "_trad_real"
        print("  Mode: real data (no clean reference)")

    print(f"\n[Pure Traditional Denoising] Applying {len(TRAD_METHODS)} methods...")
    start = time.time()
    trad_results = run_trad_only(noisy)
    elapsed = time.time() - start
    print(f"  Total time: {elapsed:.2f}s")

    if reference is not None:
        print(f"\n  {'Method':<20} {'SNR (dB)':>10} {'PSNR (dB)':>10}")
        print(f"  {'-'*42}")
        for key, arr in trad_results.items():
            met = calculate_metrics_2d(arr, reference)
            name = 'Raw' if key == 'raw' else (TRAD_METHODS[key]['name'] if key in TRAD_METHODS else key)
            print(f"  {name:<20} {met['snr']:>10.2f} {met['psnr']:>10.2f}")

    if save_results:
        figs_dir = PATHS['figures']
        res_dir = PATHS['results']
        os.makedirs(figs_dir, exist_ok=True)
        os.makedirs(res_dir, exist_ok=True)
        print("\n[Pure Traditional Denoising] Saving figures...")

        # 1. Seismic image comparison
        fig1_path = os.path.join(figs_dir, f'{data_name}{file_suffix}_comparison.png')
        plot_trad_comparison(
            trad_results, reference=reference,
            vmin=-30, vmax=30,
            save_path=fig1_path, data_name=data_name
        )

        # 2. SNR/PSNR bar chart (requires clean reference)
        fig2_path = None
        if reference is not None:
            fig2_path = os.path.join(figs_dir, f'{data_name}{file_suffix}_metrics_bar.png')
            plot_trad_metrics_bar(
                trad_results, reference,
                save_path=fig2_path, data_name=data_name
            )

        # 3. Removed noise residuals
        fig3_path = os.path.join(figs_dir, f'{data_name}{file_suffix}_noise_residuals.png')
        plot_noise_residuals(
            trad_results, raw=noisy,
            vmin=-30, vmax=30,
            save_path=fig3_path, data_name=data_name
        )

        # 4. Single-trace waveform comparison
        fig4_path = os.path.join(figs_dir, f'{data_name}{file_suffix}_trace_compare.png')
        plot_single_trace_comparison(
            trad_results, raw=noisy, reference=reference,
            save_path=fig4_path, data_name=data_name
        )

        # Save denoised arrays
        for key, arr in trad_results.items():
            if key == 'raw':
                continue
            np.save(os.path.join(res_dir, f'{data_name}{file_suffix}_{key}.npy'), arr)

        # Save metrics JSON
        metrics_out = {}
        if reference is not None:
            for key, arr in trad_results.items():
                met = calculate_metrics_2d(arr, reference)
                metrics_out[key] = {k: float(v) for k, v in met.items()}
        else:
            for key, arr in trad_results.items():
                if key == 'raw':
                    continue
                met = calculate_no_ref_metrics(noisy, arr)
                metrics_out[key] = {k: float(v) for k, v in met.items()}
        import json as _json
        json_path = os.path.join(res_dir, f'{data_name}{file_suffix}_metrics.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            _json.dump(metrics_out, f, indent=4, ensure_ascii=False)

        print(f"\n✓ Pure traditional denoising results saved to: {figs_dir}")
        print(f"  - Seismic comparison:  {os.path.basename(fig1_path)}")
        if fig2_path:
            print(f"  - Metrics bar chart:   {os.path.basename(fig2_path)}")
        print(f"  - Noise residuals:     {os.path.basename(fig3_path)}")
        print(f"  - Single trace:        {os.path.basename(fig4_path)}")
        print(f"  - Metrics JSON:        {os.path.basename(json_path)}")

    return trad_results


def create_detailed_comparison_plot(original, noisy, denoised, save_path=None):
    """创建详细对比图（类似USL DIP论文风格）"""
    
    fig = plt.figure(figsize=(18, 6))
    
    # 参数
    v = 30
    n1, n2 = original.shape
    time_interval = 0.001
    time = np.arange(0, n1) * time_interval
    extent = (1, n2, time[-1], 0)
    
    # 原始数据
    ax1 = fig.add_subplot(1, 4, 1)
    im1 = ax1.imshow(original, cmap=cseis(), vmin=-v, vmax=v,
                     aspect='auto', extent=extent)
    ax1.set_xlabel("Trace", fontsize=12)
    ax1.set_ylabel("Time (s)", fontsize=12)
    ax1.set_title('Clean Data', fontsize=14, fontweight='bold')
    
    # 噪声数据
    ax2 = fig.add_subplot(1, 4, 2)
    im2 = ax2.imshow(noisy, cmap=cseis(), vmin=-v, vmax=v,
                     aspect='auto', extent=extent)
    ax2.set_xlabel("Trace", fontsize=12)
    ax2.set_yticks([])
    ax2.set_title('Noisy Data', fontsize=14, fontweight='bold')
    
    # 去噪数据
    ax3 = fig.add_subplot(1, 4, 3)
    im3 = ax3.imshow(denoised, cmap=cseis(), vmin=-v, vmax=v,
                     aspect='auto', extent=extent)
    ax3.set_xlabel("Trace", fontsize=12)
    ax3.set_yticks([])
    ax3.set_title('Denoised Data', fontsize=14, fontweight='bold')
    
    # 去除的噪声
    ax4 = fig.add_subplot(1, 4, 4)
    im4 = ax4.imshow(noisy - denoised, cmap=cseis(), vmin=-v, vmax=v,
                     aspect='auto', extent=extent)
    ax4.set_xlabel("Trace", fontsize=12)
    ax4.set_yticks([])
    ax4.set_title('Removed Noise', fontsize=14, fontweight='bold')
    
    # 调整布局
    plt.tight_layout()
    
    # 添加colorbar
    fig.subplots_adjust(right=0.92)
    cbar_ax = fig.add_axes([0.93, 0.2, 0.01, 0.6])
    cb = plt.colorbar(im4, cax=cbar_ax)
    cb.ax.tick_params(labelsize=10)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Detailed comparison saved to {save_path}")
    
    return fig


def test_hybrid_denoise(model, data_path, noise_level=None,
                        add_synthetic_noise=False,
                        methods=None, save_results=True,
                        patch_size=None, stride=None):
    """
    深度学习 + 传统方法联合去噪测试
    生成多张对比图：
      1. 全方法 seismic 图像对比
      2. SNR / PSNR 柱状图
      3. 去除噪声残差图
      4. 单道波形对比

    Args:
        model: 训练好的DL模型
        data_path: 数据路径 (.npy)
        noise_level: 合成噪声水平
        add_synthetic_noise: 是否添加合成噪声
        methods: 传统方法列表，None则使用全部
        save_results: 是否保存图像和JSON
    """
    device = next(model.parameters()).device

    # ---------- 加载数据 ----------
    print(f"\n{'='*70}")
    print("[混合去噪] 加载数据...")
    data = np.load(data_path)
    print(f"  数据形状: {data.shape}")
    data_name = os.path.splitext(os.path.basename(data_path))[0]

    # ---------- 准备含噪/干净数据 ----------
    if add_synthetic_noise and noise_level is not None:
        data_tensor = torch.from_numpy(data).unsqueeze(0).unsqueeze(0).float()
        noisy_tensor = add_noise_2d(data_tensor, noise_level=noise_level)
        noisy = noisy_tensor.squeeze().numpy()
        reference = data
        file_suffix = f"_hybrid_noise{noise_level}"
        print(f"  模式: 合成噪声 (level={noise_level})")
    else:
        noisy = data
        reference = None
        file_suffix = "_hybrid_real"
        print("  模式: 真实数据（无干净参考）")

    # ---------- DL 去噪 ----------
    print("\n[混合去噪] 运行深度学习去噪...")
    start = time.time()
    ph, pw = patch_size or DATA_CONFIG['patch_size']
    sh, sw = stride or DATA_CONFIG['stride']
    dl_denoised, stats = denoise_2d_data(
        model, noisy,
        patch_size=(ph, pw),
        stride=(sh, sw),
        device=device,
        batch_size=128
    )
    dl_time = time.time() - start
    print(f"  DL 耗时: {dl_time:.2f}s")

    # ---------- 传统后处理 ----------
    print("\n[混合去噪] 应用传统方法后处理...")
    if methods is None:
        methods = list(TRAD_METHODS.keys())

    results = run_all_hybrid(noisy, dl_denoised, methods=methods)

    # ---------- 打印指标 ----------
    if reference is not None:
        print("\n{'='*60}")
        print(f"  {'方法':<20} {'SNR(dB)':>10} {'PSNR(dB)':>10} {'Corr':>8}")
        print(f"  {'-'*50}")
        from utils_2d import calculate_metrics_2d
        for key, arr in results.items():
            met = calculate_metrics_2d(arr, reference)
            name = 'Raw' if key == 'raw' else ('DL Only' if key == 'dl' else key)
            print(f"  {name:<20} {met['snr']:>10.2f} {met['psnr']:>10.2f} {met['correlation']:>8.4f}")
        print(f"{'='*60}\n")

    # ---------- 保存图像 ----------
    if save_results:
        figs_dir = PATHS['figures']
        res_dir  = PATHS['results']
        print("\n[混合去噪] 生成并保存可视化图像...")

        # 1. 全方法 seismic 图像对比
        fig1_path = os.path.join(figs_dir, f'{data_name}{file_suffix}_comparison.png')
        plot_hybrid_comparison(
            results, reference=reference,
            vmin=-30, vmax=30,
            save_path=fig1_path,
            data_name=data_name
        )

        # 2. SNR / PSNR 柱状图（需参考）
        if reference is not None:
            fig2_path = os.path.join(figs_dir, f'{data_name}{file_suffix}_metrics_bar.png')
            plot_metrics_bar(
                results, reference,
                save_path=fig2_path,
                data_name=data_name
            )

        # 3. 去除噪声残差图
        fig3_path = os.path.join(figs_dir, f'{data_name}{file_suffix}_noise_residuals.png')
        plot_noise_residuals(
            results, raw=noisy,
            vmin=-30, vmax=30,
            save_path=fig3_path,
            data_name=data_name
        )

        # 4. 单道波形对比（取中间道）
        fig4_path = os.path.join(figs_dir, f'{data_name}{file_suffix}_trace_compare.png')
        plot_single_trace_comparison(
            results, raw=noisy, reference=reference,
            save_path=fig4_path,
            data_name=data_name
        )

        # 5. TV 专项对比图（纯TV vs 纯DL vs DL+TV）
        tv_only_arr = None
        dl_tv_arr   = results.get('dl+tv')
        if dl_tv_arr is not None:
            # 单独计算纯TV结果
            from traditional_denoise import apply_traditional
            try:
                tv_only_arr = apply_traditional(noisy, 'tv')
                print('  ✓ 纯TV去噪完成')
            except Exception as e:
                print(f'  ✗ 纯TV去噪失败: {e}')

        if tv_only_arr is not None and dl_tv_arr is not None:
            fig5_path = os.path.join(figs_dir, f'{data_name}{file_suffix}_tv_focused.png')
            plot_tv_focused_comparison(
                noisy=noisy,
                tv_only=tv_only_arr,
                dl_only=dl_denoised,
                dl_tv=dl_tv_arr,
                reference=reference,
                vmin=-30, vmax=30,
                save_path=fig5_path,
                data_name=data_name
            )
        else:
            fig5_path = None

        # 6. 保存每种方法的去噪数据
        for key, arr in results.items():
            if key == 'raw':
                continue
            npy_path = os.path.join(res_dir,
                                    f'{data_name}{file_suffix}_{key.replace("+","_")}.npy')
            np.save(npy_path, arr)

        # 6. 保存指标 JSON
        metrics_out = {}
        if reference is not None:
            from utils_2d import calculate_metrics_2d
            for key, arr in results.items():
                met = calculate_metrics_2d(arr, reference)
                metrics_out[key] = {
                    k: float(v) for k, v in met.items()
                }
        else:
            for key, arr in results.items():
                if key == 'raw':
                    continue
                met = calculate_no_ref_metrics(noisy, arr)
                metrics_out[key] = {
                    k: float(v) for k, v in met.items()
                }
        json_path = os.path.join(res_dir, f'{data_name}{file_suffix}_metrics.json')
        import json as json_mod
        with open(json_path, 'w', encoding='utf-8') as f:
            json_mod.dump(metrics_out, f, indent=4, ensure_ascii=False)

        print(f"\n✓ 混合去噪图像已保存至: {figs_dir}")
        print(f"  - 全方法对比图: {os.path.basename(fig1_path)}")
        if reference is not None:
            print(f"  - 指标柱状图:   {os.path.basename(fig2_path)}")
        print(f"  - 噪声残差图:   {os.path.basename(fig3_path)}")
        print(f"  - 单道对比图:   {os.path.basename(fig4_path)}")
        if fig5_path:
            print(f"  - TV专项对比图: {os.path.basename(fig5_path)}")
        print(f"  - 指标JSON:     {os.path.basename(json_path)}")

    return results


def plot_tv_focused_comparison(noisy, tv_only, dl_only, dl_tv,
                               reference=None,
                               vmin=-30, vmax=30,
                               save_path=None, data_name=''):
    """
    TV专项对比图：
      列1: 原始含噪
      列2: 纯TV去噪
      列3: 纯DL去噪
      列4: DL + TV 联合去噪
      列5: (可选) 干净参考
    同时在每列下方绘制中央道波形对比。
    """
    import matplotlib.pyplot as plt
    from utils_2d import cseis, calculate_metrics_2d

    panels = [('Raw (Noisy)', noisy),
              ('TV Only', tv_only),
              ('DL Only', dl_only),
              ('DL + TV', dl_tv)]
    if reference is not None:
        panels.append(('Clean Ref', reference))

    n = len(panels)
    h, w = noisy.shape
    t = np.arange(h) * 0.001
    extent = (1, w, t[-1], 0)
    trace_idx = w // 2
    cmap = cseis()

    fig = plt.figure(figsize=(n * 4.5, 9))
    gs = fig.add_gridspec(2, n, height_ratios=[3, 1], hspace=0.35, wspace=0.25)

    colors = ['#888888', '#E65100', '#1565C0', '#2E7D32', '#000000']

    for col, ((title, arr), color) in enumerate(zip(panels, colors)):
        # ---- 地震剖面 ----
        ax_img = fig.add_subplot(gs[0, col])
        im = ax_img.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax,
                           aspect='auto', extent=extent)

        # 标题 + 可选指标
        full_title = title
        if reference is not None and title not in ('Raw (Noisy)', 'Clean Ref'):
            met = calculate_metrics_2d(arr, reference)
            full_title += f'\nSNR {met["snr"]:.1f} dB  PSNR {met["psnr"]:.1f} dB'
        elif title == 'Raw (Noisy)' and reference is not None:
            met = calculate_metrics_2d(arr, reference)
            full_title += f'\nSNR {met["snr"]:.1f} dB'

        ax_img.set_title(full_title, fontsize=10, fontweight='bold')
        ax_img.set_xlabel('Trace', fontsize=9)
        if col == 0:
            ax_img.set_ylabel('Time (s)', fontsize=9)
        else:
            ax_img.set_yticks([])
        plt.colorbar(im, ax=ax_img, fraction=0.046, pad=0.04)

        # ---- 中央道波形 ----
        ax_tr = fig.add_subplot(gs[1, col])
        ax_tr.plot(t, noisy[:, trace_idx], color='#BDBDBD',
                   linewidth=0.6, alpha=0.7, label='Raw')
        if reference is not None:
            ax_tr.plot(t, reference[:, trace_idx], color='black',
                       linewidth=1.0, linestyle='--', alpha=0.6, label='Ref')
        ax_tr.plot(t, arr[:, trace_idx], color=color,
                   linewidth=1.2, label=title)
        ax_tr.set_xlabel('Time (s)', fontsize=8)
        ax_tr.set_xlim(t[0], t[-1])
        ax_tr.tick_params(labelsize=7)
        ax_tr.set_title(f'Trace #{trace_idx + 1}', fontsize=8)
        ax_tr.grid(alpha=0.3)
        ax_tr.legend(fontsize=6, loc='upper right')

    suptitle = 'TV 去噪专项对比：纯TV  vs  纯DL  vs  DL+TV'
    if data_name:
        suptitle += f'  —  {data_name}'
    fig.suptitle(suptitle, fontsize=13, fontweight='bold', y=1.01)

    if save_path:
        plt.savefig(save_path, dpi=180, bbox_inches='tight')
        print(f'✓ TV专项对比图保存至: {save_path}')

    plt.close(fig)
    return fig


def test_combo_denoise(model, data_path, noise_level=None,
                       add_synthetic_noise=False, save_results=True,
                       patch_size=None, stride=None):
    """
    深度学习 + 多传统方法组合去噪测试。
    对比 DL Only vs DL+两方法组合 vs DL+三方法组合，并按 SNR 排名。

    生成图像：
      1. 组合方法地震剖面对比图
      2. SNR/PSNR 排名柱状图（有参考时）
      3. 各组合去除的噪声残差图
      4. 单道波形对比图

    Args:
        model: 训练好的 DL 模型
        data_path: 数据路径 (.npy)
        noise_level: 合成噪声水平（add_synthetic_noise=True 时有效）
        add_synthetic_noise: True → 添加合成噪声；False → 直接对真实数据降噪
        save_results: 是否保存图像和 JSON
    """
    device = next(model.parameters()).device

    print(f"\n{'='*70}")
    print("[多方法组合去噪] 加载数据...")
    data = np.load(data_path)
    print(f"  数据形状: {data.shape}")
    data_name = os.path.splitext(os.path.basename(data_path))[0]

    # ---------- 准备含噪 / 干净数据 ----------
    if add_synthetic_noise and noise_level is not None:
        data_tensor = torch.from_numpy(data).unsqueeze(0).unsqueeze(0).float()
        noisy_tensor = add_noise_2d(data_tensor, noise_level=noise_level)
        noisy = noisy_tensor.squeeze().numpy()
        reference = data
        file_suffix = f"_combo_noise{noise_level}"
        print(f"  模式: 合成噪声 (level={noise_level})")
    else:
        noisy = data
        reference = None
        file_suffix = "_combo_real"
        print("  模式: 真实数据（无干净参考）")

    # ---------- DL 去噪 ----------
    print("\n[多方法组合去噪] 运行深度学习去噪...")
    start = time.time()
    ph, pw = patch_size or DATA_CONFIG['patch_size']
    sh, sw = stride or DATA_CONFIG['stride']
    dl_denoised, stats = denoise_2d_data(
        model, noisy,
        patch_size=(ph, pw),
        stride=(sh, sw),
        device=device,
        batch_size=128
    )
    dl_time = time.time() - start
    print(f"  DL 耗时: {dl_time:.2f}s")

    # ---------- 多传统方法组合后处理 ----------
    print(f"\n[多方法组合去噪] 应用 {len(COMBO_METHODS)} 种组合方案...")
    combo_results = run_combo_hybrid(noisy, dl_denoised)

    # ---------- 保存结果 ----------
    if save_results:
        figs_dir = PATHS['figures']
        res_dir = PATHS['results']
        os.makedirs(figs_dir, exist_ok=True)
        os.makedirs(res_dir, exist_ok=True)
        print("\n[多方法组合去噪] 生成并保存可视化图像...")

        # 1. 组合方法地震剖面对比图
        fig1_path = os.path.join(figs_dir,
                                 f'{data_name}{file_suffix}_combo_comparison.png')
        plot_combo_comparison(
            combo_results, reference=reference,
            vmin=-30, vmax=30,
            save_path=fig1_path,
            data_name=data_name
        )

        # 2. SNR/PSNR 排名柱状图（需参考）
        fig2_path = None
        if reference is not None:
            fig2_path = os.path.join(figs_dir,
                                     f'{data_name}{file_suffix}_combo_metrics_bar.png')
            plot_combo_metrics_bar(
                combo_results, reference,
                save_path=fig2_path,
                data_name=data_name
            )

        # 3. 各组合去除的噪声残差图
        fig3_path = os.path.join(figs_dir,
                                 f'{data_name}{file_suffix}_combo_noise_residuals.png')
        plot_noise_residuals(
            combo_results, raw=noisy,
            vmin=-30, vmax=30,
            save_path=fig3_path,
            data_name=data_name
        )

        # 4. 单道波形对比
        fig4_path = os.path.join(figs_dir,
                                 f'{data_name}{file_suffix}_combo_trace_compare.png')
        plot_single_trace_comparison(
            combo_results, raw=noisy, reference=reference,
            save_path=fig4_path,
            data_name=data_name
        )

        # 5. 保存每种组合的去噪数据
        for key, arr in combo_results.items():
            if key == 'raw':
                continue
            npy_path = os.path.join(res_dir,
                                    f'{data_name}{file_suffix}'
                                    f'_{key.replace("+", "_")}.npy')
            np.save(npy_path, arr)

        # 6. 保存指标 JSON
        metrics_out = {}
        if reference is not None:
            from utils_2d import calculate_metrics_2d
            for key, arr in combo_results.items():
                met = calculate_metrics_2d(arr, reference)
                metrics_out[key] = {k: float(v) for k, v in met.items()}
        else:
            for key, arr in combo_results.items():
                if key == 'raw':
                    continue
                met = calculate_no_ref_metrics(noisy, arr)
                metrics_out[key] = {k: float(v) for k, v in met.items()}
        json_path = os.path.join(res_dir,
                                 f'{data_name}{file_suffix}_combo_metrics.json')
        import json as _json
        with open(json_path, 'w', encoding='utf-8') as f:
            _json.dump(metrics_out, f, indent=4, ensure_ascii=False)

        print(f"\n✓ 多组合去噪结果已保存至: {figs_dir}")
        print(f"  - 地震剖面对比图: {os.path.basename(fig1_path)}")
        if fig2_path:
            print(f"  - 指标排名柱状图: {os.path.basename(fig2_path)}")
        print(f"  - 噪声残差图:     {os.path.basename(fig3_path)}")
        print(f"  - 单道波形对比:   {os.path.basename(fig4_path)}")
        print(f"  - 指标JSON:       {os.path.basename(json_path)}")

    return combo_results


def batch_test(model, test_data_list, noise_levels=[0.05, 0.1, 0.15],
               patch_size=None, stride=None):
    """批量测试多个数据和噪声水平"""
    
    print("=" * 70)
    print("Batch Testing")
    print("=" * 70)
    
    all_results = []
    
    for data_path in test_data_list:
        data_name = os.path.splitext(os.path.basename(data_path))[0]
        
        print(f"\n{'='*70}")
        print(f"Testing on: {data_name}")
        print(f"{'='*70}")
        
        for noise_level in noise_levels:
            print(f"\n--- Noise level: {noise_level} ---")
            
            denoised, metrics = test_on_data(
                model, data_path, noise_level=noise_level,
                save_results=True, add_synthetic_noise=True,
                patch_size=patch_size, stride=stride
            )
            
            all_results.append({
                'data_name': data_name,
                'noise_level': noise_level,
                'metrics': metrics
            })
    
    # 保存汇总结果
    summary_path = os.path.join(PATHS['results'], 'test_summary.json')
    with open(summary_path, 'w') as f:
        def convert_types(obj):
            if isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(item) for item in obj]
            elif isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            return obj
        
        json.dump(convert_types(all_results), f, indent=4)
    
    print(f"\n{'='*70}")
    print(f"Batch testing completed!")
    print(f"Summary saved to: {summary_path}")
    print(f"{'='*70}")
    
    return all_results


if __name__ == "__main__":
    print("\n" + "="*70)
    print("2D模型测试 - 使用最佳超参数和最佳权重")
    print("="*70)

    parser = argparse.ArgumentParser(description="2D denoising model test")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint path. If relative, resolved against new/ directory.",
    )
    parser.add_argument(
        "--low-snr",
        action="store_true",
        help="Use low-SNR config (config_low_snr.py) to rebuild model and paths.",
    )
    parser.add_argument(
        "--mode",
        choices=["real", "synthetic"],
        default=None,
        help="Test mode. real=direct denoise, synthetic=add noise to clean then denoise.",
    )
    parser.add_argument(
        "--dataset",
        choices=["eq-36", "eq-68", "slice_german", "slice_german_1"],
        default=None,
        help="Which dataset to test first (if available).",
    )
    parser.add_argument(
        "--batch-all",
        action="store_true",
        help="Run batch test on all available datasets without prompting.",
    )
    args, _unknown = parser.parse_known_args()

    # Choose runtime config based on flags/checkpoint hint
    model_config, data_config, paths, use_best_params = _select_runtime_config(args)
    
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")
    
    # 加载模型（支持 --checkpoint）
    if args.checkpoint:
        checkpoint_path = _resolve_path(args.checkpoint)
    else:
        # 默认 checkpoint：常规模式 best_model_2d.pth；低SNR模式 best_model.pth
        default_name = 'best_model.pth' if paths.get('checkpoints', '').endswith('checkpoints_low_snr') else 'best_model_2d.pth'
        checkpoint_path = os.path.join(paths['checkpoints'], default_name)
    
    if not os.path.exists(checkpoint_path):
        print(f"\n❌ 模型文件不存在: {checkpoint_path}")
        if paths.get('checkpoints', '').endswith('checkpoints_low_snr'):
            print("请先训练低SNR模型: python train_low_snr.py")
        else:
            print("请先训练模型: python train_2d.py")
        sys.exit(1)
    
    # use_best_params=True 确保使用优化的超参数
    # 用选择的配置构建模型并加载权重
    # 注意：低SNR checkpoint 与常规 config_2d 结构往往不一致，需要用 low_snr config 重建模型
    print(f"\n当前配置: {'low_snr' if not use_best_params else 'default/optimized'}")
    print(f"模型checkpoint: {checkpoint_path}")

    # 无论常规还是低SNR路径，均使用 load_trained_model（内含自动结构检测）
    model = load_trained_model(checkpoint_path, device, use_best_params=use_best_params)
    
    # 测试数据列表
    test_data_paths = [
        os.path.join(paths['data'], 'eq-36.npy'),
        os.path.join(paths['data'], 'eq-68.npy'),
        os.path.join(paths['data'], 'slice_german.npy'),
        os.path.join(paths['data'], 'slice_german_1.npy'),
    ]
    
    # 检查数据文件
    available_data = [p for p in test_data_paths if os.path.exists(p)]
    
    if not available_data:
        print("\nError: No test data found!")
        print("Available datasets in DATA_CONFIG:")
        for name, path in data_config.get('datasets', {}).items():
            print(f"  - {name}: {path}")
        sys.exit(1)
    
    print(f"\nFound {len(available_data)} test dataset(s)")
    
    # 单个测试示例
    print("\n" + "=" * 70)
    print("测试选项")
    print("=" * 70)
    print("1. 真实数据模式（推荐）：直接对真实含噪数据降噪")
    print("2. 合成噪声模式：在干净数据上添加噪声后降噪")

    # 选择模式：优先使用命令行参数，否则走交互
    if args.mode is not None:
        test_mode = "1" if args.mode == "real" else "2"
    else:
        test_mode = input("\n选择测试模式 (1/2，默认=1): ").strip() or "1"

    # 选择数据集：优先使用命令行参数，否则用第一份可用数据
    if args.dataset is not None:
        preferred = os.path.join(paths['data'], f"{args.dataset}.npy")
        if preferred in available_data:
            first_data = preferred
        else:
            first_data = available_data[0]
            print(f"\n⚠ 指定数据集不存在或不可用: {preferred}，改用: {first_data}")
    else:
        first_data = available_data[0]
    
    print("\n" + "=" * 70)
    if test_mode == "1":
        print("真实数据测试")
        print("=" * 70)
        test_on_data(
            model,
            first_data,
            add_synthetic_noise=False,
            save_results=True,
            patch_size=data_config.get('patch_size'),
            stride=data_config.get('stride')
        )
    else:
        print("合成噪声测试")
        print("=" * 70)
        test_on_data(
            model,
            first_data,
            noise_level=0.1,
            add_synthetic_noise=True,
            save_results=True,
            patch_size=data_config.get('patch_size'),
            stride=data_config.get('stride')
        )
    
    # ---------- 纯传统方法去噪测试 ----------
    print("\n" + "=" * 70)
    print("Pure Traditional Denoising: comparing all 8 classic methods (no DL)")
    print("=" * 70)
    print(f"Methods: {', '.join(TRAD_METHODS.keys())}")

    run_trad = True
    if not args.batch_all:
        ans = input("\nRun pure traditional denoising test? (y/n, default=y): ").strip().lower()
        if ans == 'n':
            run_trad = False

    if run_trad:
        if test_mode == "1":
            test_trad_only(
                first_data,
                add_synthetic_noise=False,
                save_results=True
            )
        else:
            test_trad_only(
                first_data,
                noise_level=0.1,
                add_synthetic_noise=True,
                save_results=True
            )

    # ---------- 混合去噪测试 ----------
    print("\n" + "=" * 70)
    print("混合去噪：深度学习 + 传统方法联合测试")
    print("="*70)
    print("将生成以下图像：")
    print("  1. 全方法对比图 (seismic 风格)")
    print("  2. SNR/PSNR 指标柱状图")
    print("  3. 去除噪声残差图")
    print("  4. 单道波形对比")

    run_hybrid = True  # 默认运行混合测试
    if not args.batch_all:
        ans = input("\n是否运行混合去噪测试? (y/n, 默认=y): ").strip().lower()
        if ans == 'n':
            run_hybrid = False

    if run_hybrid:
        if test_mode == "1":
            test_hybrid_denoise(
                model, first_data,
                add_synthetic_noise=False,
                save_results=True,
                patch_size=data_config.get('patch_size'),
                stride=data_config.get('stride')
            )
        else:
            test_hybrid_denoise(
                model, first_data,
                noise_level=0.1,
                add_synthetic_noise=True,
                save_results=True,
                patch_size=data_config.get('patch_size'),
                stride=data_config.get('stride')
            )

    # ---------- 多方法组合去噪测试 ----------
    print("\n" + "=" * 70)
    print("多方法组合去噪：深度学习 + 两种/三种传统算法组合对比")
    print("=" * 70)
    print(f"共 {len(COMBO_METHODS)} 种预定义组合方案：")
    for key in COMBO_METHODS:
        print(f"  DL + {combo_display_name(key)}")

    run_combo = True
    if not args.batch_all:
        ans = input("\n是否运行多方法组合去噪测试? (y/n, 默认=y): ").strip().lower()
        if ans == 'n':
            run_combo = False

    if run_combo:
        if test_mode == "1":
            test_combo_denoise(
                model, first_data,
                add_synthetic_noise=False,
                save_results=True,
                patch_size=data_config.get('patch_size'),
                stride=data_config.get('stride')
            )
        else:
            test_combo_denoise(
                model, first_data,
                noise_level=0.1,
                add_synthetic_noise=True,
                save_results=True,
                patch_size=data_config.get('patch_size'),
                stride=data_config.get('stride')
            )

    # 批量测试（可选）
    if len(available_data) > 1:
        if args.batch_all:
            batch_test(
                model, available_data, noise_levels=[0.05, 0.1, 0.15],
                patch_size=data_config.get('patch_size'),
                stride=data_config.get('stride')
            )
        else:
            user_input = input("\nRun batch test on all datasets? (y/n): ")
            if user_input.lower() == 'y':
                batch_test(
                    model, available_data, noise_levels=[0.05, 0.1, 0.15],
                    patch_size=data_config.get('patch_size'),
                    stride=data_config.get('stride')
                )

    print("\n✓ Testing completed successfully!")
