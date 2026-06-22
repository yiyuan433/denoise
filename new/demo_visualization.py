"""
可视化演示脚本
对比新旧两种可视化风格，展示IPYNB风格的优势
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils_2d import (
    plot_2d_comparison, plot_ipynb_style, 
    add_noise_2d, cseis
)
from config_2d import DATA_CONFIG, PATHS


def compare_visualization_styles(data_path, noise_level=0.1):
    """
    对比两种可视化风格
    
    Args:
        data_path: 数据路径
        noise_level: 噪声水平
    """
    print("=" * 70)
    print("可视化风格对比演示")
    print("=" * 70)
    
    # 加载数据
    print(f"\n加载数据: {data_path}")
    data = np.load(data_path)
    print(f"数据形状: {data.shape}")
    
    # 模拟添加噪声
    print(f"\n添加噪声 (level={noise_level})...")
    data_tensor = torch.from_numpy(data).unsqueeze(0).unsqueeze(0).float()
    noisy_tensor = add_noise_2d(data_tensor, noise_level=noise_level)
    noisy = noisy_tensor.squeeze().numpy()
    
    # 模拟去噪（这里简单用高斯滤波模拟）
    from scipy.ndimage import gaussian_filter
    denoised = gaussian_filter(noisy, sigma=1.0)
    
    print("\n生成对比图...")
    
    # 获取数据名称
    data_name = os.path.splitext(os.path.basename(data_path))[0]
    
    # 1. 标准风格（多栏布局，包含原始数据）
    print("\n[1] 标准风格 - 多栏布局")
    fig_path_standard = os.path.join(PATHS['figures'], 
                                     f'{data_name}_standard_style.png')
    fig1 = plot_2d_comparison(
        original=noisy,
        denoised=denoised,
        noise=noisy - denoised,
        save_path=fig_path_standard,
        vmin=-30,
        vmax=30
    )
    print(f"  ✓ 已保存: {fig_path_standard}")
    
    # 2. IPYNB风格（三栏布局，论文风格）
    print("\n[2] IPYNB风格 - 三栏论文布局")
    fig_path_ipynb = os.path.join(PATHS['figures'], 
                                  f'{data_name}_ipynb_style.png')
    fig2 = plot_ipynb_style(
        data=noisy,
        denoised=denoised,
        save_path=fig_path_ipynb,
        data_name=data_name,
        vmin=-30,
        vmax=30
    )
    print(f"  ✓ 已保存: {fig_path_ipynb}")
    
    # 3. 并排对比
    print("\n[3] 生成并排对比图...")
    fig_comparison = plt.figure(figsize=(18, 8))
    
    # 标准风格
    ax1 = fig_comparison.add_subplot(2, 1, 1)
    img1 = plt.imread(fig_path_standard)
    ax1.imshow(img1)
    ax1.axis('off')
    ax1.set_title('标准风格 (Standard Style)', fontsize=16, fontweight='bold', pad=10)
    
    # IPYNB风格
    ax2 = fig_comparison.add_subplot(2, 1, 2)
    img2 = plt.imread(fig_path_ipynb)
    ax2.imshow(img2)
    ax2.axis('off')
    ax2.set_title('IPYNB风格 (USL DIP Style)', fontsize=16, fontweight='bold', pad=10)
    
    plt.tight_layout()
    comparison_path = os.path.join(PATHS['figures'], 
                                  f'{data_name}_style_comparison.png')
    plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
    print(f"  ✓ 并排对比已保存: {comparison_path}")
    
    print("\n" + "=" * 70)
    print("可视化对比完成！")
    print("\nIPYNB风格的优势:")
    print("  ✓ 三栏布局清晰简洁（Raw | Denoised | Noise）")
    print("  ✓ 标注 (a), (b), (c) 符合论文规范")
    print("  ✓ Colorbar位置统一，便于对比")
    print("  ✓ 去除冗余的y轴刻度，视觉更整洁")
    print("  ✓ 与USL DIP原文完全一致的风格")
    print("=" * 70)
    
    plt.show()


def create_demo_figure():
    """创建一个演示图展示IPYNB风格的特点"""
    
    print("\n" + "=" * 70)
    print("创建IPYNB风格特点演示图")
    print("=" * 70)
    
    # 创建一个简单的示例数据
    np.random.seed(42)
    H, W = 200, 100
    
    # 创建模拟地震信号
    t = np.linspace(0, 1, H)
    x = np.arange(W)
    signal = np.zeros((H, W))
    
    # 添加一些斜线事件（模拟地震波）
    for i in range(5):
        slope = np.random.uniform(-0.5, 0.5)
        offset = np.random.uniform(20, 80)
        for j in range(W):
            idx = int(H * 0.3 + slope * j + offset)
            if 0 <= idx < H:
                signal[max(0, idx-3):min(H, idx+3), j] = np.random.uniform(15, 25)
    
    # 添加噪声
    noise = np.random.randn(H, W) * 5
    noisy = signal + noise
    
    # "去噪"（简单平滑）
    from scipy.ndimage import gaussian_filter
    denoised = gaussian_filter(noisy, sigma=1.5)
    
    # 使用IPYNB风格绘制
    fig_path = os.path.join(PATHS['figures'], 'ipynb_style_demo.png')
    plot_ipynb_style(
        data=noisy,
        denoised=denoised,
        save_path=fig_path,
        data_name='demo',
        vmin=-20,
        vmax=20,
        figsize=(12, 5)
    )
    
    print(f"\n✓ 演示图已保存: {fig_path}")
    print("\n图中展示了:")
    print("  (a) Raw Data - 含噪声的原始数据")
    print("  (b) Denoised Data - 降噪后的数据")
    print("  (c) Removed Noise - 被去除的噪声")
    print("\n注意特点:")
    print("  • 使用地震数据专用色标 (blue-white-red)")
    print("  • Time轴单位为秒 (s)")
    print("  • Trace编号从1开始")
    print("  • 标注位置统一在左上角")
    print("  • Colorbar居右，5个均匀刻度")
    print("=" * 70)


if __name__ == "__main__":
    print("\n" + "🎨" * 35)
    print("         可视化风格对比与演示")
    print("🎨" * 35)
    
    # 首先创建演示图
    create_demo_figure()
    
    # 然后在实际数据上对比
    test_data_paths = sorted([
        os.path.join(PATHS['data'], name)
        for name in os.listdir(PATHS['data'])
        if name.endswith('.npy')
    ])
    
    # 找到可用的数据
    available_data = [p for p in test_data_paths if os.path.exists(p)]
    
    if available_data:
        print(f"\n找到 {len(available_data)} 个测试数据集")
        
        # 在第一个数据集上演示
        print(f"\n使用数据集: {os.path.basename(available_data[0])}")
        compare_visualization_styles(available_data[0], noise_level=0.1)
    else:
        print("\n⚠ 未找到测试数据，仅展示演示图")
        print("请确保data/目录下有以下文件:")
        for path in test_data_paths:
            print(f"  - {path}")
    
    print("\n✨ 完成！")
