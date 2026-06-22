# DAS Denoising Paper — 完整流水线指南与实验分析

> 本文件基于对项目结构和22轮实验现有结果的全面审查编写。
> 当前日期：2026-06-07

---

## 目录

1. [项目结构概述](#1-项目结构概述)
2. [现有实验状态评估](#2-现有实验状态评估)
3. [关键问题诊断](#3-关键问题诊断)
4. [论文所需实验清单](#4-论文所需实验清单)
5. [流水线执行顺序](#5-流水线执行顺序)
6. [脚本详细说明](#6-脚本详细说明)
7. [论文结果表格与图表规划](#7-论文结果表格与图表规划)
8. [常见问题与排查](#8-常见问题与排查)

---

## 1. 项目结构概述

```
das_denoise_3_paper/
├── paper_v2.md                    # 论文草稿
├── data/                          # 原始数据
│   ├── eq-36.npy / eq-68.npy     # 真实DAS地震数据
│   ├── slice_german_1.npy / slice_german.npy
├── new/                           # 主要工作目录
│   ├── model_2d.py                # 核心模型（FDA+NAAG+Wavelet+SparseAttn+RDC+FAG）
│   ├── model_usl.py               # USL-DIP基线模型
│   ├── train_self_supervised.py   # 自监督训练脚本
│   ├── train_2d.py                # 标准训练脚本（用于消融/基线）
│   ├── train_low_snr.py           # 低SNR训练
│   ├── train_usl.py               # USL训练脚本
│   ├── test_2d.py                 # 核心测试/评估脚本
│   ├── traditional_denoise.py     # 传统方法 + 混合/组合去噪函数
│   ├── config_2d.py               # 默认配置
│   ├── config_low_snr.py          # 低SNR配置
│   ├── utils_2d.py                # 工具函数
│   ├── data_augmentation.py       # 数据增强（FDA相关的频域增强）
│   ├── baseline_models.py         # DnCNN, UNet等基线
│   ├── ablation_runner.py         # 消融实验运行器
│   ├── paper_pipeline.py          # 论文自动化流水线
│   ├── paper_results_extractor.py # 从结果生成论文表格
│   ├── paper_sweeps.py            # 超参数搜索
│   ├── run_all_experiments.py     # 一键运行所有实验
│   ├── demo_visualization.py      # 可视化演示
│   ├── optimize_quick.py          # 快速优化
│   ├── improve_training.py        # 训练改进
│   ├── checkpoints/               # 模型权重
│   │   └── best_model_2d.pth      # 仅有此一个有效checkpoint
│   ├── checkpoints_low_snr/       # 低SNR权重
│   ├── results/                   # 实验结果
│   ├── results_low_snr/           # 低SNR结果
│   ├── paper_outputs/             # 论文输出（图表/表格）
│   ├── figures/                   # 生成的对比图
│   └── figures_low_snr/           # 低SNR对比图
```

---

## 2. 现有实验状态评估

### 2.1 已完成 ✅

| 项目 | 状态 | 详情 |
|------|------|------|
| 模型实现 | ✅ | 完整的 Complex2D 模型（FDA + NAAG + WaveletConn + SparseAttn + RDC + FAG） |
| 训练脚本 | ✅ | `train_self_supervised.py` — 自监督训练流程完整 |
| 传统方法实现 | ✅ | 8种传统方法 + 混合/组合变体 |
| 合成噪声测试 | ✅ | 运行过 `test_2d.py`，生成了 denoised 输出 |
| 真实数据测试 | ✅ | 运行过传统/混合/组合变体 |
| 对比图 | ✅ | `figures/` 下有 40 张图 |
| 论文草稿 | ✅ | `paper_v2.md` 结构完整 |
| 自动化流水线 | ✅ | `paper_pipeline.py`, `paper_results_extractor.py` |

### 2.2 部分完成 ⚠️

| 项目 | 状态 | 详情 |
|------|------|------|
| **合成噪声指标** | ⚠️ 有问题 | 不同噪声水平输出几乎相同，MSE~230（原始~0.01），模型对合成噪声无效 |
| **真实数据指标** | ⚠️ 空 | combo/hybrid/trad 的 JSON 指标文件全是 `{}` |
| **论文表格** | ⚠️ 不完整 | eq-36 合成表只有2行有值，eq-68 真实表完全为空，消融表不存在 |
| **论文图表** | ⚠️ 只有1张 | `paper_outputs/figures/` 下只有 `eq-36_noise_robustness_dl_only.png` |
| **消融实验** | ⚠️ 未运行 | `ablation_eq-36.json` 不存在，无消融 checkpoint |
| **eq-68 结果** | ⚠️ 不完整 | 真实数据指标缺失 |

### 2.3 未完成 ❌

| 项目 | 状态 | 详情 |
|------|------|------|
| **基线模型对比** | ❌ | DnCNN/UNet checkpoint 不存在，无对比数据 |
| **低SNR实验** | ❌ | `results_low_snr/` 为空或未运行，`figures_low_snr/` 为空 |
| **计算成本对比** | ❌ | 无参数/FLOPs/推理时间统计 |
| **全流水线运行** | ❌ | `paper_pipeline.py` 未成功完整运行过 |
| **noise_robustness_hybrid图** | ❌ | 只有 dl_only 版本 |
| **所有真实数据图** | ❌ | 只有合成噪声图 |

---

## 3. 关键问题诊断

### 3.1 合成测试模型输出异常

**现象**：所有噪声水平（0.05/0.1/0.15/0.2）的模型输出几乎相同
（mean_diff < 0.08），但输出幅度非常大（-329~383, clean 数据范围通常 -1~1）。

**根本原因**：
1. **训练/测试不匹配**：模型是自监督训练的（blind-spot），直接在纯噪声上预测。测试时被噪声干扰不敏感。
2. **模型未见过合成噪声**：自监督训练只在真实噪声上学习模式，对合成高斯噪声毫无抵抗力。
3. **模型可能退化为恒等映射**：自监督 loss 在有强噪声时可能使模型学习直接输出输入。

**解决方案**：
- 检查 `test_2d.py` 中合成噪声测试的预处理流程
- 确认模型 inference 时是否使用了正确的 normalization
- 验证训练时的输入范围是否与测试匹配
- 可能需要专门在合成噪声上微调或重新训练

### 3.2 真实数据指标为空

**现象**：`eq-36_combo_real_metrics.json`, `eq-36_hybrid_real_metrics.json`,
`eq-36_trad_real_metrics.json` 等文件内容为 `{}`

**原因**：指标计算代码可能在保存时出错。查看 `test_2d.py` 或 `traditional_denoise.py`
中计算并保存指标的路径。

**排查建议**：
```powershell
python -c "
import json, os
for f in ['eq-36_combo_real_metrics.json', 'eq-36_hybrid_real_metrics.json', 
          'eq-36_trad_real_metrics.json', 'eq-68_combo_real_metrics.json',
          'eq-68_hybrid_real_metrics.json', 'eq-68_trad_real_metrics.json',
          'eq-36_comparison_no_ref.json', 'eq-68_comparison_no_ref.json']:
    path = os.path.join(r'new/results', f)
    if os.path.exists(path):
        data = json.load(open(path))
        print(f'{f}: {type(data).__name__} len={len(data) if isinstance(data, (list,dict)) else \"N/A\"}')
    else:
        print(f'{f}: NOT FOUND')
"
```

### 3.3 无消融实验权重

`checkpoints/` 下仅有 `best_model_2d.pth`，没有以下消融变体：
- `best_ours_eq-36_no_fda.pth`
- `best_ours_eq-36_no_naag.pth`
- `best_ours_eq-36_no_wavelet.pth`
- `best_ours_eq-36_no_sparse_attn.pth`
- `best_ours_eq-36_no_rdc.pth`
- `best_ours_eq-36_no_fag.pth`

需要运行 `ablation_runner.py` 训练所有变体。

---

## 4. 论文所需实验清单

### 4.1 消融实验（6个变体 + 全模型）

需要训练并评估：

| 配置 | 移除模块 | 预期影响 |
|------|----------|----------|
| `full` | 完整模型（baseline） | — |
| `no_fda` | 频域适配器 | 中 |
| `no_naag` | 噪声感知门控 | 小-中 |
| `no_wavelet` | 跨尺度小波连接 | 大 |
| `no_sparse_attn` | 稀疏注意力 | 中 |
| `no_rdc` | 精炼空洞卷积 | 小 |
| `no_fag` | 特征聚合门控 | 小-中 |

### 4.2 基线模型对比（3个基线）

| 基线 | 实现 | 训练方式 |
|------|------|----------|
| DnCNN | `baseline_models.py` | 自监督（同一数据） |
| U-Net | `baseline_models.py` | 自监督（同一数据） |
| USL-DIP | `model_usl.py` | 自监督（同一数据） |

### 4.3 传统方法（8种 × 2数据集）

已完成实现的传统方法：
- Gaussian, Median, Wavelet, FK, Bandpass, SVD, TV, Wiener

混合方法（DL + 传统后缀）和组合方法（DL + 传统1 + 传统2）也已实现。

### 4.4 低SNR实验

目标：验证模型在信噪比极低条件下的鲁棒性。

### 4.5 计算成本表

需要统计：参数量、FLOPs、推理时间（单样本）

---

## 5. 流水线执行顺序

### Step 0: 环境准备

```powershell
cd f:\项目（老师）\denoise\das_denoise_3_paper\new

# 安装依赖
pip install torch numpy matplotlib scipy scikit-image pywavelets tqdm
```

### Step 1: 检查并修复合成测试问题 🔧

**这是最关键的一步——修复后才能获取有意义的合成噪声指标。**

检查 `test_2d.py` 中 `test_on_data` 函数的噪声添加和模型调用逻辑：

```powershell
# 检查模型对合成噪声的响应
python -c "
import torch, numpy as np
from model_2d import create_model_2d
from config_2d import MODEL_2D_CONFIG

model = create_model_2d(MODEL_2D_CONFIG)
ckpt = torch.load(r'checkpoints/best_model_2d.pth', map_location='cpu')
state = ckpt.get('model_state_dict', ckpt)
model.load_state_dict(state)
model.eval()

# 测试简单输入
x = torch.randn(1, 1, 128, 128)
with torch.no_grad():
    out = model(x)
print(f'Input range: {x.min():.4f} to {x.max():.4f}')
print(f'Output range: {out.min():.4f} to {out.max():.4f}')
print(f'Output mean: {out.mean():.4f}, std: {out.std():.4f}')
"
```

如果输出范围明显大于输入范围（如上所述），需要：
1. 检查模型 forward 中是否有 residual scaling 问题
2. 检查输入数据是否被正确归一化
3. 检查训练时的 data range 和测试时是否一致

### Step 2: 训练所有消融模型

```powershell
# 逐个训练消融变体
python train_2d.py --model ours --dataset eq-36 --ablation no_fda      --epochs 100
python train_2d.py --model ours --dataset eq-36 --ablation no_naag     --epochs 100
python train_2d.py --model ours --dataset eq-36 --ablation no_wavelet  --epochs 100
python train_2d.py --model ours --dataset eq-36 --ablation no_sparse_attn --epochs 100
python train_2d.py --model ours --dataset eq-36 --ablation no_rdc      --epochs 100
python train_2d.py --model ours --dataset eq-36 --ablation no_fag      --epochs 100
```

或使用 ablation_runner.py：
```powershell
python ablation_runner.py --dataset eq-36 --epochs 100
```

### Step 3: 训练基线模型

```powershell
# 训练 DnCNN
python train_self_supervised.py --model dncnn --dataset eq-36 --epochs 100

# 训练 U-Net
python train_self_supervised.py --model unet --dataset eq-36 --epochs 100

# 训练 USL-DIP
python train_usl.py --dataset eq-36 --epochs 100
```

### Step 4: 运行全部测试

```powershell
# 在合成噪声上测试完整模型
python test_2d.py --checkpoint checkpoints/best_model_2d.pth --noise-levels 0.05 0.1 0.15 0.2 --synthetic

# 在真实数据上测试完整模型 + 所有传统/混合/组合方法
python test_2d.py --checkpoint checkpoints/best_model_2d.pth --real --all-methods

# 测试消融模型
python test_2d.py --checkpoint checkpoints/best_ours_eq-36_no_fda.pth --dataset eq-36 --synthetic
# ... 为每个消融变体重复
```

### Step 5: 运行低SNR实验

```powershell
python train_low_snr.py --epochs 100
python test_2d.py --checkpoint checkpoints_low_snr/best_model.pth --low-snr --synthetic --real
```

### Step 6: 生成论文表格和图

```powershell
# 从结果提取表格
python paper_results_extractor.py --mode all

# 生成论文图（noise robustness 对比）
python test_2d.py --checkpoint checkpoints/best_model_2d.pth --plot-noise-robustness --all-methods

# 运行完整流水线（如果已修复）
python paper_pipeline.py
```

### Step 7: 生成计算成本对比表

```powershell
python -c "
from model_2d import create_model_2d
import torch

def count_params(model):
    return sum(p.numel() for p in model.parameters())

# 我们的模型
model_ours = create_model_2d(MODEL_2D_CONFIG)
params_ours = count_params(model_ours)

# 基线模型
from baseline_models import create_baseline_model
for name in ['dncnn', 'unet']:
    model_baseline = create_baseline_model(name, ...)
    params_b = count_params(model_baseline)
    print(f'{name}: {params_b/1e6:.2f}M params')

from model_usl import create_usl_model
model_usl = create_usl_model(...)
params_usl = count_params(model_usl)

print(f'Ours: {params_ours/1e6:.2f}M params')

# FLOPs 估算
input_tensor = torch.randn(1, 1, 128, 128)
# 使用 torchprofile 或 fvcore 计算 FLOPs
# pip install fvcore
from fvcore.nn import FlopCountAnalysis
flops = FlopCountAnalysis(model_ours, input_tensor)
print(f'Ours FLOPs: {flops.total()/1e9:.2f}G')
"
```

---

## 6. 脚本详细说明

### 6.1 `train_self_supervised.py` — 自监督训练

```
用法: python train_self_supervised.py [--model ours|dncnn|unet]
                                      [--dataset eq-36|eq-68]
                                      [--epochs 100]
                                      [--lr 0.001]
                                      [--ablation no_fda|no_naag|...]

注意: 目前只支持 ours 模型做消融，dncnn/unet 用此脚本训练基线。
      --ablation 参数只在 --model ours 时生效。
```

### 6.2 `train_2d.py` — 标准训练

```
用法: python train_2d.py [--model ours]
                          [--dataset eq-36|eq-68]
                          [--ablation no_fda|no_naag|no_wavelet|...]
                          [--epochs 100]

注意: 此脚本使用标准有监督训练，需要合成噪声 + 干净数据对。
      适用于消融实验训练。
```

### 6.3 `test_2d.py` — 测试与评估

```
用法: python test_2d.py [--checkpoint <path>]
                        [--dataset eq-36|eq-68]
                        [--synthetic]          # 合成噪声测试
                        [--real]               # 真实数据测试
                        [--all-methods]        # 所有传统/混合/组合方法
                        [--noise-levels 0.05 0.1 0.15 0.2]
                        [--low-snr]            # 低SNR模式
                        [--plot-noise-robustness]  # 生成噪声鲁棒性图

输出:
  - results/*_denoised.npy   去噪结果
  - results/*_metrics.json   指标
  - figures/*.png            对比图
```

### 6.4 `ablation_runner.py` — 消融运行器

```
用法: python ablation_runner.py [--dataset eq-36]
                                [--epochs 100]
                                [--variants no_fda no_naag no_wavelet ...]

功能: 自动训练所有消融变体并收集结果
```

### 6.5 `paper_pipeline.py` — 全自动流水线

```
用法: python paper_pipeline.py [--skip-training]
                               [--datasets eq-36 eq-68]
                               [--checkpoint <path>]

功能: 按顺序执行：消融训练 → 基线训练 → 合成测试 → 真实测试 → 生成图表
      但目前有依赖问题（需要先在 test_2d.py 层面修复合成效能的 bug）
```

### 6.6 `paper_results_extractor.py` — 表格提取

```
用法: python paper_results_extractor.py [--mode all|comparison|ablation|summary]
                                        [--results-dir results]

依赖:
  - results/{dataset}_comparison_no_ref.json  (对比实验结果)
  - results/ablation_{dataset}.json           (消融实验结果)
```

---

## 7. 论文结果表格与图表规划

### 7.1 表格

| 表号 | 标题 | 数据来源 | 所需实验 |
|------|------|----------|----------|
| 表1 | 合成噪声定量对比 (eq-36) | `results/eq-36_test_results.json` | 合成测试 ✓ |
| 表2 | 合成噪声定量对比 (eq-68) | `results/eq-68_test_results.json` | 合成测试 ✓ |
| 表3 | 真实数据无参考指标 (eq-36) | `results/eq-36_comparison_no_ref.json` | 真实数据测试 ✓ |
| 表4 | 真实数据无参考指标 (eq-68) | `results/eq-68_comparison_no_ref.json` | 真实数据测试 ✓ |
| 表5 | 消融实验 (eq-36) | `results/ablation_eq-36.json` | 消融训练+测试 ❌ |
| 表6 | 计算成本对比 | 手动计算 | 模型分析 ❌ |
| 表7 | 低SNR对比 | `results_low_snr/` | 低SNR实验 ❌ |

### 7.2 图表

| 图号 | 标题 | 生成方式 | 状态 |
|------|------|----------|------|
| 图1 | 模型架构图 | 手动绘制/导出 | 需手动 |
| 图2 | eq-36 合成噪声去噪可视化 | `test_2d.py --synthetic --plot` | 已有（figures/） |
| 图3 | eq-68 合成噪声去噪可视化 | 同上 | 已有 |
| 图4 | eq-36 真实数据去噪对比 | `test_2d.py --real --plot` | 已有 |
| 图5 | eq-68 真实数据去噪对比 | 同上 | 已有 |
| 图6 | 噪声鲁棒性 (DL Only) | 已有 | 已生成 but 缺指标 |
| 图7 | 噪声鲁棒性 (DL + 混合) | `test_2d.py --plot-noise-robustness --all-methods` | ❌ |
| 图8 | FDA/NAAG 可视化 | `test_2d.py --vis-fda-naag` | ❌ |
| 图9 | 消融对比柱状图 | `demo_visualization.py` | ❌ |

---

## 8. 常见问题与排查

### 8.1 合成测试输出异常

**现象**：模型对所有噪声水平输出几乎相同

**排查步骤**：
```powershell
# 1. 检查模型的 forward 结构
python -c "
import torch
from model_2d import create_model_2d
from config_2d import MODEL_2D_CONFIG
m = create_model_2d(MODEL_2D_CONFIG)
print(m)
# 检查是否有 residual paths 或 shortcut
# 确认输出层 activation
"

# 2. 验证训练时的数据预处理
grep "test_on_data" test_2d.py | head -20
# 检查是否有 noise scaling 或 normalization

# 3. 使用简单输入测试模型行为
python -c "
import torch, numpy as np
from model_2d import create_model_2d
from config_2d import MODEL_2D_CONFIG
model = create_model_2d(MODEL_2D_CONFIG)
ckpt = torch.load(r'checkpoints/best_model_2d.pth', map_location='cpu')
state = ckpt.get('model_state_dict', ckpt)
model.load_state_dict(state)
model.eval()

# 测试 clean input
x_clean = torch.zeros(1, 1, 128, 128)
x_noisy = x_clean + 0.1 * torch.randn_like(x_clean)
x_very_noisy = x_clean + 0.5 * torch.randn_like(x_clean)

with torch.no_grad():
    out_clean = model(x_clean)
    out_noisy = model(x_noisy)
    out_very_noisy = model(x_very_noisy)

print('On clean input:', out_clean.mean().item(), out_clean.std().item())
print('On noisy input:', out_noisy.mean().item(), out_noisy.std().item())
print('On very noisy:', out_very_noisy.mean().item(), out_very_noisy.std().item())
print('diff: clean vs noisy:', (out_clean - out_noisy).abs().max().item())
"
```

### 8.2 指标文件为空

**现象**：`*_metrics.json` 内容为 `{}`

**排查**：
```powershell
# 检查指标保存逻辑
grep -n "save\|save_as_json\|dump" test_2d.py | head -20

# 检查 `calculate_no_ref_metrics` 返回值
grep -n "def calculate_no_ref_metrics" utils_2d.py
```

### 8.3 模型训练不收敛

**现象**：loss 不下降

**建议**：
- 降低学习率（默认 0.001 → 0.0005）
- 增加 epoch（100 → 200）
- 检查数据增强是否过于激进
- 检查 blind-spot mask ratio 是否合适

### 8.4 对比图不够专业

**建议**：
- 统一配色方案（使用论文级 colormap，如 'seismic', 'RdBu_r'）
- 添加 scale bar
- 统一字体大小
- 调整 `plot_2d_comparison` 中的参数

---

## 附录：紧急修复清单

按优先级排列：

### 🔴 1. 修复合成效能测试问题
- [ ] 检查 `test_2d.py:test_on_data` 中的噪声添加逻辑
- [ ] 检查模型 forward pass 的输入输出范围
- [ ] 测试模型对合成噪声的响应是否合理

### 🔴 2. 修复真实数据指标为空
- [ ] 检查 `calculate_no_ref_metrics` 的调用和保存逻辑
- [ ] 重新运行真实数据测试

### 🟡 3. 运行消融实验
- [ ] 运行 `ablation_runner.py` 或逐个运行 `train_2d.py --ablation ...`
- [ ] 确认所有6个消融 checkpoint 生成
- [ ] 对每个消融变体运行合成测试
- [ ] 运行 `paper_results_extractor.py --mode ablation` 生成消融表

### 🟡 4. 训练基线模型
- [ ] DnCNN 训练
- [ ] U-Net 训练
- [ ] USL-DIP 训练
- [ ] 对每个基线运行合成+真实测试

### 🟢 5. 低SNR实验
- [ ] 运行 `train_low_snr.py`
- [ ] 运行 `test_2d.py --low-snr`

### 🟢 6. 生成缺失的论文图
- [ ] 噪声鲁棒性混合方法图
- [ ] FDA/NAAG 可视化
- [ ] 消融对比柱状图

### 🟢 7. 生成计算成本表
- [ ] 统计各模型参数量
- [ ] 统计各模型FLOPs
- [ ] 测试推理时间

### 🟢 8. 生成完整论文表格
- [ ] 运行 `paper_results_extractor.py --mode all`
- [ ] 验证 `tables_for_paper.md` 完整性
