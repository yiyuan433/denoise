# 论文实验流水线

本文档说明如何在服务器上从训练、评测、消融到论文汇总一步步产出可直接回填到 `paper_v2.md` 的图表与数据。

## 1. 目标

本项目的论文建议按以下结构补齐：

- 相关工作：补齐 DAS/地震数据去噪、传统方法、深度去噪、自监督去噪、两阶段协同去噪。
- 方法：保留两阶段框架，同时明确说明 FDA、NAAG、FAG、Noise Estimator、RDC、Wavelet、Cross-Scale、Sparse Attention 的作用。
- 实验：必须包含传统方法对比、深度基线对比、协同对比、模块消融、TV/patch 敏感性分析、低 SNR 鲁棒性曲线。

## 2. 先确认环境

在服务器上进入 `new/` 目录后，建议先检查 Python 环境和依赖：

```bash
cd new
python -V
python -c "import torch, numpy, matplotlib, scipy"
```

如果 `scikit-image`、`pywt`、`tqdm` 没有装全，先补上：

```bash
pip install scikit-image PyWavelets tqdm
```

## 3. 训练阶段

### 3.1 训练本文主模型

```bash
python train_2d.py --model ours --dataset eq-36
```

如果你想用另一组数据验证泛化：

```bash
python train_2d.py --model ours --dataset eq-68
```

输出：

- `new/checkpoints/best_ours_eq-36.pth` 或 `new/checkpoints/best_ours_eq-68.pth`
- `new/results/train_history_ours_eq-36.json`
- 控制台中的训练损失、验证损失、最佳 checkpoint

这些结果对应论文中的“实现细节”和训练设置说明，不直接贴图，但要在方法或实验小节里写清楚。

### 3.2 训练监督基线

```bash
python train_2d.py --model dncnn --dataset eq-36
python train_2d.py --model unet --dataset eq-36
```

输出：

- `new/checkpoints/best_dncnn_eq-36.pth`
- `new/checkpoints/best_unet_eq-36.pth`
- 对应训练历史 JSON

这些 checkpoint 用于论文里的深度学习基线对比表。

### 3.3 训练 USL-DIP 自监督基线

```bash
python train_usl.py --dataset eq-36 --tag A
```

输出：

- `new/checkpoints/fusion_A_best.pth`
- `new/checkpoints/fusion_A_last.pth`

如果你想把 USL-DIP 放进论文作为自监督基线，这个结果应出现在“深度学习基线对比”一节。

### 3.4 低 SNR 专用训练

如果你准备主打 EQ-36 这类极低 SNR 数据，建议跑：

```bash
python train_low_snr.py
```

输出：

- `new/checkpoints_low_snr/best_model.pth`
- `new/results_low_snr/training_history.json`

这部分更适合写进“低 SNR 鲁棒性”或“补充实验”中。

## 4. 测试与对比阶段

### 4.1 统一测试入口

建议先用批量模式一次性生成主对比结果：

```bash
python test_2d.py --batch-all
```

如果你想手动指定 checkpoint：

```bash
python test_2d.py --checkpoint checkpoints/best_ours_eq-36.pth --batch-all
```

低 SNR checkpoint 则改成：

```bash
python test_2d.py --checkpoint checkpoints_low_snr/best_model.pth --low-snr --batch-all
```

`test_2d.py` 会自动产出以下内容：

- 传统方法纯对比图
- DL Only 与 DL + 传统方法协同图
- TV 专项对比图
- 多传统组合协同图
- 单道波形图
- 指标 JSON
- 去噪后的 `.npy`
- FDA / NAAG 分析图

### 4.2 你论文里应该放哪些图

建议直接从 `new/figures/` 选用以下类型：

- `*_ipynb_style.png`：放到“定性对比”图
- `*_comparison.png`：放到“传统方法对比”或“协同对比”图
- `*_noise_residuals.png`：放到“残差分析”图
- `*_trace_compare.png`：放到“单道波形对比”图
- `*_tv_focused.png`：放到“DL Only vs DL+TV”消融图
- `*_fda_analysis.png`、`*_naag_analysis.png`、`*_fda_naag_combined.png`：放到“模块解释性分析”图

## 5. 论文主流水线

### 5.1 一键汇总主对比和表格

这是最推荐的收尾命令：

```bash
python paper_pipeline.py --mode all --datasets eq-36,eq-68 --noise-levels 0.05,0.1,0.15 --comparison-noise 0.1 --checkpoint checkpoints/best_ours_eq-36.pth --dncnn-ckpt checkpoints/best_dncnn_eq-36.pth --unet-ckpt checkpoints/best_unet_eq-36.pth --usl-ckpt checkpoints/fusion_A_best.pth --collect-ablation --plot-robustness
```

如果你只想先汇总，不想跑新测试，也可以只做 collect：

```bash
python paper_pipeline.py --mode collect --datasets eq-36,eq-68 --comparison-noise 0.1 --dncnn-ckpt checkpoints/best_dncnn_eq-36.pth --unet-ckpt checkpoints/best_unet_eq-36.pth --usl-ckpt checkpoints/fusion_A_best.pth --collect-ablation
```

输出：

- `new/paper_outputs/tables_for_paper.md`
- `new/paper_outputs/figure_manifest.md`
- `new/paper_outputs/run_summary.json`
- `new/figures/*noise_robustness.png`（如果启用 `--plot-robustness`）

### 5.2 这些文件怎么用

- `tables_for_paper.md`：直接复制里面的表格到论文“实验结果”小节。
- `figure_manifest.md`：这是图清单，方便你确认每张图是否已经生成。
- `run_summary.json`：这是运行摘要，写在实验记录里保留。

## 6. 消融实验

### 6.1 结构消融

```bash
python ablation_runner.py --dataset eq-36 --noise 0.1
```

如果显存或时间不够，可以缩短训练：

```bash
python ablation_runner.py --dataset eq-36 --noise 0.1 --epochs 20 --batch-size 64
```

输出：

- `new/results/ablation_eq-36.json`
- 每个 ablation 对应的 checkpoint
- 可用于论文中的“模块级消融”表

建议论文中至少写这些消融项：

- full
- no_fda
- no_naag
- no_wavelet
- no_sparse_attn
- no_rdc
- no_fag

### 6.2 TV 与 patch 敏感性

新脚本 `paper_sweeps.py` 已经准备好，专门做 TV 和 patch 的敏感性分析。

TV 扫描：

```bash
python paper_sweeps.py --mode tv --dataset eq-36 --noise-level 0.1 --checkpoint checkpoints/best_ours_eq-36.pth --plot
```

Patch 扫描：

```bash
python paper_sweeps.py --mode patch --dataset eq-36 --noise-level 0.1 --checkpoint checkpoints/best_ours_eq-36.pth --plot
```

如果你想在真实数据上做无参考敏感性分析：

```bash
python paper_sweeps.py --mode tv --dataset eq-36 --real --checkpoint checkpoints/best_ours_eq-36.pth --plot
python paper_sweeps.py --mode patch --dataset eq-36 --real --checkpoint checkpoints/best_ours_eq-36.pth --plot
```

输出：

- `new/results/eq-36_tv_sweep_noise0.1.json`
- `new/figures/eq-36_tv_sweep_noise0.1.png`
- `new/results/eq-36_patch_sweep_noise0.1.json`
- `new/figures/eq-36_patch_sweep_noise0.1.png`

这些图适合放到“TV 参数敏感性”和“Patch 策略消融”小节。

## 7. 论文中推荐的贴图位置

### 摘要 / 引言

- 不放具体图。
- 只引用方法框架和性能结论，最后用主结果表的结论支撑摘要。

### 方法

- 图1：两阶段框架示意图，建议使用 `paper_v2.md` 中方法描述配合你后续补的流程图。
- 图2：Wavelet-Transformer + FDA + NAAG + TV 的结构图。
- 方法小节里明确说明 `DL -> TV` 的处理顺序。

### 实验设计

- 表1：数据集信息，来自你的数据集统计。
- 表2：传统去噪参数表，来自 `traditional_denoise.py` 默认参数和你实际使用的参数。
- 表3：深度学习基线和协同方法主对比表，来自 `new/paper_outputs/tables_for_paper.md`。
- 图3：合成噪声定性对比图，来自 `new/figures/*_ipynb_style.png` 或 `*_comparison.png`。
- 图4：残差图，来自 `new/figures/*_noise_residuals.png`。

### 结果与分析

- 表4：现场数据无参考指标，来自 `new/paper_outputs/tables_for_paper.md` 中 real data 部分。
- 图5：单道波形和细节放大图，来自 `*_trace_compare.png`。
- 图6：FDA / NAAG 分析图，来自 `*_fda_analysis.png`、`*_naag_analysis.png`、`*_fda_naag_combined.png`。
- 图7：TV / patch 敏感性图，来自 `paper_sweeps.py` 生成的图。
- 表5：两阶段协同消融，来自 `DL Only`、`DL+TV`、`DL+Wiener`、`DL+Wavelet`、`DL+SVD`、`DL+Bandpass` 的对比结果。
- 表6：模块级消融，来自 `ablation_eq-36.json`。

## 8. 建议的论文填充顺序

1. 先跑 `train_2d.py`、`train_usl.py`、`train_low_snr.py`，拿到可用 checkpoint。
2. 再跑 `test_2d.py --batch-all` 生成全部可视化与 JSON。
3. 然后跑 `ablation_runner.py` 生成模块消融。
4. 最后跑 `paper_pipeline.py --mode all ... --collect-ablation --plot-robustness` 生成论文表格和图清单。
5. 需要补敏感性分析时，再跑 `paper_sweeps.py`。

## 9. 你最终会得到什么

只要把上面这条流水线跑完，论文基本可以由以下材料拼成：

- `paper_v2.md`：正文骨架
- `new/figures/`：全部定性图
- `new/results/`：全部定量 JSON 与 `.npy`
- `new/paper_outputs/tables_for_paper.md`：论文表格草稿
- `new/paper_outputs/figure_manifest.md`：图清单
- `new/paper_outputs/run_summary.json`：实验摘要

如果你愿意，下一步最有价值的是把 `paper_v2.md` 继续改成更像正式投稿稿件的版本，把“待填”表格编号和图编号统一起来。