# 项目文件总览与协同系统

> **本文档提供**：完整的文件结构、用途、和执行流程说明
> 
> **用途**：快速定位文件、理解项目结构、跟踪实验进度

---

## 📂 项目目录结构

```
das_denoise_3_paper/
│
├── paper_v2.md                         原论文模板（不修改）
├── paper.txt                           初稿备份
│
├── data/                               数据目录（4个实测数据集）
│   ├── eq-36.npy                       ✓ 地震数据集1
│   ├── eq-68.npy                       ✓ 地震数据集2
│   ├── slice_german.npy                ✓ DAS数据集1
│   └── slice_german_1.npy              ✓ DAS数据集2
│
├── new/                                主项目目录
│   │
│   ├─── 📋 论文-项目协同文件 ─────────────────────
│   ├── QUICK_START.md                  ⭐ 快速开始（必读）
│   ├── PAPER_PROJECT_INTEGRATION.md    完整集成指南（详细）
│   ├── paper_v2_INTEGRATED.md          ⭐ 论文模板+填充指南
│   ├── paper_results_extractor.py      自动提取表格脚本
│   ├── run_all_experiments.py          ⭐ 一键实验脚本
│   │
│   ├─── 🏋️ 训练脚本 ──────────────────────
│   ├── train_self_supervised.py        自监督训练（推荐）✓
│   ├── train_2d.py                     监督训练（需要标签）
│   ├── train_low_snr.py                低SNR场景训练
│   ├── train_usl.py                    USL-DIP无监督训练
│   │
│   ├─── 🧪 测试与评估脚本 ──────────────────
│   ├── test_no_reference.py            ⭐ 无参考评估+对比
│   ├── test_2d.py                      详细测试与可视化
│   ├── ablation_runner.py              ⭐ 消融实验框架
│   ├── paper_pipeline.py               论文自动汇总
│   ├── paper_sweeps.py                 参数敏感性分析
│   │
│   ├─── 🏗️ 模型架构 ──────────────────────
│   ├── model_2d.py                     ⭐ Wavelet-Transformer模型
│   ├── baseline_models.py              DnCNN, U-Net基线
│   ├── model_usl.py                    USL-DIP模型
│   │
│   ├─── 🛠️ 工具与配置 ────────────────────
│   ├── utils_2d.py                     数据处理、指标计算
│   ├── config_2d.py                    全局配置参数
│   ├── traditional_denoise.py          8种传统去噪方法
│   ├── data_augmentation.py            数据增强工具
│   ├── optimize_quick.py               快速优化脚本
│   ├── demo_visualization.py           可视化演示
│   │
│   ├─── 📊 文档与说明 ────────────────────
│   ├── README_2D.md                    项目README
│   ├── GUIDE_NO_REFERENCE.md           无参考指标完整指南
│   ├── SUMMARY.md                      项目总结
│   ├── 方法.md                          方法说明（中文）
│   ├── 创新点.md                        创新点分析（中文）
│   ├── 快速开始.md                      快速入门指南（中文）
│   ├── 2D系统总结.md                    系统设计总结（中文）
│   ├── paper_pipeline.md               论文流程说明
│   ├── CHECKLIST.txt                   方法验证检查清单
│   │
│   ├── checkpoints/                    模型权重目录（生成）
│   │   ├── best_model_2d.pth           早期保存的权重
│   │   └── best_*_ss.pth               自监督训练的权重（运行后生成）
│   │       └── best_{model}_{dataset}_ss.pth  命名格式
│   │
│   ├── checkpoints_low_snr/            低SNR模型权重
│   │
│   ├── results/                        实验结果目录（生成）
│   │   ├── *_comparison_no_ref.json    对比实验结果（4个）
│   │   ├── *_metrics.json              单个模型指标
│   │   ├── ablation_*.json             消融实验结果（4个）
│   │   ├── *_denoised.npy              去噪后的数据
│   │   └── ... 其他中间结果
│   │
│   ├── figures/                        可视化图表目录（生成）
│   │   ├── *_comparison.png            去噪对比图
│   │   ├── *_residual.png              残差分析图
│   │   ├── fda_analysis.png            FDA模块分析
│   │   ├── naag_analysis.png           NAAG模块分析
│   │   └── ... 其他分析图表
│   │
│   ├── figures_low_snr/                低SNR场景的图表
│   │
│   ├── paper_outputs/                  论文输出目录
│   │   ├── tables_for_paper.md         自动生成的论文表格
│   │   ├── run_summary.json            实验汇总数据
│   │   ├── figure_manifest.md          图表清单
│   │   ├── tables/                     详细表格数据
│   │   └── figures/                    论文用图表
│   │
│   ├── results_low_snr/                低SNR实验结果
│   │
│   └── origin/                         原始参考代码（备份）
│       ├── model.py
│       └── utils.py
```

---

## 🚀 快速执行流程

### 流程图

```
START
  ↓
检查环境 (QUICK_START.md Step 1)
  ↓
运行实验 (run_all_experiments.py) ← 需要 3-5 小时
  ├─ 阶段I：训练 (train_self_supervised.py × 12次)
  ├─ 阶段II：对比 (test_no_reference.py × 4次)
  └─ 阶段III：消融 (ablation_runner.py × 4次)
  ↓
验证结果 (QUICK_START.md Step 3)
  ├─ checkpoints/best_*_ss.pth (12 files)
  ├─ results/*_comparison_no_ref.json (4 files)
  ├─ results/ablation_*.json (4 files)
  └─ figures/*.png (多个文件)
  ↓
生成表格 (paper_results_extractor.py)
  ├─ paper_comparison_tables.md
  └─ paper_ablation_tables.md
  ↓
填充论文 (paper_v2_INTEGRATED.md)
  ├─ 复制表格数据
  ├─ 插入图表
  └─ 补充文字说明
  ↓
最终检查
  ↓
END (投稿版论文完成)
```

---

## 📖 核心脚本使用说明

### 必用脚本（一定要跑）

#### 1️⃣ run_all_experiments.py ⭐⭐⭐
**用途**：一键运行所有实验流程

```bash
# 完整流程（推荐）
python run_all_experiments.py --stage all

# 分阶段运行
python run_all_experiments.py --stage train      # 仅训练
python run_all_experiments.py --stage eval       # 仅对比
python run_all_experiments.py --stage ablation   # 仅消融

# 输出
experiment_log.txt          # 详细日志
results/*.json              # 实验数据
figures/*.png               # 可视化
checkpoints/*.pth           # 模型权重
```

#### 2️⃣ paper_results_extractor.py ⭐⭐⭐
**用途**：自动从JSON结果生成论文用表格

```bash
# 生成所有表格
python paper_results_extractor.py --mode all

# 仅生成对比表格
python paper_results_extractor.py --mode comparison

# 仅生成消融表格
python paper_results_extractor.py --mode ablation

# 输出
paper_comparison_tables.md     # 对比实验表格（直接复制到论文）
paper_ablation_tables.md       # 消融实验表格（直接复制到论文）
```

#### 3️⃣ train_self_supervised.py ⭐⭐
**用途**：训练模型（自监督，适合无标签数据）

```bash
# 训练我们的方法 (Wavelet-Transformer)
python train_self_supervised.py --model ours --dataset eq-36 --epochs 100

# 训练DnCNN基线
python train_self_supervised.py --model dncnn --dataset eq-36 --epochs 100

# 训练U-Net基线
python train_self_supervised.py --model unet --dataset eq-36 --epochs 100

# 输出
checkpoints/best_ours_eq-36_ss.pth         # 训练完的模型
results/training_log.txt                   # 训练日志
```

#### 4️⃣ test_no_reference.py ⭐⭐
**用途**：推理+对比+无参考评估

```bash
# 推理单个模型+生成指标
python test_no_reference.py --model ours --dataset eq-36

# 与所有基线对比
python test_no_reference.py --model ours --dataset eq-36 --compare-all --include-baselines

# 输出
results/eq-36_ours_metrics.json              # 单个模型指标
results/eq-36_comparison_no_ref.json         # 对比排名
figures/eq-36_ours_comparison.png            # 可视化对比
```

#### 5️⃣ ablation_runner.py ⭐⭐
**用途**：自动运行消融实验

```bash
# 运行消融实验（自动测试7个配置）
python ablation_runner.py --dataset eq-36 --noise 0.1 --epochs 80

# 输出
results/ablation_eq-36.json                 # 7个配置的性能对比
```

---

### 可选脚本（高级用途）

#### 📊 paper_pipeline.py
**用途**：自动汇总论文输出

```bash
python paper_pipeline.py --mode all --datasets auto
# 输出：paper_outputs/ 目录中的汇总表格和图表
```

#### 📈 paper_sweeps.py
**用途**：参数敏感性分析

```bash
python paper_sweeps.py --dataset eq-36
# 输出：patch大小、TV参数等的敏感性曲线
```

#### 🔬 test_2d.py
**用途**：详细的可视化和分析

```bash
# 合成数据测试
python test_2d.py --mode synthetic

# 现场数据测试
python test_2d.py --mode real

# FDA/NAAG可视化
python test_2d.py --plot-fda --plot-naag

# 输出：figures/ 目录中的详细分析图
```

---

## 📊 输出文件说明

### results/ 目录中的关键文件

| 文件名 | 产生脚本 | 内容 | 用途 |
|--------|--------|------|------|
| `*_comparison_no_ref.json` | test_no_reference.py | 对比排名（JSON） | 提取表3数据 |
| `*_metrics.json` | test_no_reference.py | 单个模型指标 | 参考性能 |
| `ablation_*.json` | ablation_runner.py | 7个消融配置的性能 | 提取表5-6数据 |
| `*_denoised.npy` | test_no_reference.py | 去噪后的数据 | 存档/后续分析 |

### figures/ 目录中的关键文件

| 文件名 | 产生脚本 | 内容 | 用途 |
|--------|--------|------|------|
| `*_comparison.png` | test_no_reference.py | 去噪对比图 | 图3-5 |
| `*_residual.png` | test_no_reference.py | 残差分析 | 图4 |
| `fda_analysis.png` | test_2d.py | FDA权重分析 | 图6(a) |
| `naag_analysis.png` | test_2d.py | NAAG权重分析 | 图6(b) |

---

## 📝 论文-项目映射

### 从实验结果到论文位置

| 论文表格 | 数据来源 | 提取命令 | 使用文件 |
|---------|--------|--------|--------|
| **Table 1** - 数据集配置 | 手动 | - | 无（手动填写） |
| **Table 2** - 传统方法参数 | config_2d.py | 查看代码 | traditional_denoise.py |
| **Table 3** - 定量对比 | test_no_reference.py | `paper_results_extractor.py` | paper_comparison_tables.md |
| **Table 4** - 现场数据指标 | test_no_reference.py | `paper_results_extractor.py` | 同上 |
| **Table 5** - 协同策略消融 | ablation_runner.py | `paper_results_extractor.py` | paper_ablation_tables.md |
| **Table 6** - 模块级消融 | ablation_runner.py | `paper_results_extractor.py` | 同上 |
| **Figure 3-5** - 去噪效果 | test_no_reference.py | 直接使用 | figures/*_comparison.png |
| **Figure 6(a)** - FDA分析 | test_2d.py | --plot-fda | figures/fda_analysis.png |
| **Figure 6(b)** - NAAG分析 | test_2d.py | --plot-naag | figures/naag_analysis.png |

---

## 🔍 配置参数速查

### config_2d.py 中的关键参数

```python
# 数据处理
PATCH_SIZE = 24              # Patch大小（像素）
PATCH_STRIDE = 6             # Patch步长（75%重叠）
NORMALIZE_MODE = "percentile"  # 归一化方式
PERCENTILE_RANGE = (1, 99)   # 异常值裁剪范围

# 模型
MODEL_DEPTH = 9              # Wavelet-Transformer的尺度数
USE_FDA = True               # 是否使用频率解耦注意力
USE_NAAG = True              # 是否使用噪声感知门控

# 训练
BATCH_SIZE = 128             # 批量大小
LEARNING_RATE = 1e-3         # 学习率
NUM_EPOCHS = 100             # 训练轮数
MASK_RATIO = 0.05            # 自监督掩码比例

# TV精修
TV_LAMBDA = 0.5              # TV参数
TV_ITERATIONS = 50           # TV迭代次数
```

修改这些参数后，重新运行实验会自动采用新配置。

---

## 🐛 故障排除快速表

| 问题 | 现象 | 解决方案 |
|------|------|--------|
| 数据缺失 | FileNotFoundError: data/*.npy | 检查 `data/` 目录，确保4个.npy文件存在 |
| 依赖缺失 | ModuleNotFoundError | `pip install -r requirements.txt`（如果有） |
| GPU显存不足 | CUDA out of memory | 修改config_2d.py中的BATCH_SIZE为64或32 |
| 训练太慢 | 1小时只完成几个epoch | 正常情况，GPU需3-5小时，CPU需更长时间 |
| 结果异常 | 指标为NaN或负数 | 检查experiment_log.txt，可能数据格式问题 |
| 表格为空 | paper_*_tables.md空白 | 确保results/*.json文件存在且内容正确 |

详细故障排除请参考 `GUIDE_NO_REFERENCE.md`。

---

## 📋 用户检查清单

### 运行前
- [ ] 数据文件都在 `data/` 目录
- [ ] 所有脚本文件都能找到
- [ ] Python环境配置正确
- [ ] GPU可用（或接受CPU训练会很慢）

### 运行中
- [ ] experiment_log.txt 有进度输出
- [ ] 没有❌错误标记在日志中
- [ ] GPU/CPU有占用（证明在运行）

### 运行后
- [ ] checkpoints/ 有12个 .pth 文件
- [ ] results/ 有 *_comparison_no_ref.json 和 ablation_*.json
- [ ] figures/ 有多个 .png 图表
- [ ] paper_*_tables.md 文件有数据

### 填论文前
- [ ] 所有【待填】的地方都有对应的输出文件
- [ ] 表格和图表都能找到来源
- [ ] 定性描述与定量数据相符

---

## 📞 获取帮助

1. **遇到错误**：查看 `experiment_log.txt` 或 `GUIDE_NO_REFERENCE.md`
2. **不确定参数**：查看 `config_2d.py` 和脚本的 --help
3. **想了解细节**：参考 `PAPER_PROJECT_INTEGRATION.md` 详细说明
4. **需要快速上手**：按照 `QUICK_START.md` 的5个步骤执行

---

## 总结

```
核心流程：

1. QUICK_START.md (5分钟) → 理解任务
            ↓
2. run_all_experiments.py (3-5小时) → 运行所有实验
            ↓
3. paper_results_extractor.py (5分钟) → 生成表格
            ↓
4. paper_v2_INTEGRATED.md (1-2小时) → 填充论文
            ↓
5. 最终检查 (30分钟) → 可投稿版本
```

**总耗时**：4-6小时（大部分是实验训练）

**最终产物**：
✅ 12个训练的模型权重
✅ 4组对比实验结果
✅ 4组消融实验结果  
✅ 完整的论文用表格和图表
✅ 一篇可投稿的学术论文

